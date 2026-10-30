#include "transcribe/inference.hpp"
#include "onnx_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace transcribe {
namespace {
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

InferenceError error(InferenceErrorCode code, std::string message) {
    return {code, std::move(message)};
}

ModelOperation validate_definition(const ModelDefinition& definition) {
    if (definition.contract_version != model_contract_version || definition.id.empty()) {
        return error(InferenceErrorCode::InvalidArgument,
            "Registration requires contract version 1 and a nonempty ID");
    }
    if (definition.objective < PitchObjective::Unspecified ||
        definition.objective > PitchObjective::IndependentMultiPitch ||
        (definition.contract == ModelContract::BasicPitchV1 &&
         definition.objective == PitchObjective::Monophonic) ||
        (definition.contract == ModelContract::Yin &&
         definition.objective == PitchObjective::IndependentMultiPitch) ||
        (definition.contract == ModelContract::Spectral &&
         definition.objective == PitchObjective::Monophonic)) {
        return error(InferenceErrorCode::InvalidArgument,
            "Declared training objective conflicts with the selected model adapter");
    }
    return std::monostate{};
}

std::pair<std::string, std::uint32_t> contract_identity(ModelContract contract) {
    switch (contract) {
    case ModelContract::Yin: return {"transcriber.pitch.yin", 1};
    case ModelContract::Spectral: return {"transcriber.pitch.spectral", 1};
    case ModelContract::BasicPitchV1: return {"transcriber.onnx.basic-pitch", 1};
    case ModelContract::LegacyPitchV1: return {"transcriber.onnx.legacy-pitch", 1};
    case ModelContract::Custom: return {};
    }
    return {};
}

InferenceResult<ModelRegistration> validate_provider_definition(
    const std::shared_ptr<IModelProvider>& provider, ModelDefinition definition) {
    const auto definition_validation = validate_definition(definition);
    if (const auto* failure = std::get_if<InferenceError>(&definition_validation)) {
        return *failure;
    }
    try {
        auto contract_result = provider->contract_for(definition);
        if (const auto* failure = std::get_if<InferenceError>(&contract_result)) return *failure;
        auto contract = std::get<std::shared_ptr<const IModelContract>>(std::move(contract_result));
        if (!contract) {
            return error(InferenceErrorCode::ModelLoad,
                "Provider '" + std::string(provider->id()) + "' returned no model contract");
        }
        const auto contract_id = contract->id();
        const auto semantic_version = contract->version();
        if (contract_id.empty() || semantic_version == 0) {
            return error(InferenceErrorCode::InvalidArgument,
                "Provider '" + std::string(provider->id()) +
                "' returned a model contract with an empty ID or zero version");
        }
        const auto contract_validation = contract->validate(definition);
        if (const auto* failure = std::get_if<InferenceError>(&contract_validation)) {
            return *failure;
        }
        return ModelRegistration{std::move(definition), std::string(provider->id()),
            std::string(contract_id), semantic_version};
    } catch (const std::exception& exception) {
        return error(InferenceErrorCode::ModelLoad,
            "Provider '" + std::string(provider->id()) + "' contract validation failed: " +
            exception.what());
    }
}

ModelMetadata describe(const ModelDefinition& definition, bool preview) {
    ModelMetadata m;
    m.definition = definition;
    const auto contract = definition.contract;
    if (contract == ModelContract::Custom) {
        throw std::invalid_argument("Custom contracts must be created by an IModelProvider");
    }
    if (contract == ModelContract::Yin || contract == ModelContract::Spectral) {
        const bool spectral = contract == ModelContract::Spectral;
        m.analysis_sample_rate = preview && !spectral ? 8000 : 11025;
        m.minimum_midi = spectral ? 28 : 36;
        m.maximum_midi = spectral ? 96 : 83;
        m.polyphonic = spectral;
        m.objective = spectral ? PitchObjective::IndependentMultiPitch : PitchObjective::Monophonic;
        m.maximum_polyphony = spectral ? 69 : 1;
        m.window_samples = spectral ? (preview ? 2048 : 4096) : (preview ? 1024 : 2048);
        m.hop_samples = preview ? 512 : 256;
        m.overlap_samples = m.window_samples - m.hop_samples;
        m.output_frame_step_samples = m.hop_samples;
        m.activation_semantics = spectral ? "selected normalized spectral salience" :
            "one-minus-CMND and heuristic voicing; not correctness probabilities";
    } else {
        const bool basic = contract == ModelContract::BasicPitchV1;
        m.analysis_sample_rate = basic ? 22050 : 16000;
        m.minimum_midi = 28;
        m.maximum_midi = 96;
        m.objective = basic ? PitchObjective::IndependentMultiPitch : definition.objective;
        m.polyphonic = m.objective == PitchObjective::IndependentMultiPitch;
        m.maximum_polyphony = m.polyphonic ? 69 : 1;
        m.window_samples = basic ? 43844 : 0;
        m.hop_samples = basic ? 36164 : 0;
        m.overlap_samples = basic ? 7680 : 0;
        m.output_frame_step_samples = basic ? 256 : 160;
        m.activation_semantics = basic ? "unclamped note-head activations; selected/clamped strengths" :
            "explicit legacy 10ms MIDI28-96 contract; softmax mono/sigmoid poly logits or clamped activations";
        m.trimmed_edge_frames = basic ? 15 : 0;
        m.output_head_semantics = basic ?
            std::vector<std::string>{"note: independent pitch activity", "onset: independent attacks",
                                     "contour: three frequency bins per semitone"} :
            std::vector<std::string>{"pitch: caller-declared training objective; optional silence"};
        if (basic) {
            m.heads = {
                {"StatefulPartitionedCall:1", ModelHeadRole::PitchActivity, 22050, 256, 0, 21, 88, 1, true, false},
                {"StatefulPartitionedCall:2", ModelHeadRole::Onset, 22050, 256, 0, 21, 88, 1, false, false},
                {"StatefulPartitionedCall:0", ModelHeadRole::Contour, 22050, 256, 0, 21, 264, 3, false, false}};
        }
    }
    m.temporal_alignment = contract == ModelContract::BasicPitchV1 ?
        "frame origin zero after 3840-sample left pad and 15-frame trim; output hop 256; "
        "window advance 36164; 142 retained frames per window; aligned_model_heads regrids "
        "using actual window positions (legacy mode concatenates retained frames)" :
        "input-relative window starts; uniform output frame step";
    return m;
}

class Adapter final : public ModelBackend {
public:
    explicit Adapter(ModelDefinition definition) : metadata_(describe(definition, false)) {}

    const ModelMetadata& metadata() const noexcept override { return metadata_; }
    ModelResources resources() const noexcept override { return resources_; }
    BackendDiagnostics diagnostics() const override { return diagnostics_; }

    ModelOperation load(const ModelLoadOptions& options) override {
        release();
        diagnostics_.requested_provider = options.provider;
        diagnostics_.device_id = options.device_id;
        const auto start = Clock::now();
        try {
            if (options.device_id < 0) throw std::invalid_argument("Device ID must be nonnegative");
            if (metadata_.definition.contract_version != model_contract_version) {
                throw std::invalid_argument("Unsupported model contract version; expected 1");
            }
            const bool neural = is_neural();
            if (neural && options.fast_preview) {
                throw std::invalid_argument("ONNX does not support preview mode");
            }
            metadata_ = describe(metadata_.definition, options.fast_preview);
            auto active_provider = options.provider;
            auto active_device_id = options.device_id;
            std::string fallback_reason;
            bool fallback_used = false;
            if (!neural && options.provider != OnnxExecutionProvider::Cpu) {
                if (!options.allow_cpu_fallback) {
                    throw std::invalid_argument("Built-in trackers require CPU");
                }
                active_provider = OnnxExecutionProvider::Cpu;
                active_device_id = 0;
                fallback_used = true;
                fallback_reason = "Built-in trackers only support CPU";
            }
            if (neural) {
                try {
                    session_ = detail::open_onnx_session(metadata_.definition, options);
                } catch (const std::exception& requested_error) {
                    if (!options.allow_cpu_fallback || options.provider == OnnxExecutionProvider::Cpu) {
                        throw;
                    }
                    fallback_reason = requested_error.what();
                    auto fallback_options = options;
                    fallback_options.provider = OnnxExecutionProvider::Cpu;
                    fallback_options.device_id = 0;
                    fallback_options.allow_cpu_fallback = false;
                    try {
                        session_ = detail::open_onnx_session(metadata_.definition, fallback_options);
                    } catch (const std::exception& fallback_error) {
                        throw std::runtime_error(
                            "Requested provider '" + inference_provider_name(options.provider) +
                            "' failed: " + fallback_reason + "; CPU fallback also failed: " +
                            fallback_error.what());
                    }
                    active_provider = OnnxExecutionProvider::Cpu;
                    active_device_id = 0;
                    fallback_used = true;
                }
                session_->describe(metadata_);
                for (auto& head : metadata_.heads) {
                    head.available = std::any_of(metadata_.outputs.begin(), metadata_.outputs.end(),
                        [&](const TensorContract& tensor) { return tensor.name == head.tensor_name; });
                }
                resources_.model_file_bytes = std::filesystem::file_size(metadata_.definition.path);
                resources_.live_sessions = 1;
                diagnostics_.runtime_version = detail::onnx_runtime_version();
            } else if (active_device_id != 0) {
                throw std::invalid_argument("Built-in CPU adapter accepts only device ID 0");
            }
            options_ = options;
            resources_.loaded = true;
            resources_.load_seconds = elapsed(start);
            diagnostics_.requested_provider = options.provider;
            diagnostics_.active_provider = active_provider;
            diagnostics_.device_id = options.device_id;
            diagnostics_.active_device_id = active_device_id;
            diagnostics_.cpu_fallback_allowed = options.allow_cpu_fallback;
            diagnostics_.cpu_fallback_disabled =
                options.provider != OnnxExecutionProvider::Cpu && !options.allow_cpu_fallback;
            diagnostics_.cpu_fallback_used = fallback_used;
            diagnostics_.fallback_reason = fallback_reason;
            diagnostics_.message = fallback_used ?
                "Requested provider '" + inference_provider_name(options.provider) +
                    "' failed; loaded using explicit CPU fallback: " + fallback_reason :
                (neural ? "Session loaded with " + inference_provider_name(active_provider) +
                    (diagnostics_.cpu_fallback_disabled ?
                        "; CPU execution-provider fallback is disabled" : "") :
                    "Built-in CPU tracker loaded; incremental inference is not supported by this adapter");
            return std::monostate{};
        } catch (const std::exception& exception) {
            const auto message = "Cannot load model '" + metadata_.definition.id + "' using " +
                inference_provider_name(options.provider) + ": " + exception.what();
            release();
            diagnostics_.requested_provider = options.provider;
            diagnostics_.device_id = options.device_id;
            diagnostics_.cpu_fallback_allowed = options.allow_cpu_fallback;
            diagnostics_.cpu_fallback_disabled =
                options.provider != OnnxExecutionProvider::Cpu && !options.allow_cpu_fallback;
            diagnostics_.message = message;
            return error(InferenceErrorCode::ModelLoad, message);
        }
    }

    void release() noexcept override {
        session_.reset();
        cached_audio_ = {};
        std::vector<Note>().swap(cached_notes_);
        cache_key_.reset();
        resources_ = {};
        diagnostics_ = {};
    }

    InferenceResult<InferenceOutput> infer(
        const AudioBuffer& audio, const InferenceRequest& request) override {
        if (!resources_.loaded) return fail(InferenceErrorCode::NotLoaded, "Load a model before inference");
        if (request.cancellation.stop_requested()) {
            return fail(InferenceErrorCode::Cancelled, "Inference cancelled before execution");
        }
        const auto start = Clock::now();
        try {
            validate(audio, request);
            InferenceOutput output;
            double progress = 0;
            const auto report_progress = [&](double value) {
                if (request.cancellation.stop_requested()) throw std::runtime_error("Inference cancelled");
                progress = std::max(progress, std::clamp(value, 0.0, 1.0));
                if (request.progress) request.progress(progress);
                if (request.cancellation.stop_requested()) throw std::runtime_error("Inference cancelled");
            };
            report_progress(0);
            resources_.result_cache_hit = false;
            const auto key = cache_key(request);
            if (request.reuse_cached_result && cache_key_ && *cache_key_ == key &&
                cached_audio_.sample_rate == audio.sample_rate && cached_audio_.samples == audio.samples &&
                !request.retain_evidence && !request.analysis.evidence_callback &&
                !request.onnx.evidence_callback && !request.analysis.timing_callback &&
                !request.analysis.progress_callback && !request.onnx.progress_callback) {
                output.notes = cached_notes_;
                resources_.result_cache_hit = true;
                resources_.last_input_bytes = audio.samples.size() * sizeof(float);
                resources_.last_evidence_frames = 0;
                report_progress(1);
                ++resources_.completed_inferences;
                resources_.last_inference_seconds = elapsed(start);
                return output;
            }
            auto analysis = request.analysis;
            analysis.polyphonic = metadata_.definition.contract == ModelContract::Spectral;
            analysis.fast_preview = options_.fast_preview;
            analysis.cancellation = request.cancellation;
            const auto analysis_progress = analysis.progress_callback;
            analysis.progress_callback = [&](double value) {
                report_progress(value);
                if (analysis_progress) analysis_progress(value);
            };
            const auto analysis_evidence = analysis.evidence_callback;
            auto onnx = request.onnx;
            if (metadata_.definition.contract == ModelContract::LegacyPitchV1) {
                if (metadata_.objective != PitchObjective::Unspecified) {
                    onnx.objective = metadata_.objective;
                }
            }
            onnx.execution_provider = options_.provider;
            onnx.device_id = options_.device_id;
            onnx.cancellation = request.cancellation;
            const auto onnx_progress = onnx.progress_callback;
            onnx.progress_callback = [&](double value) {
                report_progress(value);
                if (onnx_progress) onnx_progress(value);
            };
            const auto onnx_evidence = onnx.evidence_callback;
            const auto observe = [&](const FrameEvidenceTrack& track) {
                if (request.retain_evidence) output.evidence.push_back(track);
                if (is_neural()) {
                    if (onnx_evidence) onnx_evidence(track);
                } else if (analysis_evidence) analysis_evidence(track);
            };
            if (request.retain_evidence || analysis_evidence || onnx_evidence) {
                analysis.evidence_callback = observe;
                onnx.evidence_callback = observe;
            }
            output.notes = session_ ? session_->infer(audio, onnx) :
                transcribe_monophonic(audio, analysis);
            if (request.retain_evidence && !output.evidence.empty() &&
                (metadata_.polyphonic ||
                 (onnx.polyphonic && onnx.objective == PitchObjective::IndependentMultiPitch))) {
                output.activations = pitch_activation_grid(output.evidence,
                    metadata_.objective == PitchObjective::Unspecified ?
                        onnx.objective : metadata_.objective);
            }
            for (const auto& note : output.notes) {
                if (note.midi_pitch < metadata_.minimum_midi || note.midi_pitch > metadata_.maximum_midi ||
                    !std::isfinite(note.start_seconds) || note.start_seconds < 0 ||
                    !std::isfinite(note.duration_seconds) || note.duration_seconds <= 0 ||
                    !std::isfinite(note.confidence)) {
                    throw std::runtime_error("Backend returned a note outside its pitch/time/evidence contract");
                }
            }
            if (request.reuse_cached_result && !request.retain_evidence &&
                    !request.analysis.evidence_callback && !request.onnx.evidence_callback &&
                    !request.analysis.timing_callback && !request.analysis.progress_callback &&
                    !request.onnx.progress_callback) {
                    std::size_t bytes = audio.samples.size() * sizeof(float);
                    bool fits = bytes <= options_.maximum_result_cache_bytes;
                    for (const auto& note : output.notes) {
                        const auto size = sizeof(Note) + note.name.size() + 1;
                        if (!fits || size > options_.maximum_result_cache_bytes - bytes) {
                            fits = false; break;
                        }
                        bytes += size;
                    }
                    if (fits) {
                        auto next_audio = audio;
                        auto next_notes = output.notes;
                        bytes = next_audio.samples.capacity() * sizeof(float) +
                            next_notes.capacity() * sizeof(Note);
                        fits = bytes <= options_.maximum_result_cache_bytes;
                        for (const auto& note : next_notes) {
                            const auto size = note.name.capacity() + 1;
                            if (!fits || size > options_.maximum_result_cache_bytes - bytes) {
                                fits = false;
                                break;
                            }
                            bytes += size;
                        }
                        if (fits) {
                            std::swap(cached_audio_, next_audio);
                            cached_notes_.swap(next_notes);
                            cache_key_ = key;
                            resources_.cached_bytes = bytes;
                        }
                    }
            }
            report_progress(1);
            resources_.last_input_bytes = audio.samples.size() * sizeof(float);
            resources_.last_evidence_frames = 0;
            for (const auto& track : output.evidence) resources_.last_evidence_frames += track.frames.size();
            ++resources_.completed_inferences;
            resources_.last_inference_seconds = elapsed(start);
            diagnostics_.message = "Inference completed";
            return output;
        } catch (const std::invalid_argument& exception) {
            return fail(InferenceErrorCode::InvalidArgument, exception.what());
        } catch (const std::exception& exception) {
            return fail(request.cancellation.stop_requested() ? InferenceErrorCode::Cancelled :
                InferenceErrorCode::Inference, "Model '" + metadata_.definition.id + "': " + exception.what());
        }
    }

    ModelOperation warm_up(const InferenceRequest& request) override {
        const auto rate = metadata_.analysis_sample_rate;
        AudioBuffer silence{rate, std::vector<float>(std::max<std::size_t>(
            metadata_.window_samples, static_cast<std::size_t>(rate) * 2), 0)};
        const auto result = infer(silence, request);
        if (const auto* failure = std::get_if<InferenceError>(&result)) return *failure;
        return std::monostate{};
    }

private:
    bool is_neural() const {
        return metadata_.definition.contract == ModelContract::BasicPitchV1 ||
            metadata_.definition.contract == ModelContract::LegacyPitchV1;
    }
    InferenceError fail(InferenceErrorCode code, const std::string& message) {
        diagnostics_.message = message;
        return error(code, message);
    }
    void validate(const AudioBuffer& audio, const InferenceRequest& request) const {
        if (options_.maximum_input_bytes &&
            audio.samples.size() > options_.maximum_input_bytes / sizeof(float)) {
            throw std::invalid_argument(
                "Audio exceeds configured maximum_input_bytes; use bounded WAV inference or raise the explicit limit");
        }
        if (audio.sample_rate < metadata_.minimum_input_sample_rate ||
            (is_neural() && audio.samples.empty())) {
            throw std::invalid_argument("Expected mono audio sampled at >=2000 Hz; ONNX requires nonempty input");
        }
        if (std::any_of(audio.samples.begin(), audio.samples.end(),
                       [](float sample) { return !std::isfinite(sample); })) {
            throw std::invalid_argument("Audio contains nonfinite samples");
        }
        const auto& a = request.analysis;
        const auto& o = request.onnx;
        if (o.note_decoder != NeuralNoteDecoder::Legacy &&
            (o.note_decoder != NeuralNoteDecoder::MultiHeadEvents ||
             metadata_.definition.contract != ModelContract::BasicPitchV1 || !o.polyphonic)) {
            throw std::invalid_argument("Multi-head event decoding requires polyphonic BasicPitchV1");
        }
        if (metadata_.definition.contract == ModelContract::Yin && a.polyphonic) {
            throw std::invalid_argument("The selected YIN adapter is monophonic; select builtin:spectral for polyphony");
        }
        if (is_neural() && o.polyphonic && !metadata_.polyphonic &&
            !(metadata_.objective == PitchObjective::Unspecified &&
              o.objective == PitchObjective::IndependentMultiPitch)) {
            throw std::invalid_argument(
                "Polyphony requires an explicitly declared independent multi-pitch training objective; "
                "tensor pitch dimensions alone do not establish polyphonic capability");
        }
        if (is_neural() && (o.execution_provider != options_.provider || o.device_id != options_.device_id)) {
            throw std::invalid_argument("Inference provider/device must match the explicitly loaded session");
        }
        const auto valid_ratio = [](double v) { return std::isfinite(v) && v >= 0 && v <= 1; };
        if (!valid_ratio(a.minimum_confidence) || !valid_ratio(a.spectral_threshold) ||
            a.spectral_threshold == 0 || a.maximum_pitches_per_frame == 0 ||
            !valid_ratio(o.minimum_confidence) || !valid_ratio(o.polyphonic_threshold) ||
            o.maximum_pitches_per_frame == 0 || o.maximum_pitches_per_frame > 69 ||
            (metadata_.definition.contract == ModelContract::Spectral &&
             a.maximum_pitches_per_frame > 69)) {
            throw std::invalid_argument("Invalid confidence/activation threshold or polyphony (expected 1-69)");
        }
    }
    ModelMetadata metadata_;
    ModelLoadOptions options_;
    ModelResources resources_;
    BackendDiagnostics diagnostics_;
    std::unique_ptr<detail::OnnxSession> session_;
    using CacheKey = std::tuple<double, bool, std::size_t, bool, bool, InstrumentPrior,
        std::size_t, double, SpectralDecoder, bool, bool, double, double,
        std::size_t, PitchObjective, bool, NeuralNoteDecoder>;
    static CacheKey cache_key(const InferenceRequest& request) {
        const auto& a = request.analysis;
        const auto& o = request.onnx;
        return {a.minimum_confidence, a.parallel, a.parallel_threshold, a.fast_preview,
            a.polyphonic, a.instrument, a.maximum_pitches_per_frame, a.spectral_threshold,
            a.spectral_decoder, o.polyphonic, o.independent_probabilities, o.minimum_confidence,
            o.polyphonic_threshold, o.maximum_pitches_per_frame, o.objective, o.aligned_model_heads,
            o.note_decoder};
    }
    AudioBuffer cached_audio_;
    std::vector<Note> cached_notes_;
    std::optional<CacheKey> cache_key_;
};
}

InferenceResult<InferenceOutput> IInferenceSession::infer_incremental(
    const AudioBuffer&, bool, const InferenceRequest&) {
    return InferenceError{InferenceErrorCode::Unsupported,
        "This adapter has no incremental contract; offline windows are not streaming inference"};
}

ModelRegistry::ModelRegistry(const std::filesystem::path& onnx_path) {
    const auto factory = [](const ModelDefinition& d) { return std::make_unique<Adapter>(d); };
    (void)add({1, "builtin:yin", ModelContract::Yin, {}, "Transcriber YIN implementation", "MIT"},
        factory, "builtin");
    (void)add({1, "builtin:spectral", ModelContract::Spectral, {}, "Transcriber spectral implementation", "MIT"},
        factory, "builtin");
    if (!onnx_path.empty()) {
        (void)add_onnx({1, "onnx:basic-pitch-v1", ModelContract::BasicPitchV1, onnx_path,
            "Caller-supplied ONNX file; Basic Pitch v1 tensor contract", "Verify license in model distribution"});
        (void)add_onnx({1, "onnx:legacy-pitch-v1", ModelContract::LegacyPitchV1, onnx_path,
            "Caller-supplied ONNX file; legacy pitch v1 contract", "Verify license in model distribution"});
    }
}

ModelOperation ModelRegistry::add(
    ModelDefinition definition, Factory factory, std::string provider_id) {
    const auto definition_validation = validate_definition(definition);
    if (const auto* failure = std::get_if<InferenceError>(&definition_validation)) {
        return *failure;
    }
    if (!factory || provider_id.empty()) {
        return error(InferenceErrorCode::InvalidArgument,
            "Registration requires a factory and a nonempty provider ID");
    }
    if (definition.contract == ModelContract::Custom) {
        return error(InferenceErrorCode::InvalidArgument,
            "Custom contracts must be registered through IModelProvider");
    }
    if (std::any_of(providers_.begin(), providers_.end(), [&](const auto& provider) {
        return provider->id() == provider_id;
    })) {
        return error(InferenceErrorCode::InvalidArgument,
            "Provider ID is already registered: " + provider_id);
    }
    if (std::any_of(entries_.begin(), entries_.end(), [&](const auto& e) {
        return e.definition.id == definition.id;
    })) {
        return error(InferenceErrorCode::InvalidArgument,
            "Duplicate model ID: " + definition.id);
    }
    entries_.push_back({std::move(definition), std::move(factory), std::move(provider_id)});
    return std::monostate{};
}

ModelOperation ModelRegistry::add_onnx(ModelDefinition definition) {
    if ((definition.contract != ModelContract::BasicPitchV1 &&
         definition.contract != ModelContract::LegacyPitchV1) ||
        definition.path.empty() || definition.provenance.empty() || definition.license.empty()) {
        return error(InferenceErrorCode::InvalidArgument,
            "ONNX registration requires an explicit supported contract, path, provenance and license");
    }
    return add(std::move(definition),
        [](const ModelDefinition& d) { return std::make_unique<Adapter>(d); }, "onnxruntime");
}

ModelOperation ModelRegistry::register_provider(std::shared_ptr<IModelProvider> provider) {
    if (!provider) {
        return error(InferenceErrorCode::InvalidArgument, "Cannot register a null model provider");
    }
    const std::string provider_id(provider->id());
    if (provider_id.empty()) {
        return error(InferenceErrorCode::InvalidArgument, "Model provider ID must not be empty");
    }
    if (std::any_of(providers_.begin(), providers_.end(), [&](const auto& existing) {
        return existing->id() == provider_id;
    }) || std::any_of(entries_.begin(), entries_.end(), [&](const auto& entry) {
        return entry.provider_id == provider_id;
    })) {
        return error(InferenceErrorCode::InvalidArgument, "Duplicate model provider ID: " + provider_id);
    }
    providers_.push_back(std::move(provider));
    return std::monostate{};
}

InferenceResult<std::vector<ModelRegistration>> ModelRegistry::discover_models() const {
    std::vector<ModelRegistration> registrations;
    registrations.reserve(entries_.size());
    for (const auto& entry : entries_) {
        const auto [contract_id, semantic_version] = contract_identity(entry.definition.contract);
        if (contract_id.empty() || semantic_version == 0) {
            return error(InferenceErrorCode::Unsupported,
                "No built-in semantic contract is registered for model '" + entry.definition.id + "'");
        }
        registrations.push_back(
            {entry.definition, entry.provider_id, contract_id, semantic_version});
    }
    for (const auto& provider : providers_) {
        InferenceResult<std::vector<ModelDefinition>> discovered =
            error(InferenceErrorCode::ModelLoad, "Model provider discovery did not complete");
        try {
            discovered = provider->discover_models();
        } catch (const std::exception& exception) {
            return error(InferenceErrorCode::ModelLoad,
                "Provider '" + std::string(provider->id()) + "' discovery failed: " + exception.what());
        }
        if (const auto* failure = std::get_if<InferenceError>(&discovered)) return *failure;
        for (auto& definition : std::get<std::vector<ModelDefinition>>(discovered)) {
            auto validated = validate_provider_definition(provider, std::move(definition));
            if (const auto* failure = std::get_if<InferenceError>(&validated)) return *failure;
            auto registration = std::get<ModelRegistration>(std::move(validated));
            if (std::any_of(registrations.begin(), registrations.end(), [&](const auto& existing) {
                return existing.provider_id == registration.provider_id &&
                    existing.definition.id == registration.definition.id;
            })) {
                return error(InferenceErrorCode::InvalidArgument,
                    "Duplicate model ID '" + registration.definition.id + "' for provider '" +
                    registration.provider_id + "'");
            }
            registrations.push_back(std::move(registration));
        }
    }
    return registrations;
}

InferenceResult<std::unique_ptr<IInferenceSession>> ModelRegistry::create_session(
    std::string_view model_id, std::string_view provider_id) const {
    if (model_id.empty()) {
        return error(InferenceErrorCode::InvalidArgument, "Model ID must not be empty");
    }
    struct Candidate {
        ModelRegistration registration;
        const Entry* entry{};
        std::shared_ptr<IModelProvider> provider;
    };
    std::vector<Candidate> candidates;
    for (const auto& entry : entries_) {
        if (entry.definition.id != model_id ||
            (!provider_id.empty() && entry.provider_id != provider_id)) {
            continue;
        }
        const auto [contract_id, semantic_version] = contract_identity(entry.definition.contract);
        candidates.push_back({
            {entry.definition, entry.provider_id, contract_id, semantic_version}, &entry, {}});
    }
    for (const auto& provider : providers_) {
        if (!provider_id.empty() && provider->id() != provider_id) continue;
        InferenceResult<std::vector<ModelDefinition>> discovered =
            error(InferenceErrorCode::ModelLoad, "Model provider discovery did not complete");
        try {
            discovered = provider->discover_models();
        } catch (const std::exception& exception) {
            return error(InferenceErrorCode::ModelLoad,
                "Provider '" + std::string(provider->id()) + "' discovery failed: " + exception.what());
        }
        if (const auto* failure = std::get_if<InferenceError>(&discovered)) return *failure;
        for (auto& definition : std::get<std::vector<ModelDefinition>>(discovered)) {
            if (definition.id != model_id) continue;
            auto validated = validate_provider_definition(provider, std::move(definition));
            if (const auto* failure = std::get_if<InferenceError>(&validated)) return *failure;
            candidates.push_back({std::get<ModelRegistration>(std::move(validated)), nullptr, provider});
        }
    }
    if (candidates.empty()) {
        return error(InferenceErrorCode::InvalidArgument,
            provider_id.empty() ?
                "Unknown model ID: " + std::string(model_id) :
                "Model '" + std::string(model_id) + "' is not registered by provider '" +
                    std::string(provider_id) + "'");
    }
    if (candidates.size() > 1) {
        std::string message = "Model ID '" + std::string(model_id) +
            "' is available from multiple providers; specify provider_id:";
        for (const auto& candidate : candidates) message += " " + candidate.registration.provider_id;
        return error(InferenceErrorCode::InvalidArgument, std::move(message));
    }
    auto& candidate = candidates.front();
    try {
        InferenceResult<std::unique_ptr<IInferenceSession>> created =
            error(InferenceErrorCode::ModelLoad, "Model provider did not create a session");
        if (candidate.entry) {
            auto session = candidate.entry->factory(candidate.entry->definition);
            if (!session) {
                return error(InferenceErrorCode::ModelLoad,
                    "Model factory returned null: " + std::string(model_id));
            }
            created = std::move(session);
        } else {
            created = candidate.provider->create_session(candidate.registration.definition);
        }
        if (const auto* failure = std::get_if<InferenceError>(&created)) return *failure;
        auto session = std::get<std::unique_ptr<IInferenceSession>>(std::move(created));
        if (!session) {
            return error(InferenceErrorCode::ModelLoad,
                "Model provider '" + candidate.registration.provider_id + "' returned a null session");
        }
        const auto& actual = session->metadata().definition;
        const auto& expected = candidate.registration.definition;
        if (actual.id != expected.id || actual.contract_version != expected.contract_version ||
            actual.contract != expected.contract || actual.path != expected.path ||
            actual.provenance != expected.provenance || actual.license != expected.license ||
            actual.objective != expected.objective) {
            return error(InferenceErrorCode::ModelLoad,
                "Model provider '" + candidate.registration.provider_id +
                "' returned a session with mismatched model definition or contract");
        }
        return session;
    } catch (const std::exception& exception) {
        return error(InferenceErrorCode::ModelLoad,
            "Model provider '" + candidate.registration.provider_id + "' session creation failed: " +
            exception.what());
    }
}

std::vector<ModelDefinition> ModelRegistry::discover() const {
    std::vector<ModelDefinition> definitions;
    auto discovered = discover_models();
    if (const auto* failure = std::get_if<InferenceError>(&discovered)) {
        throw std::runtime_error(failure->message);
    }
    definitions.reserve(std::get<std::vector<ModelRegistration>>(discovered).size());
    for (const auto& registration : std::get<std::vector<ModelRegistration>>(discovered)) {
        definitions.push_back(registration.definition);
    }
    return definitions;
}

InferenceResult<std::unique_ptr<IInferenceSession>> ModelRegistry::create(const std::string& id) const {
    return create_session(id);
}

std::string inference_provider_name(OnnxExecutionProvider provider) {
    switch (provider) {
    case OnnxExecutionProvider::Cpu: return "cpu";
    case OnnxExecutionProvider::Cuda: return "cuda";
    case OnnxExecutionProvider::DirectML: return "directml";
    case OnnxExecutionProvider::Rocm: return "rocm";
    case OnnxExecutionProvider::TensorRT: return "tensorrt";
    }
    return "unknown";
}

std::vector<ProviderAvailability> inference_providers() { return detail::onnx_providers(); }
}
