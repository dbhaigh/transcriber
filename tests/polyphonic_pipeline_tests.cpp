#include "transcribe/engine.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <limits>
#include <numbers>
#include <cmath>
#include <set>
#include <stdexcept>

namespace {
using namespace transcribe;

PitchActivationGrid grid(std::size_t frames = 100) {
    PitchActivationGrid result;
    result.minimum_midi = 60;
    result.maximum_midi = 72;
    result.hop_seconds = .01;
    result.semantics = EvidenceSemantics::RelativeActivation;
    result.objective = PitchObjective::IndependentMultiPitch;
    result.method = "fixture";
    for (std::size_t f = 0; f < frames; ++f) result.timestamps.push_back(f * .01);
    result.values.resize(frames * 13);
    return result;
}

void activity(PitchActivationGrid& g, int pitch, std::size_t begin, std::size_t end, double strength = .8) {
    const auto columns = static_cast<std::size_t>(g.maximum_midi - g.minimum_midi + 1);
    for (auto f = begin; f < end; ++f)
        g.values[f * columns + static_cast<std::size_t>(pitch - g.minimum_midi)] = strength;
}

PolyphonicTranscription decode(const PitchActivationGrid& g, const PolyphonicDecodingOptions& options = {}) {
    return decode_polyphonic(g, 1000, static_cast<std::uint64_t>(g.timestamps.size() * 10),
                             options, "audio", "model");
}

CanonicalNoteEvent note(std::string id, int pitch, double onset, double offset) {
    CanonicalNoteEvent e;
    e.id = std::move(id);
    e.midi_pitch = pitch;
    e.onset_seconds = onset;
    e.offset_seconds = offset;
    return e;
}
}

TEST_CASE("polyphonic tracker preserves independent simultaneous and overlapping intervals",
          "[polyphonic-pipeline]") {
    auto g = grid();
    activity(g, 60, 10, 90);
    activity(g, 64, 10, 50);
    activity(g, 67, 25, 70);
    const auto result = decode(g);
    REQUIRE(result.events.size() == 3);
    const auto& c = result.events[0];
    REQUIRE(c.midi_pitch == 60);
    REQUIRE(c.onset_sample == 100);
    REQUIRE(c.offset_sample == 900);
    REQUIRE(c.audio_source_id == "audio");
    REQUIRE(c.source_model == "model");
    REQUIRE_FALSE(c.calibrated_confidence);
    REQUIRE(c.evidence.front().strength_semantics == EvidenceSemantics::RelativeActivation);
    REQUIRE(result.events[1].offset_sample == 500);
    REQUIRE(result.events[2].onset_sample == 250);
    REQUIRE(result.events[2].offset_sample == 700);
    const auto chord = std::find_if(result.chords.begin(), result.chords.end(),
        [](const auto& candidate) { return candidate.name == "C"; });
    REQUIRE(chord != result.chords.end());
    REQUIRE(chord->midi_pitches == std::vector<int>{60, 64, 67});
    REQUIRE(chord->note_ids.size() == 3);
    REQUIRE(chord->onset_seconds == Catch::Approx(.25));
    REQUIRE(chord->offset_seconds == Catch::Approx(.5));
}

TEST_CASE("sustain hysteresis bridges brief holes but offsets exclude release debounce",
          "[polyphonic-pipeline]") {
    auto g = grid();
    activity(g, 60, 5, 15, .8);
    activity(g, 60, 15, 40, .3);
    activity(g, 60, 43, 70, .3);
    auto result = decode(g);
    REQUIRE(result.events.size() == 1);
    REQUIRE(result.events.front().onset_sample == 50);
    REQUIRE(result.events.front().offset_sample == 700);
    activity(g, 60, 80, 100);
    result = decode(g);
    REQUIRE(result.events.size() == 2);
    REQUIRE(result.events.back().onset_sample == 800);
    REQUIRE(result.events.back().offset_sample == 1000);
}

TEST_CASE("per-pitch onset edges retrigger only the attacked pitch and do not split a plateau",
          "[polyphonic-pipeline]") {
    auto g = grid();
    activity(g, 60, 10, 90);
    activity(g, 64, 10, 90);
    g.onsets.resize(g.values.size());
    for (const int pitch : {60, 64})
        for (std::size_t f = 10; f < 14; ++f) g.onsets[f * 13 + pitch - 60] = .9;
    g.onsets[50 * 13] = .9;
    const auto result = decode(g);
    REQUIRE(result.events.size() == 3);
    REQUIRE(result.events[0].midi_pitch == 60);
    REQUIRE(result.events[0].offset_sample == 500);
    REQUIRE(result.events[1].midi_pitch == 64);
    REQUIRE(result.events[1].offset_sample == 900);
    REQUIRE(result.events[2].midi_pitch == 60);
    REQUIRE(result.events[2].onset_sample == 500);
    REQUIRE(result.events[2].offset_sample == 900);
    g.onsets.assign(g.values.size(), 0);
    REQUIRE(decode(g).events.empty());
}

TEST_CASE("polyphonic minimum duration silence truncation and frame origin are explicit",
          "[polyphonic-pipeline]") {
    auto g = grid();
    REQUIRE(decode(g).events.empty());
    activity(g, 60, 0, 3);
    REQUIRE(decode(g).events.empty());
    REQUIRE(decode(g).discarded_short_notes == 1);
    activity(g, 64, 0, 100);
    const auto result = decode(g);
    REQUIRE(result.events.size() == 1);
    REQUIRE(result.events.front().onset_sample == 0);
    REQUIRE(result.events.front().offset_sample == 1000);
    g.window_seconds = .1;
    const auto centered = decode(g);
    REQUIRE(centered.events.front().onset_sample == 50);
    REQUIRE(centered.events.front().offset_sample == 1000);
}

TEST_CASE("polyphony admission preserves sustained notes and reports capacity loss",
          "[polyphonic-pipeline]") {
    auto g = grid();
    activity(g, 60, 0, 100, .6);
    activity(g, 64, 0, 100, .8);
    activity(g, 67, 20, 100, 1);
    PolyphonicDecodingOptions options;
    options.maximum_polyphony = 1;
    const auto result = decode(g, options);
    REQUIRE(result.events.size() == 1);
    REQUIRE(result.events.front().midi_pitch == 64);
    REQUIRE(result.refused_note_starts > 0);
    REQUIRE(result.events.front().offset_sample == 1000);
}

TEST_CASE("polyphonic grids reject invalid objectives values shapes budgets and cancellation",
          "[polyphonic-pipeline]") {
    auto g = grid();
    PolyphonicDecodingOptions options;
    options.activation_off = .9;
    REQUIRE_THROWS_AS(decode(g, options), std::invalid_argument);
    options = {};
    options.maximum_grid_values = 10;
    REQUIRE_THROWS_AS(decode(g, options), std::length_error);
    options = {};
    options.maximum_audio_seconds = .5;
    REQUIRE_THROWS_AS(decode(g, options), std::length_error);
    options = {};
    options.maximum_audio_bytes = 16;
    REQUIRE_THROWS_AS(decode(g, options), std::length_error);
    options = {};
    options.maximum_events = 1;
    activity(g, 60, 0, 80);
    activity(g, 64, 0, 80);
    REQUIRE_THROWS_AS(decode(g, options), std::length_error);
    g.objective = PitchObjective::Monophonic;
    REQUIRE_THROWS_AS(decode(g), std::invalid_argument);
    g.objective = PitchObjective::IndependentMultiPitch;
    g.values[0] = 2;
    REQUIRE_THROWS_AS(decode(g), std::invalid_argument);
    g.values[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(decode(g), std::invalid_argument);
    g = grid();
    g.values.pop_back();
    REQUIRE_THROWS_AS(decode(g), std::invalid_argument);
    g = grid();
    g.timestamps.back() = 1.0;
    REQUIRE_THROWS_AS(decode(g), std::invalid_argument);
    REQUIRE_THROWS_AS(decode_polyphonic(grid(), 0, 1000, {}, "a", "m"), std::invalid_argument);
    std::stop_source stop;
    stop.request_stop();
    REQUIRE_THROWS_AS(decode_polyphonic(grid(), 1000, 1000, {}, "a", "m", stop.get_token()),
                      std::runtime_error);
}

TEST_CASE("chord templates identify roots qualities inversions and preserve unknown collections",
          "[polyphonic-pipeline]") {
    const auto recognize = [](std::vector<int> pitches) {
        std::vector<CanonicalNoteEvent> notes;
        for (std::size_t i = 0; i < pitches.size(); ++i)
            notes.push_back(note(std::to_string(i), pitches[i], .1, .9));
        return identify_chords(notes).front();
    };
    REQUIRE(recognize({60, 64, 67}).name == "C");
    const auto inversion = recognize({64, 67, 72});
    REQUIRE(inversion.name == "C/E");
    REQUIRE(inversion.inversion == 1);
    REQUIRE(inversion.bass_midi_pitch == 64);
    REQUIRE(recognize({57, 60, 64}).name == "Am");
    REQUIRE(recognize({55, 59, 62, 65}).name == "G7");
    REQUIRE(recognize({60, 64, 67, 71}).quality == "major-seventh");
    REQUIRE(recognize({60, 62, 67}).name == "Csus2");
    REQUIRE(recognize({60, 65, 67}).name == "Csus4");
    REQUIRE(recognize({60, 63, 66}).name == "Cdim");
    REQUIRE(recognize({60, 64, 68}).name == "Caug");
    REQUIRE(recognize({60, 67}).name == "C5");
    const auto unknown = recognize({60, 61, 66});
    REQUIRE(unknown.name == "unknown");
    REQUIRE_FALSE(unknown.root_pitch_class);
    REQUIRE(recognize({60, 64, 67, 72}).name == "C");
}

TEST_CASE("chords use half-open overlap intervals and validate identities and JSON escaping",
          "[polyphonic-pipeline]") {
    const std::vector touching{note("a", 60, 0, .5), note("b", 64, .5, 1)};
    REQUIRE(identify_chords(touching).empty());
    const std::vector duplicate{note("a", 60, 0, 1), note("a", 64, 0, 1)};
    REQUIRE_THROWS_AS(identify_chords(duplicate), std::invalid_argument);
    const std::vector brief{note("a", 60, 0, 1), note("b", 64, .99, 1)};
    REQUIRE(identify_chords(brief).empty());
    const std::vector notes{note("a\"\\\n", 60, 0, 1), note("b", 67, 0, 1)};
    const auto text = serialize_identified_chords(identify_chords(notes));
    REQUIRE(text.find("\"name\":\"C5\"") != std::string::npos);
    REQUIRE(text.find("a\\\"\\\\\\u000a") != std::string::npos);
    REQUIRE(text.find("\"inversion\":0") != std::string::npos);
}

TEST_CASE("spectral engine runs the tracked pipeline without changing legacy engine defaults",
          "[polyphonic-pipeline][engine]") {
    AudioBuffer audio{22050, std::vector<float>(33075)};
    for (std::size_t i = 4410; i < 26460; ++i) {
        const double time = static_cast<double>(i) / audio.sample_rate;
        for (const int pitch : {60, 64, 67}) {
            const double frequency = 440 * std::exp2((pitch - 69) / 12.);
            audio.samples[i] += static_cast<float>(.2 * std::sin(2 * std::numbers::pi * frequency * time));
        }
    }
    EngineRegistry registry;
    auto created = registry.create("builtin:spectral");
    auto engine = std::move(std::get<std::unique_ptr<TranscriptionEngine>>(created));
    REQUIRE(std::holds_alternative<std::monostate>(engine->initialize()));
    EngineConfiguration configuration;
    configuration.polyphonic_decoding.emplace();
    REQUIRE(std::holds_alternative<std::monostate>(engine->configure(configuration)));
    auto analyzed = engine->transcribe(audio);
    if (const auto* error = std::get_if<InferenceError>(&analyzed)) INFO(error->message);
    REQUIRE(std::holds_alternative<EngineUpdate>(analyzed));
    const auto& update = std::get<EngineUpdate>(analyzed);
    REQUIRE(update.polyphonic);
    std::set<int> found;
    for (const auto& e : update.polyphonic->events) found.insert(static_cast<int>(e.midi_pitch));
    REQUIRE(found == std::set<int>{60, 64, 67});
    REQUIRE(std::any_of(update.polyphonic->chords.begin(), update.polyphonic->chords.end(),
        [](const auto& c) { return c.name == "C"; }));
    REQUIRE(update.events.size() == update.polyphonic->events.size());
    REQUIRE(update.activations->semantics == EvidenceSemantics::RelativeActivation);
    REQUIRE(std::holds_alternative<InferenceError>(engine->incremental()->begin({})));
    configuration = {};
    REQUIRE(std::holds_alternative<std::monostate>(engine->configure(configuration)));
    analyzed = engine->transcribe(audio);
    REQUIRE(std::holds_alternative<EngineUpdate>(analyzed));
    REQUIRE_FALSE(std::get<EngineUpdate>(analyzed).polyphonic);
}

TEST_CASE("tracked engine rejects monophonic configuration and input over budget",
          "[polyphonic-pipeline][engine]") {
    EngineRegistry registry;
    for (const auto& id : {"builtin:yin", "builtin:spectral"}) {
        auto created = registry.create(id);
        auto engine = std::move(std::get<std::unique_ptr<TranscriptionEngine>>(created));
        REQUIRE(std::holds_alternative<std::monostate>(engine->initialize()));
        EngineConfiguration config;
        config.polyphonic_decoding.emplace();
        config.polyphonic_decoding->maximum_audio_seconds = .5;
        const auto configured = engine->configure(config);
        if (std::string(id) == "builtin:yin") {
            REQUIRE(std::holds_alternative<InferenceError>(configured));
        } else {
            REQUIRE(std::holds_alternative<std::monostate>(configured));
            const auto result = engine->transcribe({8000, std::vector<float>(8000)});
            REQUIRE(std::holds_alternative<InferenceError>(result));
            REQUIRE(std::get<InferenceError>(result).message.find("budget") != std::string::npos);
        }

    }
}

#ifdef TRANSCRIBE_ENABLE_ONNX
TEST_CASE("Basic Pitch tracked engine consumes aligned onset heads and emits finite canonical events",
          "[polyphonic-pipeline][onnx]") {
    AudioBuffer audio{22050, std::vector<float>(22050 * 3)};
    for (std::size_t i = 4410; i < 22050 * 2; ++i) {
        const double time = static_cast<double>(i) / audio.sample_rate;
        for (const int pitch : {60, 64, 67}) {
            const double frequency = 440 * std::exp2((pitch - 69) / 12.);
            audio.samples[i] += static_cast<float>(.2 * std::sin(2 * std::numbers::pi * frequency * time));
        }
    }
    EngineRegistry registry(TRANSCRIBE_BASIC_PITCH_MODEL_PATH);
    auto created = registry.create("onnx:basic-pitch-v1");
    auto engine = std::move(std::get<std::unique_ptr<TranscriptionEngine>>(created));
    REQUIRE(std::holds_alternative<std::monostate>(engine->initialize()));
    EngineConfiguration configuration;
    configuration.polyphonic_decoding.emplace();
    REQUIRE(std::holds_alternative<std::monostate>(engine->configure(configuration)));
    auto result = engine->transcribe(audio);
    if (const auto* error = std::get_if<InferenceError>(&result)) INFO(error->message);
    REQUIRE(std::holds_alternative<EngineUpdate>(result));
    const auto& update = std::get<EngineUpdate>(result);
    REQUIRE(update.polyphonic);
    REQUIRE_FALSE(update.activations->onsets.empty());
    REQUIRE(update.activations->timestamps.back() < 3);
    REQUIRE(update.polyphonic->events.size() >= 2);
    for (const auto& event : update.polyphonic->events) {
        REQUIRE(event.onset_sample < event.offset_sample);
        REQUIRE(event.offset_sample <= audio.samples.size());
        REQUIRE(event.evidence.front().onset_strength);
        REQUIRE(event.source_model == "onnx:basic-pitch-v1");
    }
}
#endif
