#include "../src/gui_transcription.hpp"
#include "transcribe/engine.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numbers>
#include <stdexcept>

namespace {
struct WavFixture {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("transcriber-polyphonic-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".wav");
    WavFixture() {
        std::ofstream out(path, std::ios::binary);
        const auto integer = [&](std::uint32_t value, int bytes) {
            for (int i = 0; i < bytes; ++i) out.put(static_cast<char>((value >> (i * 8)) & 255));
        };
        constexpr std::uint32_t rate = 22050;
        constexpr std::uint32_t frames = rate * 3;
        out << "RIFF"; integer(36 + frames * 2, 4); out << "WAVEfmt ";
        integer(16, 4); integer(1, 2); integer(1, 2);
        integer(rate, 4); integer(rate * 2, 4); integer(2, 2); integer(16, 2);
        out << "data"; integer(frames * 2, 4);
        for (std::uint32_t i = 0; i < frames; ++i) {
            const double time = static_cast<double>(i) / rate;
            double value = 0;
            if (time >= .2 && time < 2.0) {
                for (const int pitch : {60, 64, 67}) {
                    value += .2 * std::sin(2 * std::numbers::pi *
                        440 * std::exp2((pitch - 69) / 12.) * time);
                }
            }
            integer(static_cast<std::uint16_t>(static_cast<std::int16_t>(value * 32767)), 2);
        }
        if (!out) throw std::runtime_error("Could not write polyphonic GUI fixture");
    }
    ~WavFixture() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};
}

TEST_CASE("offline workstation polyphonic mode uses DSP without ONNX and preserves canonical evidence",
          "[polyphonic-pipeline][gui]") {
    WavFixture fixture;
    transcribe::gui::Request request;
    request.input_path = fixture.path;
    request.polyphonic = true;
    request.proposal_only = true;
    const auto result = transcribe::gui::run_transcription(request, {}, {}, {});
    REQUIRE(result.polyphonic);
    REQUIRE(result.polyphonic->events.size() >= 3);
    REQUIRE(result.audio_sample_rate == 22050);
    REQUIRE(result.audio_frame_count == 66150);
    REQUIRE(result.score.source_sample_rate == 22050);
    REQUIRE_FALSE(result.polyphonic->chords.empty());
    REQUIRE(std::any_of(result.polyphonic->chords.begin(), result.polyphonic->chords.end(),
        [](const auto& chord) { return chord.name == "C"; }));
    for (const auto& source : result.score.source_events) {
        REQUIRE(source.source_sample_rate == 22050);
        REQUIRE(source.source_model == "builtin:spectral");
        REQUIRE_FALSE(source.evidence.empty());
        REQUIRE(source.onset_sample < source.offset_sample);
    }
#ifdef TRANSCRIBE_ENABLE_ONNX
    request.model_path = TRANSCRIBE_BASIC_PITCH_MODEL_PATH;
    request.instrument = transcribe::InstrumentPrior::Guitar;
    const auto neural = transcribe::gui::run_transcription(request, {}, {}, {});
    REQUIRE(neural.polyphonic);
    transcribe::EngineRegistry registry(request.model_path);
    auto created = registry.create("onnx:basic-pitch-v1");
    auto engine = std::move(std::get<std::unique_ptr<transcribe::TranscriptionEngine>>(created));
    REQUIRE(std::holds_alternative<std::monostate>(engine->initialize()));
    transcribe::EngineConfiguration config;
    config.polyphonic_decoding.emplace();
    REQUIRE(std::holds_alternative<std::monostate>(engine->configure(config)));
    auto inferred = engine->transcribe(transcribe::read_wav(fixture.path));
    REQUIRE(std::holds_alternative<transcribe::EngineUpdate>(inferred));
    const auto& expected = std::get<transcribe::EngineUpdate>(inferred).polyphonic->events;
    REQUIRE_FALSE(expected.empty());
    REQUIRE(neural.polyphonic->events.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(neural.polyphonic->events[i].midi_pitch == expected[i].midi_pitch);
        REQUIRE(neural.polyphonic->events[i].onset_sample == expected[i].onset_sample);
        REQUIRE(neural.polyphonic->events[i].offset_sample == expected[i].offset_sample);
    }
    REQUIRE(neural.score.instrument.name == "Guitar");
    request.model_path.clear();
#endif
    request.bounded_inference = true;
    REQUIRE_THROWS_AS(transcribe::gui::run_transcription(request, {}, {}, {}), std::invalid_argument);
    std::stop_source stop;
    stop.request_stop();
    REQUIRE_THROWS_AS(transcribe::gui::run_transcription(request, stop.get_token(), {}, {}, {}),
                      std::runtime_error);
}
