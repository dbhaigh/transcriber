#pragma once

#include "transcribe/decoding.hpp"
#include "transcribe/polyphonic.hpp"

namespace transcribe {

struct PolyphonicDecodingOptions {
    double activation_on{0.4};
    double activation_off{0.25};
    double onset_threshold{0.5};
    double minimum_duration_seconds{0.055};
    double release_seconds{0.06};
    double minimum_retrigger_seconds{0.07};
    double minimum_chord_seconds{0.05};
    std::size_t maximum_polyphony{16};
    std::size_t maximum_events{100000};
    std::size_t maximum_grid_values{8 * 1024 * 1024};
    std::size_t maximum_audio_bytes{64 * 1024 * 1024};
    double maximum_audio_seconds{120};
};

struct IdentifiedChord {
    double onset_seconds{};
    double offset_seconds{};
    std::vector<std::string> note_ids;
    std::vector<int> midi_pitches;
    std::optional<int> root_pitch_class;
    int bass_midi_pitch{};
    std::string quality{"unknown"};
    std::string name{"unknown"};
    std::optional<int> inversion;
};

struct PolyphonicTranscription {
    std::vector<CanonicalNoteEvent> events;
    std::vector<IdentifiedChord> chords;
    std::size_t discarded_short_notes{};
    std::size_t refused_note_starts{};
};

void validate_polyphonic_decoding_options(const PolyphonicDecodingOptions& options);

// Exact pitch-class templates over half-open sounding intervals, not a key/harmony model.
IdentifiedChord identify_sounding_chord(std::span<const int> midi_pitches);
std::vector<IdentifiedChord> identify_chords(
    std::span<const CanonicalNoteEvent> events, double minimum_duration_seconds = 0.05,
    std::stop_token cancellation = {});
std::string serialize_identified_chords(std::span<const IdentifiedChord> chords);

// Finite recording, frame-major independent activity; each pitch has its own lifecycle.
// Window-start grids use their window centers. Pointwise model heads use timestamps unchanged.
PolyphonicTranscription decode_polyphonic(
    const PitchActivationGrid& grid, std::uint32_t source_sample_rate,
    std::uint64_t source_frames, const PolyphonicDecodingOptions& options,
    const std::string& audio_source_id, const std::string& model_id,
    std::stop_token cancellation = {});

} // namespace transcribe
