#include "transcribe/polyphonic_pipeline.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace transcribe {
namespace {

void check_cancelled(std::stop_token cancellation) {
    if (cancellation.stop_requested()) throw std::runtime_error("Polyphonic transcription cancelled");
}

constexpr std::array<const char*, 12> pitch_names{
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

struct ChordTemplate {
    const char* quality;
    const char* suffix;
    std::vector<int> intervals;
};

void label_chord(IdentifiedChord& chord) {
    static const std::vector<ChordTemplate> templates{
        {"major", "", {0, 4, 7}}, {"minor", "m", {0, 3, 7}},
        {"dominant-seventh", "7", {0, 4, 7, 10}},
        {"major-seventh", "maj7", {0, 4, 7, 11}},
        {"minor-seventh", "m7", {0, 3, 7, 10}},
        {"diminished", "dim", {0, 3, 6}}, {"augmented", "aug", {0, 4, 8}},
        {"suspended-second", "sus2", {0, 2, 7}},
        {"suspended-fourth", "sus4", {0, 5, 7}},
        {"half-diminished-seventh", "m7b5", {0, 3, 6, 10}},
        {"diminished-seventh", "dim7", {0, 3, 6, 9}},
        {"major-sixth", "6", {0, 4, 7, 9}}, {"minor-sixth", "m6", {0, 3, 7, 9}},
        {"power", "5", {0, 7}}};
    std::set<int> classes;
    for (const int pitch : chord.midi_pitches) classes.insert(pitch % 12);
    int best_inversion = std::numeric_limits<int>::max();
    for (const auto& pattern : templates) {
        if (pattern.intervals.size() != classes.size()) continue;
        for (const int root : classes) {
            std::set<int> expected;
            for (const int interval : pattern.intervals) expected.insert((root + interval) % 12);
            if (expected != classes) continue;
            const int bass_interval = (chord.bass_midi_pitch % 12 - root + 12) % 12;
            const auto bass = std::find(pattern.intervals.begin(), pattern.intervals.end(), bass_interval);
            const int inversion = static_cast<int>(bass - pattern.intervals.begin());
            if (inversion >= best_inversion) continue;
            best_inversion = inversion;
            chord.root_pitch_class = root;
            chord.quality = pattern.quality;
            chord.inversion = inversion;
            chord.name = std::string(pitch_names[root]) + pattern.suffix;
            if (inversion != 0) chord.name += "/" + std::string(pitch_names[chord.bass_midi_pitch % 12]);
        }
    }
}

std::uint64_t sample_position(double time, std::uint32_t rate, std::uint64_t frames) {
    const long double position = std::round(static_cast<long double>(time) * rate);
    return position >= frames ? frames : static_cast<std::uint64_t>(position);
}

struct PitchState {
    bool active{};
    double onset{};
    double last_supported_end{};
    double previous_onset_strength{};
    EventEvidence attack;
};

} // namespace

IdentifiedChord identify_sounding_chord(std::span<const int> midi_pitches) {
    IdentifiedChord chord;
    for (const int pitch : midi_pitches) {
        if (pitch < 0 || pitch > 127)
            throw std::invalid_argument("Chord pitches must be MIDI values between 0 and 127");
        chord.midi_pitches.push_back(pitch);
    }
    std::sort(chord.midi_pitches.begin(), chord.midi_pitches.end());
    chord.midi_pitches.erase(std::unique(chord.midi_pitches.begin(), chord.midi_pitches.end()),
                            chord.midi_pitches.end());
    if (!chord.midi_pitches.empty()) {
        chord.bass_midi_pitch = chord.midi_pitches.front();
        label_chord(chord);
    }
    return chord;
}

void validate_polyphonic_decoding_options(const PolyphonicDecodingOptions& o) {
    const auto ratio = [](double x) { return std::isfinite(x) && x >= 0 && x <= 1; };
    if (!ratio(o.activation_on) || o.activation_on == 0 || !ratio(o.activation_off) ||
        o.activation_off > o.activation_on || !ratio(o.onset_threshold) || o.onset_threshold == 0 ||
        !std::isfinite(o.minimum_duration_seconds) || o.minimum_duration_seconds < 0 ||
        !std::isfinite(o.release_seconds) || o.release_seconds < 0 ||
        !std::isfinite(o.minimum_retrigger_seconds) || o.minimum_retrigger_seconds <= 0 ||
        !std::isfinite(o.minimum_chord_seconds) || o.minimum_chord_seconds < 0 ||
        o.maximum_polyphony == 0 || o.maximum_polyphony > 128 ||
        o.maximum_events == 0 || o.maximum_grid_values == 0 || o.maximum_audio_bytes < sizeof(float) ||
        !std::isfinite(o.maximum_audio_seconds) || o.maximum_audio_seconds <= 0) {
        throw std::invalid_argument("Invalid polyphonic thresholds, timing or resource limits");
    }
}

std::vector<IdentifiedChord> identify_chords(
    std::span<const CanonicalNoteEvent> events, double minimum_duration_seconds,
    std::stop_token cancellation) {
    check_cancelled(cancellation);
    if (!std::isfinite(minimum_duration_seconds) || minimum_duration_seconds < 0)
        throw std::invalid_argument("Chord duration must be finite and nonnegative");
    struct Boundary { double time; std::size_t index; bool start; };
    std::vector<Boundary> boundaries;
    std::set<std::string> ids;
    for (std::size_t i = 0; i < events.size(); ++i) {
        check_cancelled(cancellation);
        const auto& e = events[i];
        if (!std::isfinite(e.midi_pitch) || e.midi_pitch < 0 || e.midi_pitch > 127 ||
            !std::isfinite(e.onset_seconds) || !std::isfinite(e.offset_seconds) ||
            e.onset_seconds < 0 || e.offset_seconds <= e.onset_seconds ||
            e.id.empty() || !ids.insert(e.id).second)
            throw std::invalid_argument("Chord identification requires unique IDs and valid note intervals");
        boundaries.push_back({e.onset_seconds, i, true});
        boundaries.push_back({e.offset_seconds, i, false});
    }
    std::sort(boundaries.begin(), boundaries.end(), [](const auto& a, const auto& b) {
        return std::tie(a.time, a.start, a.index) < std::tie(b.time, b.start, b.index);
    });
    std::set<std::size_t> active;
    std::vector<IdentifiedChord> result;
    for (std::size_t i = 0; i < boundaries.size();) {
        check_cancelled(cancellation);
        const double time = boundaries[i].time;
        do {
            const auto& b = boundaries[i++];
            if (b.start) active.insert(b.index);
            else active.erase(b.index);
        } while (i < boundaries.size() && boundaries[i].time == time);
        if (i == boundaries.size() || active.size() < 2) continue;
        IdentifiedChord chord;
        chord.onset_seconds = time;
        chord.offset_seconds = boundaries[i].time;
        for (const auto index : active) {
            chord.note_ids.push_back(events[index].id);
            chord.midi_pitches.push_back(static_cast<int>(std::lround(events[index].midi_pitch)));
        }
        std::sort(chord.midi_pitches.begin(), chord.midi_pitches.end());
        std::sort(chord.note_ids.begin(), chord.note_ids.end());
        chord.midi_pitches.erase(std::unique(chord.midi_pitches.begin(), chord.midi_pitches.end()),
                                chord.midi_pitches.end());
        if (chord.midi_pitches.size() < 2) continue;
        chord.bass_midi_pitch = chord.midi_pitches.front();
        label_chord(chord);
        if (chord.offset_seconds - chord.onset_seconds + 1e-10 >= minimum_duration_seconds)
            result.push_back(std::move(chord));
    }
    return result;
}

std::string serialize_identified_chords(std::span<const IdentifiedChord> chords) {
    const auto quoted = [](const std::string& text) {
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << '"';
        for (const unsigned char c : text) {
            if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
            else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
            else out << static_cast<char>(c);
        }
        return out.str() + '"';
    };
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << '[';
    for (std::size_t i = 0; i < chords.size(); ++i) {
        const auto& c = chords[i];
        if (!std::isfinite(c.onset_seconds) || !std::isfinite(c.offset_seconds) ||
            c.onset_seconds < 0 || c.offset_seconds <= c.onset_seconds)
            throw std::invalid_argument("Cannot serialize invalid chord interval");
        if (i) out << ',';
        out << "{\"onset_seconds\":" << c.onset_seconds << ",\"offset_seconds\":" << c.offset_seconds
            << ",\"name\":" << quoted(c.name) << ",\"quality\":" << quoted(c.quality)
            << ",\"root_pitch_class\":";
        if (c.root_pitch_class) out << *c.root_pitch_class; else out << "null";
        out << ",\"bass_midi_pitch\":" << c.bass_midi_pitch << ",\"inversion\":";
        if (c.inversion) out << *c.inversion; else out << "null";
        out << ",\"midi_pitches\":[";
        for (std::size_t p = 0; p < c.midi_pitches.size(); ++p) {
            if (p) out << ',';
            out << c.midi_pitches[p];
        }
        out << "],\"note_ids\":[";
        for (std::size_t n = 0; n < c.note_ids.size(); ++n) {
            if (n) out << ',';
            out << quoted(c.note_ids[n]);
        }
        out << "]}";
    }
    return out.str() + ']';
}

PolyphonicTranscription decode_polyphonic(
    const PitchActivationGrid& grid, std::uint32_t source_sample_rate,
    std::uint64_t source_frames, const PolyphonicDecodingOptions& options,
    const std::string& audio_source_id, const std::string& model_id,
    std::stop_token cancellation) {
    check_cancelled(cancellation);
    validate_polyphonic_decoding_options(options);
    if (grid.values.size() > options.maximum_grid_values)
        throw std::length_error("Polyphonic activation grid exceeds the configured resource budget");
    validate_pitch_activation_grid(grid);
    if (source_sample_rate == 0 || audio_source_id.empty() || model_id.empty() ||
        grid.objective != PitchObjective::IndependentMultiPitch)
        throw std::invalid_argument("Polyphonic decoding requires an independent multi-pitch objective and source identity");
    const double duration = static_cast<double>(source_frames) / source_sample_rate;
    if (duration > options.maximum_audio_seconds ||
        source_frames > options.maximum_audio_bytes / sizeof(float))
        throw std::length_error("Polyphonic recording exceeds the configured duration/audio budget");
    for (const auto* head : {&grid.values, &grid.onsets}) {
        for (const double value : *head) {
            if (value < 0 || value > 1)
                throw std::invalid_argument("Polyphonic decoder requires normalized activity/onset strengths in [0,1]");
        }
    }
    if (!grid.timestamps.empty() && grid.timestamps.back() >= duration)
        throw std::invalid_argument("Polyphonic grid extends beyond the source recording");
    const auto pitches = static_cast<std::size_t>(grid.maximum_midi - grid.minimum_midi + 1);
    std::vector<PitchState> states(pitches);
    std::vector<std::size_t> candidates;
    candidates.reserve(pitches);
    PolyphonicTranscription result;
    std::size_t active_count = 0;
    std::size_t starts = 0;
    const auto close = [&](std::size_t pitch, double end) {
        auto& state = states[pitch];
        state.active = false;
        --active_count;
        end = std::min(end, duration);
        const auto onset_sample = sample_position(state.onset, source_sample_rate, source_frames);
        const auto offset_sample = sample_position(end, source_sample_rate, source_frames);
        if (end - state.onset + 1e-10 < options.minimum_duration_seconds || offset_sample <= onset_sample) {
            ++result.discarded_short_notes;
            return;
        }
        CanonicalNoteEvent event;
        event.id = audio_source_id + ":poly-" + std::to_string(++starts);
        event.midi_pitch = grid.minimum_midi + static_cast<int>(pitch);
        event.onset_sample = onset_sample;
        event.offset_sample = offset_sample;
        event.source_sample_rate = source_sample_rate;
        event.onset_seconds = static_cast<double>(onset_sample) / source_sample_rate;
        event.offset_seconds = static_cast<double>(offset_sample) / source_sample_rate;
        event.audio_source_id = audio_source_id;
        event.source_model = model_id;
        event.detection_method = grid.method + ":polyphonic-state-tracker";
        event.evidence_kind = std::string(evidence_semantics_name(grid.semantics));
        event.evidence.push_back(state.attack);
        if (result.events.size() == options.maximum_events)
            throw std::length_error("Polyphonic note count exceeds the configured event budget");
        result.events.push_back(std::move(event));
    };
    for (std::size_t frame = 0; frame < grid.timestamps.size(); ++frame) {
        check_cancelled(cancellation);
        const double time = std::min(duration, grid.timestamps[frame] + grid.window_seconds * .5);
        if (time >= duration) break;
        const double end = std::min(duration, time + grid.hop_seconds);
        candidates.clear();
        for (std::size_t p = 0; p < pitches; ++p) {
            const auto index = frame * pitches + p;
            auto& state = states[p];
            const double value = grid.values[index];
            const double onset = grid.onsets.empty() ? 0 : grid.onsets[index];
            const bool attack = onset >= options.onset_threshold &&
                state.previous_onset_strength < options.onset_threshold;
            state.previous_onset_strength = onset;
            if (state.active && attack && value >= options.activation_on &&
                time - state.onset + 1e-10 >= options.minimum_retrigger_seconds) {
                close(p, time);
            }
            if (state.active) {
                if (value >= options.activation_off && value > 0) state.last_supported_end = end;
                else if (end - state.last_supported_end + 1e-10 >= options.release_seconds)
                    close(p, state.last_supported_end);
            }
            if (!state.active && value >= options.activation_on &&
                (grid.onsets.empty() || attack || frame == 0))
                candidates.push_back(p);
        }
        std::stable_sort(candidates.begin(), candidates.end(), [&](auto a, auto b) {
            return grid.values[frame * pitches + a] > grid.values[frame * pitches + b];
        });
        for (const auto p : candidates) {
            if (active_count >= options.maximum_polyphony) {
                ++result.refused_note_starts;
                continue;
            }
            auto& state = states[p];
            state.active = true;
            ++active_count;
            state.onset = time;
            state.last_supported_end = end;
            const auto index = frame * pitches + p;
            state.attack = {};
            state.attack.time_seconds = time;
            state.attack.midi_pitch = grid.minimum_midi + static_cast<int>(p);
            state.attack.raw_midi_pitch = state.attack.midi_pitch;
            state.attack.raw_strength = grid.values[index];
            state.attack.strength_semantics = grid.semantics;
            if (!grid.onsets.empty()) state.attack.onset_strength = grid.onsets[index];
            if (!grid.fractional_midi.empty()) state.attack.fractional_midi = grid.fractional_midi[index];
        }
    }
    for (std::size_t p = 0; p < pitches; ++p) {
        if (states[p].active) close(p, states[p].last_supported_end);
    }
    std::sort(result.events.begin(), result.events.end(), [](const auto& a, const auto& b) {
        return std::tie(a.onset_sample, a.midi_pitch, a.id) < std::tie(b.onset_sample, b.midi_pitch, b.id);
    });
    result.chords = identify_chords(result.events, options.minimum_chord_seconds, cancellation);
    return result;
}

} // namespace transcribe
