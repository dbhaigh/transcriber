#include "transcribe/musical_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <locale>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace transcribe {
namespace {

ScaleAnalysis make_scale(int tonic, std::string mode, std::span<const int> intervals,
                         const std::array<double, 12>& weights) {
    ScaleAnalysis scale;
    scale.tonic_pitch_class = tonic;
    scale.mode = std::move(mode);
    double inside = 0;
    for (int interval : intervals) {
        const int pitch = (scale.tonic_pitch_class + interval) % 12;
        scale.pitch_classes.push_back(pitch);
        inside += weights[pitch];
    }
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    scale.duration_coverage = total > 0 ? std::clamp(inside / total, 0.0, 1.0) : 0;
    return scale;
}

ScaleAnalysis scale_for_key(const KeyHypothesis& key, const std::array<double, 12>& weights,
                           int written_transposition = 0) {
    const int tonic = ((7 * key.fifths + (key.minor ? 9 : 0) -
        (key.user_constrained ? written_transposition : 0)) % 12 + 12) % 12;
    const std::vector<int> intervals = key.minor ? std::vector<int>{0, 2, 3, 5, 7, 8, 10} :
                                                  std::vector<int>{0, 2, 4, 5, 7, 9, 11};
    return make_scale(tonic, key.minor ? "natural-minor" : "major", intervals, weights);
}

bool contains(const ScaleAnalysis& scale, int midi) {
    return std::find(scale.pitch_classes.begin(), scale.pitch_classes.end(), midi % 12) !=
           scale.pitch_classes.end();
}

KeyHypothesis active_key(const ScoreHypothesis& score, double position) {
    const double bar = score.meter.numerator * 4.0 / score.meter.denominator;
    const double measure = score.pickup_quarters > 0 ?
        (position < score.pickup_quarters ? 0 :
         1 + std::floor((position - score.pickup_quarters) / bar)) :
        std::floor(position / bar);
    auto key = score.key;
    if (!key.user_constrained) {
        for (const auto& region : score.local_tonal_context) {
            if (region.first_measure <= measure && measure < region.end_measure &&
                region.key.confidence >= 0.5) key = region.key;
        }
    }
    return key;
}

} // namespace

MusicalAnalysis analyze_musical_structure(const ScoreHypothesis& score) {
    if (score.key.fifths < -7 || score.key.fifths > 7 ||
        score.meter.numerator <= 0 || score.meter.denominator <= 0 ||
        score.written_transposition_semitones < -48 || score.written_transposition_semitones > 48 ||
        !std::isfinite(score.pickup_quarters) || score.pickup_quarters < 0)
        throw std::invalid_argument("Musical analysis requires a valid key, meter and pickup");
    MusicalAnalysis result;
    struct Boundary { double position; std::size_t note; bool start; };
    std::vector<Boundary> boundaries;
    for (std::size_t i = 0; i < score.notes.size(); ++i) {
        const auto& note = score.notes[i];
        if (note.sounding_midi_pitch < 0 || note.sounding_midi_pitch > 127 ||
            !std::isfinite(note.onset_quarters) || note.onset_quarters < 0 ||
            !std::isfinite(note.duration_quarters) || note.duration_quarters <= 0 ||
            !std::isfinite(note.onset_quarters + note.duration_quarters) ||
            !std::isfinite(note.onset_residual_seconds) ||
            !std::isfinite(note.offset_residual_seconds))
            throw std::invalid_argument("Musical analysis requires finite score notes and sounding pitches");
        result.pitch_class_duration_quarters[note.sounding_midi_pitch % 12] += note.duration_quarters;
        result.symbolic_rhythm_fraction += note.rhythm.symbolic ? 1 : 0;
        result.mean_absolute_onset_residual_seconds += std::abs(note.onset_residual_seconds);
        result.mean_absolute_offset_residual_seconds += std::abs(note.offset_residual_seconds);
        boundaries.push_back({note.onset_quarters, i, true});
        boundaries.push_back({note.onset_quarters + note.duration_quarters, i, false});
    }
    result.note_count = score.notes.size();
    if (result.note_count) {
        result.symbolic_rhythm_fraction /= result.note_count;
        result.mean_absolute_onset_residual_seconds /= result.note_count;
        result.mean_absolute_offset_residual_seconds /= result.note_count;
    }
    result.key_scale = scale_for_key(score.key, result.pitch_class_duration_quarters,
                                     score.written_transposition_semitones);
    const std::vector<std::pair<std::string, std::vector<int>>> modes{
        {"major", {0, 2, 4, 5, 7, 9, 11}}, {"natural-minor", {0, 2, 3, 5, 7, 8, 10}},
        {"harmonic-minor", {0, 2, 3, 5, 7, 8, 11}}, {"melodic-minor", {0, 2, 3, 5, 7, 9, 11}},
        {"dorian", {0, 2, 3, 5, 7, 9, 10}}, {"phrygian", {0, 1, 3, 5, 7, 8, 10}},
        {"lydian", {0, 2, 4, 6, 7, 9, 11}}, {"mixolydian", {0, 2, 4, 5, 7, 9, 10}},
        {"locrian", {0, 1, 3, 5, 6, 8, 10}}};
    for (int tonic = 0; result.note_count && tonic < 12; ++tonic) {
        for (const auto& [mode, intervals] : modes) {
            auto scale = make_scale(tonic, mode, intervals, result.pitch_class_duration_quarters);
            if (scale.tonic_pitch_class != result.key_scale.tonic_pitch_class ||
                scale.mode != result.key_scale.mode)
                result.alternative_scales.push_back(std::move(scale));
        }
    }
    std::stable_sort(result.alternative_scales.begin(), result.alternative_scales.end(),
        [](const auto& a, const auto& b) { return a.duration_coverage > b.duration_coverage; });
    if (result.alternative_scales.size() > 3) result.alternative_scales.resize(3);
    for (std::size_t i = 0; i < score.notes.size(); ++i) {
        const auto& note = score.notes[i];
        const auto key = active_key(score, note.onset_quarters);
        const auto scale = scale_for_key(key, result.pitch_class_duration_quarters,
                                         score.written_transposition_semitones);
        if (!contains(scale, note.sounding_midi_pitch)) {
            ++result.out_of_scale_notes;
            const int interval = (note.sounding_midi_pitch % 12 - scale.tonic_pitch_class + 12) % 12;
            result.checks.push_back({i, key.minor && (interval == 9 || interval == 11) ?
                "minor-raised-degree" : "chromatic-note"});
        }
    }
    std::sort(boundaries.begin(), boundaries.end(), [](const auto& a, const auto& b) {
        return std::tie(a.position, a.start, a.note) < std::tie(b.position, b.start, b.note);
    });
    std::set<std::size_t> active;
    for (std::size_t i = 0; i < boundaries.size();) {
        const double onset = boundaries[i].position;
        do {
            const auto& boundary = boundaries[i++];
            if (boundary.start) active.insert(boundary.note);
            else active.erase(boundary.note);
        } while (i < boundaries.size() && boundaries[i].position == onset);
        if (i == boundaries.size() || active.empty()) continue;
        const double offset = boundaries[i].position;
        result.sounding_quarters += offset - onset;
        if (active.size() < 2) continue;
        std::vector<int> pitches;
        for (auto index : active) pitches.push_back(score.notes[index].sounding_midi_pitch);
        auto chord = identify_sounding_chord(pitches);
        const auto scale = scale_for_key(active_key(score, onset),
                                         result.pitch_class_duration_quarters,
                                         score.written_transposition_semitones);
        const bool diatonic = std::all_of(pitches.begin(), pitches.end(),
            [&](int pitch) { return contains(scale, pitch); });
        std::set<int> classes;
        for (int pitch : pitches) classes.insert(pitch % 12);
        const bool exportable = chord.root_pitch_class.has_value() && classes.size() >= 3 &&
                                offset - onset >= 0.25 &&
                                chord.quality != "augmented" &&
                                chord.quality != "diminished-seventh";
        if (chord.root_pitch_class) result.named_chord_quarters += offset - onset;
        result.harmony.push_back({onset, offset, std::move(chord),
            {active.begin(), active.end()}, diatonic, exportable});
    }
    return result;
}

std::string serialize_musical_analysis(const MusicalAnalysis& analysis) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    const auto scale = [&out](const ScaleAnalysis& value) {
        out << "{\"tonic_pitch_class\":" << value.tonic_pitch_class
            << ",\"mode\":\"" << value.mode << "\",\"duration_coverage\":"
            << value.duration_coverage << ",\"pitch_classes\":[";
        for (std::size_t i = 0; i < value.pitch_classes.size(); ++i) {
            if (i) out << ',';
            out << value.pitch_classes[i];
        }
        out << "]}";
    };
    out << "{\"version\":1,\"derived\":true,\"pitch_reference\":\"sounding\",\"key_scale\":";
    scale(analysis.key_scale);
    out << ",\"alternative_scales\":[";
    for (std::size_t i = 0; i < analysis.alternative_scales.size(); ++i) {
        if (i) out << ',';
        scale(analysis.alternative_scales[i]);
    }
    out << "],\"pitch_class_duration_quarters\":[";
    for (std::size_t i = 0; i < analysis.pitch_class_duration_quarters.size(); ++i) {
        if (i) out << ',';
        out << analysis.pitch_class_duration_quarters[i];
    }
    out << "],\"harmony\":[";
    for (std::size_t i = 0; i < analysis.harmony.size(); ++i) {
        if (i) out << ',';
        const auto& h = analysis.harmony[i];
        out << "{\"onset_quarters\":" << h.onset_quarters << ",\"offset_quarters\":"
            << h.offset_quarters << ",\"quality\":\"" << h.chord.quality << "\",\"root_pitch_class\":";
        if (h.chord.root_pitch_class) out << *h.chord.root_pitch_class;
        else out << "null";
        out << ",\"inversion\":";
        if (h.chord.inversion) out << *h.chord.inversion;
        else out << "null";
        out << ",\"name\":\"" << h.chord.name << "\",\"bass_midi_pitch\":" << h.chord.bass_midi_pitch
            << ",\"diatonic\":" << (h.diatonic ? "true" : "false")
            << ",\"export_symbol\":" << (h.export_symbol ? "true" : "false")
            << ",\"note_indices\":[";
        for (std::size_t n = 0; n < h.note_indices.size(); ++n) {
            if (n) out << ',';
            out << h.note_indices[n];
        }
        out << "]}";
    }
    out << "],\"checks\":[";
    for (std::size_t i = 0; i < analysis.checks.size(); ++i) {
        if (i) out << ',';
        out << "{\"note_index\":" << analysis.checks[i].note_index << ",\"code\":\""
            << analysis.checks[i].code << "\"}";
    }
    out << "],\"metrics\":{\"note_count\":" << analysis.note_count
        << ",\"out_of_scale_notes\":" << analysis.out_of_scale_notes
        << ",\"sounding_quarters\":" << analysis.sounding_quarters
        << ",\"named_chord_quarters\":" << analysis.named_chord_quarters
        << ",\"named_chord_coverage\":" << (analysis.sounding_quarters > 0 ?
            analysis.named_chord_quarters / analysis.sounding_quarters : 0)
        << ",\"symbolic_rhythm_fraction\":" << analysis.symbolic_rhythm_fraction
        << ",\"mean_absolute_onset_residual_seconds\":" << analysis.mean_absolute_onset_residual_seconds
        << ",\"mean_absolute_offset_residual_seconds\":" << analysis.mean_absolute_offset_residual_seconds
        << "}}";
    return out.str();
}

} // namespace transcribe
