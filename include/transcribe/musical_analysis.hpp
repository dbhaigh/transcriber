#pragma once

#include "transcribe/interpretation.hpp"
#include "transcribe/polyphonic_pipeline.hpp"

#include <array>

namespace transcribe {

struct ScaleAnalysis {
    int tonic_pitch_class{};
    std::string mode;
    std::vector<int> pitch_classes;
    double duration_coverage{};
};

struct HarmonicInterval {
    double onset_quarters{};
    double offset_quarters{};
    IdentifiedChord chord;
    std::vector<std::size_t> note_indices;
    bool diatonic{};
    // Exact template, at least three pitch classes, lasting >= 1/4 quarter.
    // This is a conservative notation gate, not a calibrated probability.
    bool export_symbol{};
};

struct HarmonicCheck {
    std::size_t note_index{};
    std::string code;
};

struct MusicalAnalysis {
    ScaleAnalysis key_scale;
    std::vector<ScaleAnalysis> alternative_scales;
    std::array<double, 12> pitch_class_duration_quarters{};
    std::vector<HarmonicInterval> harmony;
    std::vector<HarmonicCheck> checks;
    std::size_t note_count{};
    std::size_t out_of_scale_notes{};
    double sounding_quarters{};
    double named_chord_quarters{};
    double symbolic_rhythm_fraction{};
    double mean_absolute_onset_residual_seconds{};
    double mean_absolute_offset_residual_seconds{};
};

// Derived from the current score, never mutates source events or score pitches.
// Pitches/scales are sounding; MusicXML transposes symbols to written pitch.
MusicalAnalysis analyze_musical_structure(const ScoreHypothesis& hypothesis);
std::string serialize_musical_analysis(const MusicalAnalysis& analysis);

} // namespace transcribe
