#pragma once

#include "transcribe/score.hpp"

namespace transcribe {

struct NoteConfidence {
    std::string note_id;
    std::vector<std::string> source_event_ids;
    std::optional<double> evidence_score;
    std::optional<double> calibrated_probability;
    std::optional<double> detector_support;
    std::optional<double> temporal_coverage;
    double notation_evidence{};
    std::string band{"unassessed"};
    std::vector<std::string> reasons;
};

struct ConfidenceSummary {
    std::size_t note_count{};
    std::size_t assessed_notes{};
    std::size_t calibrated_notes{};
    std::size_t low_notes{};
    std::optional<double> mean_evidence_score;
    std::optional<double> mean_calibrated_probability;
};

struct PhraseConfidence {
    double start_seconds{};
    double end_seconds{};
    std::vector<std::string> note_ids;
    ConfidenceSummary summary;
};

struct ConfidenceReport {
    std::vector<NoteConfidence> notes;
    std::vector<PhraseConfidence> phrases;
    ConfidenceSummary transcription;
    double phrase_gap_seconds{1.0};
};

struct ConfidenceOptions {
    double phrase_gap_seconds{1.0};
};

// Derived review metadata, not a detector filter or a project-schema mutation.
// Evidence scores are heuristic, calibrated probabilities must already be supplied.
ConfidenceReport score_confidence(const ScoreDocument& score,
                                 const ConfidenceOptions& options = {});
std::string serialize_note_confidence(const NoteConfidence& confidence);
std::string serialize_confidence_report(const ConfidenceReport& report);

} // namespace transcribe
