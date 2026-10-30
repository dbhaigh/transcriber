#include "transcribe/confidence.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <locale>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace transcribe {
namespace {

void quoted(std::ostream& out, const std::string& value) {
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 0x20) {
            constexpr char hex[] = "0123456789abcdef";
            out << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else out << static_cast<char>(c);
    }
    out << '"';
}

void optional_number(std::ostream& out, std::optional<double> value) {
    if (value) {
        if (!std::isfinite(*value)) throw std::invalid_argument("Nonfinite confidence value");
        out << *value;
    } else out << "null";
}

void ids(std::ostream& out, const std::vector<std::string>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        quoted(out, values[i]);
    }
    out << ']';
}

void summary_json(std::ostream& out, const ConfidenceSummary& summary) {
    out << "{\"note_count\":" << summary.note_count << ",\"assessed_notes\":"
        << summary.assessed_notes << ",\"calibrated_notes\":" << summary.calibrated_notes
        << ",\"low_notes\":" << summary.low_notes << ",\"mean_evidence_score\":";
    optional_number(out, summary.mean_evidence_score);
    out << ",\"mean_calibrated_probability\":";
    optional_number(out, summary.mean_calibrated_probability);
    out << '}';
}

ConfidenceSummary summarize(const std::vector<NoteConfidence>& notes,
                            const std::vector<std::size_t>& indices) {
    ConfidenceSummary result;
    double evidence_sum = 0, probability_sum = 0;
    std::size_t evidence_count = 0;
    result.note_count = indices.size();
    for (auto index : indices) {
        const auto& note = notes[index];
        result.assessed_notes += note.evidence_score.has_value() || note.calibrated_probability.has_value();
        if (note.evidence_score) {
            evidence_sum += *note.evidence_score;
            ++evidence_count;
        }
        if (note.calibrated_probability) {
            probability_sum += *note.calibrated_probability;
            ++result.calibrated_notes;
        }
        result.low_notes += note.band == "low";
    }
    if (evidence_count) result.mean_evidence_score = evidence_sum / evidence_count;
    if (result.calibrated_notes)
        result.mean_calibrated_probability = probability_sum / result.calibrated_notes;
    return result;
}

} // namespace

ConfidenceReport score_confidence(const ScoreDocument& score, const ConfidenceOptions& options) {
    validate_score_document(score);
    if (!std::isfinite(options.phrase_gap_seconds) || options.phrase_gap_seconds <= 0)
        throw std::invalid_argument("Confidence phrase gap must be finite and positive");
    ConfidenceReport report;
    report.phrase_gap_seconds = options.phrase_gap_seconds;
    std::unordered_map<std::string, const CanonicalNoteEvent*> sources;
    for (const auto& event : score.source_events) sources.emplace(event.id, &event);
    for (const auto& note : score.notes) {
        NoteConfidence result;
        result.note_id = note.id;
        result.source_event_ids = note.source_event_ids;
        result.notation_evidence = note.interpretation_evidence;
        bool mismatch = false;
        for (const auto& id : note.source_event_ids) {
            const auto found = sources.find(id);
            if (found != sources.end() &&
                std::lround(found->second->midi_pitch) != note.sounding_midi_pitch) mismatch = true;
        }
        if (note.user_authored || mismatch) {
            result.band = "user-edited";
            result.reasons.push_back("detector-confidence-does-not-assess-user-edits");
        } else {
            result.calibrated_probability = note.calibrated_confidence;
            struct Evidence {
                double sum{};
                std::size_t count{};
                double first{std::numeric_limits<double>::infinity()};
                double last{-std::numeric_limits<double>::infinity()};
                double peak{};
                double alternative_peak{};
            };
            std::map<EvidenceSemantics, Evidence> groups;
            for (const auto& sample : note.raw_evidence) {
                if (sample.strength_semantics == EvidenceSemantics::Unspecified ||
                    sample.raw_strength < 0 || sample.raw_strength > 1 ||
                    sample.time_seconds < note.measured_onset_seconds ||
                    sample.time_seconds > note.measured_offset_seconds) continue;
                auto& group = groups[sample.strength_semantics];
                if (sample.midi_pitch == note.sounding_midi_pitch) {
                    group.sum += sample.raw_strength;
                    ++group.count;
                    group.peak = std::max(group.peak, sample.raw_strength);
                    group.first = std::min(group.first, sample.time_seconds);
                    group.last = std::max(group.last, sample.time_seconds);
                } else group.alternative_peak = std::max(group.alternative_peak, sample.raw_strength);
            }
            const Evidence* selected = nullptr;
            for (const auto& [semantics, group] : groups) {
                (void)semantics;
                if (group.count && (!selected || group.count > selected->count)) selected = &group;
            }
            if (selected) {
                const double duration = note.measured_offset_seconds - note.measured_onset_seconds;
                result.detector_support = selected->sum / selected->count;
                result.temporal_coverage = duration > 0 ?
                    std::clamp((selected->last - selected->first) / duration, 0.0, 1.0) : 0;
                const bool conflict = selected->alternative_peak > 0 &&
                                      selected->alternative_peak >= selected->peak * .8;
                result.evidence_score = *result.detector_support *
                    (.5 + .5 * *result.temporal_coverage) * (conflict ? .75 : 1);
                if (*result.temporal_coverage < .5) result.reasons.push_back("sparse-temporal-evidence");
                if (conflict) result.reasons.push_back("competing-pitch-evidence");
                result.reasons.push_back("heuristic-not-correctness-probability");
            } else result.reasons.push_back("no-compatible-bounded-detector-evidence");
            const auto value = result.calibrated_probability ?
                result.calibrated_probability : result.evidence_score;
            if (value) result.band = *value < .5 ? "low" : *value < .8 ? "medium" : "high";
        }
        report.notes.push_back(std::move(result));
    }
    struct Interval { std::size_t index; double start; double end; };
    std::vector<Interval> intervals;
    std::vector<std::size_t> all(report.notes.size());
    std::iota(all.begin(), all.end(), 0);
    for (auto index : all) {
        const auto& note = score.notes[index];
        intervals.push_back({index, score_position_seconds(score, note.onset_quarters),
            score_position_seconds(score, note.onset_quarters + note.duration_quarters)});
    }
    std::stable_sort(intervals.begin(), intervals.end(), [](const auto& a, const auto& b) {
        return a.start < b.start;
    });
    std::vector<std::size_t> phrase_indices;
    for (const auto& interval : intervals) {
        if (report.phrases.empty() ||
            interval.start - report.phrases.back().end_seconds + 1e-9 >= options.phrase_gap_seconds) {
            if (!report.phrases.empty())
                report.phrases.back().summary = summarize(report.notes, phrase_indices);
            phrase_indices.clear();
            report.phrases.push_back({interval.start, interval.end, {}, {}});
        }
        auto& phrase = report.phrases.back();
        phrase.end_seconds = std::max(phrase.end_seconds, interval.end);
        phrase.note_ids.push_back(score.notes[interval.index].id);
        phrase_indices.push_back(interval.index);
    }
    if (!report.phrases.empty())
        report.phrases.back().summary = summarize(report.notes, phrase_indices);
    report.transcription = summarize(report.notes, all);
    return report;
}

std::string serialize_note_confidence(const NoteConfidence& confidence) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << "{\"note_id\":";
    quoted(out, confidence.note_id);
    out << ",\"source_event_ids\":";
    ids(out, confidence.source_event_ids);
    out << ",\"evidence_score\":";
    optional_number(out, confidence.evidence_score);
    out << ",\"calibrated_probability\":";
    optional_number(out, confidence.calibrated_probability);
    out << ",\"detector_support\":";
    optional_number(out, confidence.detector_support);
    out << ",\"temporal_coverage\":";
    optional_number(out, confidence.temporal_coverage);
    out << ",\"notation_evidence\":";
    optional_number(out, confidence.notation_evidence);
    out << ",\"band\":";
    quoted(out, confidence.band);
    out << ",\"reasons\":";
    ids(out, confidence.reasons);
    out << '}';
    return out.str();
}

std::string serialize_confidence_report(const ConfidenceReport& report) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17)
        << "{\"version\":1,\"method\":\"bounded-evidence-v1\",\"derived\":true,"
           "\"aggregate_semantics\":\"note-weighted-means-not-all-notes-correct-probability\","
           "\"phrase_gap_seconds\":" << report.phrase_gap_seconds << ",\"notes\":[";
    for (std::size_t i = 0; i < report.notes.size(); ++i) {
        if (i) out << ',';
        out << serialize_note_confidence(report.notes[i]);
    }
    out << "],\"phrases\":[";
    for (std::size_t i = 0; i < report.phrases.size(); ++i) {
        if (i) out << ',';
        const auto& phrase = report.phrases[i];
        out << "{\"start_seconds\":" << phrase.start_seconds << ",\"end_seconds\":"
            << phrase.end_seconds << ",\"note_ids\":";
        ids(out, phrase.note_ids);
        out << ",\"summary\":";
        summary_json(out, phrase.summary);
        out << '}';
    }
    out << "],\"transcription\":";
    summary_json(out, report.transcription);
    out << '}';
    return out.str();
}

} // namespace transcribe
