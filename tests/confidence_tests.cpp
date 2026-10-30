#include "transcribe/confidence.hpp"
#include "transcribe/musicxml.hpp"
#include "transcribe/review.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>

namespace {

transcribe::CanonicalNoteEvent event(int midi, double start, double end, double strength) {
    transcribe::CanonicalNoteEvent result;
    result.id = std::to_string(midi) + "-" + std::to_string(start);
    result.midi_pitch = midi;
    result.onset_seconds = start;
    result.offset_seconds = end;
    result.source_sample_rate = 48000;
    result.onset_sample = static_cast<std::uint64_t>(start * 48000);
    result.offset_sample = static_cast<std::uint64_t>(end * 48000);
    result.source_model = "confidence-fixture";
    for (double position : {start, (start + end) / 2, end}) {
        transcribe::EventEvidence evidence;
        evidence.midi_pitch = midi;
        evidence.time_seconds = position;
        evidence.raw_strength = strength;
        evidence.strength_semantics = transcribe::EvidenceSemantics::RelativeActivation;
        result.evidence.push_back(evidence);
    }
    return result;
}

transcribe::ScoreDocument document(std::vector<transcribe::CanonicalNoteEvent> events) {
    transcribe::InterpretationOptions options;
    options.tempo_bpm = 120;
    options.key_fifths = 0;
    options.meter = transcribe::MeterConstraint{4, 4, {}};
    options.quantization_mode = transcribe::QuantizationMode::Expressive;
    const auto interpretation = transcribe::interpret_note_events(events, options);
    return transcribe::make_score_document(interpretation, 0, "confidence-source",
        transcribe::metadata_for_hypothesis(interpretation.hypotheses.front()));
}

} // namespace

TEST_CASE("confidence exposes bounded evidence without manufacturing calibrated probabilities",
          "[confidence]") {
    auto score = document({event(60, 0, 1, .9), event(64, 0, 1, .6), event(67, 0, 1, .2)});
    const auto before = transcribe::serialize_score_document(score);
    const auto report = transcribe::score_confidence(score);
    REQUIRE(report.notes.size() == 3);
    REQUIRE(report.notes[0].evidence_score == Catch::Approx(.9));
    REQUIRE(report.notes[0].band == "high");
    REQUIRE(report.notes[1].band == "medium");
    REQUIRE(report.notes[2].band == "low");
    REQUIRE_FALSE(report.notes[0].calibrated_probability);
    REQUIRE(report.notes[0].temporal_coverage == Catch::Approx(1));
    REQUIRE(report.transcription.assessed_notes == 3);
    REQUIRE(report.transcription.low_notes == 1);
    REQUIRE(report.transcription.mean_evidence_score == Catch::Approx(1.7 / 3));
    REQUIRE_FALSE(report.transcription.mean_calibrated_probability);
    REQUIRE(transcribe::serialize_score_document(score) == before);
}

TEST_CASE("sparse and competing pitch evidence lower only the derived confidence score",
          "[confidence]") {
    auto source = event(60, 0, 1, .8);
    source.evidence.resize(1);
    auto score = document({source});
    const auto sparse = transcribe::score_confidence(score);
    REQUIRE(sparse.notes[0].evidence_score == Catch::Approx(.4));
    REQUIRE(sparse.notes[0].band == "low");
    auto alternative = score.notes[0].raw_evidence.front();
    alternative.midi_pitch = 61;
    score.notes[0].raw_evidence.push_back(alternative);
    const auto conflict = transcribe::score_confidence(score);
    REQUIRE(conflict.notes[0].evidence_score == Catch::Approx(.3));
    REQUIRE(conflict.notes[0].reasons.size() == 3);
    REQUIRE(score.source_events[0].evidence[0].raw_strength == .8);
}

TEST_CASE("confidence leaves missing unbounded and unspecified evidence explicitly unassessed",
          "[confidence]") {
    auto score = document({event(60, 0, 1, .8)});
    score.notes[0].raw_evidence.clear();
    REQUIRE(transcribe::score_confidence(score).notes[0].band == "unassessed");
    auto evidence = score.source_events[0].evidence.front();
    evidence.strength_semantics = transcribe::EvidenceSemantics::Unspecified;
    score.notes[0].raw_evidence = {evidence};
    REQUIRE_FALSE(transcribe::score_confidence(score).notes[0].evidence_score);
    evidence.strength_semantics = transcribe::EvidenceSemantics::RawActivation;
    evidence.raw_strength = 9;
    score.notes[0].raw_evidence = {evidence};
    const auto report = transcribe::score_confidence(score);
    REQUIRE_FALSE(report.transcription.mean_evidence_score);
    REQUIRE(report.transcription.assessed_notes == 0);
    REQUIRE(report.transcription.note_count == 1);
}

TEST_CASE("phrase confidence uses global silence union and an inclusive one second gap",
          "[confidence]") {
    const auto score = document({event(48, 0, 5, .9), event(60, 1, 2, .5),
        event(62, 3, 4, .5), event(64, 6, 7, .8), event(65, 7.999, 8.5, .6)});
    const auto report = transcribe::score_confidence(score);
    REQUIRE(report.phrases.size() == 2);
    REQUIRE(report.phrases[0].note_ids.size() == 3);
    REQUIRE(report.phrases[0].end_seconds == Catch::Approx(5));
    REQUIRE(report.phrases[1].start_seconds == Catch::Approx(6));
    REQUIRE(report.phrases[1].summary.note_count == 2);
    REQUIRE(report.transcription.mean_evidence_score == Catch::Approx(3.3 / 5));
    REQUIRE(report.phrases[0].summary.mean_evidence_score == Catch::Approx(1.9 / 3));
}

TEST_CASE("confidence does not mix detector semantics or use evidence outside the measured interval",
          "[confidence]") {
    auto score = document({event(60, 0, 1, .9)});
    auto sample = score.notes[0].raw_evidence.front();
    sample.strength_semantics = transcribe::EvidenceSemantics::RawActivation;
    sample.raw_strength = .1;
    score.notes[0].raw_evidence.push_back(sample);
    sample.time_seconds = 1;
    score.notes[0].raw_evidence.push_back(sample);
    sample.midi_pitch = 61;
    sample.raw_strength = 1;
    score.notes[0].raw_evidence.push_back(sample);
    sample.strength_semantics = transcribe::EvidenceSemantics::RelativeActivation;
    sample.time_seconds = 2;
    score.notes[0].raw_evidence.push_back(sample);
    sample.midi_pitch = 60;
    sample.raw_strength = 0;
    score.notes[0].raw_evidence.push_back(sample);
    const auto report = transcribe::score_confidence(score);
    REQUIRE(report.notes[0].evidence_score == Catch::Approx(.9));
    REQUIRE(report.notes[0].detector_support == Catch::Approx(.9));
    REQUIRE(report.notes[0].temporal_coverage == Catch::Approx(1));
    REQUIRE(report.notes[0].reasons == std::vector<std::string>{
        "heuristic-not-correctness-probability"});

    for (auto& evidence : score.notes[0].raw_evidence)
        evidence.strength_semantics = transcribe::EvidenceSemantics::CalibratedProbability;
    const auto frame_probabilities = transcribe::score_confidence(score);
    REQUIRE(frame_probabilities.notes[0].evidence_score);
    REQUIRE_FALSE(frame_probabilities.notes[0].calibrated_probability);
}

TEST_CASE("calibrated and evidence means have separate denominators and edits are unassessed",
          "[confidence]") {
    auto score = document({event(60, 0, 1, .9), event(62, 1, 2, .5), event(64, 2, 3, .7)});
    score.notes[0].calibrated_confidence = 0.0;
    score.notes[1].calibrated_confidence = 1.0;
    score.notes[1].raw_evidence.clear();
    score.notes[2].user_authored = true;
    score.notes[2].calibrated_confidence = .99;
    const auto report = transcribe::score_confidence(score);
    REQUIRE(report.notes[0].band == "low");
    REQUIRE(report.notes[1].band == "high");
    REQUIRE(report.notes[2].band == "user-edited");
    REQUIRE_FALSE(report.notes[2].calibrated_probability);
    REQUIRE(report.transcription.assessed_notes == 2);
    REQUIRE(report.transcription.calibrated_notes == 2);
    REQUIRE(report.transcription.mean_evidence_score == Catch::Approx(.9));
    REQUIRE(report.transcription.mean_calibrated_probability == Catch::Approx(.5));
    REQUIRE(score.notes[2].calibrated_confidence == .99);
}

TEST_CASE("confidence recomputes after score edits without altering project serialization",
          "[confidence][score]") {
    const auto score = document({event(60, 0, 1, .9)});
    const auto json = transcribe::serialize_score_document(score);
    REQUIRE(transcribe::serialize_confidence_report(transcribe::score_confidence(
        transcribe::deserialize_score_document(json))) ==
        transcribe::serialize_confidence_report(transcribe::score_confidence(score)));
    transcribe::ScoreEditor editor(score);
    transcribe::ScoreTransaction change;
    change.base_revision = score.revision;
    change.commands.push_back(transcribe::ChangeScorePitch{score.notes[0].id, 61});
    editor.apply(change);
    REQUIRE(transcribe::score_confidence(editor.document()).notes[0].band == "user-edited");
    REQUIRE(editor.document().source_events[0].midi_pitch == 60);
    editor.undo(editor.document().revision);
    REQUIRE(transcribe::score_confidence(editor.document()).notes[0].band == "high");
    auto mismatched = score;
    mismatched.source_events[0].midi_pitch = 61;
    REQUIRE(transcribe::score_confidence(mismatched).notes[0].band == "user-edited");
}

TEST_CASE("confidence handles empty scores validates options and emits JSON safe note IDs",
          "[confidence]") {
    const auto empty = transcribe::make_empty_score_document("empty");
    const auto report = transcribe::score_confidence(empty);
    REQUIRE(report.phrases.empty());
    REQUIRE(report.transcription.note_count == 0);
    REQUIRE_FALSE(report.transcription.mean_evidence_score);
    REQUIRE_THROWS_AS(transcribe::score_confidence(empty, {0}), std::invalid_argument);
    REQUIRE_THROWS_AS(transcribe::score_confidence(empty,
        {std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
    transcribe::NoteConfidence note;
    note.note_id = "quote\"\n";
    REQUIRE(transcribe::serialize_note_confidence(note).find("quote\\\"\\u000a") != std::string::npos);
}

TEST_CASE("MusicXML confidence metadata retains note links and phrase summary without project changes",
          "[confidence][interpretation]") {
    const auto score = document({event(60, 0, 1, .9)});
    const auto path = std::filesystem::temp_directory_path() /
        ("confidence-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".musicxml");
    transcribe::write_musicxml(path, score);
    std::ifstream input(path);
    const std::string contents{std::istreambuf_iterator<char>(input), {}};
    input.close();
    std::filesystem::remove(path);
    REQUIRE(contents.find("name=\"transcriber-confidence-v1\"") != std::string::npos);
    REQUIRE(contents.find(score.notes[0].id) != std::string::npos);
    REQUIRE(contents.find("mean_evidence_score") != std::string::npos);
    REQUIRE(contents.find("note-weighted-means-not-all-notes-correct-probability") != std::string::npos);
}

TEST_CASE("calibration diagnostics report exact Brier and expected calibration error with data gates",
          "[confidence][review]") {
    const transcribe::CalibrationDatasetInfo info{"held-out-fixture", true, true, true};
    const std::vector<transcribe::CalibrationSample> samples{
        {.75, true}, {.75, true}, {.75, true}, {.75, false}};
    const auto report = transcribe::evaluate_calibration(samples, info, 10, 4);
    REQUIRE(report.available);
    REQUIRE(report.brier_score == Catch::Approx(.1875));
    REQUIRE(report.expected_calibration_error == Catch::Approx(0));
    const auto uncalibrated = transcribe::evaluate_calibration(
        std::vector<transcribe::CalibrationSample>{{.5, true}, {.5, true}}, info, 10, 2);
    REQUIRE(uncalibrated.expected_calibration_error == Catch::Approx(.5));
    const auto insufficient = transcribe::evaluate_calibration(samples, info, 10, 100);
    REQUIRE_FALSE(insufficient.available);
    REQUIRE_FALSE(insufficient.expected_calibration_error);
    const auto unequal_bins = transcribe::evaluate_calibration(
        std::vector<transcribe::CalibrationSample>{
            {.2, true}, {.2, false}, {.2, false}, {.8, true}}, info, 10, 4);
    REQUIRE(unequal_bins.brier_score == Catch::Approx(.19));
    REQUIRE(unequal_bins.expected_calibration_error == Catch::Approx(.15));
    REQUIRE(unequal_bins.reliability_bins.size() == 10);
}
