#include "transcribe/musical_analysis.hpp"
#include "transcribe/musicxml.hpp"
#include "transcribe/score.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>

namespace {

transcribe::CanonicalNoteEvent event(int pitch, double start, double end) {
    transcribe::CanonicalNoteEvent value;
    value.id = std::to_string(pitch) + "-" + std::to_string(start);
    value.midi_pitch = pitch;
    value.onset_seconds = start;
    value.offset_seconds = end;
    value.state = transcribe::EventLifecycle::Stabilized;
    value.source_sample_rate = 48000;
    value.onset_sample = static_cast<std::uint64_t>(start * 48000);
    value.offset_sample = static_cast<std::uint64_t>(end * 48000);
    value.source_model = "musical-fixture";
    value.calibrated_confidence = .37;
    return value;
}

transcribe::InterpretationOptions fixed_context() {
    transcribe::InterpretationOptions options;
    options.tempo_bpm = 120;
    options.meter = transcribe::MeterConstraint{4, 4, {}};
    options.key_fifths = 0;
    options.minor_key = false;
    options.infer_local_keys = false;
    return options;
}

std::string xml(const transcribe::ScoreInterpretation& score) {
    const auto path = std::filesystem::temp_directory_path() /
        ("musical-analysis-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".musicxml");
    transcribe::write_musicxml(path, score);
    std::ifstream input(path);
    const std::string result{std::istreambuf_iterator<char>(input), {}};
    input.close();
    std::filesystem::remove(path);
    return result;
}

} // namespace

TEST_CASE("musical analysis preserves evidence and exports overlapping chord harmony",
          "[musical-analysis][interpretation]") {
    const std::vector events{event(60, 0, 2), event(64, .5, 1.5), event(67, .5, 1.5),
                            event(61, 2, 2.5)};
    const auto interpretation = transcribe::interpret_note_events(events, fixed_context());
    const auto& hypothesis = interpretation.hypotheses.front();
    const auto analysis = transcribe::analyze_musical_structure(hypothesis);
    REQUIRE(analysis.note_count == 4);
    REQUIRE(analysis.out_of_scale_notes == 1);
    REQUIRE(analysis.checks.front().code == "chromatic-note");
    REQUIRE(analysis.key_scale.duration_coverage == Catch::Approx(4.0 / 4.5));
    REQUIRE(analysis.sounding_quarters == Catch::Approx(5));
    REQUIRE(analysis.named_chord_quarters == Catch::Approx(2));
    REQUIRE(analysis.harmony.size() == 1);
    const auto& chord = analysis.harmony.front();
    REQUIRE(chord.chord.root_pitch_class == 0);
    REQUIRE(chord.chord.quality == "major");
    REQUIRE(chord.onset_quarters == Catch::Approx(1));
    REQUIRE(chord.offset_quarters == Catch::Approx(3));
    REQUIRE(chord.export_symbol);
    REQUIRE(chord.diatonic);
    REQUIRE(chord.note_indices.size() == 3);
    for (std::size_t i = 0; i < events.size(); ++i) {
        REQUIRE(interpretation.source_events[i]->midi_pitch == events[i].midi_pitch);
        REQUIRE(interpretation.source_events[i]->onset_sample == events[i].onset_sample);
        REQUIRE(interpretation.source_events[i]->offset_sample == events[i].offset_sample);
        REQUIRE(interpretation.source_events[i]->calibrated_confidence == .37);
    }
    const auto rendered = xml(interpretation);
    REQUIRE(rendered.find("<harmony><root><root-step>C</root-step></root><kind>major</kind>") !=
            std::string::npos);
    REQUIRE(rendered.find("<offset>960</offset><staff>1</staff></harmony>") != std::string::npos);
}

TEST_CASE("unknown dyad and fleeting harmony do not become confident MusicXML symbols",
          "[musical-analysis][interpretation]") {
    auto options = fixed_context();
    options.quantization_mode = transcribe::QuantizationMode::Expressive;
    const std::vector events{event(60, 0, .1), event(64, 0, .1), event(67, 0, .1),
                            event(60, 1, 2), event(61, 1, 2),
                            event(60, 3, 4), event(67, 3, 4)};
    const auto interpreted = transcribe::interpret_note_events(events, options);
    const auto analysis = transcribe::analyze_musical_structure(interpreted.hypotheses.front());
    REQUIRE(analysis.harmony.size() == 3);
    REQUIRE(std::none_of(analysis.harmony.begin(), analysis.harmony.end(),
                        [](const auto& value) { return value.export_symbol; }));
    REQUIRE(xml(interpreted).find("<harmony>") == std::string::npos);
}

TEST_CASE("analysis uses current score edits without changing canonical project serialization",
          "[musical-analysis][score]") {
    const std::vector events{event(60, 0, 1), event(64, 0, 1), event(67, 0, 1)};
    const auto interpreted = transcribe::interpret_note_events(events, fixed_context());
    auto document = transcribe::make_score_document(interpreted, 0, "musical-fixture",
        transcribe::metadata_for_hypothesis(interpreted.hypotheses.front()));
    const auto before = transcribe::serialize_score_document(document);
    const auto analyze = [](const transcribe::ScoreDocument& score) {
        return transcribe::serialize_musical_analysis(transcribe::analyze_musical_structure(
            transcribe::to_score_interpretation(score).hypotheses.front()));
    };
    REQUIRE(before.find("\"musical_analysis\"") == std::string::npos);
    REQUIRE(analyze(document).find("\"quality\":\"major\"") != std::string::npos);
    REQUIRE(transcribe::serialize_score_document(transcribe::deserialize_score_document(before)) == before);
    transcribe::ScoreEditor editor(document);
    transcribe::ScoreTransaction transaction;
    transaction.base_revision = document.revision;
    transaction.commands.push_back(transcribe::ChangeScorePitch{document.notes[1].id, 63});
    editor.apply(transaction);
    const auto after = analyze(editor.document());
    REQUIRE(after.find("\"quality\":\"minor\"") != std::string::npos);
    REQUIRE(editor.document().source_events[1].midi_pitch == 64);
    REQUIRE(editor.document().notes[1].sounding_midi_pitch == 63);
}

TEST_CASE("minor raised degrees and local tonal contexts are advisory not pitch repairs",
          "[musical-analysis][interpretation]") {
    auto options = fixed_context();
    options.minor_key = true;
    const std::vector events{event(69, 0, 1), event(68, 1, 2), event(66, 2, 3)};
    auto score = transcribe::interpret_note_events(events, options).hypotheses.front();
    const auto minor = transcribe::analyze_musical_structure(score);
    REQUIRE(minor.out_of_scale_notes == 2);
    REQUIRE(std::all_of(minor.checks.begin(), minor.checks.end(),
        [](const auto& check) { return check.code == "minor-raised-degree"; }));
    score.key = {0, false, .9, false};
    score.local_tonal_context = {{1, 2, {2, false, .9, false}}};
    const auto local = transcribe::analyze_musical_structure(score);
    // Measure one is C major; measure two is D major, so F# is in the local scale.
    REQUIRE(local.out_of_scale_notes == 1);
    score.key.user_constrained = true;
    REQUIRE(transcribe::analyze_musical_structure(score).out_of_scale_notes == 2);
}

TEST_CASE("written transposition changes chord symbols not sounding harmonic evidence",
          "[musical-analysis][interpretation]") {
    auto options = fixed_context();
    options.written_transposition_semitones = 2;
    const std::vector events{event(64, 0, 1), event(67, 0, 1), event(72, 0, 1)};
    const auto interpreted = transcribe::interpret_note_events(events, options);
    const auto analysis = transcribe::analyze_musical_structure(interpreted.hypotheses.front());
    REQUIRE(analysis.harmony.front().chord.root_pitch_class == 0);
    REQUIRE(analysis.harmony.front().chord.inversion == 1);
    // Explicit C-major is a written-key constraint; with +2 it sounds in Bb.
    REQUIRE(analysis.key_scale.tonic_pitch_class == 10);
    const auto rendered = xml(interpreted);
    REQUIRE(rendered.find("<root-step>D</root-step>") != std::string::npos);
    REQUIRE(rendered.find("<bass-step>F</bass-step><bass-alter>1</bass-alter>") != std::string::npos);
    REQUIRE(rendered.find("<inversion>1</inversion>") != std::string::npos);
}

TEST_CASE("analysis handles empty input and rejects invalid score coordinates",
          "[musical-analysis]") {
    transcribe::ScoreHypothesis score;
    const auto empty = transcribe::analyze_musical_structure(score);
    REQUIRE(empty.note_count == 0);
    REQUIRE(empty.sounding_quarters == 0);
    REQUIRE(empty.key_scale.duration_coverage == 0);
    transcribe::InterpretedNote note;
    note.sounding_midi_pitch = 60;
    note.duration_quarters = 1;
    score.notes.push_back(note);
    score.notes[0].onset_quarters = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(transcribe::analyze_musical_structure(score), std::invalid_argument);
    REQUIRE_THROWS_AS(transcribe::identify_sounding_chord(std::vector<int>{60, 128}),
                      std::invalid_argument);
}

TEST_CASE("performed accents rank meter without turning detection confidence into loudness",
          "[musical-analysis][interpretation]") {
    std::vector<transcribe::CanonicalNoteEvent> events;
    for (int i = 0; i < 32; ++i) {
        auto value = event(60, i * .5, (i + 1) * .5);
        value.intensity_rms = i % 4 == 0 ? .9 : .3;
        value.calibrated_confidence = .8;
        events.push_back(value);
    }
    transcribe::InterpretationOptions options;
    options.tempo_bpm = 120;
    options.key_fifths = 0;
    const auto inferred = transcribe::interpret_note_events(events, options);
    REQUIRE(inferred.hypotheses.front().meter.numerator == 4);
    REQUIRE(inferred.hypotheses.front().meter.denominator == 4);
    const auto& hypotheses = inferred.hypotheses;
    for (std::size_t i = 1; i < hypotheses.size(); ++i)
        REQUIRE(hypotheses[i].confidence <= hypotheses[i - 1].confidence);
    for (auto& value : events) {
        value.velocity = static_cast<int>(*value.intensity_rms * 127);
        value.intensity_rms.reset();
    }
    REQUIRE(transcribe::interpret_note_events(events, options).hypotheses.front().meter.numerator == 4);
    options.meter = transcribe::MeterConstraint{3, 4, {}};
    REQUIRE(transcribe::interpret_note_events(events, options).hypotheses.front().meter.numerator == 3);
    options.meter.reset();
    for (auto& value : events) {
        value.calibrated_confidence = *value.velocity / 127.0;
        value.velocity.reset();
    }
    REQUIRE(transcribe::interpret_note_events(events, options).hypotheses.front().meter_ambiguous);
    events[0].intensity_rms = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(transcribe::interpret_note_events(events, options), std::invalid_argument);
}

TEST_CASE("rejected evidence is retained but excluded from musical conclusions",
          "[musical-analysis][interpretation]") {
    std::vector events{event(60, 0, 1), event(64, 0, 1), event(67, 0, 1), event(61, 0, 1)};
    events.back().state = transcribe::EventLifecycle::Rejected;
    transcribe::EventEvidence evidence;
    evidence.time_seconds = .1;
    evidence.midi_pitch = 60;
    evidence.raw_strength = .42;
    evidence.raw_voicing = .61;
    events[0].evidence.push_back(evidence);
    const auto interpreted = transcribe::interpret_note_events(events, fixed_context());
    const auto analysis = transcribe::analyze_musical_structure(interpreted.hypotheses.front());
    REQUIRE(interpreted.source_events.size() == 4);
    REQUIRE(interpreted.source_events.back()->state == transcribe::EventLifecycle::Rejected);
    REQUIRE(interpreted.source_events[0]->evidence.front().raw_strength == .42);
    REQUIRE(interpreted.source_events[0]->evidence.front().raw_voicing == .61);
    REQUIRE(analysis.note_count == 3);
    REQUIRE(analysis.out_of_scale_notes == 0);
    REQUIRE(analysis.harmony.front().chord.quality == "major");
}

TEST_CASE("scale alternatives retain modal and harmonic minor readings without changing the key",
          "[musical-analysis][interpretation]") {
    auto options = fixed_context();
    options.key_fifths = -3;
    options.minor_key = true;
    std::vector<transcribe::CanonicalNoteEvent> events;
    for (const int pitch : {60, 62, 63, 65, 67, 68, 71}) {
        const double start = events.size() * .5;
        events.push_back(event(pitch, start, start + .5));
    }
    const auto interpretation = transcribe::interpret_note_events(events, options);
    const auto analysis = transcribe::analyze_musical_structure(interpretation.hypotheses.front());
    REQUIRE(analysis.key_scale.mode == "natural-minor");
    REQUIRE(analysis.key_scale.tonic_pitch_class == 0);
    REQUIRE(analysis.key_scale.duration_coverage == Catch::Approx(6.0 / 7));
    REQUIRE(std::any_of(analysis.alternative_scales.begin(), analysis.alternative_scales.end(),
        [](const auto& scale) {
            return scale.mode == "harmonic-minor" && scale.tonic_pitch_class == 0 &&
                   scale.duration_coverage > .999;
        }));
    REQUIRE(analysis.checks.size() == 1);
    REQUIRE(analysis.checks.front().code == "minor-raised-degree");
    REQUIRE(interpretation.hypotheses.front().key.fifths == -3);
}

TEST_CASE("repeated identical chords produce one symbol and barline changes remain positioned",
          "[musical-analysis][interpretation]") {
    std::vector<transcribe::CanonicalNoteEvent> events;
    for (int beat = 0; beat < 8; ++beat) {
        for (int pitch : (beat < 4 ? std::vector<int>{60, 64, 67} :
                                    std::vector<int>{62, 65, 69}))
            events.push_back(event(pitch, beat * .5, (beat + 1) * .5));
    }
    const auto interpreted = transcribe::interpret_note_events(events, fixed_context());
    const auto rendered = xml(interpreted);
    std::size_t count = 0, position = 0;
    while ((position = rendered.find("<harmony>", position)) != std::string::npos) {
        ++count;
        ++position;
    }
    REQUIRE(count == 2);
    const auto second_measure = rendered.find("<measure number=\"2\"");
    const auto second_symbol = rendered.find("<harmony>", rendered.find("<harmony>") + 1);
    REQUIRE(second_measure != std::string::npos);
    REQUIRE(second_symbol > second_measure);
    REQUIRE(rendered.find("<kind>minor</kind><offset>0</offset>", second_symbol) != std::string::npos);
}

TEST_CASE("MusicXML seventh chord kinds use schema vocabulary and ambiguous symmetric roots abstain",
          "[musical-analysis][interpretation]") {
    const std::vector events{event(60, 0, 1), event(64, 0, 1), event(67, 0, 1), event(70, 0, 1),
                            event(62, 1, 2), event(65, 1, 2), event(68, 1, 2), event(72, 1, 2),
                            event(60, 2, 3), event(64, 2, 3), event(68, 2, 3)};
    const auto interpretation = transcribe::interpret_note_events(events, fixed_context());
    const auto rendered = xml(interpretation);
    REQUIRE(rendered.find("<kind>dominant</kind>") != std::string::npos);
    REQUIRE(rendered.find("<kind>half-diminished</kind>") != std::string::npos);
    REQUIRE(rendered.find("<kind>dominant-seventh</kind>") == std::string::npos);
    REQUIRE(rendered.find("<kind>augmented</kind>") == std::string::npos);
}
