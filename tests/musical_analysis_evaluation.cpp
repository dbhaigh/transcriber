#include "transcribe/musical_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <stdexcept>

namespace {

struct Fixture {
    std::string name;
    int tonic{};
    bool minor{};
    double tempo{};
    int beats{4};
    std::vector<transcribe::CanonicalNoteEvent> notes;
    struct Chord { double start; double end; int root; std::string quality; };
    std::vector<Chord> chords;
};

Fixture progression(int tonic, bool minor, double tempo, int beats) {
    Fixture fixture{"tonic-" + std::to_string(tonic) + (minor ? "-minor" : "-major") +
                    "-" + std::to_string(beats) + "-4", tonic, minor, tempo, beats, {}, {}};
    // Two I-IV-V-I / i-iv-v-i cadences, with simultaneous attacks and half-open releases.
    const int degrees[]{0, 5, 7, 0, 0, 5, 7, 0};
    const double beat_seconds = 60.0 / tempo;
    for (int bar = 0; bar < 8; ++bar) {
        const int root = (tonic + degrees[bar]) % 12;
        const bool minor_chord = minor;
        fixture.chords.push_back({double(bar * beats), double((bar + 1) * beats),
                                  root, minor_chord ? "minor" : "major"});
        for (int beat = 0; beat < beats; ++beat) {
            for (int interval : {0, minor_chord ? 3 : 4, 7}) {
                transcribe::CanonicalNoteEvent event;
                event.id = "note-" + std::to_string(fixture.notes.size());
                event.midi_pitch = 48 + root + interval;
                event.onset_seconds = (bar * beats + beat) * beat_seconds;
                event.offset_seconds = (bar * beats + beat + 1) * beat_seconds;
                event.source_sample_rate = 48000;
                event.onset_sample = static_cast<std::uint64_t>(std::llround(event.onset_seconds * 48000));
                event.offset_sample = static_cast<std::uint64_t>(std::llround(event.offset_seconds * 48000));
                event.state = transcribe::EventLifecycle::Stabilized;
                event.calibrated_confidence = .8;
                event.intensity_rms = beat == 0 ? .9 : .3;
                event.source_model = "symbolic-ground-truth";
                transcribe::EventEvidence evidence;
                evidence.time_seconds = event.onset_seconds;
                evidence.midi_pitch = static_cast<int>(event.midi_pitch);
                evidence.raw_strength = .73;
                evidence.onset_strength = .81;
                event.evidence.push_back(evidence);
                fixture.notes.push_back(event);
            }
        }
    }
    return fixture;
}

int evaluate(std::ostream& out) {
    std::size_t count = 0, keys_correct = 0, meters_correct = 0, tempos_correct = 0;
    std::size_t meter_abstentions = 0, key_abstentions = 0, source_count = 0, preserved = 0;
    double tempo_error_sum = 0, chord_correct = 0, chord_total = 0, scale_sum = 0;
    out.imbue(std::locale::classic());
    out << std::setprecision(10)
        << "{\"schema_version\":1,\"corpus\":\"musical-cadences-v1\","
           "\"scope\":\"symbolic fixtures, not real-audio accuracy\","
           "\"protocol\":{\"tempo_tolerance_relative\":0.04,"
           "\"chord_metric\":\"duration-weighted root and quality accuracy\","
           "\"meter_metric\":\"exact numerator and denominator; ambiguity reported separately\","
           "\"key_metric\":\"exact tonic and major/minor, enharmonic equivalent\"},"
           "\"per_fixture\":[";
    for (int tonic = 0; tonic < 12; ++tonic) {
        for (bool minor : {false, true}) {
            const double tempi[]{60, 90, 120, 180};
            const auto fixture = progression(tonic, minor, tempi[tonic % 4],
                                             tonic % 2 ? 3 : 4);
            transcribe::InterpretationOptions options;
            options.infer_local_keys = false;
            options.detect_tempo_changes = false;
            const auto interpreted = transcribe::interpret_note_events(fixture.notes, options);
            const auto& score = interpreted.hypotheses.front();
            const auto analysis = transcribe::analyze_musical_structure(score);
            const bool key_ok = analysis.key_scale.tonic_pitch_class == fixture.tonic &&
                                score.key.minor == fixture.minor;
            const bool meter_ok = score.meter.numerator == fixture.beats &&
                                  score.meter.denominator == 4;
            const double tempo_error = std::abs(score.tempo_map.front().beats_per_minute -
                                                fixture.tempo) / fixture.tempo;
            ++count;
            keys_correct += key_ok;
            meters_correct += meter_ok;
            tempos_correct += tempo_error <= .04;
            tempo_error_sum += tempo_error;
            meter_abstentions += score.meter_ambiguous;
            key_abstentions += score.key_ambiguous;
            scale_sum += analysis.key_scale.duration_coverage;
            double correct = 0;
            const double reference_seconds_per_quarter = 60.0 / fixture.tempo;
            const double estimated_seconds_per_quarter = 60.0 / score.tempo_map.front().beats_per_minute;
            for (const auto& reference : fixture.chords) {
                chord_total += (reference.end - reference.start) * reference_seconds_per_quarter;
                for (const auto& hypothesis : analysis.harmony) {
                    if (hypothesis.chord.root_pitch_class == reference.root &&
                        hypothesis.chord.quality == reference.quality) {
                        const double overlap = std::max(0.0, std::min(
                            reference.end * reference_seconds_per_quarter,
                            hypothesis.offset_quarters * estimated_seconds_per_quarter) - std::max(
                            reference.start * reference_seconds_per_quarter,
                            hypothesis.onset_quarters * estimated_seconds_per_quarter));
                        correct += overlap;
                    }
                }
            }
            chord_correct += correct;
            for (std::size_t i = 0; i < fixture.notes.size(); ++i) {
                ++source_count;
                const auto& a = fixture.notes[i];
                const auto& b = *interpreted.source_events[i];
                preserved += a.id == b.id && a.midi_pitch == b.midi_pitch &&
                    a.onset_seconds == b.onset_seconds && a.offset_seconds == b.offset_seconds &&
                    a.onset_sample == b.onset_sample && a.offset_sample == b.offset_sample &&
                    a.source_sample_rate == b.source_sample_rate &&
                    a.calibrated_confidence == b.calibrated_confidence &&
                    a.intensity_rms == b.intensity_rms && a.state == b.state &&
                    a.source_model == b.source_model && a.evidence.size() == b.evidence.size() &&
                    a.evidence.front().time_seconds == b.evidence.front().time_seconds &&
                    a.evidence.front().midi_pitch == b.evidence.front().midi_pitch &&
                    a.evidence.front().raw_strength == b.evidence.front().raw_strength &&
                    a.evidence.front().onset_strength == b.evidence.front().onset_strength;
            }
            if (count > 1) out << ',';
            out << "{\"name\":\"" << fixture.name << "\",\"key_correct\":" << (key_ok ? "true" : "false")
                << ",\"meter_correct\":" << (meter_ok ? "true" : "false")
                << ",\"estimated_meter_numerator\":" << score.meter.numerator
                << ",\"estimated_meter_denominator\":" << score.meter.denominator
                << ",\"meter_ambiguous\":" << (score.meter_ambiguous ? "true" : "false")
                << ",\"key_ambiguous\":" << (score.key_ambiguous ? "true" : "false")
                << ",\"tempo_relative_error\":" << tempo_error
                << ",\"chord_root_quality_accuracy\":" << correct /
                    (8 * fixture.beats * reference_seconds_per_quarter)
                << ",\"scale_duration_coverage\":" << analysis.key_scale.duration_coverage << '}';
        }
    }
    out << "],\"metrics\":{\"fixtures\":" << count
        << ",\"key_accuracy\":" << double(keys_correct) / count
        << ",\"key_ambiguity_rate\":" << double(key_abstentions) / count
        << ",\"meter_accuracy\":" << double(meters_correct) / count
        << ",\"meter_ambiguity_rate\":" << double(meter_abstentions) / count
        << ",\"tempo_accuracy\":" << double(tempos_correct) / count
        << ",\"mean_tempo_relative_error\":" << tempo_error_sum / count
        << ",\"chord_root_quality_accuracy\":" << chord_correct / chord_total
        << ",\"mean_scale_duration_coverage\":" << scale_sum / count
        << ",\"source_events\":" << source_count
        << ",\"evidence_preservation\":" << double(preserved) / source_count << "}}\n";
    return preserved == source_count && tempos_correct == count &&
           keys_correct == count && meters_correct == count &&
           chord_correct / chord_total >= .99 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("Usage: transcriber-musical-evaluate [output.json]");
        if (argc == 1) return evaluate(std::cout);
        std::ofstream output(argv[1]);
        if (!output) throw std::runtime_error("Cannot open musical evaluation output");
        const int result = evaluate(output);
        output.close();
        if (!output) throw std::runtime_error("Cannot write musical evaluation output");
        return result;
    } catch (const std::exception& error) {
        std::cerr << "Musical evaluation failed: " << error.what() << '\n';
        return 1;
    }
}
