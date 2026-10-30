"""Unit tests for the real-recording polyphonic evaluator (stdlib only)."""

import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from evaluate_polyphonic import (chord_frame_counts, chord_summary, markdown_summary,  # noqa: E402
                                 note_counts, overlapping_reference_indices, paired_bootstrap,
                                 parse_harte, predicted_family, read_midi_notes,
                                 read_performed_chords)


def variable_length(value: int) -> bytes:
    groups = [value & 0x7F]
    value >>= 7
    while value:
        groups.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(groups))


def track(events: list[tuple[int, bytes]]) -> bytes:
    body = b"".join(variable_length(delta) + payload for delta, payload in events)
    body += b"\x00\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(body)) + body


def midi_file(*tracks: bytes, division: int = 480) -> bytes:
    return b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), division) + b"".join(tracks)


class HarteTests(unittest.TestCase):
    def test_roots_and_families(self):
        self.assertEqual(parse_harte("C:maj"), (0, "maj"))
        self.assertEqual(parse_harte("A:min7"), (9, "min"))
        self.assertEqual(parse_harte("Bb:7/1"), (10, "maj"))
        self.assertEqual(parse_harte("D#:maj6(2)/5"), (3, "maj"))
        self.assertEqual(parse_harte("Cb:min"), (11, "min"))
        self.assertEqual(parse_harte("G#:hdim7"), (8, None))
        self.assertEqual(parse_harte("E:(1,5)/1"), (4, None))
        self.assertEqual(parse_harte("F"), (5, "maj"))
        self.assertEqual(parse_harte("N"), (None, None))
        with self.assertRaises(ValueError):
            parse_harte("H:maj")

    def test_predicted_family(self):
        self.assertEqual(predicted_family("dominant-seventh"), "maj")
        self.assertEqual(predicted_family("minor-sixth"), "min")
        self.assertIsNone(predicted_family("power"))
        self.assertIsNone(predicted_family("unknown"))

    def test_selects_performed_annotation(self):
        jams = {"annotations": [
            {"namespace": "chord", "annotation_metadata": {"data_source": ""},
             "data": [{"time": 0, "duration": 1, "value": "C:maj"}]},
            {"namespace": "chord", "annotation_metadata": {
                "data_source": "Semi-automatic chord transcription with manual verification"},
             "data": [{"time": 0, "duration": 1, "value": "A:min7/1"},
                      {"time": 1, "duration": 0, "value": "C:maj"}]}]}
        self.assertEqual(read_performed_chords(jams),
                         [{"start": 0.0, "end": 1.0, "root": 9, "family": "min"}])
        with self.assertRaises(ValueError):
            read_performed_chords({"annotations": jams["annotations"][:1]})


class ChordScoringTests(unittest.TestCase):
    def test_root_majmin_and_coverage(self):
        references = [{"start": 0.0, "end": 1.0, "root": 0, "family": "maj"},
                      {"start": 1.0, "end": 2.0, "root": 9, "family": None},
                      {"start": 2.0, "end": 3.0, "root": None, "family": None}]
        predictions = [
            {"onset_seconds": 0.0, "offset_seconds": 0.5, "root_pitch_class": 0, "quality": "major"},
            {"onset_seconds": 0.5, "offset_seconds": 1.0, "root_pitch_class": 0, "quality": "minor"},
            {"onset_seconds": 1.0, "offset_seconds": 1.5, "root_pitch_class": 9, "quality": "power"},
            {"onset_seconds": 1.5, "offset_seconds": 3.0, "root_pitch_class": None, "quality": "unknown"},
        ]
        counts = chord_frame_counts(references, predictions, 3.0)
        self.assertEqual(counts, dict(reference_frames=200, covered_frames=150, root_correct=150,
                                      majmin_frames=100, majmin_correct=50))
        summary = chord_summary(counts)
        self.assertAlmostEqual(summary["coverage"], 0.75)
        self.assertAlmostEqual(summary["root_recall"], 0.75)
        self.assertAlmostEqual(summary["majmin_recall"], 0.5)
        self.assertEqual(chord_summary(chord_frame_counts([], predictions, 3.0))["root_recall"], 0.0)


class NoteScoringTests(unittest.TestCase):
    def test_overlap_requires_simultaneous_different_pitches(self):
        references = [{"midi": 60, "start": 0.0, "duration": 1.0},
                      {"midi": 64, "start": 0.5, "duration": 1.0},
                      {"midi": 67, "start": 2.0, "duration": 0.5},
                      {"midi": 67, "start": 2.2, "duration": 0.5}]
        self.assertEqual(overlapping_reference_indices(references), {0, 1})

    def test_counts(self):
        references = [{"midi": 60, "start": 0.0, "duration": 1.0},
                      {"midi": 64, "start": 0.0, "duration": 1.0},
                      {"midi": 67, "start": 2.0, "duration": 0.5}]
        predictions = [{"midi": 60, "start": 0.02, "duration": 0.98},
                       {"midi": 64, "start": 0.01, "duration": 0.3},
                       {"midi": 72, "start": 2.0, "duration": 0.5}]
        counts = note_counts(references, predictions, 3.0)
        self.assertEqual(counts["onset_pitch"], (2, 1, 1))
        self.assertEqual(counts["note_event"], (1, 2, 2))
        self.assertEqual(counts["overlapping_onset_recall"], (2, 0, 0))

    def test_paired_bootstrap_is_deterministic_and_bounded(self):
        def clip(candidate, baseline):
            return {"dataset": "x", "systems": {
                "new": {"metrics": {"note_event": dict(zip(("tp", "fp", "fn"), candidate))}},
                "old": {"metrics": {"note_event": dict(zip(("tp", "fp", "fn"), baseline))}}}}
        clips = [clip((8, 2, 2), (4, 6, 6)), clip((5, 5, 5), (5, 5, 5)), clip((9, 1, 1), (3, 7, 7))]
        first = paired_bootstrap(clips, "new", "old", "note_event", resamples=200)
        self.assertEqual(first, paired_bootstrap(clips, "new", "old", "note_event", resamples=200))
        self.assertAlmostEqual(first["delta_f1"], 22 / 30 - 12 / 30)
        self.assertLessEqual(first["ci95_low"], first["delta_f1"])
        self.assertGreaterEqual(first["ci95_high"], first["delta_f1"])
        self.assertGreaterEqual(first["ci95_low"], 0.0)
        self.assertIsNone(paired_bootstrap(clips, "new", "missing", "note_event"))
        summary = markdown_summary({"results": {}, "chords": {},
                                    "paired_differences": {"x": {"new-vs-old": {"note_event": first}}}})
        self.assertIn("| new-vs-old | note_event | +0.333 |", summary)


class MidiTests(unittest.TestCase):
    def test_tempo_map_running_status_and_multiple_tracks(self):
        conductor = track([(0, b"\xff\x51\x03" + (500000).to_bytes(3, "big")),
                           (960, b"\xff\x51\x03" + (1000000).to_bytes(3, "big"))])
        part = track([(0, b"\x90\x3c\x64"), (480, b"\x3c\x00"),  # running-status note-off
                      (480, b"\x91\x40\x50"), (480, b"\x81\x40\x00"),
                      (0, b"\xc0\x05"), (0, b"\xf0\x01\xf7")])
        notes = read_midi_notes(midi_file(conductor, part))
        self.assertEqual([n["midi"] for n in notes], [60, 64])
        self.assertAlmostEqual(notes[0]["start"], 0.0)
        self.assertAlmostEqual(notes[0]["duration"], 0.5)
        self.assertAlmostEqual(notes[1]["start"], 1.0)
        self.assertAlmostEqual(notes[1]["duration"], 1.0)

    def test_overlapping_same_key_is_fifo(self):
        part = track([(0, b"\x90\x3c\x64"), (240, b"\x90\x3c\x64"),
                      (240, b"\x80\x3c\x00"), (240, b"\x80\x3c\x00")])
        notes = read_midi_notes(midi_file(part))
        self.assertEqual([(round(n["start"], 3), round(n["duration"], 3)) for n in notes],
                         [(0.0, 0.5), (0.25, 0.5)])

    def test_rejects_invalid_files(self):
        with self.assertRaises(ValueError):
            read_midi_notes(b"RIFF")
        with self.assertRaises(ValueError):
            read_midi_notes(b"MThd" + struct.pack(">IHHH", 6, 1, 1, 0xE728) + track([]))


if __name__ == "__main__":
    unittest.main()
