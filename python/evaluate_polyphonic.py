#!/usr/bin/env python3
"""Real-recording evaluation of the tracked polyphonic pipeline against existing paths.

Runs the built CLI per recording (process-cold), scores notes with the shared
evaluation metrics, and scores identified chords against GuitarSet's performed
chord annotations. Datasets are read in place and never copied into the repository:

* GuitarSet v1.1.0 mono-microphone audio (performer-grouped split; default: test).
* TRIOS chamber-music mixes: real, separately recorded parts with aligned MIDI.
  It contains no guitar, giving an instrument-disjoint check of the same thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import wave
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from evaluation_metrics import checked_metrics, distribution, event_pairs, frame_counts
from evaluation_datasets import read_midi_notes
from evaluate_guitarset import (
    ANNOTATION_MD5, AUDIO_MD5, BUNDLED_MODEL_SHA256, guitarset_dataset_split,
    guitarset_split_policy, md5_file, read_reference_events, sha256_file,
)

SCHEMA_VERSION = 1
CHORD_FRAME_SECONDS = 0.01
NOTE_METRICS = ("onset_pitch", "note_event", "frame_10ms", "overlapping_onset_recall")
SYSTEMS: dict[str, tuple[bool, list[str]]] = {
    "yin": (False, []),
    "spectral": (False, ["--polyphonic"]),
    "spectral_tracked": (False, ["--polyphonic", "--polyphonic-decoder", "tracked"]),
    "basic_pitch": (True, ["--polyphonic"]),
    "basic_pitch_tracked": (True, ["--polyphonic", "--polyphonic-decoder", "tracked"]),
}
# TRIOS percussion parts are unpitched General MIDI drum notes, not pitch references.
TRIOS_UNPITCHED_PARTS = frozenset({"kick", "snare", "ride", "hihat", "drums"})
PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MAJOR_QUALITIES = frozenset({"maj", "maj7", "maj6", "maj9", "7", "9", "11", "13"})
MINOR_QUALITIES = frozenset({"min", "min7", "min6", "min9", "min11", "min13"})
PREDICTED_MAJOR = frozenset({"major", "major-seventh", "dominant-seventh", "major-sixth"})
PREDICTED_MINOR = frozenset({"minor", "minor-seventh", "minor-sixth"})


def parse_harte(label: str) -> tuple[int | None, str | None]:
    """Return (root pitch class, MIREX maj/min class or None) for a Harte chord label."""
    if label in ("N", "X", ""):
        return None, None
    root_text, _, quality_text = label.partition(":")
    if not root_text or root_text[0] not in PITCH_CLASSES:
        raise ValueError(f"Unsupported chord root in {label!r}")
    root = PITCH_CLASSES[root_text[0]]
    for modifier in root_text[1:]:
        if modifier == "#":
            root += 1
        elif modifier == "b":
            root -= 1
        else:
            raise ValueError(f"Unsupported chord root in {label!r}")
    base = (quality_text or "maj").split("/", 1)[0].split("(", 1)[0]
    if base in MAJOR_QUALITIES:
        family = "maj"
    elif base in MINOR_QUALITIES:
        family = "min"
    else:
        family = None
    return root % 12, family


def predicted_family(quality: str) -> str | None:
    if quality in PREDICTED_MAJOR:
        return "maj"
    if quality in PREDICTED_MINOR:
        return "min"
    return None


def read_performed_chords(jams_data: dict[str, Any]) -> list[dict[str, Any]]:
    """GuitarSet's verified performed-chord annotation (not the instructed lead sheet)."""
    chords = [a for a in jams_data.get("annotations", []) if a.get("namespace") == "chord"]
    performed = [a for a in chords if "Semi-automatic" in
                 str(a.get("annotation_metadata", {}).get("data_source", ""))]
    if len(performed) != 1:
        raise ValueError(f"Expected one performed chord annotation, found {len(performed)}")
    result = []
    for observation in performed[0]["data"]:
        start, duration = float(observation["time"]), float(observation["duration"])
        if duration > 0:
            root, family = parse_harte(str(observation["value"]))
            result.append({"start": start, "end": start + duration, "root": root, "family": family})
    return result


def chord_frame_counts(references: list[dict[str, Any]], predictions: list[dict[str, Any]],
                       duration: float, frame_seconds: float = CHORD_FRAME_SECONDS) -> dict[str, int]:
    """Frame-centre chord recall counts (MIREX-style root and majmin vocabularies).

    Reference frames without a chord are ignored. A reference frame with no named
    predicted chord counts as incorrect; maj/min recall excludes references outside
    that vocabulary, as MIREX does.
    """
    def at(intervals: list[dict[str, Any]], start_key: str, end_key: str, time_value: float,
           cursor: list[int]) -> dict[str, Any] | None:
        while cursor[0] < len(intervals) and intervals[cursor[0]][end_key] <= time_value:
            cursor[0] += 1
        if cursor[0] < len(intervals) and intervals[cursor[0]][start_key] <= time_value:
            return intervals[cursor[0]]
        return None

    refs = sorted(references, key=lambda c: c["start"])
    named = sorted((c for c in predictions if c.get("root_pitch_class") is not None),
                   key=lambda c: c["onset_seconds"])
    counts = dict(reference_frames=0, covered_frames=0, root_correct=0,
                  majmin_frames=0, majmin_correct=0)
    ref_cursor, prediction_cursor = [0], [0]
    for frame in range(int(duration / frame_seconds)):
        time_value = (frame + 0.5) * frame_seconds
        reference = at(refs, "start", "end", time_value, ref_cursor)
        if reference is None or reference["root"] is None:
            continue
        prediction = at(named, "onset_seconds", "offset_seconds", time_value, prediction_cursor)
        counts["reference_frames"] += 1
        root_ok = prediction is not None and prediction["root_pitch_class"] == reference["root"]
        counts["covered_frames"] += prediction is not None
        counts["root_correct"] += root_ok
        if reference["family"] is not None:
            counts["majmin_frames"] += 1
            counts["majmin_correct"] += root_ok and predicted_family(
                prediction["quality"]) == reference["family"]
    return counts


def chord_summary(counts: dict[str, int]) -> dict[str, float | int]:
    ratio = lambda a, b: counts[a] / counts[b] if counts[b] else 0.0
    return dict(counts, coverage=ratio("covered_frames", "reference_frames"),
                root_recall=ratio("root_correct", "reference_frames"),
                majmin_recall=ratio("majmin_correct", "majmin_frames"))


def overlapping_reference_indices(references: list[dict[str, Any]]) -> set[int]:
    """References sounding at the same time as another reference of a different pitch."""
    order = sorted(range(len(references)), key=lambda i: references[i]["start"])
    result: set[int] = set()
    active: list[int] = []
    for index in order:
        start = references[index]["start"]
        active = [i for i in active if references[i]["start"] + references[i]["duration"] > start]
        for other in active:
            if references[other]["midi"] != references[index]["midi"]:
                result.update((index, other))
        active.append(index)
    return result


def note_counts(references, predictions, duration: float) -> dict[str, tuple[int, int, int]]:
    onset_pairs = event_pairs(references, predictions, require_offset=False)
    note_tp = len(event_pairs(references, predictions, require_offset=True))
    overlapping = overlapping_reference_indices(references)
    overlap_tp = sum(1 for reference, _ in onset_pairs if reference in overlapping)
    return {
        "onset_pitch": (len(onset_pairs), len(predictions) - len(onset_pairs),
                        len(references) - len(onset_pairs)),
        "note_event": (note_tp, len(predictions) - note_tp, len(references) - note_tp),
        "frame_10ms": frame_counts(references, predictions, duration),
        # Recall only: unmatched predictions cannot be attributed to overlap regions.
        "overlapping_onset_recall": (overlap_tp, 0, len(overlapping) - overlap_tp),
    }


def run_cli(executable: Path, model: Path | None, system: str, audio: Path,
            directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None, float]:
    needs_model, arguments = SYSTEMS[system]
    tracked = "tracked" in arguments
    notes_path, events_path = directory / f"{system}.json", directory / f"{system}.events.json"
    for path in (notes_path, events_path):
        path.unlink(missing_ok=True)
    command = [str(executable), str(audio), "--json", str(notes_path), "--quiet", *arguments]
    if needs_model:
        command[2:2] = ["--model", str(model)]
    if tracked:
        command += ["--events-json", str(events_path)]
    environment = dict(os.environ, PATH=os.pathsep.join([str(executable.parent), os.environ.get("PATH", "")]))
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    elapsed = time.perf_counter() - started
    if result.returncode:
        message = (result.stderr + result.stdout).strip()
        if "No stable pitches were detected" in message and not notes_path.exists():
            return [], None, elapsed
        raise RuntimeError(f"{system} failed for {audio.name} (exit {result.returncode}): {message}")
    notes = [{"midi": int(n["midi"]), "start": float(n["start"]), "duration": float(n["duration"])}
             for n in json.loads(notes_path.read_text(encoding="utf-8")).get("notes", [])]
    events = json.loads(events_path.read_text(encoding="utf-8")) if tracked else None
    return notes, events, elapsed


def guitarset_items(audio_zip: Path, annotation_zip: Path, split: str, limit: int | None,
                    workspace: Path) -> Iterable[dict[str, Any]]:
    with zipfile.ZipFile(audio_zip) as audio, zipfile.ZipFile(annotation_zip) as annotations:
        wavs = {PurePosixPath(n).name: n for n in audio.namelist() if n.lower().endswith(".wav")}
        names = sorted(n for n in annotations.namelist() if n.lower().endswith(".jams")
                       and (split == "all" or guitarset_dataset_split(PurePosixPath(n).stem) == split))
        for name in names[:limit]:
            stem = PurePosixPath(name).stem
            jams = json.loads(annotations.read(name))
            wav_path = workspace / f"{stem}_mic.wav"
            wav_path.write_bytes(audio.read(wavs[f"{stem}_mic.wav"]))
            yield dict(dataset="guitarset", track=stem, audio=wav_path, temporary=True,
                       references=read_reference_events(jams),
                       chords=read_performed_chords(jams) if stem.endswith("_comp") else None,
                       style="comp" if stem.endswith("_comp") else "solo")


def trios_items(root: Path) -> Iterable[dict[str, Any]]:
    pieces = sorted(p for p in root.iterdir() if (p / "mix.wav").is_file())
    if not pieces:
        raise FileNotFoundError(f"No TRIOS piece directories with mix.wav under {root}")
    for piece in pieces:
        parts = sorted(p for p in piece.glob("*.mid") if p.stem.lower() not in TRIOS_UNPITCHED_PARTS)
        references = [note for part in parts for note in read_midi_notes(part.read_bytes())]
        yield dict(dataset="trios", track=piece.name, audio=piece / "mix.wav", temporary=False,
                   references=sorted(references, key=lambda n: (n["start"], n["midi"])),
                   chords=None, style="ensemble", parts=[p.stem for p in parts])


def add_counts(target: dict[str, list[int]], counts: dict[str, Iterable[int]]) -> None:
    for metric, values in counts.items():
        totals = target.setdefault(metric, [0, 0, 0])
        for index, value in enumerate(values):
            totals[index] += value


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    executable = args.transcriber.resolve()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown or not systems:
        raise ValueError(f"Unknown systems {unknown}; choose from {sorted(SYSTEMS)}")
    model = args.model.resolve() if any(SYSTEMS[s][0] for s in systems) else None
    for path in [executable] + ([model] if model else []):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    if (args.guitarset_audio is None) != (args.guitarset_annotations is None):
        raise ValueError("Supply both GuitarSet archives or neither")
    if args.guitarset_audio is None and args.trios is None:
        raise ValueError("Supply GuitarSet archives and/or a TRIOS directory")
    dataset_info: dict[str, Any] = {}
    if args.guitarset_audio is not None:
        hashes = {"audio_md5": md5_file(args.guitarset_audio),
                  "annotation_md5": md5_file(args.guitarset_annotations)}
        if hashes != {"audio_md5": AUDIO_MD5, "annotation_md5": ANNOTATION_MD5}:
            raise ValueError(f"GuitarSet archive checksums do not match v1.1.0: {hashes}")
        dataset_info["guitarset"] = dict(version="1.1.0", doi="10.5281/zenodo.3371780",
                                         license="CC BY 4.0", audio_source="audio_mono-mic.zip",
                                         selected_split=args.guitarset_split,
                                         split_policy=guitarset_split_policy(), **hashes)
    if args.trios is not None:
        dataset_info["trios"] = dict(doi="10.5281/zenodo.6797837", audio_source="mix.wav",
                                     license="CC BY-NC-SA 2.0 UK",
                                     excluded_unpitched_parts=sorted(TRIOS_UNPITCHED_PARTS))

    totals: dict[str, dict[str, dict[str, list[int]]]] = {}
    chord_totals: dict[str, dict[str, int]] = {}
    timings: dict[str, list[float]] = {s: [] for s in systems}
    failures: list[dict[str, str]] = []
    clips: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="transcriber-polyphonic-eval-") as temporary:
        workspace = Path(temporary)
        sources: list[Iterable[dict[str, Any]]] = []
        if args.guitarset_audio is not None:
            sources.append(guitarset_items(args.guitarset_audio, args.guitarset_annotations,
                                           args.guitarset_split, args.limit, workspace))
        if args.trios is not None:
            sources.append(trios_items(args.trios))
        for source in sources:
            for item in source:
                with wave.open(str(item["audio"]), "rb") as wav:
                    duration = wav.getnframes() / wav.getframerate()
                clip = dict(dataset=item["dataset"], track=item["track"], style=item["style"],
                            duration_seconds=duration, reference_notes=len(item["references"]),
                            overlapping_reference_notes=len(
                                overlapping_reference_indices(item["references"])),
                            systems={})
                for system in systems:
                    error = None
                    try:
                        notes, events, elapsed = run_cli(executable, model, system, item["audio"],
                                                         workspace)
                    except RuntimeError as failure:
                        # A failed transcription is scored as producing no notes, not skipped.
                        notes, events, elapsed, error = [], None, float("nan"), str(failure)
                        failures.append(dict(dataset=item["dataset"], track=item["track"],
                                             system=system, error=error.splitlines()[0]))
                    if error is None:
                        timings[system].append(elapsed)
                    counts = note_counts(item["references"], notes, duration)
                    for group in (item["dataset"], f"{item['dataset']}:{item['style']}"):
                        add_counts(totals.setdefault(group, {}).setdefault(system, {}), counts)
                    result: dict[str, Any] = dict(
                        predicted_notes=len(notes),
                        process_wall_seconds=None if error else elapsed, error=error,
                        metrics={m: checked_metrics(*c) for m, c in counts.items()})
                    if events is not None:
                        result["identified_chords"] = len(events.get("chords", []))
                        result["named_chords"] = sum(
                            1 for c in events.get("chords", []) if c.get("root_pitch_class") is not None)
                        result["polyphonic_diagnostics"] = events.get("polyphonic_diagnostics")
                        if item["chords"] is not None:
                            frame_result = chord_frame_counts(item["chords"], events["chords"], duration)
                            result["chord_frames"] = chord_summary(frame_result)
                            target = chord_totals.setdefault(system, {})
                            for key, value in frame_result.items():
                                target[key] = target.get(key, 0) + value
                    clip["systems"][system] = result
                if item["temporary"]:
                    item["audio"].unlink()
                clips.append(clip)
                print(f"\rEvaluated {len(clips)}: {item['dataset']}/{item['track']}",
                      end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    model_hash = sha256_file(model) if model else None
    return {
        "schema_version": SCHEMA_VERSION,
        "executable_sha256": sha256_file(executable),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "model": None if model is None else dict(
            sha256=model_hash,
            name="Spotify Basic Pitch" if model_hash == BUNDLED_MODEL_SHA256 else "User-supplied ONNX model"),
        "systems": {s: dict(arguments=SYSTEMS[s][1], uses_model=SYSTEMS[s][0]) for s in systems},
        "datasets": dataset_info,
        "protocol": dict(
            onset_tolerance_seconds=0.05,
            note_event_offset_tolerance="max(50 ms, 20% of reference duration)",
            frame_seconds=0.01, chord_frame_seconds=CHORD_FRAME_SECONDS,
            chord_reference="GuitarSet performed-chord annotation (comp recordings)",
            thresholds="CLI defaults; nothing tuned on these recordings",
            timing="One process-cold CLI run per recording, including decode and JSON export",
            failed_runs="Scored as no predicted notes; excluded from timing; listed in failed_runs",
            python=platform.python_version(), platform=platform.platform()),
        "results": {group: {s: {m: checked_metrics(*v) for m, v in metrics.items()}
                            for s, metrics in by_system.items()}
                    for group, by_system in totals.items()},
        "chords": {s: chord_summary(c) for s, c in chord_totals.items()},
        "paired_differences": paired_comparisons(clips),
        "failed_runs": failures,
        "timing": {s: distribution(v) for s, v in timings.items()},
        "per_recording": clips,
    }


def paired_bootstrap(clips: list[dict[str, Any]], candidate: str, baseline: str, metric: str,
                     resamples: int = 2000, seed: int = 20261029) -> dict[str, float | int] | None:
    """Recording-resampled 95% interval for micro F1(candidate) - F1(baseline)."""
    rows = []
    for clip in clips:
        systems = clip["systems"]
        if candidate in systems and baseline in systems:
            rows.append(tuple(systems[s]["metrics"][metric][k]
                              for s in (candidate, baseline) for k in ("tp", "fp", "fn")))
    if not rows:
        return None

    def delta(sample: list[tuple[int, ...]]) -> float:
        sums = [sum(row[i] for row in sample) for i in range(6)]
        return checked_metrics(*sums[:3])["f1"] - checked_metrics(*sums[3:])["f1"]

    generator = random.Random(seed)
    values = sorted(delta([rows[generator.randrange(len(rows))] for _ in rows])
                    for _ in range(resamples))
    return dict(recordings=len(rows), delta_f1=delta(rows), resamples=resamples, seed=seed,
                ci95_low=values[int(0.025 * (resamples - 1))],
                ci95_high=values[int(0.975 * (resamples - 1))])


def paired_comparisons(clips: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset in sorted({c["dataset"] for c in clips}):
        subset = [c for c in clips if c["dataset"] == dataset]
        for candidate, baseline in (("basic_pitch_tracked", "basic_pitch"),
                                    ("spectral_tracked", "spectral")):
            for metric in ("onset_pitch", "note_event", "frame_10ms"):
                value = paired_bootstrap(subset, candidate, baseline, metric)
                if value is not None:
                    result.setdefault(dataset, {}).setdefault(
                        f"{candidate}-vs-{baseline}", {})[metric] = value
    return result


def markdown_summary(report: dict[str, Any]) -> str:
    lines = []
    for group, by_system in report["results"].items():
        lines += [f"### {group}", "",
                  "| System | Onset F1 | Onset+offset F1 | Frame F1 | Overlapping-note onset recall |",
                  "|---|---:|---:|---:|---:|"]
        for system, metrics in by_system.items():
            lines.append(f"| {system} | " + " | ".join(
                f"{metrics[m]['recall' if m == 'overlapping_onset_recall' else 'f1']:.3f}"
                for m in NOTE_METRICS) + " |")
        lines.append("")
    if report["chords"]:
        lines += ["### chords (GuitarSet comp, performed annotation)", "",
                  "| System | Coverage | Root recall | Maj/min recall |", "|---|---:|---:|---:|"]
        for system, values in report["chords"].items():
            lines.append(f"| {system} | {values['coverage']:.3f} | {values['root_recall']:.3f} | "
                         f"{values['majmin_recall']:.3f} |")
        lines.append("")
    for dataset, comparisons in report.get("paired_differences", {}).items():
        lines += [f"### paired F1 differences ({dataset}, recording bootstrap 95% CI)", "",
                  "| Comparison | Metric | Delta F1 | 95% CI |", "|---|---|---:|---:|"]
        for comparison, metrics in comparisons.items():
            for metric, value in metrics.items():
                lines.append(f"| {comparison} | {metric} | {value['delta_f1']:+.3f} | "
                             f"[{value['ci95_low']:+.3f}, {value['ci95_high']:+.3f}] |")
        lines.append("")
    if report.get("failed_runs"):
        lines += ["### failed runs (scored as no output)", ""]
        lines += [f"- {f['system']} on {f['dataset']}/{f['track']}: {f['error']}"
                  for f in report["failed_runs"]]
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transcriber", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parent / "models" / "basic_pitch.onnx")
    parser.add_argument("--systems", default="yin,spectral,spectral_tracked,basic_pitch,basic_pitch_tracked")
    parser.add_argument("--guitarset-audio", type=Path, help="GuitarSet audio_mono-mic.zip")
    parser.add_argument("--guitarset-annotations", type=Path, help="GuitarSet annotation.zip")
    parser.add_argument("--guitarset-split", choices=("all", "train", "validation", "test"), default="test")
    parser.add_argument("--trios", type=Path, help="Extracted 'TRIOS Dataset' directory")
    parser.add_argument("--limit", type=int, help="First N GuitarSet recordings (sorted) in the split")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = evaluate(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    summary = markdown_summary(report)
    print(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
