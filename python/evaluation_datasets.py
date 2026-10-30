"""Local, reproducible adapters for symbolic/audio transcription datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import struct
import wave
from pathlib import Path, PurePosixPath
from typing import Any

from evaluation_metrics import validate_events


MUSICNET_ANNOTATION_RATE = 44_100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_midi_notes(data: bytes) -> list[dict[str, float | int]]:
    """Read note events from a type-0/1 Standard MIDI File into seconds."""
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("Not a Standard MIDI File")
    header_length = struct.unpack(">I", data[4:8])[0]
    if header_length < 6 or header_length > len(data) - 8:
        raise ValueError("Malformed MIDI header")
    format_type, track_count, division = struct.unpack(">HHH", data[8:14])
    if format_type not in (0, 1):
        raise ValueError(f"Unsupported MIDI format: {format_type}")
    if track_count < 1 or division == 0:
        raise ValueError("Invalid MIDI track count or time division")
    if division & 0x8000:
        raise ValueError("SMPTE MIDI time division is unsupported")

    position = 8 + header_length
    events: list[tuple[int, int, int, int, int]] = []
    for track in range(track_count):
        if position + 8 > len(data) or data[position:position + 4] != b"MTrk":
            raise ValueError("Malformed MIDI track header")
        length = struct.unpack(">I", data[position + 4:position + 8])[0]
        cursor, end = position + 8, position + 8 + length
        if end > len(data):
            raise ValueError("Truncated MIDI track")
        position = end
        tick, status, order = 0, 0, 0

        def variable() -> int:
            nonlocal cursor
            value = 0
            for _ in range(4):
                if cursor >= end:
                    raise ValueError("Truncated MIDI variable-length quantity")
                byte = data[cursor]
                cursor += 1
                value = (value << 7) | (byte & 0x7F)
                if not byte & 0x80:
                    return value
            raise ValueError("Invalid MIDI variable-length quantity")

        while cursor < end:
            tick += variable()
            if cursor >= end:
                raise ValueError("Truncated MIDI event")
            if data[cursor] & 0x80:
                status = data[cursor]
                cursor += 1
            elif status == 0 or status >= 0xF0:
                raise ValueError("MIDI running status has no channel status")
            kind = status & 0xF0
            if status == 0xFF:
                if cursor >= end:
                    raise ValueError("Truncated MIDI meta event")
                meta = data[cursor]
                cursor += 1
                size = variable()
                if cursor + size > end:
                    raise ValueError("Truncated MIDI meta payload")
                if meta == 0x51 and size == 3:
                    tempo = int.from_bytes(data[cursor:cursor + 3], "big")
                    if tempo <= 0:
                        raise ValueError("Invalid MIDI tempo")
                    events.append((tick, order, 0, 0, tempo))
                cursor += size
            elif status in (0xF0, 0xF7):
                size = variable()
                if cursor + size > end:
                    raise ValueError("Truncated MIDI system-exclusive event")
                cursor += size
                status = 0
            elif kind in (0x80, 0x90):
                if cursor + 2 > end:
                    raise ValueError("Truncated MIDI note event")
                key, velocity = data[cursor], data[cursor + 1]
                cursor += 2
                on = kind == 0x90 and velocity > 0
                events.append((tick, order, 2 if on else 1,
                               (status & 0x0F) << 8 | key, track))
            elif kind in (0xC0, 0xD0):
                if cursor + 1 > end:
                    raise ValueError("Truncated MIDI channel event")
                cursor += 1
            elif 0x80 <= status <= 0xEF:
                if cursor + 2 > end:
                    raise ValueError("Truncated MIDI channel event")
                cursor += 2
            else:
                raise ValueError(f"Unsupported MIDI status byte: {status:#x}")
            order += 1

    events.sort(key=lambda event: (event[0], event[2] != 0, event[2], event[1]))
    seconds, last_tick, tempo = 0.0, 0, 500_000
    starts: dict[tuple[int, int], list[float]] = {}
    notes: list[dict[str, float | int]] = []
    for tick, _, kind, key, value in events:
        seconds += (tick - last_tick) * tempo / (division * 1_000_000)
        last_tick = tick
        if kind == 0:
            tempo = value
        elif kind == 2:
            starts.setdefault((value, key), []).append(seconds)
        elif starts.get((value, key)):
            start = starts[(value, key)].pop(0)
            if seconds > start:
                notes.append({"midi": key & 0x7F, "start": start, "duration": seconds - start})
    validate_events(notes)
    return sorted(notes, key=lambda note: (note["start"], note["midi"]))


def _dataset_root(path: Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")
    return root


def _resolve_relative_file(root: Path, value: str, dataset: str) -> Path:
    relative = PurePosixPath(str(value).replace("\\", "/"))
    if (not relative.parts or relative.is_absolute() or
            any(part == ".." for part in relative.parts) or
            ":" in relative.parts[0]):
        raise ValueError(f"Invalid {dataset} relative path: {value!r}")
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{dataset} path escapes the dataset root: {value!r}") from error
    if not path.is_file():
        raise FileNotFoundError(f"Missing {dataset} file: {path}")
    return path


def _wav_info(path: Path) -> tuple[int, float]:
    try:
        with wave.open(str(path), "rb") as audio:
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            if sample_rate <= 0 or frame_count <= 0 or audio.getnchannels() <= 0:
                raise ValueError(f"Invalid WAV timing or channel count: {path}")
            return sample_rate, frame_count / sample_rate
    except (EOFError, wave.Error) as error:
        raise ValueError(f"Invalid PCM WAV file: {path}: {error}") from error


def _make_clip(
    root: Path,
    identifier: str,
    audio_path: Path,
    annotation_path: Path,
    notes: list[dict[str, Any]],
    split: str,
    scenarios: list[str],
) -> dict[str, Any]:
    audio_path = audio_path.resolve()
    annotation_path = annotation_path.resolve()
    try:
        relative_audio = audio_path.relative_to(root).as_posix()
        annotation_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Dataset input escapes the dataset root: {identifier}") from error
    if not notes:
        raise ValueError(f"No reference notes found for {identifier}")
    validate_events(notes)
    notes.sort(key=lambda note: (note["start"], note["midi"], note["duration"]))
    sample_rate, duration = _wav_info(audio_path)
    return {
        "id": identifier,
        "audio": relative_audio,
        "audio_path": str(audio_path),
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "notes": notes,
        "sha256": sha256_file(audio_path),
        "annotation_sha256": sha256_file(annotation_path),
        "dataset_split": split,
        "scenarios": scenarios,
    }


def _selected(items: list[Any], limit: int | None, seed: int, key) -> list[Any]:
    ordered = sorted(items, key=key)
    if limit is None or limit >= len(ordered):
        return ordered
    if limit < 1:
        raise ValueError("Evaluation limit must be at least one")
    return sorted(random.Random(seed).sample(ordered, limit), key=key)


def _finish_corpus(
    name: str,
    version: str,
    seed: int,
    limit: int | None,
    split: str,
    split_policy: dict[str, Any],
    provenance: dict[str, Any],
    clips: list[dict[str, Any]],
) -> dict[str, Any]:
    corpus = {
        "name": name,
        "version": version,
        "seed": seed,
        "selection": {"limit": limit, "seed": seed},
        "provenance": provenance,
        "dataset_split_policy": split_policy,
        "selected_split": split,
        "clips": clips,
    }
    identity = dict(corpus)
    identity["clips"] = [
        {key: value for key, value in clip.items() if key != "audio_path"}
        for clip in clips
    ]
    corpus["sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return corpus


def maestro_corpus(
    dataset_root: Path,
    split: str,
    limit: int | None,
    seed: int,
) -> dict[str, Any]:
    root = _dataset_root(dataset_root)
    if split not in ("all", "train", "validation", "test"):
        raise ValueError(f"Unsupported MAESTRO split: {split}")
    metadata_files = sorted(path for path in root.iterdir()
                            if path.is_file() and path.suffix.lower() == ".csv" and
                            (path.name.lower().startswith("maestro-v") or
                             path.name.lower() == "metadata.csv"))
    if len(metadata_files) != 1:
        raise ValueError(f"Expected one MAESTRO metadata CSV at the dataset root; found {len(metadata_files)}")
    metadata = metadata_files[0]
    match = re.search(r"maestro-v(\d+(?:\.\d+)*)", metadata.name, re.IGNORECASE)
    version = match.group(1) if match else "unspecified"
    records: list[tuple[str, str, str]] = []
    with metadata.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = {name.strip().lower() for name in (reader.fieldnames or [])}
        required = {"split", "audio_filename", "midi_filename"}
        if not required <= columns:
            raise ValueError(f"MAESTRO metadata is missing columns: {sorted(required - columns)}")
        for row_number, row in enumerate(reader, start=2):
            values = {str(key).strip().lower(): value for key, value in row.items() if key}
            row_split = str(values.get("split", "")).strip().lower()
            if row_split not in ("train", "validation", "test"):
                raise ValueError(f"Invalid MAESTRO split at metadata row {row_number}: {row_split!r}")
            audio_name = str(values.get("audio_filename", "")).strip()
            midi_name = str(values.get("midi_filename", "")).strip()
            if not audio_name or not midi_name:
                raise ValueError(f"Missing MAESTRO audio/MIDI path at metadata row {row_number}")
            if split == "all" or row_split == split:
                records.append((row_split, audio_name, midi_name))
    if not records:
        raise ValueError(f"MAESTRO split has no recordings: {split}")
    if len({midi_name.casefold() for _, _, midi_name in records}) != len(records):
        raise ValueError("MAESTRO metadata contains duplicate MIDI paths")
    records = _selected(records, limit, seed, key=lambda row: row[2].casefold())
    clips = []
    for row_split, audio_name, midi_name in records:
        audio_path = _resolve_relative_file(root, audio_name, "MAESTRO")
        midi_path = _resolve_relative_file(root, midi_name, "MAESTRO")
        identifier = PurePosixPath(midi_name.replace("\\", "/")).with_suffix("").as_posix()
        notes = read_midi_notes(midi_path.read_bytes())
        for note in notes:
            note["instrument"] = "piano"
            note["velocity"] = None
        clips.append(_make_clip(root, identifier, audio_path, midi_path, notes, row_split,
                                ["piano", "polyphonic"]))
    return _finish_corpus(
        "MAESTRO", version, seed, limit, split,
        {"kind": "upstream metadata split", "metadata_file": metadata.name,
         "assignments": "train/validation/test are preserved without reassignment"},
        {"source": "MAESTRO", "source_url": "https://magenta.tensorflow.org/datasets/maestro",
         "metadata_sha256": sha256_file(metadata)},
        clips,
    )


def _index_by_stem(paths: list[Path], dataset: str, kind: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        key = path.stem.casefold()
        if key in indexed:
            raise ValueError(
                f"Ambiguous {dataset} {kind} stem {path.stem!r}: "
                f"{indexed[key]} and {path}"
            )
        indexed[key] = path
    return indexed


def maps_corpus(
    dataset_root: Path,
    split: str,
    limit: int | None,
    seed: int,
) -> dict[str, Any]:
    root = _dataset_root(dataset_root)
    if split != "all":
        raise ValueError(
            "MAPS has no canonical split in this adapter; pass a preselected subset root and use split 'all'"
        )
    audio = _index_by_stem(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"],
        "MAPS", "audio",
    )
    midi = _index_by_stem(
        [path for path in root.rglob("*") if path.is_file() and
         path.suffix.lower() in (".mid", ".midi")],
        "MAPS", "MIDI",
    )
    if not audio or not midi:
        raise ValueError(f"Expected paired MAPS WAV/MIDI files under {root}")
    if audio.keys() != midi.keys():
        missing_midi = sorted(audio.keys() - midi.keys())
        missing_audio = sorted(midi.keys() - audio.keys())
        raise ValueError(
            f"Unpaired MAPS files; missing MIDI for {missing_midi[:5]}, "
            f"missing audio for {missing_audio[:5]}"
        )
    pairs = _selected(list(audio), limit, seed, key=str.casefold)
    clips = []
    for stem in pairs:
        audio_path, midi_path = audio[stem], midi[stem]
        identifier = audio_path.relative_to(root).with_suffix("").as_posix()
        notes = read_midi_notes(midi_path.read_bytes())
        for note in notes:
            note["instrument"] = "piano"
            note["velocity"] = None
        clips.append(_make_clip(root, identifier, audio_path, midi_path, notes, "all",
                                ["piano", "polyphonic"]))
    return _finish_corpus(
        "MAPS", "unversioned-local", seed, limit, "all",
        {"kind": "no universal split imposed",
         "assignments": "all paired WAV/MIDI recordings under the selected root"},
        {"source": "MAPS piano transcription dataset",
         "license_notice": "Use the upstream dataset terms; no audio or labels are redistributed."},
        clips,
    )


def _read_musicnet_notes(path: Path) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = {str(name).strip().lower(): name for name in (reader.fieldnames or [])}
        required = {"start_time", "end_time", "note"}
        if not required <= columns.keys():
            raise ValueError(f"MusicNet annotation is missing columns: {sorted(required - columns.keys())}")
        for row_number, row in enumerate(reader, start=2):
            try:
                start_sample = int(str(row[columns["start_time"]]).strip())
                end_sample = int(str(row[columns["end_time"]]).strip())
                pitch = int(str(row[columns["note"]]).strip())
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Malformed MusicNet annotation at {path}:{row_number}") from error
            if start_sample < 0 or end_sample <= start_sample:
                raise ValueError(f"Invalid MusicNet timing at {path}:{row_number}")
            note: dict[str, Any] = {
                "midi": pitch,
                "start": start_sample / MUSICNET_ANNOTATION_RATE,
                "duration": (end_sample - start_sample) / MUSICNET_ANNOTATION_RATE,
            }
            if "instrument" in columns:
                note["instrument"] = str(row.get(columns["instrument"], "")).strip()
            if "note_label" in columns:
                note["note_label"] = str(row.get(columns["note_label"], "")).strip()
            notes.append(note)
    validate_events(notes)
    return notes


def musicnet_corpus(
    dataset_root: Path,
    split: str,
    limit: int | None,
    seed: int,
) -> dict[str, Any]:
    root = _dataset_root(dataset_root)
    if split not in ("all", "train", "test"):
        raise ValueError(f"MusicNet supports only train/test/all splits, not {split!r}")
    available: list[str] = []
    for candidate in ("train", "test"):
        audio_dir, labels_dir = root / f"{candidate}_data", root / f"{candidate}_labels"
        if audio_dir.exists() != labels_dir.exists():
            raise ValueError(f"MusicNet {candidate} audio and label directories must both exist")
        if audio_dir.is_dir() and labels_dir.is_dir():
            available.append(candidate)
    requested = available if split == "all" else [split]
    if not requested or any(candidate not in available for candidate in requested):
        raise FileNotFoundError(f"MusicNet split is not available under {root}: {split}")

    records: list[tuple[str, str, Path, Path]] = []
    for candidate in requested:
        audio_files = _index_by_stem(
            [path for path in (root / f"{candidate}_data").rglob("*")
             if path.is_file() and path.suffix.lower() == ".wav"],
            "MusicNet", "audio",
        )
        label_files = _index_by_stem(
            [path for path in (root / f"{candidate}_labels").rglob("*")
             if path.is_file() and path.suffix.lower() == ".csv"],
            "MusicNet", "annotation",
        )
        if not audio_files or not label_files or audio_files.keys() != label_files.keys():
            missing_labels = sorted(audio_files.keys() - label_files.keys())
            missing_audio = sorted(label_files.keys() - audio_files.keys())
            raise ValueError(
                f"Unpaired MusicNet {candidate} files; missing labels for {missing_labels[:5]}, "
                f"missing audio for {missing_audio[:5]}"
            )
        records.extend((candidate, stem, audio_files[stem], label_files[stem])
                       for stem in audio_files)
    records = _selected(records, limit, seed, key=lambda row: (row[0], row[1]))
    clips = []
    for row_split, stem, audio_path, labels_path in records:
        notes = _read_musicnet_notes(labels_path)
        clips.append(_make_clip(root, f"{row_split}/{stem}", audio_path, labels_path, notes,
                                row_split, ["polyphonic", f"musicnet_{row_split}"]))
    return _finish_corpus(
        "MusicNet", "unversioned-local", seed, limit, split,
        {"kind": "published train/test directory assignment",
         "available_splits": available, "annotation_timebase_hz": MUSICNET_ANNOTATION_RATE},
        {"source": "MusicNet note-annotated audio corpus",
         "annotation_timebase_hz": MUSICNET_ANNOTATION_RATE,
         "license_notice": "Use the upstream dataset terms; no audio or labels are redistributed."},
        clips,
    )
