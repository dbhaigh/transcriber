const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const { mkdtempSync, writeFileSync, readFileSync, readdirSync, unlinkSync, rmdirSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join, resolve } = require("node:path");

const executable = resolve(process.argv[2]);
const directory = mkdtempSync(join(tmpdir(), "transcriber-polyphonic-cli-"));
try {
  const rate = 22050;
  const frames = rate * 2;
  const wav = Buffer.alloc(44 + frames * 2);
  wav.write("RIFF", 0);
  wav.writeUInt32LE(wav.length - 8, 4);
  wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(rate, 24);
  wav.writeUInt32LE(rate * 2, 28);
  wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);
  wav.write("data", 36);
  wav.writeUInt32LE(frames * 2, 40);
  for (let frame = 0; frame < frames; ++frame) {
    const time = frame / rate;
    let value = 0;
    for (const [pitch, start, end] of [[60, .2, 1.6], [64, .2, 1.1], [67, .5, 1.4]]) {
      if (time >= start && time < end)
        value += .2 * Math.sin(2 * Math.PI * 440 * 2 ** ((pitch - 69) / 12) * time);
    }
    wav.writeInt16LE(Math.round(value * 32767), 44 + frame * 2);
  }
  writeFileSync(join(directory, "chord.wav"), wav);
  const run = (...args) => spawnSync(executable, args, {
    cwd: directory, encoding: "utf8", timeout: 30000,
  });
  const result = run("chord.wav", "--polyphonic", "--polyphonic-decoder", "tracked",
    "--events-json", "events.json", "--json", "notes.json",
    "--score-project", "score.json", "--midi", "notes.mid", "--quiet");
  assert.equal(result.error, undefined);
  assert.equal(result.status, 0, result.stderr);
  const events = JSON.parse(readFileSync(join(directory, "events.json"), "utf8"));
  const score = JSON.parse(readFileSync(join(directory, "score.json"), "utf8"));
  assert.equal(events.source.sample_rate_hz, rate);
  assert.equal(result.stdout, "", "--quiet must also silence confidence summaries");
  const confidence = events.confidence_report;
  assert.equal(confidence.version, 1);
  assert.equal(confidence.notes.length, score.notes.length);
  assert.equal(confidence.transcription.note_count, score.notes.length);
  assert.equal(confidence.transcription.calibrated_notes, 0,
    "Uncalibrated detectors must not invent probabilities");
  assert.equal(confidence.transcription.mean_calibrated_probability, null);
  assert.ok(confidence.notes.every(note => sourceIdsFor(note, score)),
    "Confidence must retain canonical note and source-event links");
  function sourceIdsFor(note, document) {
    const source = document.notes.find(item => item.id === note.note_id);
    return source && JSON.stringify(source.source_event_ids) === JSON.stringify(note.source_event_ids);
  }
  const legacy = JSON.parse(readFileSync(join(directory, "notes.json"), "utf8"));
  assert.deepEqual(legacy.confidence_report, confidence);
  assert.equal(legacy.legacy_note_confidence_semantics, "heuristic_not_probability");
  const c = events.chords.find(chord => chord.name === "C");
  assert.deepEqual([...new Set(events.notes.map(note => note.midi_pitch))].sort((a, b) => a - b),
    [60, 64, 67], "Synthetic fixture must not introduce extra pitch classes");
  assert.ok(c, "C major must be identified over the sounding overlap");
  assert.deepEqual(c.midi_pitches, [60, 64, 67]);
  assert.equal(c.inversion, 0);
  assert.equal(c.root_pitch_class, 0);
  assert.ok(c.onset_seconds >= .3 && c.onset_seconds <= .75);
  assert.ok(c.offset_seconds > .8 && c.offset_seconds < 1.4);
  const sourceIds = new Set(score.source_events.map(event => event.id));
  assert.ok(c.note_ids.every(id => sourceIds.has(id)), "Chord members must reference canonical score source events");
  for (const pitch of [60, 64, 67]) {
    const note = events.notes.find(note => note.midi_pitch === pitch);
    assert.ok(note, `Missing pitch ${pitch}`);
    assert.ok(note.offset_frame > note.onset_frame);
  }
  assert.equal(events.polyphonic_diagnostics.refused_note_starts, 0);
  const analysis = events.musical_analysis;
  assert.equal(analysis.version, 1);
  assert.equal(analysis.derived, true);
  assert.equal(analysis.pitch_reference, "sounding");
  assert.equal(analysis.metrics.note_count, score.notes.length);
  assert.equal(analysis.pitch_class_duration_quarters.length, 12);
  assert.ok(analysis.harmony.some(interval =>
    interval.root_pitch_class === 0 && interval.quality === "major" && interval.export_symbol),
  "Score-level analysis must identify the sustained C-major interval");
  assert.ok(analysis.harmony.every(interval => interval.note_indices.every(index =>
    Number.isInteger(index) && index >= 0 && index < score.notes.length)));
  assert.ok(analysis.metrics.named_chord_coverage >= 0 && analysis.metrics.named_chord_coverage <= 1);
  assert.equal(score.musical_analysis, undefined,
    "Derived analysis must not change canonical project integrity payloads");
  assert.equal(score.confidence_report, undefined,
    "Derived confidence must not change canonical project integrity payloads");
  assert.equal(readFileSync(join(directory, "notes.mid")).subarray(0, 4).toString(), "MThd");
  assert.match(readFileSync(join(directory, "transcription.musicxml"), "utf8"), /<backup>/,
    "Unequal overlapping durations must use independent MusicXML voices");
  assert.match(readFileSync(join(directory, "transcription.musicxml"), "utf8"),
    /<harmony><root><root-step>C<\/root-step><\/root><kind>major<\/kind>/,
    "MusicXML must include the supported chord symbol without removing note evidence");
  const xmlConfidence = readFileSync(join(directory, "transcription.musicxml"), "utf8")
    .match(/<miscellaneous-field name="transcriber-confidence-v1">([^<]+)<\/miscellaneous-field>/);
  assert.ok(xmlConfidence, "MusicXML must retain note, phrase and transcription confidence metadata");
  const decoded = xmlConfidence[1].replaceAll("&quot;", '"').replaceAll("&apos;", "'")
    .replaceAll("&lt;", "<").replaceAll("&gt;", ">").replaceAll("&amp;", "&");
  assert.deepEqual(JSON.parse(decoded), confidence, "MusicXML and JSON confidence must agree");
  const detailed = run("chord.wav", "--polyphonic", "--polyphonic-decoder", "tracked",
    "--confidence-details");
  assert.equal(detailed.status, 0, detailed.stderr);
  assert.match(detailed.stdout, /Confidence: evidence mean.*heuristic/);
  assert.match(detailed.stdout, /Phrase 1/);
  assert.match(detailed.stdout, /Note .*; evidence .*; calibrated unavailable/);
  for (const invalid of [
    ["--polyphonic-decoder", "tracked"],
    ["--polyphonic", "--polyphonic-decoder", "tracked", "--backend", "builtin:yin"],
    ["--polyphonic", "--polyphonic-decoder", "tracked", "--bounded-inference"],
    ["--polyphonic", "--polyphonic-decoder", "tracked", "--live-pipeline"],
  ]) {
    const failure = run("chord.wav", ...invalid);
    assert.notEqual(failure.status, 0, `Unexpected acceptance of ${invalid}`);
    assert.match(failure.stderr, /tracked|Tracked/);
  }
  if (process.argv[3]) {
    const neural = run("chord.wav", "--polyphonic", "--polyphonic-decoder", "tracked",
      "--model", resolve(process.argv[3]), "--instrument", "guitar",
      "--events-json", "neural-events.json", "--score-project", "neural-score.json", "--quiet");
    assert.equal(neural.error, undefined);
    assert.equal(neural.status, 0, neural.stderr);
    const output = JSON.parse(readFileSync(join(directory, "neural-events.json"), "utf8"));
    assert.ok(output.notes.length >= 2, "Basic Pitch must emit simultaneous note evidence");
    assert.ok(Array.isArray(output.chords));
    assert.equal(output.musical_analysis.metrics.note_count, output.notes.length);
    assert.equal(output.confidence_report.notes.length, output.notes.length);
    assert.equal(output.confidence_report.transcription.calibrated_notes, 0);
    assert.equal(output.confidence_report.transcription.mean_calibrated_probability, null);
    assert.ok(output.confidence_report.notes.every(note =>
      note.evidence_score === null || (note.evidence_score >= 0 && note.evidence_score <= 1)));
    assert.equal(output.analysis.engine_id, "onnx:basic-pitch-v1");
    console.log("Basic Pitch CLI: tracked inference with guitar score interpretation passed");
  }
  console.log("Polyphonic CLI: overlaps, named chords, source IDs, MusicXML/MIDI and invalid combinations passed");
} finally {
  for (const entry of readdirSync(directory)) unlinkSync(join(directory, entry));
  rmdirSync(directory);
}
