# Note, phrase and transcription confidence

The detector output remains evidence, not a correctness guarantee.
[`confidence.hpp`](../include/transcribe/confidence.hpp) exposes
`score_confidence(const ScoreDocument&, const ConfidenceOptions&)` and shared
JSON serializers. The implementation is
[`confidence.cpp`](../src/confidence.cpp). Scoring never filters notes, edits
pitches, overwrites source evidence, or changes project integrity payloads.
Reports are derived again after accepted score edits.

## What the values mean

Each note retains its canonical note ID and source-event IDs, and exposes:

- `evidence_score`: a bounded **heuristic**, not a probability.
- `calibrated_probability`: the existing explicit event-level calibrated
  confidence, if supplied by the producer. No such probabilities are fitted
  or manufactured by this scorer.
- `detector_support`: mean selected-pitch strength from retained samples with
  declared semantics and values in [0,1].
- `temporal_coverage`: retained selected-pitch timestamp span divided by measured
  note duration, clamped to [0,1]. This is a span proxy, not proof that every
  intervening audio frame supports the note.
- `notation_evidence`: existing interpretation evidence, **not blended** into
  detector confidence. Chromatic music and expressive rhythm are not errors.
- `band` and `reasons`: review labels and the evidence limitations.

For `bounded-evidence-v1`:

```text
support = mean(selected-pitch bounded detector strengths)
span = clamp((last selected sample - first selected sample) / measured duration, 0, 1)
conflict = alternative peak >= 0.8 * selected peak, with a positive alternative
evidence_score = support * (0.5 + 0.5 * span) * (conflict ? 0.75 : 1)
```

Only samples within the measured note interval participate. Values above 1
are not silently normalized. Samples with unspecified semantics are not used.
Different strength semantics are never averaged: the selected-pitch group with
the most retained samples is used (ties follow the enum order). Competing
pitches must use that same semantics group. Even a frame-level
`CalibratedProbability` is only evidence here; it does not become an event-level
probability. Raw frame count, loudness and note duration do not increase an
aggregate's weight.

For review, `<0.5` is low, `[0.5,0.8)` medium and `>=0.8` high. An explicit
calibrated probability determines the band when present; otherwise the evidence
score does. These heuristic thresholds are not validated risk guarantees.
Missing compatible evidence is **unassessed**, not zero or perfect confidence.
User-authored/edited notes, including source-pitch mismatches, are not assessed by
the original detector's confidence. Their original evidence and supplied
confidence remain intact in the canonical source; the derived report does not
present them as confidence in the edit.

Attack-only retained evidence has zero timestamp span and is discounted. This
is common in the tracked polyphonic decoder, which currently retains attack
evidence rather than a complete sustain trace. Low scores can therefore mean
limited retained evidence, not necessarily incorrect notes. Distribution shift
is not assessed.

## Phrase and transcription summaries

Phrases are whole-transcription groups, across voices, separated by **at least
1 second of silence** (configurable in the C++ API). Timing follows the current
score's playback map. The union of sounding intervals is used, so a held bass
prevents a false phrase break in a moving melody. A 1 ns numerical tolerance
handles exact-boundary roundoff.

Each summary contains:

- all-note count, assessed-note count, calibrated-note count and low-note count;
- mean evidence score over notes with evidence scores;
- mean calibrated probability over notes with supplied calibrated probabilities.

The means have independent denominators. Missing or edited notes are included
in the all-note count and disclosed through coverage, not silently assigned
confidence. Means are note-weighted, not duration-weighted; a long held note
cannot hide several uncertain short notes. A mean calibrated probability
estimates a correct-note fraction only for its calibrated subset, provided
those individual probabilities are valid for the deployment. It is **not the
probability that a phrase or entire transcription is correct**. Probabilities
are never multiplied under an unjustified independence assumption.
An empty score has null means and no phrases.

## Export and interface contracts

- **CLI:** the normal summary reports evidence/calibrated means, assessed
  coverage and low-note count. `--confidence-details` prints every note and
  phrase; the matching config key is supported. `--quiet` suppresses both.
- **Legacy JSON (`--json`):** adds `confidence_report`. The original
  `notes[].confidence` is retained for compatibility and explicitly labelled
  by `legacy_note_confidence_semantics: "heuristic_not_probability"`.
- **Events JSON (`--events-json`):** adds the identical `confidence_report`.
  The per-note confidence object also includes `evidence_score`; the old
  `heuristic_score` is explicitly identified as notation interpretation evidence.
- **MusicXML:** interpretation-aware and canonical score exports embed the
  complete JSON report as an XML-escaped
  `identification/miscellaneous/miscellaneous-field` named
  `transcriber-confidence-v1`. Canonical exports retain the same note IDs as
  JSON; tied fragments keep the existing note/fragment IDs. Direct interpretation
  exports use generated canonical IDs and retained source-event links. Metadata
  does not alter notes, rhythmic cursors, ties or playback dynamics.
  The legacy `vector<Note>` MusicXML exporters remain compatibility paths;
  use an interpretation or canonical score for confidence metadata.
- **WebView2:** note messages carry `confidence_assessment`, and score metadata
  carries the shared report. The heatmap uses dashed borders for heuristic
  evidence, solid calibrated estimates and gray unassessed notes. Low-note
  outlines, text, accessible labels and tooltips do not rely on color alone.
  The expandable panel shows transcription/phrase means and coverage (first
  100 phrases, with an explicit truncation notice; exports retain all phrases).
  Local edits invalidate detector labels immediately and withhold aggregate
  summaries until accepted export, rather than displaying stale confidence.

Example:

```powershell
.\build\transcriber.exe take.wav --polyphonic --polyphonic-decoder tracked `
  --confidence-details --json notes.json --events-json events.json `
  --musicxml score.musicxml
```

## Calibration methodology

No deployable calibration profile or real-world calibration claim is bundled.
Before publishing event correctness probabilities:

1. **Define the target.** For a strict note-event target, require the same MIDI
   pitch, onset within 50 ms, and offset within max(50 ms, 20% of reference
   duration). Use the repository's maximum-cardinality one-to-one matcher;
   duplicate predictions count as errors. If calibrating pitch alone, use a
   separate named target, not a probability labelled as full-event correctness.
2. **Build independent partitions.** Separate detector training, calibration
   fitting, calibration/model selection and final evaluation. Group by performer,
   session and source recording; retain instrument-disjoint/OOD strata. Verify
   pretrained-model training overlap explicitly. GuitarSet is not clean held-out
   data for Basic Pitch simply because this application's performer split is
   held out.
3. **Record the recipe.** Pin backend/model SHA, decoder configuration, evidence
   retention policy, `bounded-evidence-v1` version, target definition and dataset
   provenance. Changes to retained evidence can change this score even with
   unchanged pitches, so they require calibration revalidation.
4. **Create labels without threshold leakage.** Keep all emitted notes, including
   false positives and duplicates. Keep ambiguous ground-truth annotations
   separately and disclose excluded-label coverage. Never select only high-score
   notes before calculating reliability.
5. **Fit only on calibration data.** Compare a monotonic isotonic mapping with
   Platt/logistic scaling of this note score (not raw frame activation or the
   legacy `Note.confidence`). Select on the separate validation partition. Use
   a backend-specific model and enough labelled errors in every deployment
   stratum; regularize sparse bins. Do not copy a curve across engines.
6. **Evaluate on untouched recordings.** Report Brier score, expected calibration
   error (ECE), reliability-bin sample counts/mean predictions/observed accuracy,
   uncertainty intervals resampled by recording, and selective precision/recall
   and retained coverage. Measure omitted-note recall separately: calibration
   of emitted notes says nothing about notes never detected.
7. **Gate deployment.** Reject a profile outside its backend/model/target/recipe
   scope. With insufficient labels or unreviewed overlap, retain heuristic
   scores and null probabilities. Reassess drift and instrument strata; neither
   quietness nor unfamiliar harmony should be used as evidence of incorrectness.

The existing `evaluate_calibration` API in
[`review.hpp`](../include/transcribe/review.hpp) now reports ECE alongside Brier
and reliability bins. It requires a dataset identifier, reviewed training
overlap, held-out/representative flags and a minimum labelled count (100 by
default). Those checks are necessary, not proof of statistical adequacy.
Use **independent final-test predictions** as its input; it is a diagnostic,
not a fitting routine. Only a separately validated producer should populate
`ScoreNote::calibrated_confidence`.

## Tests

[`confidence_tests.cpp`](../tests/confidence_tests.cpp) verifies exact score
formulas, sparse/competing evidence, separate semantics and denominators,
null/unbounded evidence, phrase union and the 1 s boundary, edits/undo, source
preservation, empty/invalid input, safe JSON, MusicXML metadata, and exact
synthetic Brier/ECE values. These are semantic tests, not measured acoustic
calibration.

CLI tests compare the same report across both JSON formats and parsed
MusicXML metadata, exercise detailed output and quiet mode, and run DSP and
Basic Pitch inference. The browser workflow tests note labels/ARIA, low-note
markers, phrase/transcription summaries and edit invalidation, alongside
existing responsive-layout and contrast checks. The native WebView2 smoke
(`tests\workstation_host.py --webview <transcriber-gui.exe> --polyphonic`)
asserts the real host renders the transcription summary, phrase list and a
confidence band on every note row.

```powershell
cmake --build build --target transcriber transcriber-tests transcriber-notation
.\build\transcriber-tests.exe "[confidence],[review],[interpretation],[score],[project]"
ctest --test-dir build -R "polyphonic_cli|cli_export_validation|interpretation_semantics|ui_asset_smoke" --output-on-failure
```
