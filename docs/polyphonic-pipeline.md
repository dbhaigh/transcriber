# Tracked polyphonic transcription

## Architecture decision

Use the existing **Basic Pitch ONNX adapter plus deterministic per-pitch
temporal decoding**, with an explicitly selected spectral DSP backend for
model-free operation. Preserve the existing default algorithms.

The pipeline is:

```text
audio decode / existing preprocessing
    -> ModelBackend (Basic Pitch or spectral)
    -> aligned PitchActivationGrid
    -> independent per-pitch onset / sustain / release states
    -> sample-referenced CanonicalNoteEvent intervals
    -> sounding-interval chord identification
    -> interpret_note_events
    -> canonical editable score
    -> MusicXML / MIDI / tablature / project
```

This is an ML-first architecture with deterministic temporal postprocessing,
not an average of incomparable ML probabilities and spectral magnitudes. The
spectral mode uses its selected **relative** strengths. Basic Pitch uses its
aligned activity and onset heads. Neither mode labels detector strength as
calibrated event confidence.

### Approach comparison

- **[Basic Pitch](https://github.com/spotify/basic-pitch)**: lightweight,
  instrument-agnostic multipitch model with ONNX deployment already implemented
  here. Upstream says it works best on one instrument at a time. Its
  per-pitch activity/onset/contour heads fit the current engine interface and
  preserve the native C++ desktop deployment model. **Recommended first
  backend**, not an accuracy guarantee for mixtures or distorted guitar.
- **[CREPE](https://github.com/marl/crepe)**: a monophonic fundamental-frequency
  estimator. Useful for a solo voice, single string, or isolated stem; it
  cannot satisfy simultaneous note detection on a polyphonic mixture.
  Running it repeatedly and subtracting detected pitches is not a robust
  replacement for a multipitch model. Keep it a possible monophonic specialist,
  not the polyphonic foundation.
- **[Onsets and Frames](https://magenta.tensorflow.org/onsets-frames)**:
  explicitly designed for polyphonic piano; the onset-conditioned frame
  architecture is an excellent model for temporal decoding. A piano-specific
  adapter is a sensible next backend. It needs reviewed weights, conversion,
  tensor/alignment tests, runtime/provider qualification and piano evaluation;
  it should not be assumed to generalize to guitar.
- **[MT3](https://github.com/magenta/mt3)**: multi-instrument, multitrack
  transcription using T5X and sequence events. It is the most relevant option
  for future ensemble/instrument assignment, but does not fit a frame-head
  adapter without a separate event-token adapter and stateful segment
  reconciliation. Its upstream deployment stack is substantially different
  from this C++/ONNX desktop application. Defer until multitrack product scope,
  resource budgets and deployment requirements justify that work.
- **Hybrid DSP+ML**: recommended as an architectural direction, with clear
  responsibilities: ML proposes independent pitches/attacks; deterministic
  tracking enforces lifecycles; optional DSP features can later refine timing
  or reject unsupported harmonics after paired evaluation. Do not silently
  fall back to another engine on ML failure, and do not combine relative
  salience with neural activity as if both were calibrated probabilities.
  The current implementation offers an explicit DSP mode, **not learned
  fusion or a demonstrated DSP+ML accuracy improvement**.

These are suitability comparisons based on upstream capabilities and this
repository's contracts. Only the Basic Pitch and DSP paths are measured here;
see [Real-recording evaluation](#real-recording-evaluation). CREPE, Onsets and
Frames and MT3 were not benchmarked. Sources were consulted on 2026-10-29.
Earlier real-guitar measurements and limitations are in
[polyphonic.md](polyphonic.md) and [evaluation.md](evaluation.md).

## Implemented contracts

[polyphonic_pipeline.hpp](../include/transcribe/polyphonic_pipeline.hpp)
defines `PolyphonicDecodingOptions`, `PolyphonicTranscription`,
`IdentifiedChord`, `decode_polyphonic`, and `identify_chords`.

The engine composes these through the optional
`EngineConfiguration::polyphonic_decoding`. Ordinary configurations retain
their previous behavior. The result is available as `EngineUpdate::polyphonic`
alongside the existing engine events and legacy note projection.

### Note lifecycle

- Every MIDI pitch has an independent active state. C, E and G can start
  together, end independently, or overlap a melody introduced later.
- Activity must reach `activation_on` to start. An active note can sustain
  down to `activation_off`; brief gaps shorter than `release_seconds` are
  bridged.
- With a model onset head, starts require a rising onset-threshold edge
  (or supported activity in the first frame, for a recording starting mid-note).
  An onset plateau is one attack, not a series of retriggers.
- A new onset edge retriggers **only that pitch**, subject to
  `minimum_retrigger_seconds`. It closes the old event and opens the new one
  at the same sample boundary without splitting other sustained notes.
- Offsets are the end of the last supporting frame, not the end of the
  release debounce. Offset detection is inferred from activity decay; Basic
  Pitch does not provide a separately consumed offset head here.
- At end of input, remaining states close at their last supported frame,
  clipped to the original recording. Short notes are discarded and counted.
- The polyphony cap admits strongest start candidates deterministically
  (MIDI order breaks ties), but never steals an already sustained note.
  Refused starts are reported as **candidate-frame refusals**, not as a count
  of distinct missed physical notes.

The input grid must declare `IndependentMultiPitch`, have finite aligned
frame-major heads, and have activity/onset values in `[0,1]`. Source identity,
sample rate, timestamps, options and limits are validated before decoding.
Input/model clock conversion remains in the adapter. Window-start DSP grids
are interpreted at window centers; pointwise model timestamps are unchanged.
Reported events use rounded, clipped source samples, with seconds derived from
those samples. No musical quantization is applied to measured event times.

Each event retains bounded attack evidence and provenance. This is not a full
posterior-history archive. Existing retained frame evidence remains available
on the engine update when consumers need it.

### Chord identification

`identify_chords` sweeps note onsets/offsets using half-open sounding intervals:
notes ending exactly when another begins are not simultaneous. The sweep
therefore includes sustained bass notes and arpeggios once they actually
overlap; it does not invent an overlap for sequential notes.

Each result contains measured onset/offset, member source-event IDs, unique
MIDI pitches, bass pitch, optional root pitch class, quality, display name and
optional inversion (0 = root position). Current exact pitch-class templates
cover major/minor, diminished/augmented, suspended second/fourth, power chords,
major/minor sixths and common sevenths.

Unmatched sets remain `unknown` with null root/inversion; notes are never
discarded to force a chord name. Octave doublings do not alter the pitch-class
match. Ambiguous templates prefer the lowest inversion, then fixed template
order: for example, C6 versus Am7 is not resolved using key, musical context
or a probability model. Accidentals use deterministic sharp spelling. Chord
segments shorter than `minimum_chord_seconds` are omitted.

These labels describe detected **sounding collections**, not ground-truth
harmonic analysis. They are analysis results, not persistent editable
MusicXML harmony annotations. Canonical source notes survive project
serialization; chord labels can be regenerated with `identify_chords`.
After user edits, regenerate harmony from the desired edited timeline rather
than treating the original analysis summary as authoritative.

### Resource and execution boundaries

Default limits are 120 seconds / 64 MiB of decoded mono float audio,
8,388,608 grid values, 100,000
accepted events and 16 concurrent tracked pitches. The existing model pitch
range remains MIDI 28-96; instrument priors can further restrict it. The
backend is requested to retain up to its supported 69 pitches before the
tracker's own admission cap; instrument-specific candidate limits still apply.
These are operational limits, not promises of recall at that polyphony.

The engine estimates grid size and checks duration **before inference**, then
the decoder checks the actual grid. WAV hosts also check duration before
loading samples. Compressed formats still use the existing optional FFmpeg
full-buffer reader: the inference limits do not bound compressed-file decode
memory. Prefer WAV for predictable preflight behavior.

Tracking uses `O(frames * pitches)` state updates plus sorting at most 128
start candidates per frame, with `O(pitches)` active state. Chord boundaries
are sorted in `O(events log events)`; emitted memberships scale with sounding
polyphony. The existing adapters still materialize per-pitch evidence tracks
before the activation grid, so this change does **not** solve all upstream
long-recording memory costs.

Cancellation is checked per decoded frame and at chord-sweep boundaries.
Failures use the existing engine `InferenceError` contract; missing models,
unsupported objectives and exhausted budgets are explicit errors, never
automatic success-shaped fallbacks.

This new tracker is **finite-recording/offline only**. Engine `begin()` rejects
it explicitly. The existing live windowed spectral/neural modes remain
separate; this change does not certify their seam matching, callback latency
or same-pitch voice identity.

## CLI

Existing `--polyphonic` behavior is unchanged unless `tracked` is selected.
The tracked mode chooses the spectral engine without a model, or the explicit
Basic Pitch adapter when a model is supplied:

```powershell
.\build\transcriber.exe input.wav --polyphonic --polyphonic-decoder tracked `
  --events-json events.json --midi notes.mid --score-project score.json

.\build\transcriber.exe input.wav --polyphonic --polyphonic-decoder tracked `
  --model .\python\models\basic_pitch.onnx --events-json events.json
```

The second command requires an ONNX-enabled build and a matching ONNX Runtime
DLL. `--events-json` adds `chords` and `polyphonic_diagnostics` fields in this
opt-in mode; its existing note fields remain intact. Chord member IDs reference
canonical source events (also retained in score/project output), not quantized
notation-note IDs. The ordinary `--json` legacy note format is unchanged.

`--minimum-confidence` sets the tracked activation-on threshold; activation-off
is 0.625 times that threshold. Other lifecycle/resource settings are currently
library options. The CLI rejects combining tracked mode with ASIO, live,
bounded-window inference, a monophonic backend, or the separate neural-event
decoder. Batch job fingerprints include the tracked choice.

## Workstation

Select **Polyphonic notes and chords** for offline analysis. With no model
selected, the host uses spectral DSP even in a model-free build. With a model
selected, it requires the Basic Pitch adapter; an incompatible model fails
explicitly. Completion and proposal summaries display measured chord times
and heuristic names, plus capacity-refusal diagnostics.
Instrument selection remains a notation/interpretation preference for Basic
Pitch, not a claim that the generic model is conditioned on that instrument.

The selection is disabled in live mode, whose backend selector remains
authoritative. Bounded neural windows and tracked polyphony cannot be combined
yet. The existing preview/proposal/acceptance rules are preserved; reanalysis
does not silently replace the accepted score.

## Library example

```cpp
transcribe::EngineRegistry registry;  // or registry(model_path)
auto created = registry.create("builtin:spectral"); // or "onnx:basic-pitch-v1"
if (const auto* error = std::get_if<transcribe::InferenceError>(&created))
    throw std::runtime_error(error->message);
auto engine = std::move(
    std::get<std::unique_ptr<transcribe::TranscriptionEngine>>(created));
auto initialized = engine->initialize();
if (const auto* error = std::get_if<transcribe::InferenceError>(&initialized))
    throw std::runtime_error(error->message);
transcribe::EngineConfiguration config;
config.audio_source_id = "take-1";
config.polyphonic_decoding.emplace();
auto configured = engine->configure(config);
if (const auto* error = std::get_if<transcribe::InferenceError>(&configured))
    throw std::runtime_error(error->message);
auto result = engine->transcribe(audio, cancellation);
if (const auto* error = std::get_if<transcribe::InferenceError>(&result))
    throw std::runtime_error(error->message);
const auto& transcription = std::get<transcribe::EngineUpdate>(result).polyphonic.value();
// transcription.events -> interpret_note_events; transcription.chords -> analysis display.
```

## Real-recording evaluation

[`python/evaluate_polyphonic.py`](../python/evaluate_polyphonic.py) runs the
built CLI once per recording (process-cold) with every system below and
scores notes with the shared metric code: onset+pitch within 50 ms,
onset+offset (offset within max(50 ms, 20% of reference duration)) and 10 ms
frames. "Overlapping-note onset recall" is onset recall restricted to
reference notes that sound together with a different pitch. All thresholds are
CLI defaults; **nothing was tuned on these recordings**. Results, per-recording
metrics, tracker diagnostics, executable/model/evaluator SHA-256 and dataset
checksums are stored in
[`polyphonic-tracked-guitarset-test.json`](../python/evaluation/polyphonic-tracked-guitarset-test.json)
and [`polyphonic-tracked-trios.json`](../python/evaluation/polyphonic-tracked-trios.json).
Dataset audio is read in place and is not redistributed.

| System | CLI selection |
|---|---|
| `yin` | default monophonic path |
| `spectral` | `--polyphonic` (existing spectral decoder) |
| `spectral_tracked` | `--polyphonic --polyphonic-decoder tracked` |
| `basic_pitch` | `--model basic_pitch.onnx --polyphonic` (existing decoder) |
| `basic_pitch_tracked` | `--model basic_pitch.onnx --polyphonic --polyphonic-decoder tracked` |

Datasets:

- **GuitarSet v1.1.0** (CC BY 4.0, MD5-verified), mono-microphone audio, the
  performer-held-out **test split (performer 05)**: 60 recordings, 30.5 min,
  8,715 reference notes, 30 comping and 30 solo takes. **Caveat:** Basic Pitch
  was trained with GuitarSet data, so these Basic Pitch numbers may be
  optimistic; the comparison between decoders shares that bias equally.
- **TRIOS** (Zenodo 6797837, CC BY-NC-SA 2.0 UK): five real chamber/jazz
  mixes (horn, violin, bassoon, trumpet, clarinet, viola, cello, saxophone and
  piano), 3.2 min, 1,534 pitched reference notes from the aligned part MIDI;
  drum parts are excluded. TRIOS contains **no guitar** and is not one of Basic
  Pitch's training sets. This makes it the out-of-domain, guitar-disjoint check. It is
  not disjoint from piano, which Basic Pitch saw in training.

Note results (micro-averaged F1 over all recordings):

| Dataset | System | Onset F1 | Onset+offset F1 | Frame F1 | Overlapping-note onset recall |
|---|---|---:|---:|---:|---:|
| GuitarSet test | yin | 0.226 | 0.035 | 0.261 | 0.077 |
| GuitarSet test | spectral | 0.202 | 0.045 | 0.398 | 0.531 |
| GuitarSet test | spectral_tracked | 0.096 | 0.040 | 0.505 | 0.147 |
| GuitarSet test | basic_pitch | 0.310 | 0.124 | 0.726 | 0.478 |
| GuitarSet test | **basic_pitch_tracked** | **0.719** | **0.498** | **0.783** | **0.754** |
| TRIOS | yin | 0.111 | 0.004 | 0.091 | 0.074 |
| TRIOS | spectral | 0.234 | 0.016 | 0.457 | 0.406 |
| TRIOS | spectral_tracked | 0.238 | 0.086 | 0.584 | 0.325 |
| TRIOS | basic_pitch | 0.149 | 0.022 | 0.551 | 0.198 |
| TRIOS | **basic_pitch_tracked** | **0.621** | **0.266** | **0.650** | **0.570** |

On GuitarSet, `basic_pitch_tracked` reached onset F1 0.692 on comping and 0.800
on solo takes, with onset+offset F1 of 0.471 and 0.577 respectively. The
legacy `basic_pitch` and `yin` figures agree with the repository's earlier
full-GuitarSet run (onset F1 0.331 and 0.223), which supports that the new
harness is consistent with the existing evaluator.

Paired differences (same recordings, same model weights), with a 95% interval
from 2,000 seeded recording-level bootstrap resamples:

| Dataset | Comparison | Onset F1 delta | Onset+offset F1 delta | Frame F1 delta |
|---|---|---:|---:|---:|
| GuitarSet test | Basic Pitch: tracked - existing | +0.409 [+0.376, +0.445] | +0.374 [+0.336, +0.414] | +0.057 [+0.042, +0.072] |
| GuitarSet test | Spectral: tracked - existing | -0.106 [-0.133, -0.080] | -0.005 [-0.030, +0.016] | +0.107 [+0.090, +0.129] |
| TRIOS | Basic Pitch: tracked - existing | +0.472 [+0.360, +0.584] | +0.244 [+0.210, +0.275] | +0.100 [+0.063, +0.128] |
| TRIOS | Spectral: tracked - existing | +0.004 [-0.007, +0.049] | +0.071 [+0.052, +0.089] | +0.127 [+0.104, +0.207] |

TRIOS intervals come from only five recordings and are therefore coarse.

Chord identification was scored against GuitarSet's verified *performed*
chord annotation on the 30 comping recordings. Scores use 10 ms frames. A
reference chord frame without a named predicted chord counts as wrong. Maj/min
recall uses MIREX-style reduction and excludes reference chords outside that
vocabulary:

| System | Named-chord coverage | Root recall | Maj/min recall |
|---|---:|---:|---:|
| spectral_tracked | 0.298 | 0.228 | 0.111 |
| basic_pitch_tracked | 0.447 | 0.377 | 0.279 |

Robustness and cost:

- The tracked decoder completed every recording with both backends. The existing
  spectral decoder exited with "Fingering exceeds simultaneous stretch or
  fretting-finger limits" on 3 of 60 GuitarSet recordings and on the
  `take_five` mix. Those runs are scored as producing no notes and are listed
  in each JSON file's `failed_runs`.
- Median / p95 process-cold wall time per GuitarSet recording (about 30 s of
  audio, including decode, model load and JSON export, on one Windows
  workstation, GCC 16 Release, ONNX CPU):
  - `basic_pitch_tracked`: 0.94 / 1.26 s.
  - `basic_pitch`: 0.99 / 1.68 s.
  - `spectral_tracked`: 0.51 / 1.06 s.
  - `spectral`: 0.72 / 1.91 s.

  These are single runs, not a performance qualification.

### Readiness assessment

- **Notes:** `basic_pitch_tracked` is the recommended polyphonic path. On both
  real-recording sets it beats the existing decoder by large margins. Each
  interval excludes zero, including on guitar-free ensemble audio that the
  thresholds were never adjusted for. Onset, sustain and offset tracking on
  solo or comped guitar is suitable for an opt-in production feature. Offsets
  in dense ensemble mixtures remain weak (onset+offset F1 0.266).
- **Spectral DSP mode:** keep it as the explicit model-free fallback, not a
  recommended equivalent. The tracker improves its frame accuracy and makes
  it robust, but on guitar it lowers onset F1. The model-free path remains far
  below Basic Pitch.
- **Chords:** naming is a convenience over exact sounding pitch-class sets.
  Fewer than half of the reference chord frames receive a name, and root recall is
  0.377. Present chord names as provisional annotations, not production-grade
  harmonic analysis. The next step is key/context-aware chord estimation that
  tolerates missing and extra tones, evaluated on the same protocol.

## Verification and release qualification

New Catch2 tests cover:

- exact simultaneous/overlapping intervals and independent offsets
- sustained activity and release debounce
- onset edges/plateaus and same-pitch retriggers
- end-of-input flushing, short-note filtering and window centering
- deterministic capacity limits, cancellation, malformed grids and resource budgets
- chord roots/inversions/unknown sets and JSON escaping
- real DSP engine integration and optional real Basic Pitch inference

[`tests/polyphonic_evaluation_tests.py`](../tests/polyphonic_evaluation_tests.py)
(CTest `polyphonic_evaluation_framework`) covers the evaluator: Harte chord
parsing, performed-annotation selection, chord-frame scoring, overlap
detection, note counts, the deterministic paired bootstrap, and a stdlib
Standard MIDI File reader with a tempo map, running status and FIFO
same-key overlap handling.

The production workstation backend is compiled into the test build and tested
without requiring WebView2. A Node CLI integration test generates overlapping
C/E/G audio, verifies named chords and source IDs, checks MusicXML multi-voice
and MIDI output, and exercises incompatible options. Browser workflow tests
exercise DSP availability without ONNX, named-chord summaries, proposal
summaries and capacity warnings.

Validation on Windows (rerun with the same binary used for the evaluation;
`transcriber.exe` SHA-256 `227bb0a6…df9e71`):

- Related C++ regression selection
  `"[polyphonic],[polyphonic-pipeline],[engine],[decoding],[interpretation],[score]"`:
  86 cases, 6,758 assertions passed.
- `ctest -R "polyphonic|cli_export_validation|ui_asset_smoke"`: 12/12
  passed. This covers the DSP and ONNX CLI pipelines, the evaluator unit tests,
  the synthetic polyphonic suite, CLI export validation and UI asset smoke.
- The Chrome workstation browser workflow passed, including the new UI
  assertions.
- The actual WebView2 host passed the `--polyphonic` workstation smoke. It
  generated overlapping C/E/G audio and displayed C-major harmony, canonical
  evidence and real exports. Run it with
  `python tests\workstation_host.py --webview <gui-executable> --polyphonic`.
  The native MSVC Release build needed `NOMINMAX` in local compiler flags and
  warnings-as-errors disabled for existing warnings. No repository-wide warning
  suppression was added. The broader existing native smoke stops at its
  spectrogram frequency-accessibility-metadata assertion and is not counted
  as passing.
- Full ONNX-enabled C++ suite: 218/220 passed. The two failures were
  independently reproduced against unchanged `2a4c943`:
  - the ASIO discontinuity-storm `overloaded` expectation
  - the tablature `"confidence"` wording

Remaining release work:

1. Measure peak RSS and p50/p95 processing time on target CPUs/GPUs with
   repeated runs, and validate model/DLL packaging and failure injection.
2. Calibrate any per-instrument thresholds only on GuitarSet's validation
   split (or another validation set); never on the test recordings above.
3. Add a non-guitar, non-piano real-recording set with more recordings so that
   out-of-domain intervals are tighter.
4. Add bounded window reconciliation before lifting the 120 s duration cap.
   Simultaneous same-pitch unisons from different instruments require
   source/voice information beyond one activity cell per pitch.

No new weights, source separator, CREPE, Onsets and Frames or MT3 runtime is
bundled by this change.

Reproduce:

```powershell
cd python
python evaluate_polyphonic.py --transcriber <build>\transcriber.exe `
  --guitarset-audio <dir>\audio_mono-mic.zip --guitarset-annotations <dir>\annotation.zip `
  --guitarset-split test --output-json evaluation\polyphonic-tracked-guitarset-test.json
python evaluate_polyphonic.py --transcriber <build>\transcriber.exe `
  --trios "<dir>\TRIOS Dataset" --output-json evaluation\polyphonic-tracked-trios.json
```
