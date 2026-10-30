# Measured C++20 DSP performance

## Scope and evidence

This change optimizes the existing algorithms, not the transcription model,
thresholds, sample rates, chunk size, note assembly or public API. No fast-math,
architecture-specific intrinsics, extra production dependency or additional
worker pool is introduced. The normal lightweight build is unchanged.

Before editing DSP, the existing `AnalysisTimings` reports were inspected.
The synthetic YIN report attributed about 0.502 s to pitch analysis and
0.085 s to onset/FFT processing; spectral attributed 0.166 s and 0.080 s,
respectively. These were **stage wall-clock profiles**, not sampled
instruction profiles. No existing WPR/VTune sampling trace was available,
and no sampling-profiler result is claimed.

The new [performance runner](../tests/performance_runner.cpp) then measured
the unmodified engine before the production change. Its expanded workloads
confirmed YIN and FFT as the priorities:

| Cold stage, summed across workloads | Before YIN | After YIN | Before spectral | After spectral |
|---|---:|---:|---:|---:|
| Preprocessing | 0.0394 s | 0.0392 s | 0.0389 s | 0.0417 s |
| Onset FFT | 0.5198 s | 0.3808 s | 0.5269 s | 0.3815 s |
| Resampling | 0.0219 s | 0.0206 s | 0.0223 s | 0.0212 s |
| Pitch analysis (includes spectral FFT/salience) | 1.7158 s | 0.5684 s | 0.5506 s | 0.4139 s |

Stages exclude note assembly and streaming decode. Stage sums are therefore
not the end-to-end totals below. Resampling was a small fraction of this
profile; changing its interpolation/anti-aliasing would be a separate quality
change, not an appropriate speed shortcut.

## Changes retained

In [transcriber.cpp](../src/transcriber.cpp):

- **YIN:** accumulate four independent lag differences in one contiguous
  input scan. Load the shared leading sample once. Each lag's double-precision
  additions remain in the original order; the short unequal tails and
  remaining lags use scalar loops. Compilers can vectorize across independent
  lags without reassociating a reduction. No explicit ISA requirement or
  altered voicing/pitch threshold is used.
- **FFT:** immutable, lazily initialized plans reuse bit-reversal swaps and
  twiddle factors. Twiddles use the original root-multiplication recurrence,
  not a numerically different direct trigonometric formula. Plans cover
  256/512/1024/2048/4096, including every public GUI-preview size.
  Initialization is thread-safe; FFT workspaces remain private per worker.
- **Windows/allocations:** reuse cached 2048/4096 Hann windows for spectral
  analysis and reserve the known onset flux/energy frame counts. Existing
  spectrum buffers, YIN workspaces and candidate-vector capacity were already
  reused; they were not replaced with per-frame allocations.
- **Cache synchronization:** retire the old spectrogram outside the cache
  mutex, avoiding a potentially multi-megabyte deallocation under that lock.
  This lock improvement has no isolated speed claim.

No sliding subtraction of YIN differences, approximate FFT, pitch pruning,
changed candidate ordering or reordered floating-point reductions is used.
Those could change boundaries/confidence despite unchanged average pitch.
Exact recomputation is retained where it protects output identity.
Clang's `-Rpass=slp-vectorizer` diagnostic confirmed SLP vectorization in
`yin_with_workspace`; emitted assembly contained packed double arithmetic
without fast-math or target-specific flags. This is compiler evidence on the
measured toolchain, not a requirement that every supported compiler vectorize.

## Workloads and protocol

Toolchain: LLVM-MinGW Clang 23.1.1, C++20, Windows 11 AMD64, Release
`-O3 -DNDEBUG`, 12 logical CPUs, Intel family 6/model 165. Python 3.12.10
orchestrates the runs; it is not a production dependency.

[performance.py](../python/performance.py) generates:

- All 16 corpus-v1 fixtures, not just the historical guitar scale.
- Six-second low notes (MIDI 36/40).
- A six-second dense chord (MIDI 40/47/52/55/59/64).
- 90-second silence and dense-chord WAVs, analyzed with the unchanged
  15-second streaming chunks and one-second overlap.
- An optional local real recording, supplied for this experiment: 10.720 s,
  stereo PCM16, 48 kHz. It is not redistributed or uploaded. There are no
  annotations or independently established difficulty/provenance claims for
  that recording.

Each of 21 clips runs once per backend in a fresh process: **42 workloads**.
Each process makes one cold and three warm calls. Cold means first invocation
of engine plans/cache in that process, **not** flushed disk/OS caches.
Warm means subsequent calls on identical audio, including eligible spectral
cache hits. Each full-buffer decode happens once outside analysis timing;
streaming timings include decoding on every call. These timings are not
interchangeable with the process-cold CLI/export timings in the
[quality framework](evaluation.md).

Reports include compiler/build flags, executable digest, current checkout
source digests, provider/model identity, audio rates/channels/hashes,
parallel/chunk settings, predictions, labeled metrics, CPU time, elapsed time
and process peak resident memory. The baseline binary was saved **before**
modifying DSP; source and executable identity are recorded separately.
Do not infer that an old executable matches a subsequently changed checkout.

CPU time is kernel+user process time (Windows `GetProcessTimes`; POSIX
`getrusage`). CPU/core equivalents are CPU seconds divided by wall seconds:
1.0 means one busy logical core, not 100% of the whole machine. Short Windows
measurements can quantize to zero CPU time; do not interpret that as free work.
Peak resident memory is a process-lifetime high-water mark, including retained
cache/decode/results, not allocation totals or an isolated per-call peak.
Sub-clock-resolution speed ratios are `null`, not fabricated infinities.

Measurements ran sequentially, without concurrent test/build workloads.
Power policy, antivirus, turbo and thermal state were not controlled.
One cold observation and three warm observations per case do not establish
confidence intervals or universal speedups; raw observations are retained in
local JSON reports. A second optimized run also produced identical outputs,
but is not substituted for the final measurements.

## Release measurement schema

The performance runner now records a distinct first inference before any
warmup, then excludes the configured warmup calls from its `runs` array.
`performance.py --warmups N --repetitions M` therefore produces one
`cold_run`, N discarded warmups, and M reported warm samples. It also measures
the whole subprocess wall time and runner startup separately; runner startup
includes decode and model/session initialization, while model-load time is
reported independently when available.

Schema-v2 reports contain warm p50/p95/p99, CPU/wall time, RTF, audio-seconds
per second, peak resident memory, first-window time, build/compiler/features,
model hash, provider/runtime, CPU identity and best-effort GPU inventory.
Quantiles are linearly interpolated. A p95/p99 based on fewer than 100 warm
samples is explicitly descriptive and must not be described as statistically
stable. RSS is process-lifetime high-water memory, including warmups.

When comparing reports, matching hardware and build configuration are
recorded explicitly. Timing deltas with a hardware or build mismatch remain
observations, not speedup claims. Shared CI does not gate wall-clock timings;
the historical results above use the earlier protocol and must not be
compared with schema-v2 reports as though measurement order were identical.
The clean-checkout command, release report, deterministic per-fixture gates,
data-provenance policy and manual acceptance record are documented in
[Release evidence](release-evidence.md).

## Results

Each backend processed **244.130 seconds** of audio once in the cold total.
Warm totals sum per-workload medians; they do not combine cache hits and cold
misses into one apparent result.

| Backend | Cold before / after | Cold speedup | Warm before / after | Warm speedup |
|---|---|---:|---|---:|
| YIN | 2.330 / 1.043 s | 2.23x | 2.273 / 1.043 s | 2.18x |
| Spectral | 1.159 / 0.880 s | 1.32x | 0.972 / 0.762 s | 1.28x |

This corresponds to approximately 105 -> 234 audio-seconds/second for cold
YIN and 211 -> 278 for cold spectral on these workloads, not a hardware
real-time guarantee. Total process CPU across all four invocations fell from
15.0625 -> 6.1875 CPU seconds for YIN and 6.0000 -> 4.4844 for spectral.

Selected per-workload wall times (milliseconds):

| Workload | Backend | Cold before / after | Warm median before / after |
|---|---|---|---|
| Low notes, 6 s | YIN | 127.35 / 48.88 | 127.35 / 48.78 |
| Dense chord, 6 s | YIN | 131.42 / 48.72 | 132.40 / 48.02 |
| Silence, 1 s | YIN | 2.10 / 1.82 | 2.40 / 1.63 |
| Silence, 90 s streaming | YIN | 218.57 / 170.93 | 219.38 / 168.51 |
| Chord, 90 s streaming | YIN | 1100.23 / 463.36 | 1048.10 / 469.14 |
| Local recording | YIN | 147.75 / 71.32 | 149.19 / 70.85 |
| Low notes, 6 s | Spectral | 30.70 / 25.49 | 14.75 / 11.98 |
| Dense chord, 6 s | Spectral | 33.29 / 25.28 | 14.41 / 11.44 |
| Silence, 1 s | Spectral | 7.20 / 5.50 | 2.16 / 1.69 |
| Silence, 90 s streaming | Spectral | 273.98 / 211.56 | 264.31 / 207.46 |
| Chord, 90 s streaming | Spectral | 455.38 / 341.27 | 476.89 / 367.91 |
| Local recording | Spectral | 80.87 / 62.49 | 50.67 / 42.64 |

Peak resident MiB, before -> after:

| Workload | YIN | Spectral |
|---|---|---|
| Low notes | 5.68 -> 5.75 | 10.88 -> 11.06 |
| 90 s silence, streaming | 8.01 -> 8.12 | 23.04 -> 23.17 |
| 90 s chord, streaming | 8.14 -> 8.08 | 29.39 -> 29.57 |
| Local recording | 9.21 -> 9.27 | 17.59 -> 17.77 |

Plans/windows intentionally trade a small bounded amount of memory for CPU.
No claim of reduced peak memory is made. For the long chord, pooled measured
CPU/core equivalents were 2.26 -> 1.98 (YIN) and 1.76 -> 1.67 (spectral):
elapsed-time savings were not obtained by increasing parallelism.

## Quality and regression checks

All **42 before/after workloads produced exactly equal notes**, including
MIDI pitch, onset, duration and confidence. All warm repetitions also matched
their cold predictions exactly. Labeled workload event-F1 deltas were zero.
The local recording checks output preservation only, not absolute accuracy.

Independent three-repetition comparisons against the checked-in corpus-v1
baselines passed a **zero event-F1 drop** gate for YIN streaming, spectral full
and realtime YIN:

| Mode | Strict event F1 before / after | Frame F1 before / after |
|---|---|---|
| YIN streaming | 48.15% / 48.15% | 78.33% / 78.33% |
| Spectral full | 26.67% / 26.67% | 81.01% / 81.01% |
| YIN realtime | 69.57% / 69.57% | 80.65% / 80.65% |

No tolerance was widened. Original checked-in baselines were not overwritten.
Real-recording ground-truth accuracy, broad held-out generalization and model
training overlap remain unverified. The historical GuitarSet report was not
rerun; one unlabeled recording is not a substitute for that dataset.

Validation:

- Default Release: 43 passed, one external-recording test initially skipped
  across 44 CTests; that test subsequently **passed** with the supplied clip.
- ONNX/ASIO Release: 45 passed, one external-recording test initially skipped
  across 46 CTests; that test subsequently **passed** with the supplied clip.
- New scalar YIN oracle covers 175 combinations of sample rate, short/odd/
  full frame length, silence, low and high pitches; pitch/confidence/voicing
  agree exactly.
- Added concurrent normal/preview FFT/cache tests, sample-content/sample-rate
  invalidation, all public preview FFT sizes, and CPU/memory/warm-cache report
  tests. Existing tests are retained.
- In the local validation recorded above, Linux/macOS builds, alternate
  compilers, GPU providers, physical ASIO devices and GUI responsiveness under
  interactive load were not measured. Catch2's existing Clang `__COUNTER__`
  warnings remain.

The CI workflow is configured for Windows and Linux Debug/Release matrices,
Linux GCC/Clang builds, an optional FFmpeg build, CPU ONNX packaging, and a
Windows WebView2/Verovio package check. It also defines Linux sanitizer and
clang-tidy jobs. Those workflow definitions do not update the historical
measurements above; consult the exact Actions run for executed results.
The workflow uploads per-build evaluation reports but intentionally does not gate on
single-run timing noise. The default and ONNX CPU packages are exercised from
an extracted archive; optional FFmpeg still depends on host runtime libraries.

## Cache, concurrency and deferred work

The spectrogram cache remains a single immutable entry, keyed by full
float-bit audio hash, sample count, sample rate, window and hop. Instrument,
threshold and candidate-count options do not affect magnitudes and correctly
do not invalidate it; frame geometry/content/rate changes do. The 64 MiB
limit bounds **retained magnitudes**, not active computations or the entire
process. Hash collisions remain a theoretical pre-existing limitation; no
sampled/partial hash replaced the full-content key.

Hashing and FFT processing happen outside the cache lock. Concurrent misses
may compute duplicate work, and active readers may temporarily retain the
old entry. Retired-cache destruction now also happens outside the lock.
Do not interpret the cache limit as a global concurrent-memory cap.

Long WAV streaming still decodes blocks of at most 4096 samples and keeps
one configured chunk plus overlap, output notes and bounded retained cache.
Its output-note list can grow with recording length. Full-buffer spectral and
ONNX analysis still scale with input length; this change does **not** make
those APIs bounded-memory. Frame spectra retain their existing per-worker
buffers; resampled audio and overlapping chunks are not duplicated further.

The existing offline scheduler remains capped at eight workers with at least
256 frames per worker and does not nest a new thread pool. No unmeasured
scheduler change or interactive-responsiveness improvement is claimed.
Realtime analysis still runs on its analysis thread and holds its existing
state mutex during DSP; changing its ownership/public thread-safety contract
was deferred. The ASIO device callback is unchanged: bounded ring writes and
atomics only, with no FFT/model/UI/disk work, allocation or mutex added.
FFT plan initialization happens on analysis/preview paths, not that callback.

Spectral salience/candidate extraction, redundant ONNX conversions and same-rate
buffer copies remain deferred: the available profiles do not isolate them as
dominant costs. Approximate sliding-window YIN and fast-math SIMD were
specifically avoided. The ASIO ring transfer loop is the exception: a focused
microbenchmark justified changing its per-sample copies to contiguous copies,
without changing the data or publication order.

## Follow-up audit: profiling plan and ring benchmark

The ASIO [CaptureTransport](../src/asio_transport.hpp) preallocates separate
recording and analysis SPSC rings. Both consumers need independent lifetimes, so
the two copies are intentional; the callback adds no allocation, lock, FFT,
inference, UI or disk work. The previous ring implementation advanced and
checked its wrap position for every sample. [AudioRingBuffer read/write](../src/audio_ring_buffer.cpp)
now copies each block in one contiguous range, or two ranges when it wraps.
Atomic publication and partial-read/write behavior are unchanged. Existing
concurrent tests in [asio_tests.cpp](../tests/asio_tests.cpp) exercise arbitrary
capacities and verify every sample.

The [audio ring benchmark](../tests/audio_ring_buffer_benchmark.cpp) isolates
the transfer loop across 32-4096 sample blocks. Each run transfers 4,194,304
samples through a non-power-of-two ring capacity, with five repetitions; ring
construction and the final exact sample check are outside the timed interval.
On the local Release build, the 512-sample median fell from 4.9536 ms to
0.3711 ms (13.35x); aggregate read plus write throughput increased from
6,459.9 to 86,230.1 MiB/s. The 32-1024 sample cases improved 2.80x-14.15x.
The 4096-sample case had visibly higher run-to-run variance. This is a
cache-resident, single-thread microbenchmark, not an ASIO callback latency
measurement or a hard real-time claim.

Other findings and measurement limits:

- The existing immutable FFT plans, cached Hann windows and worker-private
  workspaces avoid repeated plan/window construction and per-frame allocation.
  Current onset and pitch stage timers include more than FFT work, so they
  cannot attribute a change to an individual transform.
- The offline scheduler is bounded to eight workers and does not create nested
  pools. Stage wall time and process CPU time reveal scaling, but not context
  switching or cache misses.
- The long-file runner reports process high-water RSS, not allocation counts,
  allocation sites or per-call peaks. A sample-content hash and spectrogram
  cache lookup also require a full audio scan.
- `capture_asio` requests 512 frames at 44.1 kHz. The unit tests validate ring
  concurrency and sample integrity, but they do not measure a physical ASIO
  driver or establish a callback deadline.

### Profiling plan

1. Use a Release build and record the executable/source digests, compiler,
   build flags, CPU, provider, input hashes, sample rate, channel count, chunk
   size and scheduling mode. Compare only matching configurations.
2. Run [performance.py](../python/performance.py) on the generated labeled
   workloads and, where available, a consented local recording. Preserve its
   cold and warm runs, stage timings, process CPU/wall time, peak RSS, exact
   note output and labeled metrics. Compare serial and parallel runs separately.
3. Use a sampled CPU profiler to find instruction-level hotspots: Windows
   Performance Recorder/Analyzer, Visual Studio CPU Usage, or VTune; on Linux,
   `perf record -g` and `perf stat -d`. Attribute FFT time by sampled call
   stacks and normalize by frame count. Keep cold plan initialization separate
   from warm transforms.
4. Measure heap allocations with a native allocation trace (Visual Studio
   Native Memory/ETW heap events on Windows, or `heaptrack` on Linux). Record
   count, size and call site separately from RSS. Also identify copies by
   call stack and bytes moved; do not infer allocation totals from working-set
   size.
5. For cache locality, capture hardware cache-miss and memory-bandwidth
   counters on representative long inputs. Treat the ring microbenchmark as a
   transfer-kernel check only; use an end-to-end trace to decide whether
   spectrogram traversal or other DSP buffers are cache-bound.
6. For threading, capture worker count, CPU-core equivalents, context switches
   and wall time for the same workload in serial and parallel modes. Do not add
   a second thread pool or interpret more CPU consumption as lower latency.
7. For real-time safety, profile the physical ASIO callback externally without
   logging or adding locks/allocations to it. Record callback-duration p50/p99/
   maximum against the configured buffer period, along with driver overflow,
   refused blocks and queue high-water marks. Repeat on supported hardware;
   simulated ring tests are not a substitute.
8. Reject any optimization that changes emitted notes, confidence or sample
   ordering. Keep exact ring sample checks, exact benchmark note comparison and
   labeled quality gates; report hardware-only timing as an observation, not a
   CI threshold or a real-time guarantee.

Build and run the focused microbenchmark with:

```powershell
cmake --build build-performance --config Release --target transcriber-audio-ring-benchmark
.\build-performance\transcriber-audio-ring-benchmark.exe
```

Its JSON output goes to stdout. The CTest smoke case checks that the benchmark
executes and preserves samples; it intentionally has no machine-dependent
speed threshold.

## Reproduce

Build with the existing test switch (no new dependencies):

```powershell
cmake -S . -B build-performance -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_TESTS=ON
cmake --build build-performance --parallel
ctest --test-dir build-performance -C Release --output-on-failure

python .\python\performance.py `
  --runner .\build-performance\transcriber-performance.exe `
  --build-info .\build-performance\evaluation-build-Release.json `
  --recording "C:\path\to\local-recording.wav" `
  --output .\build-performance\evaluation\performance-before.json
```

Keep the recording local and retain consent/provenance independently. Omit
`--recording` only if real-recording validation is explicitly reported absent.
The generated stress fixtures require no download. Use `--serial` for a
separate serial experiment; do not compare serial to parallel as equal configs.

Preserve the baseline report and rebuild the candidate with identical flags,
then run with a distinct output:

```powershell
python .\python\performance.py `
  --runner .\build-performance\transcriber-performance.exe `
  --build-info .\build-performance\evaluation-build-Release.json `
  --recording "C:\path\to\local-recording.wav" `
  --compare .\build-performance\evaluation\performance-before.json `
  --output .\build-performance\evaluation\performance-after.json
```

The comparison requires identical audio hashes and configuration, and fails
on **any** changed emitted note, not only a drop in average F1. Inspect raw
per-call CPU/peak/stage data as well as ratios. Never overwrite a baseline
report with the comparison output.

To reconstruct this experiment, the unoptimized production source is commit
`721d254`. Use a separate checkout of that revision with only the current
performance harness/CMake/test additions applied, **not** the production
`src/transcriber.cpp` optimization. Build/run that baseline before building
the optimized checkout. Keep metadata matched to each build directory.
An old engine binary and a new harness are comparable only when the binary
and source identities are recorded honestly.

Local measured reports for this run are under the ignored
`build-release-mingw\evaluation` directory: `performance-before.json`,
`performance-final.json` and `optimized-*.json`. They contain private local
recording-derived note events and are intentionally not committed. The public
tables above disclose timings, not the recording or its note content.

## Subsequent model-interface measurements

The versioned inference adapters have a separate
[persistent-session CPU benchmark](../python/evaluation/model-interface-windows-clang23-cpu.json)
and [contract/measurement documentation](model-inference.md#persistent-session-cpu-measurements).
It covers YIN, spectral and Basic Pitch on the same public synthetic/stress
fixtures, reports session loading separately, and checks exact legacy outputs.
It contains no private recording. These are descriptive interface measurements,
not a rerun or replacement of the before/after optimization experiment above.
No GPU execution provider was available for that experiment.

## Bounded long-recording pipeline

See [streaming contracts, limits and reproducible measurements](streaming.md)
for the opt-in WAV/Basic Pitch file pipeline, repeated-result cache, cancellation,
two-hour protocol and explicit non-streaming backend limitations.
