# Engineering Decisions

Decisions that are non-obvious from reading the code, and the constraints that
forced them.

---

## 1. No librosa — the DSP is hand-written

**Constraint.** `librosa.effects.time_stretch` depends on `numba`, which
JIT-compiles at runtime and emits `.pyd` binaries. The enterprise AppLocker
policy on the target Windows machine blocks dynamically generated binaries, so
any librosa call touching numba failed outright — with no administrative
workaround available.

**Decision.** Implement the phase vocoder (`src/augmentation/axis_c_temporal.py`)
and the entire Mel filterbank + DCT chain (`src/features/extractor.py`) directly
on `scipy.signal` and `numpy`.

**Consequence.** More code to maintain and test, but it also serves the academic
requirement that the DSP be auditable rather than a library call. `librosa` is
not a dependency and must not be reintroduced.

---

## 2. int8 quantization is forced, not chosen

**Constraint.** Whisper `large-v3` needs roughly 7–8 GB of VRAM in float16. The
target GPU is a 4 GB RTX 3050.

**Decision.** `WHISPER_COMPUTE_TYPE = "int8"` in `config.py`, with a fallback
chain: configured → CUDA/int8 → CPU/int8.

**Consequence.** The measured clean baseline is 3.44 % rather than the ~2.7 %
published for the unquantized model. Absolute WER figures are therefore specific
to this configuration; the relative comparisons between conditions — which is
what the study is about — are unaffected, since every condition runs through the
same quantized model.

---

## 3. `config.py` mutates `PATH` at import time

**Constraint.** CTranslate2 needs `cublas64_12.dll` and `cudnn64_9.dll`. When
installed via pip these land in `venv/Lib/site-packages/nvidia/*/bin`, which is
not on the Windows DLL search path.

**Decision.** `config.py` appends those directories to `os.environ["PATH"]` on
import.

**Consequence — this is load-bearing.** `import config` **must** happen before
`import faster_whisper`, or CTranslate2 cannot find the DLLs. Every module
already satisfies this; reordering imports will break GPU inference with an
opaque error. `src/inference/whisper_engine.py` documents it inline.

Both directory additions are guarded by `.exists()`, so a non-venv install
degrades silently rather than crashing — which is why the OOM fallback chain
deliberately re-raises non-OOM errors instead of quietly demoting to CPU.

---

## 4. Out-of-memory skips the row instead of recording it

**Problem.** The original handler caught OOM and returned `""`. That empty
transcript flowed into `evaluate_metrics`, producing a row with
`deletions == len(reference)` and `wer == 1.0` — indistinguishable from a
genuine total recognition failure.

**Why it mattered.** OOM is most likely during beam expansion on the most
degraded audio, i.e. 5 dB café — precisely the experimental cell the headline
finding rests on. A handful of fabricated 100 %-WER rows would have inflated
exactly that number, with nothing in the output recording that it happened.

**Decision.** `transcribe()` raises `TranscriptionOOMError`. Callers skip the
row entirely, increment a counter, and warn at the end of the stage. Because
resumption is keyed on rows *present* in the output CSV, a subsequent run
naturally retries the skipped ones.

**Related.** The OOM detector no longer matches the bare substring `"cuda"`.
That swept up unrelated failures — missing drivers, missing DLLs, wrong device —
and silently demoted them to an empty transcript or a days-long CPU run. It now
matches only genuine memory markers, and everything else propagates.

---

## 5. Noise is drawn before the resume check

**Problem.** `scripts/02_run_axis_a.py` originally skipped already-processed
rows *before* calling `noise_source.get_noise()`. The RNG streams therefore
advanced only for rows actually computed, so a run resumed at row 900 drew 900
fewer samples than a fresh run and produced **different noise for every
remaining file**. The results CSV became a mixture of two incompatible noise
realizations with nothing recording it.

**Decision.** Draw the noise first, then check the resume set. The RNG advances
once per `(utterance, noise_type, snr)` combination regardless of how many rows
already exist.

**Also fixed.** The resume key parsed `snr_db` with `int()` after an `isdigit()`
guard, so a value serialized as `"5.0"` fell through as a string and raised
inside `int()` — an exception that was swallowed by a bare `except`, emptying the
resume set and duplicating all 1,800 rows. It now uses `int(float(...))`.

**Loop order is fixed** at utterance → noise_type → snr. The café cursor slides
through a pre-generated buffer, so iteration order determines which babble
segment each file receives. Reordering the loops would change every generated
file for no benefit.

---

## 6. Three independent RNG streams

`NoiseSource` keeps `_rng_white`, `_rng_pink` and `_rng_cafe` seeded at 42, 43
and 44 rather than sharing one generator.

With a single shared generator, the number of draws taken by one noise type
would shift every subsequent value produced for the others — so adding an SNR
level, or changing utterance count, would silently alter noise that had nothing
to do with the change. Separate streams make each noise type independently
reproducible.

---

## 7. Café babble comes from a disjoint speaker pool

**Problem.** Two contradictory generators existed. `NoiseSource` built babble
from a FLAC pool filtered to exclude the test utterances; `scripts/02` built it
from **the 150 test utterances themselves**. Which one ran depended purely on
whether `cafe.wav` already existed — and if the latter, an utterance could be
masked by babble containing itself.

**Decision.** The duplicate in `scripts/02` is removed. All café generation goes
through `NoiseSource._ensure_cafe_loaded`, which filters
`_raw_download/LibriSpeech/dev-clean/**.flac` by stem against the test set.

The generated babble is cached to `data/noise_sources/cafe.wav` and its exact
speaker membership written to `cafe.provenance.json`, so the disjointness claim
is **verifiable rather than asserted**.

---

## 8. Resume refuses to start from a partial read

`load_resume_keys` raises on any read error rather than returning what it
managed to parse.

The previous `except Exception: pass` meant a truncated final row — from a hard
kill mid-write — would abort the read partway, leave a half-filled resume set,
and cause the run to **append rows the file already contained**. Aggregate WER
would then double-count those utterances, in a way nothing in the output would
reveal. A partial resume set is strictly worse than refusing to start.

---

## 9. Per-row flush, not batched writes

Every inference stage calls `f.flush()` after each row. At ~2,850 rows the
syscall cost is negligible, and it bounds crash loss to a single row.

This is also why the inference stages use the standard-library `csv` module
rather than pandas: the append-on-resume pattern needs streaming writes. pandas
appears only in the analysis stage and the corpus loader, where whole-frame
operations are the natural fit.

---

## 10. One implementation of the reported metric

Global WER was previously computed in four places — the analysis stage, two
document generators and the diagnostic tool — each maintained by hand.

It now lives once in `src/analysis/metrics.py`, alongside `mean_utterance_wer`
so the distinction between the two definitions is explicit in code rather than
folklore. `tools/diagnostic_report.py` prints both side by side.

---

## 11. Condition names come from a registry

`silence_sta_0.5s` and friends were re-typed as string literals in four files.
They are baked into filenames *and* into the result CSVs, so a mismatch fails
silently — a wrong name simply finds no files.

`config.py` now owns `stretch_condition()`, `silence_condition()` and
`AXIS_C_CONDITIONS`. Adding a condition is a one-line change, and
`tests/test_degradations.py` pins the exact on-disk strings so a rename cannot
silently orphan the existing corpus.

---

## 12. Verification measures the property, not a proxy

`tools/verify_acoustics.py` independently re-derives every degradation parameter
from the files on disk. Two of its checks needed to be more careful than they
first appeared:

**SNR must not assume the clean component is unscaled.** When the anti-clip path
fires, the degraded file contains `α·clean` for some α < 1. Subtracting the
unscaled clean leaves an `(α−1)·clean` residual in the "noise" term and inflates
the measurement — this reported 14 correctly-mixed files as failures. The check
now recovers α by least-squares projection, which reduces to plain subtraction
when α = 1.

**Bandpass suppression must not depend on what the source contained.** The
original check required the in-band/out-of-band *ratio* to improve tenfold,
which failed on utterances that were already nearly band-limited — little
out-of-band energy to remove — despite the filter working perfectly. It now
measures the retained out-of-band energy directly. The threshold (25 %) is set
from the measured distribution across all 150 files (median 2.4 %, max 11.3 %,
that spread being the intrinsic roll-off of a 4th-order Butterworth), leaving a
wide margin below the ~100 % a missing filter would produce.

**Silence runs may legitimately exceed the inserted duration.** Some recordings
open with their own near-silence, which merges with a start-position insertion.
The check asserts *at least* the target; exact total length is asserted
separately.

The script exits non-zero on failure and runs as stage 4 of the pipeline.
`--all` sweeps the corpus: 3,150 checks, 0 failures.

---

## 13. Test-mode outputs are fully isolated

`--test-mode` previously read from `results/test_mode/` but wrote plots to the
production directory, so a one-utterance smoke run silently overwrote the
published figures. All test-mode paths now derive from a single `out_dir`.

---

## 14. The Mel filterbank is cached

`build_mel_filterbank` is a pure function of `(n_fft, sr, n_mels)` and was being
rebuilt — via a Python double loop — on all 5,400 calls in the feature stage. It
is now `@lru_cache`d and returned read-only so callers cannot mutate the shared
array. The delta-MFCC computation was vectorized in the same pass; the result is
bit-identical to the previous loop.

A full inverse STFT was also being computed and immediately discarded on every
call. Removed.

---

## 15. What was deliberately left alone

- **`src/api/service.py`** — a demo endpoint, not part of the experiment. It has
  known defects (an unbound `temp_path` in its `finally`, an `async def` handler
  making a blocking call, unbounded upload size, `str(e)` in the 500 response)
  and is out of scope for this pass.
- **Existing generated audio.** Every degradation was verified correct against
  its stated parameters, so nothing was regenerated. All committed results
  remain valid.
- **`config.py` creating directories on import.** Convenient, and no test yet
  needs a side-effect-free import.