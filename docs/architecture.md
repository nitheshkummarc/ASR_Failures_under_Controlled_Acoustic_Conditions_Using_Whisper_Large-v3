# Architecture

## Overview

A three-tier batch pipeline. Stages communicate exclusively through the
filesystem — CSVs and WAVs — so every stage is independently runnable and
independently resumable.

```
┌──────────────────────────── ORCHESTRATION ─────────────────────────────┐
│  main.py — subprocess sequencer + JSON checkpoint                      │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ spawns (process isolation)
┌───────────────────────────────▼────────────────────────────────────────┐
│  DRIVERS   scripts/00..06                                              │
│  Own: CLI args, CSV I/O, resume sets, loop structure, progress bars    │
│  Do NOT own: any DSP or ASR logic                                      │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ imports pure functions + two classes
┌───────────────────────────────▼────────────────────────────────────────┐
│  LIBRARY   src/                                                        │
│  augmentation/  mix_at_target_snr, bandpass, phase vocoder, silence    │
│  inference/     WhisperEngine (stateful), evaluate_metrics, runner     │
│  features/      extract_all_features                                   │
│  analysis/      global_wer                                             │
│  data/          normalize_text, corpus loader                          │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ imports
┌───────────────────────────────▼────────────────────────────────────────┐
│  FOUNDATION   config.py (constants + import-time side effects)         │
│               src/logger.py                                            │
└────────────────────────────────────────────────────────────────────────┘

       ┌───────────────── PARALLEL / DETACHED ──────────────────┐
       │  src/api/service.py — reuses the library, bypasses     │
       │  the drivers entirely. Not part of the experiment.     │
       └────────────────────────────────────────────────────────┘
```

Dependencies flow strictly downward: `main → scripts → src → config`.
There are no circular imports. `config.py` imports only the standard library.

## Why subprocesses rather than function calls

`main.py` spawns each stage with `subprocess.run([sys.executable, "-u", script])`
and waits. This is not for parallelism — nothing runs concurrently. Three
reasons:

1. **GPU memory.** Each inference stage loads Whisper `large-v3`. Process exit
   is the only reliable way to fully release VRAM on Windows.
2. **Fault isolation.** A stage crash cannot corrupt the orchestrator's state;
   `check=True` gives a clean `CalledProcessError` boundary.
3. **Independence.** Every stage stays runnable on its own from the CLI.

The `-u` flag keeps stdout unbuffered so `tqdm` progress bars stream live
through the pipe.

## Pipeline stages

| # | Script | Reads | Writes |
|---|---|---|---|
| 0 | `scripts/00_prepare_corpus.py` | LibriSpeech archive | `metadata/utterance_manifest.csv`, 150 WAVs |
| 1 | `scripts/01_run_baseline.py` | manifest | `results/baseline_results.csv` |
| 2 | `scripts/02_run_axis_a.py` | manifest | 1,800 WAVs, `results/axis_a_results.csv` |
| 3 | `scripts/03_run_axis_b_c.py` | manifest | 900 WAVs, `results/axis_b_results.csv`, `axis_c_results.csv` |
| 4 | `tools/verify_acoustics.py` | generated WAVs | PASS/FAIL, non-zero exit on failure |
| 5 | `scripts/04_run_feature_extraction.py` | result CSVs + WAVs | `results/feature_deviation.csv` |
| 6 | `scripts/05_run_analysis.py` | all result CSVs | summary charts, logged p-values |
| 7 | `scripts/06_generate_spectrograms.py` | one utterance's WAVs | per-utterance figures |

Stage 5 uses the **result CSVs** as its work-list, not the manifests, which is
why `feature_deviation.csv` has exactly as many rows as there were successful
transcriptions.

## The inner loop

Identical in shape across stages 1–3:

```
manifest row
  ↓  sf.read(clean_wav)                        float64 ndarray
  ↓  [stage 2/3] degrade(y)                    ndarray
  ↓  sf.write(path, PCM_16)                    ← round-trip to disk is mandatory:
  ↓                                              WhisperEngine takes a path
  ↓  engine.transcribe(path)                   raw string
  ↓  normalize_text(raw)                       normalized hypothesis
  ↓  evaluate_metrics(ref, hyp)                {wer, cer, S, I, D, hits}
  ↓  writer.writerow(); f.flush()              ← crash-safe: ≤1 row ever lost
```

The disk round-trip also means every downstream measurement analyses the same
16-bit quantized signal the model actually saw, which is the correct thing to
measure.

## Resumption

Two independent layers, both of which matter when debugging a partial run.

**Stage level.** `results/.pipeline_checkpoint` holds `{"completed": [...]}`.
A listed stage is skipped entirely. Persisted after every successful stage, and
deleted on full success unless `--keep-checkpoint` is passed. A malformed
checkpoint is treated as "nothing completed" — safe, since it only costs a
re-run.

**Row level.** Each script rebuilds a `processed` set by re-reading its own
output CSV, then appends only missing rows and flushes after each write.

| Stage | Resume key |
|---|---|
| baseline | `(utterance_id,)` |
| axis A | `(utterance_id, noise_type, snr_db)` |
| axis B/C | `(utterance_id, condition)` |

Stage 5 is the exception — it always rewrites `feature_deviation.csv` in full.

If the resume set cannot be read completely, `load_resume_keys` raises rather
than returning a partial set. Appending on top of a half-read file would
duplicate rows the file already contained and silently double-count them in the
aggregate.

## Data storage

There is no database. Persistence is flat CSV and WAV on the local filesystem.

| File | Rows | Natural key |
|---|---|---|
| `metadata/utterance_manifest.csv` | 150 | `utterance_id` |
| `results/baseline_results.csv` | 150 | `utterance_id` |
| `results/axis_a_results.csv` | 1,800 | `(utterance_id, noise_type, snr_db)` |
| `results/axis_b_results.csv` | 150 | `(utterance_id, condition)` |
| `results/axis_c_results.csv` | 750 | `(utterance_id, condition)` |
| `results/feature_deviation.csv` | 2,700 | `(utterance_id, condition)` |

Joins are on `utterance_id` and are enforced by nothing. The "indexes" are the
in-memory `processed` sets. The only atomic write in the codebase is the
manifest rewrite in stage 1, which goes through `os.replace()`.

Schema differs by axis: `hits` and `word_count_ref` exist only in the baseline
CSV; `noise_type`/`snr_db` only in Axis A; `condition` only in B/C. Stages 5
and 6 special-case accordingly.

## Concurrency

None. The pipeline is entirely single-threaded and synchronous — no threads,
no multiprocessing, no asyncio outside the demo API.

This is deliberate, not an oversight. Three pieces of shared mutable state
define the reproducibility guarantee:

- `NoiseSource._rng_white` / `_rng_pink` — sequential draw order determines the
  noise every file receives.
- `NoiseSource._cafe_offset` — a sliding cursor into the babble buffer.
- The append-mode CSV handle — concurrent writers would interleave partial rows.

Parallelising Axis A would require per-worker seed derivation and would change
every generated file. The cost of staying serial is wall-clock: roughly 2,850
sequential Whisper inferences.

## Error handling

`main.py` fails fast at the stage boundary: non-zero exit → log → `sys.exit(1)`.
No retries. Given that a stage takes hours, the checkpoint makes this the right
trade — fix the cause and resume.

Within a stage, three failure classes are handled explicitly rather than
silently:

| Failure | Behaviour |
|---|---|
| Missing manifest / result CSV | log with a remediation hint, exit 1 |
| Unreadable resume set | raise — never resume from a partial read |
| Transcription OOM | skip the row, count it, warn at the end — never record it |
| Empty ground truth | skip before any expensive work, count it |
| Missing audio in stage 5 | count and report; abort if nothing at all was found |

The OOM case is the important one. Returning an empty transcript would write a
row with `deletions == len(reference)` and `wer == 1.0`, indistinguishable from
a genuine total recognition failure — and OOM is most likely at the highest
degradation levels, precisely where the headline results live.