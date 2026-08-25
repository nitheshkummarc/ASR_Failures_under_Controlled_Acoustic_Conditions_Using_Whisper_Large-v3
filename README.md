# ASR Failures under Controlled Acoustic Conditions

A controlled-experiment framework that measures **how and why** Whisper
`large-v3` fails, rather than reporting a single benchmark number.

150 clean LibriSpeech utterances are degraded along three orthogonal acoustic
axes — one variable at a time — so every change in Word Error Rate can be
attributed causally to a specific condition at a specific severity.

---

## Experimental design

| Axis | Condition | Parameters | Files |
|---|---|---|---|
| **A** — Additive noise | white / pink / café babble | SNR 20, 15, 10, 5 dB | 1,800 |
| **B** — Channel | telephonic bandpass | 300–3400 Hz, Butterworth order 4, zero-phase | 150 |
| **C** — Temporal | speaking-rate change | rate 1.2×, 1.5× (faster → shorter) | 300 |
| **C** — Temporal | silence insertion | 0.5 s start, 1.0 s mid, 2.0 s mid | 450 |

18 conditions × 150 utterances + 150 clean = **2,850 inference trials.**

Corpus: LibriSpeech `dev-clean`, 3–15 s clips, 16 kHz / 16-bit PCM / mono,
stratified across 40 speakers, seed 42.

## Results

Global (corpus-level) WER = `Σ(S+D+I) / Σ(reference words)`.

| Condition | Global WER |
|---|---|
| Clean baseline | **3.44 %** |
| Telephonic (300–3400 Hz) | 3.27 % — no measurable cost |
| White 20 / 15 / 10 / 5 dB | 3.53 / 3.89 / 4.25 / **6.28 %** |
| Pink 20 / 15 / 10 / 5 dB | 3.83 / 4.02 / 3.96 / **4.78 %** |
| Café 20 / 15 / 10 / 5 dB | 2.91 / 3.24 / 4.12 / **9.39 %** |
| Silence 0.5 s start / 1.0 s mid / 2.0 s mid | 4.12 / 3.40 / 3.53 % |
| Rate 1.2× / 1.5× | 5.86 / **11.16 %** |

Three findings:

1. **Café babble is the worst additive noise** at low SNR — 9.39 % versus
   6.28 % (white) and 4.78 % (pink) at the *same measured SNR*. Consistent with
   informational rather than energetic masking: the model can filter stationary
   noise, but competing speech competes for the same attention.
2. **Band-limiting is free.** Discarding everything outside 300–3400 Hz costs
   nothing, so Whisper's usable information lives inside the vocal band.
3. **The largest single degradation is temporal** — 11.16 % at rate 1.5×, on
   audio containing no added noise at all. That points at decoder cross-attention
   alignment rather than the acoustic front-end.

Feature-space deviation (MFCC / STE / ZCR MSE) shows **no significant
correlation** with WER across Axis A (all p > 0.05), reinforcing point 3: the
magnitude of spectral distortion is a poor predictor of recognition failure.

---

## Setup

Requires Python 3.11 and, for GPU inference, a CUDA-capable device.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

`large-v3` needs ~7–8 GB in float16; `config.py` forces `int8` so it fits the
target 4 GB RTX 3050. Without CUDA the engine falls back to CPU — correct, but
expect days rather than hours.

`config.py` injects the pip-installed `nvidia/cublas/bin` and `nvidia/cudnn/bin`
directories into `PATH` on import, so **`import config` must precede
`import faster_whisper`**. Every module already does this; keep it that way.

## Running

```bash
python main.py                    # run, or resume from results/.pipeline_checkpoint
python main.py --fresh            # wipe the checkpoint and start over
python main.py --keep-checkpoint  # don't clear the checkpoint on success
```

Stages, in order:

| Stage | Script | Output |
|---|---|---|
| 0 | `scripts/00_prepare_corpus.py` | `metadata/utterance_manifest.csv`, 150 WAVs |
| 1 | `scripts/01_run_baseline.py` | `results/baseline_results.csv` |
| 2 | `scripts/02_run_axis_a.py` | 1,800 WAVs, `results/axis_a_results.csv` |
| 3 | `scripts/03_run_axis_b_c.py` | 900 WAVs, `results/axis_b_results.csv`, `axis_c_results.csv` |
| 4 | `tools/verify_acoustics.py` | PASS/FAIL — asserts the degradations are what they claim |
| 5 | `scripts/04_run_feature_extraction.py` | `results/feature_deviation.csv` |
| 6 | `scripts/05_run_analysis.py` | summary charts, Pearson p-values |
| 7 | `scripts/06_generate_spectrograms.py` | per-utterance spectrograms and MFCC heatmaps |

Each stage also runs standalone. Stages 1–6 accept `--test-mode` (one utterance,
outputs redirected to `results/test_mode/` and `scratch/`). Stage 7 additionally
accepts `--utterance <id>`.

**Always run from the project root** — several scripts resolve paths relative to
the working directory.

### Tests and tools

```bash
pytest tests/                              # 52 unit tests

python tools/verify_acoustics.py           # verify one utterance in detail
python tools/verify_acoustics.py --all     # sweep the corpus (3,150 checks)
python tools/diagnostic_report.py          # per-condition WER + jiwer alignments
python tools/check_determinism.py          # confirm GPU decoding is reproducible
python tools/run_api.py                    # start the demo API on :8000
```

`verify_acoustics.py` exits non-zero on any failure, so it doubles as pipeline
stage 4.

## Layout

```
config.py        Single source of truth: paths, constants, condition registry
main.py          Stage orchestrator with checkpointing

src/             Importable library — no CLI, no side effects beyond logging
  augmentation/  The DSP: SNR mixing, noise generators, bandpass, phase vocoder
  inference/     Whisper wrapper, jiwer scoring, shared stage plumbing
  features/      Hand-written STE / ZCR / log-Mel / MFCC extraction
  analysis/      Corpus-level WER metric
  data/          Corpus builder, text normalization
  api/           FastAPI demo (not part of the experiment)

scripts/         Pipeline stages — one driver per stage, nothing else
tools/           Verification, diagnostics, and the API launcher
tests/           pytest suite
docs/            architecture.md, methodology.md, engineering-decisions.md

metadata/        utterance_manifest.csv — the pipeline's input contract
results/         Result CSVs (committed) and plots (generated)
data/            Audio (generated, gitignored)
```

## Design notes

**Reproducibility.** Seeded at 42 throughout. `NoiseSource` keeps three
independent RNG streams (white / pink / café) so the draw count of one noise
type cannot perturb another. Noise is drawn *before* the resume check so a
restarted run produces identical audio to a fresh one. Café babble is built only
from speakers disjoint from the test set, and the exact membership is recorded
in `data/noise_sources/cafe.provenance.json`.

**No librosa.** The phase vocoder and the entire Mel/DCT chain are written on
`scipy` and `numpy` directly. Partly academic — this is a signal-processing
project, so the DSP should be auditable — and partly practical: librosa depends
on `numba`, whose JIT-emitted `.pyd` binaries are blocked by AppLocker on the
target machine.

**Two WER definitions.** The `wer` column in every CSV is per-utterance. Every
*reported* figure is corpus-level (`src/analysis/metrics.py`), which weights
utterances by length. They differ by roughly half a point; do not mix them.

**Crash safety.** Two independent resumption layers: a JSON stage checkpoint,
and per-row resume where each script rebuilds a `processed` set from its own
output CSV and flushes after every write. A crash costs at most one row out of
2,850. Out-of-memory failures are *skipped and counted*, never recorded — an
empty transcript would otherwise be written as a genuine 100 % WER result.

**Verification, not assumption.** `tools/verify_acoustics.py` independently
re-derives every degradation parameter from the WAV files on disk rather than
trusting the code that produced them. Full corpus sweep: 3,150 checks, 0
failures.

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Component layout, stage graph, resumption, data storage, error handling |
| [docs/methodology.md](docs/methodology.md) | Experimental design, DSP derivations, evaluation protocol, results, limitations |
| [docs/engineering-decisions.md](docs/engineering-decisions.md) | Non-obvious decisions and the constraints that forced them |

`CLAUDE.md` holds working notes for AI-assisted development.