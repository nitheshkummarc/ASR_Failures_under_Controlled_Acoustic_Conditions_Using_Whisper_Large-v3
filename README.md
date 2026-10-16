# ASR Failures under Controlled Acoustic Conditions

A controlled-experiment framework that measures **how and why** Whisper
`large-v3` fails, rather than reporting a single benchmark number.

150 clean LibriSpeech utterances are degraded along three orthogonal acoustic
axes — one variable at a time — so every change in Word Error Rate can be
attributed causally to a specific condition at a specific severity. A fourth
noise type exists purely as an experimental control, to separate the two
mechanisms by which noise damages recognition.

---

## Experimental design

| Axis | Condition | Parameters | Files |
|---|---|---|---|
| **A** — Additive noise | white / pink / café babble / **SSN** | SNR 20, 15, 10, 5 dB | 2,400 |
| **B** — Channel | telephonic bandpass | 300–3400 Hz, Butterworth order 4, zero-phase | 150 |
| **C** — Temporal | speaking-rate change | rate 1.2×, 1.5× (faster → shorter) | 300 |
| **C** — Temporal | silence insertion | 0.5 s start, 1.0 s mid, 2.0 s mid | 450 |

22 conditions × 150 utterances + 150 clean = **3,450 inference trials.**

Corpus: LibriSpeech `dev-clean`, 3–15 s clips, 16 kHz / 16-bit PCM / mono,
stratified across 40 speakers, seed 42.

## Results

Global (corpus-level) WER = `Σ(S+D+I) / Σ(reference words)`.

| Condition | Global WER |
|---|---|
| Clean baseline | **3.01 %** |
| Telephonic (300–3400 Hz) | 3.11 % — no measurable cost |
| White 20 / 15 / 10 / 5 dB | 3.27 / 3.63 / 3.99 / **6.02 %** |
| Pink 20 / 15 / 10 / 5 dB | 3.66 / 3.76 / 3.70 / **4.35 %** |
| SSN 20 / 15 / 10 / 5 dB | 2.68 / 2.94 / 3.21 / **4.97 %** |
| Café 20 / 15 / 10 / 5 dB | 2.55 / 2.72 / 3.70 / **9.00 %** |
| Silence 0.5 s start / 1.0 s mid / 2.0 s mid | 3.70 / 3.14 / 3.11 % |
| Rate 1.2× / 1.5× | 5.76 / **11.09 %** |

These supersede an earlier set that was 0.2–0.6 pp higher everywhere: the text
normalizer turned Whisper's digits into words that did not match the reference
("9,500" → "ninefive hundred", "1707" → "one thousand seven hundred and seven"
against "seventeen o seven"). Fixed, and every stage re-run.

### 1. Babble damages recognition through two separable mechanisms

Café babble differs from white noise in two ways at once: its energy sits in the
speech bands, **and** it carries linguistic content. Comparing the two cannot
attribute the damage to either cause.

SSN (speech-shaped noise) resolves this. It is the café babble with its phase
spectrum randomized — the same long-term average spectrum, no temporal structure
and no intelligible content. Validated: `|SII(café) − SII(SSN)| = 0.0005`.

That isolates one variable at a time:

| SNR | energetic (SSN − white) | informational (café − SSN) |
|---|---|---|
| 20 dB | −0.59 | −0.13 |
| 15 dB | −0.69 | −0.23 |
| 10 dB | −0.79 | +0.49 |
| **5 dB** | **−1.05** | **+4.02** |
| **pooled** | **−0.78** `[−1.14, −0.41]` p = 0.00002 | **+1.04** `[+0.63, +1.47]` p = 0.00001 |

Both terms are significant pooled across 600 paired cells. The energetic term is
also significant on its own at 15, 10 and 5 dB; the informational term only at 5 dB.

**The energetic term is negative.** Concentrating noise energy in the speech
bands is *less* damaging than spreading it flat, at the same broadband SNR.
White noise puts 87.5 % of its energy above 1 kHz where the consonant cues live
(in-band SNR −1.05 dB at nominal 5 dB); speech-shaped energy sits under the
F0/F1 region where speech is redundant (+3.77 dB). Pink is gentler still
(+4.52 dB): about 63 % of its power lies below 50 Hz, counted in the broadband
SNR but masking nothing — which is why pink has the lowest WER at 5 dB.

**Informational masking is threshold-like.** It is indistinguishable from zero at
20, 15 and 10 dB, then accounts for +4.02 pp at 5 dB — 67 % of café's total
excess over baseline. Since café and SSN are spectrally matched and mixed at the
same measured SNR, that gap cannot be energetic.

### 2. Band-limiting is free

Discarding everything outside 300–3400 Hz costs nothing measurable (3.11 %
against a 3.01 % baseline), so the information the model relies on lives inside
the telephone band.

### 3. The largest single degradation is temporal

Rate 1.5× yields 11.09 % on audio containing no added noise whatsoever — a signal whose long-term spectrum is
almost unchanged, simply traversed faster. An audibility index predicts only
~0.6 pp of it. That points at decoder alignment, though local phase-vocoder
artefacts (smeared onsets) could contribute and this design cannot separate them.

### 4. An end-of-audio hallucination confounds the raw comparison

On 6.7 % of clean utterances Whisper appends a stock closing phrase —
overwhelmingly `"thank you"` — absent from the reference, supplying 20 of the
baseline's 25 insertions. The effect is condition-dependent, not a constant
offset: café suppresses it completely at every SNR by filling the trailing
silence, while white and pink leave it intact.

Uncorrected, that lets a mildly degraded condition post a *lower* WER than clean
audio while recognising the speech slightly worse. A hallucination-robust
secondary metric is reported alongside (never instead of) the primary WER:

| | raw | robust |
|---|---|---|
| clean baseline | 3.01 % | **2.36 %** |
| café 20 dB | 2.55 % (−0.46 below) | 2.55 % (**+0.20 above**) |

After correction no condition sits below the clean baseline.

Note this is **not** purely spectral: SSN is spectrally identical to café yet only
partially suppresses the phrase (0–4 %), so café's temporal structure matters too.

### 5. Spectral distortion is informative within a masker and misleading across maskers

The naive test — Pearson correlation pooled over all 2,400 Axis A trials — finds
almost nothing: MFCC `r = 0.046, p = 0.024` (significant, but under 0.3 % of the
variance), STE `r = 0.014, p = 0.495`, ZCR `r = 0.024, p = 0.247`. But Pearson tests a *linear* relationship, and pooling
hides a sign reversal.

Spearman on the same data is significant for two of the three
(MFCC `ρ = 0.083, p = 4.7e-05`; STE `ρ = 0.144, p = 1.4e-12`), and **within** a
fixed noise type the correlation is positive:

| Noise | MFCC r | p |
|---|---|---|
| café | **0.208** | 2.7e-07 |
| SSN | **0.129** | 0.001 |
| pink | **0.124** | 0.002 |
| white | 0.049 | 0.233 |

*Between* noise types the relationship runs the other way, and strongly:

| Condition | mean MFCC MSE | Global WER |
|---|---|---|
| **telephonic** | **831.8** | **3.11 %** (no measurable cost) |
| white 5 dB | 244.9 | 6.02 % |
| pink 5 dB | 93.7 | 4.35 % |
| SSN 5 dB | 49.7 | 4.97 % |
| **café 5 dB** | **33.5** | **9.00 %** (worst) |

Telephonic produces **25× more MFCC deviation than café** and costs nothing,
while café barely moves the MFCCs and is the most damaging additive condition.
Pooling the positive within-masker trend against this negative between-masker
trend nearly cancels both — a Simpson's paradox. So the honest claim is not "feature
deviation is uninformative"; it is that a spectral-distance feature tracks
severity *within* a masker and actively misleads *across* maskers, which is the
same conclusion the masking decomposition reaches by another route.

### Statistical note

Only 9 of 22 conditions are significantly worse than clean under a paired
bootstrap and Wilcoxon test together: white 10/5 dB, pink 15/5 dB, café 5 dB,
SSN 5 dB, both rates, and the 0.5 s start silence. Everything at 20 dB, telephonic
and both mid-utterance silence conditions have intervals straddling zero. With 3,056 reference words
the resolution limit is roughly ±0.6 pp; resolving a 0.25 pp difference would
need ~780 utterances.

---

## Setup

Requires Python 3.11 and, for GPU inference, a CUDA-capable device.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

`large-v3` needs ~7–8 GB in float16; `config.py` forces `int8` so it fits the
target 4 GB RTX 3050. `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` are required
for the GPU path and are **not** pulled in by faster-whisper — without them model
load fails outright rather than silently demoting to CPU.

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
| 2 | `scripts/02_run_axis_a.py` | 2,400 WAVs, `results/axis_a_results.csv` |
| 3 | `scripts/03_run_axis_b_c.py` | 900 WAVs, `results/axis_b_results.csv`, `axis_c_results.csv` |
| 4 | `tools/verify_acoustics.py --all` | PASS/FAIL — 3,750 checks across the corpus |
| 5 | `scripts/04_run_feature_extraction.py` | `results/feature_deviation.csv` |
| 6 | `scripts/05_run_analysis.py` | 12 summary charts, `hallucination_audit.csv`, `masking_decomposition.csv`, correlations |
| 7 | `scripts/06_generate_spectrograms.py` | per-utterance spectrograms and MFCC heatmaps |

`masking_decomposition.png` carries the headline result: the four maskers as
degradation curves beside the signed energetic/informational split per SNR.
`all_conditions_overview.png` puts all 22 conditions and the baseline on one
scale, raw against hallucination-robust. Every bar and point is labelled with
its value, since several conditions sit within 0.2 pp of the baseline.

Every axis gets the same three views, so they can be read against each other:
WER against the clean baseline, a severity curve wherever the axis has a severity
parameter, and error composition (S/I/D). Axis A sweeps SNR; Axis C sweeps
speaking rate and silence duration separately, since those are two independent
parameters rather than one ordered scale; Axis B has a single condition, so it
gets the composition plot against clean instead of a curve.

A full run is roughly 1 h 45 min on an RTX 3050.

Each stage also runs standalone. Stages 1–6 accept `--test-mode` (one utterance,
outputs redirected to `results/test_mode/` and `scratch/`). Stage 7 additionally
accepts `--utterance <id>`.

**Always run from the project root** — several scripts resolve paths relative to
the working directory.

### Tests and tools

```bash
pytest tests/                              # 93 unit tests

python tools/verify_acoustics.py           # verify one utterance in detail
python tools/verify_acoustics.py --all     # sweep the corpus (3,750 checks)
python tools/masking_analysis.py           # energetic vs informational report
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
  analysis/      Corpus WER, masking decomposition, hallucination audit
  results_io.py  Shared result-CSV reading/writing
  data/          Corpus builder, text normalization
  api/           FastAPI demo (not part of the experiment)

scripts/         Pipeline stages — one driver per stage, nothing else
tools/           Verification, diagnostics, masking report, API launcher
tests/           pytest suite
docs/            architecture.md, methodology.md, engineering-decisions.md

metadata/        utterance_manifest.csv — the pipeline's input contract
results/         Result CSVs and plots (generated, gitignored)
data/            Audio (generated, gitignored)
```

## Design notes

**Reproducibility.** Seeded at 42 throughout. `NoiseSource` keeps four
independent RNG streams (white / pink / café / SSN) so the draw count of one
noise type cannot perturb another. Noise is drawn *before* the resume check so a
restarted run produces identical audio to a fresh one. Café babble is built only
from speakers disjoint from the test set, and the exact membership is recorded in
`data/noise_sources/cafe.provenance.json`. Verified in practice: a full re-run on
different hardware with different CUDA libraries reproduced every condition
exactly.

**Noise sources are peak-normalized before writing.** The generators normalize to
unit standard deviation, which puts ~32 % of a Gaussian's samples outside
[−1, 1]; writing that as PCM_16 hard-clips every one of them. Because clipping is
phase-dependent it also breaks the spectral match between café and its SSN
control, which the entire informational-masking result depends on. See
`write_noise_source()`.

**No pandas, no seaborn, no librosa.** All three are blocked by the same
AppLocker policy on the target machine, which rejects JIT- and
extension-compiled binaries (`DLL load failed while importing timestamps` for
pandas). Every stage uses the stdlib `csv` module plus numpy, and plots are drawn
with bare matplotlib.

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
3,450. Out-of-memory failures are *skipped and counted*, never recorded — an
empty transcript would otherwise be written as a genuine 100 % WER result.

**Verification, not assumption.** `tools/verify_acoustics.py` independently
re-derives every degradation parameter from the WAV files on disk rather than
trusting the code that produced them. Full corpus sweep: 3,750 checks, 0
failures.

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Component layout, stage graph, resumption, data storage, error handling |
| [docs/methodology.md](docs/methodology.md) | Experimental design, DSP derivations, evaluation protocol, results, limitations |
| [docs/engineering-decisions.md](docs/engineering-decisions.md) | Non-obvious decisions and the constraints that forced them |

`CLAUDE.md` holds working notes for AI-assisted development.
