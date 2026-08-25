# Methodology

## Research question

Most ASR robustness studies evaluate against naturally noisy corpora, where
background noise, reverberation and channel effects co-occur. That yields an
aggregate robustness score but cannot attribute failure to any single factor.

This study applies **precisely one degradation at a time** to a fixed corpus of
clean speech, so that every change in Word Error Rate is causally attributable
to a specific condition at a specific severity.

## Corpus

| Property | Value |
|---|---|
| Source | LibriSpeech `dev-clean` |
| Utterances | 150 |
| Duration filter | 3.0–15.0 s |
| Format | 16 kHz, 16-bit PCM, mono |
| Sampling | Stratified by speaker |
| Speakers | 40 |
| Seed | 42 |

**Stratified selection.** Utterances are grouped by `speaker_id`; each of the 40
speakers contributes a base of `150 // 40 = 3`, and the remaining 30 slots go to
randomly ranked speakers. Speaker keys are **sorted before permuting**, so the
selection reproduces across machines regardless of filesystem iteration order.
If a speaker has too few eligible utterances the shortfall is backfilled from
the global pool.

## Degradation axes

18 conditions per utterance, plus the clean baseline: **2,850 inference trials**.

### Axis A — Additive noise (parameter: SNR)

Three noise types at four SNR levels: 20, 15, 10, 5 dB → 1,800 files.

The mixing is written by hand rather than delegated to a library, so the DSP is
auditable:

```
RMS(x)            = sqrt(mean(x²))
target_rms_noise  = RMS(clean) / 10^(SNR_dB / 20)
k                 = target_rms_noise / RMS(noise)
degraded          = clean + k · noise
if max|degraded| > 1:  degraded /= max|degraded|
```

The factor of 20 (not 10) is used because RMS is an *amplitude* quantity and
power ∝ amplitude².

The anti-clipping step divides the **entire mixture**, scaling signal and noise
identically, so the SNR ratio is preserved exactly while avoiding 16-bit PCM
wraparound. Both silent-signal cases (`RMS < 1e-10` for either input) return the
clean signal unchanged, because SNR is undefined against silence and nothing
should be fabricated.

**Noise types:**

- **White** — samples drawn i.i.d. from 𝒩(0,1). Flat power spectral density.
- **Pink** — white noise → `rfft` → multiplied by a `1/√f` amplitude filter →
  `irfft` → normalised. `1/√f` in amplitude gives PSD ∝ `1/f` in power. The DC
  bin is handled by setting `freqs[0] = 1.0` to avoid the division, then zeroing
  that filter tap.
- **Café babble** — 15 speakers normalised to unit RMS, tiled to length, placed
  at random offsets and summed. With 15 uncorrelated talkers no individual word
  survives; only speech *texture* remains, which is what makes it an
  **informational** rather than energetic masker.

Café speakers are drawn from a pool **strictly disjoint** from the 150-utterance
test set, so no utterance can be masked by babble containing itself. The exact
membership is recorded in `data/noise_sources/cafe.provenance.json`.

### Axis B — Channel distortion (parameter: bandwidth)

A 4th-order Butterworth bandpass limited to **300–3400 Hz**, the ITU-T G.712
narrowband telephone passband.

Applied with `filtfilt` (forward then backward), giving **zero phase
distortion**. This matters experimentally: a phase shift would become an
uncontrolled second variable alongside the intended band-limiting. The cost is
that the effective magnitude response is that of an 8th-order filter.

### Axis C — Temporal manipulation (parameter: rate, duration)

**Time-scale modification** at rates 1.2× and 1.5× via a hand-written phase
vocoder.

Rate semantics follow the standard convention: **rate > 1 means the speech is
delivered faster**, so the output is *shorter* by that factor. A 9.37 s
utterance yields 6.24 s at rate 1.5×. This matches the proposal's parameter
("speaking rate: 1× / 1.2× / 1.5×"), where the quantity is the rate, not a
duration multiplier.

The algorithm resamples the STFT frame grid, linearly interpolates magnitudes,
and reconstructs phase by estimating each bin's instantaneous frequency from the
inter-frame phase difference (minus the expected advance, wrapped to [−π, π])
and integrating it across output frames. Naively reusing the original phases
produces the classic "phasiness" artifact. Pitch is preserved because bin centre
frequencies are never altered — only the time axis is resampled.

**Silence insertion** of 0.5 s at the start, 1.0 s at the midpoint, and 2.0 s at
the midpoint. Exact digital zeros. The midpoint split is made at the *sample*
midpoint rather than a word boundary, so the gap generally falls mid-word —
intentionally the worst case for decoder alignment.

## Inference configuration

Whisper `large-v3` via faster-whisper (CTranslate2 backend).

| Parameter | Value | Rationale |
|---|---|---|
| `compute_type` | `int8` | Forced, not preferred — large-v3 needs ~7–8 GB in fp16 and the target GPU has 4 GB |
| `beam_size` | 5 | Matches the published Whisper setup |
| `language` | `"en"` | Removes language identification as a confound under noise |
| `condition_on_previous_text` | `False` | **Experimental control.** With it enabled, a hallucination in one segment feeds the next segment's prompt and cascades. Disabling it isolates per-segment failure. |

## Evaluation

**Normalization** is applied identically to reference and hypothesis before any
metric is computed: lowercase → strip ordinal suffixes → numbers to words →
strip punctuation (hyphen → space first) → collapse whitespace. Numbers are
expanded so `"1929"` and `"nineteen twenty nine"` are not counted as errors;
decimals are spoken digit-by-digit (`"3.14"` → `"three point one four"`), which
is how a speaker actually reads them.

**Two WER definitions coexist** and must not be mixed:

- *Per-utterance WER* — the `wer` column in every result CSV, straight from
  jiwer. Averaging it weights a 5-word utterance the same as a 40-word one.
- *Global (corpus) WER* — `Σ(S+D+I) / Σ(reference words) × 100`. Utterances are
  weighted by length. **This is the reported metric**, implemented once in
  `src/analysis/metrics.py`.

On this corpus the two differ by roughly half a percentage point;
`tools/diagnostic_report.py` prints both side by side.

Storing raw substitution / insertion / deletion counts — not just WER — is what
makes corpus-level aggregation and the error-composition analysis possible.

## Feature-space analysis

For every clean/degraded pair, MFCC, short-time energy and zero-crossing rate
are extracted and the MSE between them recorded, to test whether the *magnitude*
of acoustic distortion predicts recognition failure.

The MFCC chain: pre-emphasis (`y[n] − 0.97·y[n−1]`) → Hann-windowed STFT →
power spectrum → 40 triangular Mel filters → log → DCT-II. The output is sliced
`[1:14]`, so **C0 (frame energy) is dropped** and the coefficients are c1–c13.

**The two deviation branches are not on the same scale:**

- Non-stretch conditions compare the **mean feature vectors** (one time-average
  per signal), since both signals are the same length.
- Stretch conditions compare full per-frame **sequences** after resampling the
  degraded one onto the clean frame grid, because time-averaging would hide
  exactly the temporal distortion being measured.

Time-averaging cancels sign-alternating deviations, so the first branch yields
systematically smaller numbers for equivalent distortion. The statistical test
uses only Axis A rows, which are all first-branch, so it is internally
consistent — but `mfcc_mse` must not be compared across axes.

## Results

Global WER:

| Condition | WER |
|---|---|
| Clean baseline | **3.44 %** |
| Telephonic (300–3400 Hz) | 3.27 % |
| White 20 / 15 / 10 / 5 dB | 3.53 / 3.89 / 4.25 / **6.28 %** |
| Pink 20 / 15 / 10 / 5 dB | 3.83 / 4.02 / 3.96 / **4.78 %** |
| Café 20 / 15 / 10 / 5 dB | 2.91 / 3.24 / 4.12 / **9.39 %** |
| Silence 0.5 s start / 1.0 s mid / 2.0 s mid | 4.12 / 3.40 / 3.53 % |
| Rate 1.2× / 1.5× | 5.86 / **11.16 %** |

**1. Café babble is the worst additive noise.** At 5 dB it reaches 9.39 %,
versus 6.28 % (white) and 4.78 % (pink) at the *same measured SNR*. Since SNR is
held constant, the difference cannot be energetic — it is consistent with
informational masking, where competing linguistic content contends for the same
attention the target speech needs. Note also that café at 20 dB (2.91 %) sits
slightly *below* the clean baseline, so the effect is strongly non-linear in SNR.

**2. Band-limiting is free.** Discarding everything outside 300–3400 Hz costs
nothing measurable (3.27 % versus a 3.44 % baseline), so the information the
model relies on lives inside the telephone band.

**3. The largest single degradation is temporal.** Rate 1.5× yields 11.16 % on
audio containing no added noise whatsoever — a spectrally clean signal, simply
traversed faster. This points at decoder cross-attention alignment rather than
the acoustic front end.

**4. Café at high SNR scores BELOW the clean baseline — and it is a
hallucination artifact, not better recognition.** Café at 20 dB (2.91 %) and
15 dB (3.24 %) both beat the clean baseline (3.44 %). Decomposing the error
counts shows the cause is not improved accuracy:

| baseline vs café 20 dB | subs | ins | dels | total |
|---|---|---|---|---|
| clean baseline | 65 | **32** | 8 | 105 |
| café 20 dB | 70 | **8** | 11 | 89 |
| delta | +5 | **−24** | +3 | −16 |

Substitutions and deletions both *rise*; the entire gain comes from insertions
collapsing. Inspecting those rows identifies the mechanism precisely: on clean
audio Whisper appends a stock trailing phrase — overwhelmingly `"thank you"` —
that is absent from the reference. It is the classic Whisper end-of-audio
hallucination, where trailing low-energy audio lets the language-model prior
take over before the EOS token.

Counting utterances whose transcript ends in such a phrase absent from the
reference:

| condition | rate |
|---|---|
| clean baseline | 8.0 % (12/150) |
| white 20/15/10/5 dB | 7.3 / 8.7 / 8.0 / 4.0 % |
| pink 20/15/10/5 dB | 12.0 / 13.3 / 8.7 / 8.0 % |
| telephonic | 8.0 % |
| silence start / mid | 12.0 / 6.7–7.3 % |
| rate 1.2× / 1.5× | 2.0 / 0.7 % |
| **café — all four SNRs** | **0.0 % (0/150)** |

Café babble is the only condition that suppresses the effect completely, at
every SNR. Its non-stationary, speech-shaped energy leaves no trailing
low-energy region for the decoder to fill from its prior. Stationary white and
pink noise do not have this effect, so it is not simply "any added energy".

Re-scoring with the trailing phrase stripped removes the anomaly entirely:

|  | as measured | hallucination-free |
|---|---|---|
| clean baseline | 3.44 % | 2.68 % |
| café 20 dB | 2.91 % | 2.91 % |
| verdict | café beats clean by 0.52 pp | café is **worse** by 0.23 pp |

A paired Wilcoxon signed-rank test over the 150 utterances confirms the
direction: as measured, café 20 dB is significantly better than clean
(p = 0.045); with hallucinations removed the advantage disappears
(p = 0.244, café worse on 12 utterances and better on 6).

So the correct claim is **not** "mild café babble improves recognition". It is
that mild café babble *suppresses end-of-audio hallucination*, and because WER
charges those hallucinated words as insertions, the suppression shows up as an
apparent accuracy gain. Underlying recognition is marginally worse, as expected.

**5. Feature deviation does not predict WER.** Across all 1,800 Axis A trials:

| Feature | Pearson r | p |
|---|---|---|
| MFCC MSE | 0.013 | 0.595 |
| STE MSE | 0.007 | 0.782 |
| ZCR MSE | 0.019 | 0.421 |

None significant at α = 0.05. White noise at 5 dB produces enormous MFCC
deviation but only moderate WER, while time-scaled audio produces near-zero
spectral deviation and the worst WER in the study. Together with finding 3, this
argues the dominant failure mode is architectural rather than acoustic.

## Verification

The degradations are not assumed correct — `tools/verify_acoustics.py`
independently re-derives each parameter from the WAV files on disk and exits
non-zero on any failure. It runs as stage 4 of the pipeline.

| Check | Method | Tolerance |
|---|---|---|
| Axis A SNR | Recover the noise by least-squares projection, then measure | ±0.5 dB |
| Axis B bandwidth | Out-of-band energy retained after filtering | < 25 % |
| Axis C rate | Output duration versus `input / rate` | ±0.1 s |
| Axis C silence | Total length, plus longest run of true zeros | ±0.05 s |

`--all` sweeps the whole corpus: **3,150 checks, 0 failures.**

Two subtleties the verifier has to account for:

- **SNR measurement must not assume the clean component is unscaled.** When the
  anti-clip path fires, the degraded file contains `α·clean` for some α < 1.
  Plain subtraction leaves an `(α−1)·clean` residual in the "noise" term and
  inflates the result — this mis-reported 14 correctly-mixed files as failures
  before the projection-based estimate was adopted.
- **Silence runs may exceed the inserted duration.** Some recordings open with
  up to ~0.4 s of their own near-silence, which merges with a start-position
  insertion. The check asserts *at least* the target duration; the exact total
  length is asserted separately.

## Reproducibility

A complete from-scratch rebuild was performed on 2026-08-21: the corpus archive,
the 150-utterance selection, the cafe babble, all 2,700 degraded files and all
2,850 transcriptions were regenerated from nothing but source code.

| Artifact | Result |
|---|---|
| Utterance selection | 150/150 identical IDs |
| Clean corpus audio | all 150 bit-identical |
| White noise mixtures | all 600 bit-identical |
| Pink noise mixtures | all 600 bit-identical |
| Telephonic / stretch / silence | all 750 bit-identical |
| Cafe mixtures | all 600 differ — by design (see below) |
| Transcriptions | 2,849 / 2,850 identical |

**The cafe source was corrected.** The previously committed `cafe.wav` was shown,
by exact md5 match against a deliberate reproduction of the removed code path, to
have been synthesised from the 150 TEST utterances rather than the disjoint pool.
Fifteen test utterances were therefore masked by babble containing themselves.
All cafe noise now comes from the 2,553-utterance disjoint pool. The corrected
figures are 0.1-0.4 points lower at 10 and 5 dB; the ranking is unchanged.

**Decoding is not bit-deterministic.** One row out of 2,850 (0.035 %) produced a
different transcript from bit-identical audio — utterance `6313-66129-0005` under
rate 1.5x. The old decode degenerated into character-level spew (23 errors
against an 11-word reference, 209 % WER); the new one stayed coherent (8 errors).
That single row moved the rate-1.5x figure by 0.49 points, from 11.65 % to
11.16 %.

This is worth stating plainly: **at the failure boundary, beam search is
metastable.** Non-deterministic GPU reduction order is enough to flip a decode
between a contained error and a runaway hallucination. Single-run WER for the
most severe conditions therefore carries roughly +/-0.5 points of run-to-run
uncertainty, and conclusions there should rest on the ranking rather than the
exact value. Milder conditions were perfectly stable across the rebuild.
`tools/check_determinism.py` probes this directly.

## Limitations

- Conditions are applied independently. Compound degradations (noise *and*
  band-limiting) are not measured, and the axes may not combine additively.
- 150 utterances from a single read-speech corpus. Findings may not transfer to
  spontaneous or accented speech.
- Café babble is synthetic (LibriSpeech speakers overlaid), not recorded
  ambience. It lacks room acoustics and non-speech restaurant sounds.
- `int8` quantization raises the clean baseline slightly (3.44 % measured versus
  ~2.7 % published for the unquantized model), so absolute WER figures are
  specific to this configuration; the relative comparisons are the result.
- Feature deviation is measured only as MSE of MFCC/STE/ZCR. A null correlation
  rules out those specific measures, not all acoustic predictors.