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

22 conditions per utterance, plus the clean baseline: **3,450 inference trials**.

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
  survives; only speech *texture* remains. Whether that texture actually acts as
  an informational masker is not assumed here — it is measured, against the SSN
  control below.
- **SSN (speech-shaped noise)** — the café babble with its phase spectrum
  replaced by uniform draws on [0, 2π) and inverse-transformed. This preserves
  the magnitude spectrum exactly while destroying the temporal envelope and any
  intelligible content. It exists solely as an experimental control; see
  *Separating the two masking mechanisms* below.

Café speakers are drawn from a pool **strictly disjoint** from the 150-utterance
test set, so no utterance can be masked by babble containing itself. The exact
membership is recorded in `data/noise_sources/cafe.provenance.json`.

**Noise sources are peak-normalised before being written to disk.** The
generators normalise to unit standard deviation, which places roughly a third of
a Gaussian's samples outside [−1, 1]; writing that as PCM_16 hard-clips every one
of them. A previously cached `cafe.wav` had 29.9 % of its samples pinned to the
rails and a crest factor of 1.42, against 3–5 for natural babble. Because
clipping is a phase-dependent nonlinearity it distorted the babble and its
phase-randomised control *differently*, breaking the spectral match the
informational-masking estimate depends on. Only the waveform shape needs to
survive the round trip — every consumer re-normalises on load, and
`mix_at_target_snr` rescales to the target SNR — so the file is written at 0.95
peak.

### Separating the two masking mechanisms

Noise damages recognition in two distinct ways, and a babble masker exhibits
both at once:

- **Energetic masking** is peripheral. Noise energy overlaps the speech energy in
  frequency and swamps it. It is predictable from the spectra alone.
- **Informational masking** is central. The masker is audible and separable from
  the target but still interferes, because it competes for the same interpretive
  machinery. It is *not* predictable from the spectra.

Comparing café against white noise cannot distinguish them, because the two
differ in spectrum **and** in linguistic content simultaneously. Adding SSN —
spectrally identical to café, structurally inert — isolates one variable per
comparison:

```
ssn  - white   ->  energetic     (energy moved into the speech bands)
cafe - ssn     ->  informational (same spectrum, linguistic content added)
```

The decomposition is only valid if café and SSN really are spectrally matched.
That is asserted rather than assumed: `tools/masking_analysis.py` computes a
speech-weighted audibility index (a simplified SII, ANSI S3.5-1997 octave-band
importance weights, ±15 dB dynamic range) for both and requires them to agree.
Measured: `max |SII(café) − SII(SSN)| = 0.0005`.

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
| Clean baseline | **3.01 %** |
| Telephonic (300–3400 Hz) | 3.11 % |
| White 20 / 15 / 10 / 5 dB | 3.27 / 3.63 / 3.99 / **6.02 %** |
| Pink 20 / 15 / 10 / 5 dB | 3.66 / 3.76 / 3.70 / **4.35 %** |
| SSN 20 / 15 / 10 / 5 dB | 2.68 / 2.94 / 3.21 / **4.97 %** |
| Café 20 / 15 / 10 / 5 dB | 2.55 / 2.72 / 3.70 / **9.00 %** |
| Silence 0.5 s start / 1.0 s mid / 2.0 s mid | 3.70 / 3.14 / 3.11 % |
| Rate 1.2× / 1.5× | 5.76 / **11.09 %** |

An earlier run reported every figure 0.2–0.6 pp higher. The text normalizer
mis-converted Whisper's digits — "9,500" became "ninefive hundred", and "1707"
became "one thousand seven hundred and seven" against a reference of "seventeen o
seven" — so formatting was being scored as recognition error. The normalizer now
strips thousands separators, reads 1100–1999 as years the way LibriSpeech writes
them, and converts ordinals to ordinal words; the whole pipeline was re-run.
Every stage now also stores Whisper's raw output, so a future normalization
change can be re-scored without re-running inference.

Significance is assessed by a paired bootstrap over utterances (4,000 resamples)
alongside a Wilcoxon signed-rank test. Only **9 of the 22 conditions** are
significantly worse than clean: white 10/5 dB, pink 15/5 dB, café 5 dB, SSN 5 dB,
silence-at-start, and both rate conditions. Everything at 20 dB, telephonic, and
both mid-utterance silence conditions have intervals straddling zero.

**1. Babble damages recognition through two separable mechanisms.** At 5 dB café
reaches 9.00 %, versus 6.02 % (white) and 4.35 % (pink) at the same measured SNR.
The naive reading — "SNR is held constant, so the difference cannot be
energetic" — does not hold, because café and white differ in spectrum as well as
content. The SSN control settles it:

| SNR | energetic (SSN − white) | informational (café − SSN) |
|---|---|---|
| 20 dB | −0.59 | −0.13 |
| 15 dB | −0.69 | −0.23 |
| 10 dB | −0.79 | +0.49 |
| **5 dB** | **−1.05** | **+4.02** |
| **pooled (600 cells)** | **−0.78** `[−1.14, −0.41]`, p = 0.00002 | **+1.04** `[+0.63, +1.47]`, p = 0.00001 |

The energetic term is significant on its own at 15, 10 and 5 dB (20 dB is
disputed: bootstrap yes, Wilcoxon p = 0.07); the informational term only at 5 dB.
`tools/masking_contrasts.py` extends this to every noise as the reference.

Two results follow.

*The energetic term is negative.* Concentrating noise energy in the speech bands
is **less** damaging than spreading it flat at the same broadband SNR. The
spectra explain why: the experiment equalises broadband RMS, but recognition
depends on the SNR where the phonetic information sits. At a nominal 5 dB the
in-band SNR in the 1–4 kHz consonant region is −1.05 dB for white, +3.77 dB for
SSN and +4.52 dB for pink — white dumps 87.5 % of its energy above 1 kHz, where
speech carries only 11.2 % of its own. Pink is gentlest because about 63 % of its
power lies below 50 Hz: it counts toward the broadband SNR but masks nothing,
and pink has the lowest WER of the four at 5 dB. (An earlier version reported
pink at +1.63 dB: the band analysis normalised spectra by their Welch total,
which cannot see that infrasonic power and spread it into the speech bands.)

*Informational masking is threshold-like.* It is statistically indistinguishable
from zero at 20, 15 and 10 dB, then accounts for +4.02 pp at 5 dB — about 67 % of
café's total excess over baseline, with 65 of 150 utterances worse against 11
better (p < 0.00001). Since café and SSN share a magnitude spectrum to within
`|ΔSII| = 0.0005` and are mixed at the same verified SNR, that gap cannot be
energetic.

A caveat on the label: SSN is stationary by construction while babble retains
some syllabic-rate fluctuation (envelope CV 0.246 against 0.113, with 1.2 % of
frames more than 6 dB below the median against 0.0 %). The `café − SSN` term is
therefore strictly *informational plus modulation* masking. Since 15 overlaid
talkers fill in most of the dips — real speech reaches CV 1.069 and 37.7 % deep
dips — the modulation contribution is small, but it is not zero and should not
be described as pure informational masking.

**2. Band-limiting is free.** Discarding everything outside 300–3400 Hz costs
nothing measurable (3.11 % versus a 3.01 % baseline), so the information the
model relies on lives inside the telephone band.

**3. The largest single degradation is temporal.** Rate 1.5× yields 11.09 % on
audio containing no added noise whatsoever — a signal whose long-term spectrum is
almost unchanged, simply traversed faster. An audibility index predicts only
~0.6 pp of it. That points at decoder alignment, though local phase-vocoder
artefacts (smeared onsets) could contribute and this design cannot separate them.

**4. Café at high SNR scores BELOW the clean baseline — and it is a
hallucination artifact, not better recognition.** Café at 20 dB (2.55 %) and
15 dB (2.72 %) both beat the clean baseline (3.01 %). Decomposing the error
counts shows the cause is not improved accuracy:

| baseline vs café 20 dB | subs | ins | dels | total |
|---|---|---|---|---|
| clean baseline | 61 | **25** | 6 | 92 |
| café 20 dB | 68 | **1** | 9 | 78 |
| delta | +7 | **−24** | +3 | −14 |

Substitutions and deletions both *rise*; the entire gain comes from insertions
collapsing. Inspecting those rows identifies the mechanism precisely: on clean
audio Whisper appends a stock trailing phrase — overwhelmingly `"thank you"` —
that is absent from the reference. It is the classic Whisper end-of-audio
hallucination, where trailing low-energy audio lets the language-model prior
take over before the EOS token.

Counting utterances whose transcript contains such a phrase as inserted words
(absent from the reference):

| condition | rate |
|---|---|
| clean baseline | 6.7 % (10/150) |
| white 20/15/10/5 dB | 7.3 / 8.0 / 7.3 / 4.0 % |
| pink 20/15/10/5 dB | 10.7 / 12.7 / 8.0 / 7.3 % |
| **SSN 20/15/10/5 dB** | **3.3 / 4.0 / 1.3 / 0.0 %** |
| telephonic | 7.3 % |
| silence start / mid | 11.3 / 6.7–7.3 % |
| rate 1.2× / 1.5× | 2.0 / 0.0 % |
| **café — all four SNRs** | **0.0 % (0/150)** |

The phrases are not typed in by hand. Following the "Bag of Hallucinations"
method (Barański et al., ICASSP 2025, arXiv:2501.11378), they are learned from
Whisper itself: `tools/build_hallucination_bag.py` feeds the same Whisper
configuration 150 clips that contain no speech (digital silence, and white, pink
and speech-shaped noise at −60 to −20 dBFS, 1–30 s), so anything it transcribes is
a hallucination by construction. Outputs that recur and are improbable as ordinary
English (log10 unigram probability ≤ −4 under the LibriSpeech transcripts disjoint
from the test set, standing in for the paper's n-gram LM) form the bag. On this
setup Whisper produced text for all 150 clips: `"thank you"` 141 times and
`"thanks for watching"` 7 times — the same top two the paper reports — and those
two are the bag (`metadata/hallucination_bag.json`, every probe's output included).

Detection is deliberately conservative: only words that jiwer aligns as
**insertions** (no reference counterpart) can be removed, and within an insertion
run only a bag phrase or a repetition loop of two or more words. A word that
matched or substituted a reference word is never touched, so a genuine
mis-transcription keeps its errors. The bag comes only from non-speech audio,
never from the results, with its parameters fixed in `config.py` before it was
built, so the correction cannot quietly grow to fit a desired result; it is
applied identically to every condition including clean. Using the alignment
rather than "the words after the last reference word" also fixed two misses of
the earlier hand-written rule — a reference ending in "you" (the anchor landed
inside "thank you"), and a transcript with deletions elsewhere — which is why
white 20 dB, pink 20 dB and telephonic each gained one detected utterance.

Café is the only condition that suppresses the effect completely at every SNR.
The SSN control shows the cause is *partly but not wholly* spectral: SSN shares
café's magnitude spectrum exactly and suppresses the phrase substantially
(3.3–0.0 % against white's 4.0–8.0 %), but only café reaches zero everywhere.
Filling the trailing silence with speech-shaped energy accounts for most of the
effect; café's residual temporal structure accounts for the rest. Flat white and
pink noise barely help at all, so it is not simply "any added energy".

Re-scoring with the trailing phrase stripped removes the anomaly entirely:

|  | as measured | hallucination-robust |
|---|---|---|
| clean baseline | 3.01 % | 2.36 % |
| café 20 dB | 2.55 % | 2.55 % |
| verdict | café beats clean by 0.46 pp | café is **worse** by 0.20 pp |

The full per-condition audit is written to `results/hallucination_audit.csv` by
stage 6 and plotted as `results/plots/summary/hallucination_analysis.png`. The
robust figure is a **secondary** metric: `global_wer` remains the reported
number, because deleting words from a hypothesis before scoring is not standard
WER and must not masquerade as it.

A paired bootstrap over the 150 utterances shows the apparent advantage was
never significant to begin with: café 20 dB against clean is −0.46 pp with a 95 %
interval of [−1.06, +0.13] (Wilcoxon p = 0.10). After the hallucination
correction the sign reverses and café sits +0.20 pp above clean. No condition
remains below the corrected baseline; the closest are telephonic (+0.03) and
1.0 s mid-utterance silence (+0.07).

So the correct claim is **not** "mild café babble improves recognition". It is
that mild café babble *suppresses end-of-audio hallucination*, and because WER
charges those hallucinated words as insertions, the suppression shows up as an
apparent accuracy gain. Underlying recognition is marginally worse, as expected.

**5. Spectral distortion predicts WER within a masker and inverts across
maskers.** Pooled over all 2,400 Axis A trials, Pearson finds almost nothing:

| Feature | Pearson r | p | Spearman ρ | p |
|---|---|---|---|---|
| MFCC MSE | 0.046 | 0.024 | 0.083 | **4.7e-05** |
| STE MSE | 0.014 | 0.495 | 0.144 | **1.4e-12** |
| ZCR MSE | 0.024 | 0.247 | 0.004 | 0.843 |

(MFCC's Pearson p = 0.024 is nominally significant, but r = 0.046 explains under
0.3 % of the variance.)

Two of the three are significant under Spearman, so the relationship exists but
is not linear — reporting Pearson alone (as earlier versions of this document
did) understates it. Within a fixed noise type the correlation is positive and
significant for café (r = 0.208, p = 2.7e-07), SSN (r = 0.129, p = 0.001) and
pink (r = 0.124, p = 0.002), and absent for white (r = 0.049, p = 0.23).

Between noise types it runs the other way:

| Condition | mean MFCC MSE | Global WER |
|---|---|---|
| telephonic | **831.8** | 3.11 % |
| white 5 dB | 244.9 | 6.02 % |
| pink 5 dB | 93.7 | 4.35 % |
| SSN 5 dB | 49.7 | 4.97 % |
| café 5 dB | **33.5** | 9.00 % |

Telephonic generates the largest feature deviation in the entire study — 25×
café's — at no measurable cost in WER, while café generates the smallest and is
the most damaging additive condition. The positive within-masker trend and the
negative between-masker trend nearly cancel when pooled, which is why the naive
test reads almost zero: a Simpson's paradox rather than a null result.

The implication is the same one finding 1 reaches from the masking side and
finding 3 from the temporal side: what determines recognition failure is not how
far the spectrum moved. A spectral-distance feature can rank severity inside a
single masker, but comparing that distance across different kinds of corruption
is meaningless.

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

`--all` sweeps the whole corpus: **3,750 checks, 0 failures.** Stage 4 runs it
with `--all`; without that flag it would check only the single showcase
utterance — 25 checks, 0.67 % of the generated audio — and a generation fault
in any other utterance would pass unnoticed.

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

A complete from-scratch rebuild was performed on 2026-08-21, and the experiment
was re-run end to end on 2026-09-25 after the SSN control was added — on
different hardware, a rebuilt virtual environment and different CUDA library
versions.

| Artifact | 2026-09-25 re-run result |
|---|---|
| Utterance selection | 150/150 identical IDs |
| Clean corpus audio | all 150 bit-identical |
| Baseline / white / pink WER | identical to 4 decimal places |
| Telephonic / silence / rate 1.2x WER | identical to 4 decimal places |
| Cafe WER | identical once the clipping defect was fixed (see below) |

Reproducing every condition exactly across a hardware and library change is a
stronger reproducibility result than the seeded-RNG claim alone, because it also
exercises the decoder.

**The cafe source was corrected (2026-08-21).** The previously committed
`cafe.wav` was shown, by exact md5 match against a deliberate reproduction of the
removed code path, to have been synthesised from the 150 TEST utterances rather
than the disjoint pool. Fifteen test utterances were therefore masked by babble
containing themselves. All cafe noise now comes from the 2,553-utterance disjoint
pool.

**A clipping defect was found and fixed (2026-09-25).** The cached `cafe.wav` was
written at unit standard deviation as PCM_16, clipping 29.9 % of its samples. The
defect was latent in the original run, which used the freshly generated
in-memory buffer; it only manifested on a later run that LOADED the cached file,
which shifted every cafe figure by up to 0.4 pp. Regenerating with peak
normalisation restored the original numbers exactly (2.91 / 3.24 / 4.12 / 9.39,
as scored before the 2026-10-11 normalizer fix),
which is what identifies the cache round-trip rather than the generator as the
culprit. The same defect affected the SSN control and broke its spectral match to
cafe (`|ΔSII|` 0.0172, now 0.0005).

**Decoding is not bit-deterministic.** One row out of 2,850 (0.035 %) produced a
different transcript from bit-identical audio — utterance `6313-66129-0005` under
rate 1.5x. The old decode degenerated into character-level spew (23 errors
against an 11-word reference, 209 % WER); the new one stayed coherent (8 errors).
That single row moved the rate-1.5x figure by 0.49 points, from 11.65 % to
11.16 % in that pair of runs. (The 2026-09-25 re-run settled at 11.19 %, a
further 0.03 from the same instability.)

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
- `int8` quantization raises the clean baseline slightly (3.01 % measured versus
  ~2.7 % published for the unquantized model), so absolute WER figures are
  specific to this configuration; the relative comparisons are the result.
- Feature deviation is measured only as MSE of MFCC/STE/ZCR. A null correlation
  rules out those specific measures, not all acoustic predictors.
- **Statistical resolution is ~±0.6 pp.** With 3,056 reference words, only 9 of
  22 conditions separate from clean at 95 % confidence. Differences below about
  half a point — including every 20 dB condition and the telephonic result —
  cannot be resolved by this corpus. Resolving 0.25 pp would need roughly 780
  utterances.
- **The `café − SSN` term is informational *plus* modulation masking.** SSN is
  stationary while babble retains some syllabic fluctuation. The modulation
  contribution is small (envelope CV 0.246 against 0.113, with no dips deeper
  than 12 dB in either) but non-zero, so the term is an upper bound on the purely
  informational component.
- The energetic/informational split is measured on one masker at one talker
  count. Babble becomes a more effective informational masker at *low* talker
  counts, where individual words remain intelligible; 15 talkers sits near the
  stationary end of that range.
- The speech-weighted audibility index is a simplified SII: it omits upward
  spread of masking and the absolute hearing threshold, both of which model a
  human listener rather than a neural network. It is used here as a
  spectrum-only predictor and a matched-spectrum validity check, not as an
  intelligibility prediction.