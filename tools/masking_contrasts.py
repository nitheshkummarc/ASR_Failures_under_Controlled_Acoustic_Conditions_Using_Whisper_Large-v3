"""
Energetic vs informational masking against EVERY noise, not only white.

tools/masking_analysis.py and stage 6 decompose the cafe damage with white as
the single reference. This report widens that in two ways.

1. PAIRWISE CONTRASTS (behavioural, model-free)
   Every ordered pair of maskers, at every SNR, as a paired WER difference with
   a bootstrap CI and a Wilcoxon p (src/analysis/masking.paired_significance):

       contrast(A, B) = 100 * sum_u (e_A,u - e_B,u) / sum_u N_u

   What a contrast MEANS depends on what differs between the two maskers:

       pink - white, ssn - white, ssn - pink   ENERGETIC: all three are stationary
                                               and content-free, only the spectral
                                               shape differs
       cafe - ssn                              INFORMATIONAL (+ modulation): same
                                               magnitude spectrum, content added
       cafe - white, cafe - pink               MIXED. Split exactly by the identity
           cafe - X = (cafe - ssn) + (ssn - X)
                    = informational + energetic(relative to X)
       which holds exactly because every cell is scored on the same utterances.

2. REFERENCE-FREE DECOMPOSITION (acoustic model + behaviour)
   The standard approach in the hearing literature: predict the energetic part
   from an acoustic audibility metric, and call whatever the metric cannot
   explain non-energetic. Two metrics, measured on the actual degraded files:

     SII  -- simplified Speech Intelligibility Index (ANSI S3.5), from long-term
             spectra. Blind to timing, so it cannot see dips in a masker.
     GP   -- glimpse proportion (Cooke 2006): % of time-frequency cells whose
             local SNR exceeds a threshold. Sees spectro-temporal dips, so it
             credits a fluctuating masker with its glimpsing opportunity.

   A line WER = a + b * metric is fitted ONLY to the stationary, content-free
   maskers (white, pink, ssn), which by construction carry no informational
   masking. Then, for every masker and SNR:

       predicted     = a + b * metric
       energetic     = predicted - WER(clean)
       non-energetic = WER(observed) - predicted

   For the stationary maskers the non-energetic term is just the fit residual,
   which is the model's error band. For cafe it is the informational estimate.
   SII and GP disagreeing about cafe measures how much of the cafe - ssn term
   could be glimpsing (modulation) rather than content.

3. EVERY NOISE AS THE REFERENCE, BOTH TERMS
   For each reference R and every other noise X, WER(X) - WER(R) is split into
   an energetic and an informational part, by two independent methods.

   (a) Control subtraction (literature definition, model-free). ctrl(X) is X's
       phase-randomised, spectrally identical control: ssn for cafe, and X
       itself for the stationary noises (randomising the phase of a steady
       noise changes nothing). Controlling BOTH sides, exactly:

           WER(X) - WER(R) = [WER(ctrl X) - WER(ctrl R)]
                           + [(WER(X) - WER(ctrl X)) - (WER(R) - WER(ctrl R))]
                              energetic first, informational second

       Informational is zero BY CONSTRUCTION between two stationary noises (no
       content on either side), and energetic is zero by construction for cafe
       vs ssn (same spectrum). With cafe as the reference, informational is
       NEGATIVE: the other noise lacks the content cafe has.

   (b) SII model. Using the stationary-only fit from section 2:

           energetic     = pred(X) - pred(R)
           informational = [WER(X) - pred(X)] - [WER(R) - pred(R)]

       Here informational is MEASURED for every noise, not assumed zero, so the
       white/pink/ssn values show how close to zero they really are. Error band
       on it: sqrt(2) x fit residual SD (difference of two residuals).

4. EVERY CONDITION, ALL THREE AXES, ON ONE AUDIBILITY SCALE
   Energetic and informational masking are defined for a MASKER. Axes B and C
   add no sound, so informational masking is zero there by definition. The
   question behind the split still applies to every condition: how much damage
   does lost audible speech explain, and how much not? Each condition gets an
   SII (Axis A: exact masker SII; Axes B/C: masking.distortion_sii on the
   time-aligned signal, where a removed band scores 0 and inserted silence,
   once aligned, scores 1). Relative to clean, with the stationary-only fit:

       audibility (energetic) = b * (SII(cond) - 1)
       non-audibility         = [WER(cond) - WER(clean)] - audibility

   For cafe the non-audibility term is informational masking; for B and C it is
   decoder-level damage (alignment, rate, hallucination) that no loss of
   audible speech can account for. Reported on raw and hallucination-robust WER.

HOW THE MASKER IS RECOVERED
   Each degraded file is (clean + k * noise) / p, where p >= 1 is the anti-clip
   peak normalisation. Projecting the degraded signal onto the clean one,

       a = <d, c> / <c, c>,   speech = a * c,   masker = d - a * c

   recovers the exact noise realisation that was used, including the cafe/ssn
   segment the cursor selected. The noise is uncorrelated with the speech, so
   the projection error is negligible; the measured broadband SNR is printed as
   a check against the nominal one.

Outputs:
    results/masking_{contrasts,contrast_matrix,acoustics,model,by_reference,all_conditions}.csv
    results/plots/masking/*.png

Prints rather than logs, like every tool here.

Usage:
    python tools/masking_contrasts.py
    python tools/masking_contrasts.py --utterances 30          # quicker acoustic pass
    python tools/masking_contrasts.py --glimpse-threshold 0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from collections import defaultdict

import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from config import (
    AXIS_A_DIR, AXIS_B_DIR, AXIS_C_DIR, DATA_DIR, NOISE_TYPES, RESULTS_DIR, SNR_LEVELS,
    NOISE_DISPLAY, display_condition,
    TARGET_SR, AXIS_B_CONDITION, STRETCH_RATES, SILENCE_CONFIGS,
    stretch_condition, silence_condition,
)
from src.analysis.hallucination import robust_global_wer
from src.analysis.masking import distortion_sii, paired_significance, speech_weighted_snr
from src.augmentation.axis_c_temporal import remove_inserted_silence
from src.analysis.metrics import global_wer
from src.augmentation.axis_a_noise import compute_rms
from src.features.extractor import FRAME_LENGTH, HOP_LENGTH, N_MELS, build_mel_filterbank
from src.results_io import error_count, read_rows, word_count, write_rows

OUT_DIR = RESULTS_DIR          # top level, so the results/*.csv ignore rule covers it
PLOTS_DIR = RESULTS_DIR / "plots" / "masking"

STRUCTURED = "cafe"     # the masker with linguistic content
CONTROL = "ssn"         # its phase-randomised, spectrally identical control
STATIONARY = [n for n in NOISE_TYPES if n != STRUCTURED]   # white, pink, ssn

# (A, B, kind): contrast = WER(A) - WER(B). See the module docstring for why
# each pair gets the label it does.
CONTRASTS = [
    ("pink", "white", "energetic"),
    ("ssn", "white", "energetic"),
    ("ssn", "pink", "energetic"),
    ("cafe", "ssn", "informational"),
    ("cafe", "white", "mixed"),
    ("cafe", "pink", "mixed"),
]

# Same colours as scripts/05_run_analysis.py so the figures read together.
NOISE_COLOURS = {"white": "#d62728", "pink": "#ff7f0e", "cafe": "#1f77b4", "ssn": "#2ca02c"}
ENERGETIC_COLOUR = "#2ca02c"
INFORMATIONAL_COLOUR = "#1f77b4"
TOTAL_COLOUR = "#7f7f7f"


# =============================================================================
# Behavioural: paired contrasts
# =============================================================================

def cell(rows: list, noise: str, snr: int) -> list:
    return [r for r in rows if r["noise_type"] == noise and int(float(r["snr_db"])) == snr]


def paired_diffs(rows_a: list, rows_b: list):
    """Per-utterance error difference A - B over the utterances both cells share."""
    by_a = {r["utterance_id"]: r for r in rows_a}
    by_b = {r["utterance_id"]: r for r in rows_b}
    diffs, words = [], []
    for uid in sorted(set(by_a) & set(by_b)):
        diffs.append(error_count(by_a[uid]) - error_count(by_b[uid]))
        words.append(word_count(by_b[uid]["ground_truth"]))
    return diffs, words


def contrast_rows(axis_a: list, snrs: list) -> list:
    """Every contrast in CONTRASTS at every SNR, plus pooled over SNRs."""
    out = []
    for a, b, kind in CONTRASTS:
        pooled_d, pooled_w = [], []
        for snr in snrs + ["pooled"]:
            if snr == "pooled":
                d, w = pooled_d, pooled_w
            else:
                d, w = paired_diffs(cell(axis_a, a, snr), cell(axis_a, b, snr))
                pooled_d += d
                pooled_w += w
            if not d:
                continue
            st = paired_significance(d, w)
            out.append({
                "contrast": f"{a} - {b}", "kind": kind, "snr_db": snr,
                "diff_pp": round(st["diff_pp"], 4),
                "ci_low": round(st["ci_low"], 4), "ci_high": round(st["ci_high"], 4),
                "wilcoxon_p": round(st["wilcoxon_p"], 6),
                "n_worse": st["n_worse"], "n_better": st["n_better"],
                "verdict": st["verdict"],
            })
    return out


def contrast_matrix(axis_a: list, snr: int):
    """
    4x4 matrix M[i, j] = WER(noise_i) - WER(noise_j), with verdicts.

    Antisymmetric, so only i < j is bootstrapped; the transpose is negated. The
    verdict does not depend on direction.
    """
    n = len(NOISE_TYPES)
    values = np.full((n, n), np.nan)
    verdicts = [["" for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d, w = paired_diffs(cell(axis_a, NOISE_TYPES[i], snr),
                                cell(axis_a, NOISE_TYPES[j], snr))
            if not d:
                continue
            st = paired_significance(d, w)
            values[i, j], values[j, i] = st["diff_pp"], -st["diff_pp"]
            verdicts[i][j] = verdicts[j][i] = st["verdict"]
    return values, verdicts


# =============================================================================
# Acoustic: recover the masker from each file, measure SII and glimpses
# =============================================================================

def split_mixture(clean: np.ndarray, degraded: np.ndarray):
    """Project the degraded file onto the clean one: (speech part, masker part)."""
    n = min(len(clean), len(degraded))
    c, d = clean[:n], degraded[:n]
    a = float(np.dot(d, c) / np.dot(c, c))
    speech = a * c
    return speech, d - speech


def glimpse_proportion(speech: np.ndarray, masker: np.ndarray, sr: int,
                       threshold_db: float) -> float:
    """
    Cooke (2006) glimpse proportion, simplified.

        GP = 100 * #{(t, f) : S(t, f) > N(t, f) * 10^(threshold/10)} / #{(t, f)}

    S and N are the speech and masker energies in 40 Mel bands, 32 ms frames,
    10 ms hop -- the project's feature grid, in place of Cooke's 55-channel
    gammatone filterbank. No minimum glimpse area and no hearing threshold.
    Both simplifications apply to every masker identically, so the comparison
    between maskers is what this measures, not an absolute intelligibility.
    """
    stft = dict(fs=sr, nperseg=FRAME_LENGTH, noverlap=FRAME_LENGTH - HOP_LENGTH, window="hann")
    _, _, s = scipy.signal.stft(speech, **stft)
    _, _, m = scipy.signal.stft(masker, **stft)
    fbank = build_mel_filterbank(FRAME_LENGTH, sr, N_MELS)
    s_mel = fbank @ (np.abs(s) ** 2)
    m_mel = fbank @ (np.abs(m) ** 2)
    glimpsed = s_mel > m_mel * 10.0 ** (threshold_db / 10.0)
    return float(glimpsed.mean() * 100.0)


def measure_acoustics(axis_a: list, n_utterances, threshold_db: float) -> list:
    """SII, GP and measured SNR for every Axis A file on disk."""
    uids = sorted({r["utterance_id"] for r in axis_a})
    if n_utterances:
        uids = uids[:n_utterances]

    out, missing = [], 0
    for k, uid in enumerate(uids, 1):
        clean_path = DATA_DIR / f"{uid}.wav"
        if not clean_path.exists():
            missing += len(NOISE_TYPES) * len(SNR_LEVELS)
            continue
        clean, sr = sf.read(str(clean_path), dtype="float64")

        for noise in NOISE_TYPES:
            for snr in SNR_LEVELS:
                path = AXIS_A_DIR / f"{uid}_{noise}_{snr}db.wav"
                if not path.exists():
                    missing += 1
                    continue
                degraded, _ = sf.read(str(path), dtype="float64")
                speech, masker = split_mixture(clean, degraded)
                measured_snr = 20.0 * np.log10(compute_rms(speech) / compute_rms(masker))
                out.append({
                    "utterance_id": uid, "noise_type": noise, "snr_db": snr,
                    "measured_snr_db": measured_snr,
                    "sii": speech_weighted_snr(speech, masker, sr, measured_snr),
                    "gp": glimpse_proportion(speech, masker, sr, threshold_db),
                })
        if k % 25 == 0 or k == len(uids):
            print(f"  acoustic pass: {k}/{len(uids)} utterances")

    if missing:
        print(f"  [WARN] {missing} degraded/clean files missing from disk -- skipped.")
    return out


def mean_by_cell(acoustic: list) -> dict:
    groups = defaultdict(list)
    for r in acoustic:
        groups[(r["noise_type"], r["snr_db"])].append(r)
    return {
        key: {m: float(np.mean([r[m] for r in rows])) for m in ("sii", "gp", "measured_snr_db")}
        for key, rows in groups.items()
    }


# =============================================================================
# Model: energetic prediction from the stationary maskers
# =============================================================================

def fit_line(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(slope), "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        # ddof=2: two parameters fitted
        "resid_sd": float(np.sqrt(ss_res / max(len(x) - 2, 1))),
    }


def model_rows(wer: dict, acoustics: dict, baseline: float, metric: str, snrs: list):
    stationary = [(n, s) for n in STATIONARY for s in snrs if (n, s) in acoustics and (n, s) in wer]
    fit = fit_line([acoustics[k][metric] for k in stationary], [wer[k] for k in stationary])

    rows = []
    for noise in NOISE_TYPES:
        for snr in snrs:
            if (noise, snr) not in acoustics or (noise, snr) not in wer:
                continue
            x = acoustics[(noise, snr)][metric]
            predicted = fit["intercept"] + fit["slope"] * x
            rows.append({
                "metric": metric, "noise_type": noise, "snr_db": snr,
                "metric_value": round(x, 4),
                "observed_wer": round(wer[(noise, snr)], 4),
                "predicted_wer": round(predicted, 4),
                "energetic_pp": round(predicted - baseline, 4),
                "non_energetic_pp": round(wer[(noise, snr)] - predicted, 4),
                "in_fit": noise in STATIONARY,
            })
    return fit, rows


# =============================================================================
# Every noise as the reference: energetic + informational for every other noise
# =============================================================================

def control_of(noise: str) -> str:
    """Phase-randomised, spectrally identical control. A steady noise is its own."""
    return CONTROL if noise == STRUCTURED else noise


def _term(axis_a: list, terms: list, snr) -> dict:
    """
    Paired signed combination of WERs, e.g. [(cafe, +1), (ssn, -1)] = WER(cafe) -
    WER(ssn), at one SNR or pooled over all SNRs. Per utterance:

        d_u = sum_n coef_n * errors_n,u        diff = 100 * sum_u d_u / sum_u N_u

    so a four-noise informational difference gets a proper paired CI. Terms that
    cancel (e.g. white - white) leave nothing: zero by construction.
    """
    coefs = defaultdict(int)
    for noise, coef in terms:
        coefs[noise] += coef
    coefs = {n: c for n, c in coefs.items() if c}
    if not coefs:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "verdict": "zero by construction"}

    snrs = sorted(SNR_LEVELS, reverse=True) if snr == "pooled" else [snr]
    d, w = [], []
    for s in snrs:
        by_noise = {n: {r["utterance_id"]: r for r in cell(axis_a, n, s)} for n in coefs}
        for uid in sorted(set.intersection(*(set(m) for m in by_noise.values()))):
            d.append(sum(c * error_count(by_noise[n][uid]) for n, c in coefs.items()))
            w.append(word_count(next(iter(by_noise.values()))[uid]["ground_truth"]))
    st = paired_significance(d, w)
    return {"diff": st["diff_pp"], "lo": st["ci_low"], "hi": st["ci_high"],
            "verdict": st["verdict"]}


def by_reference_subtraction(axis_a: list, snrs: list) -> list:
    """
    Method (a): control subtraction, applied to BOTH sides, with paired CIs:

        energetic     = WER(ctrl X) - WER(ctrl R)
        informational = [WER(X) - WER(ctrl X)] - [WER(R) - WER(ctrl R)]

    The two sum exactly to WER(X) - WER(R). Controlling only X was wrong: for
    R = cafe it pushed cafe's own informational term into "energetic".
    """
    rows = []
    for ref in NOISE_TYPES:
        for noise in NOISE_TYPES:
            if noise == ref:
                continue
            cx, cr = control_of(noise), control_of(ref)
            for snr in snrs + ["pooled"]:
                total = _term(axis_a, [(noise, 1), (ref, -1)], snr)
                energetic = _term(axis_a, [(cx, 1), (cr, -1)], snr)
                informational = _term(axis_a, [(noise, 1), (cx, -1), (ref, -1), (cr, 1)], snr)
                rows.append({
                    "method": "control subtraction", "reference": ref, "noise": noise,
                    "snr_db": snr,
                    "total_pp": round(total["diff"], 4),
                    "energetic_pp": round(energetic["diff"], 4),
                    "energetic_ci_low": round(energetic["lo"], 4),
                    "energetic_ci_high": round(energetic["hi"], 4),
                    "energetic_verdict": energetic["verdict"],
                    "informational_pp": round(informational["diff"], 4),
                    "informational_ci_low": round(informational["lo"], 4),
                    "informational_ci_high": round(informational["hi"], 4),
                    "informational_verdict": informational["verdict"],
                })
    return rows


def by_reference_model(model: list, fit: dict, snrs: list, metric: str) -> list:
    """
    Method (b): SII model. Energetic = difference of predictions; informational =
    difference of residuals, measured for every noise. Pooled = mean over SNRs
    (every cell has the same reference words, so this equals the word-weighted
    pooled difference).
    """
    lookup = {(r["noise_type"], r["snr_db"]): r for r in model}
    band = float(np.sqrt(2.0) * fit["resid_sd"])
    rows = []
    for ref in NOISE_TYPES:
        for noise in NOISE_TYPES:
            if noise == ref:
                continue
            per_snr = []
            for snr in snrs:
                x, r = lookup.get((noise, snr)), lookup.get((ref, snr))
                if x is None or r is None:
                    continue
                per_snr.append((snr,
                                x["observed_wer"] - r["observed_wer"],
                                x["predicted_wer"] - r["predicted_wer"],
                                x["non_energetic_pp"] - r["non_energetic_pp"]))
            if per_snr:
                per_snr.append(("pooled",) + tuple(float(np.mean([p[i] for p in per_snr]))
                                                   for i in (1, 2, 3)))
            for snr, total, energetic, informational in per_snr:
                rows.append({
                    "method": f"{metric.upper()} model", "reference": ref, "noise": noise,
                    "snr_db": snr, "total_pp": round(total, 4),
                    "energetic_pp": round(energetic, 4),
                    "informational_pp": round(informational, 4),
                    "informational_band_pp": round(band, 4),
                })
    return rows


# =============================================================================
# Plots
# =============================================================================

def save(fig, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {PLOTS_DIR / name}")


def bar_labels(ax, xs, values, fontsize=7):
    for x, v in zip(xs, values):
        if v is None or np.isnan(v):
            continue
        v = round(float(v), 2) + 0.0          # no "-0.00"
        ax.text(x, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=fontsize)


def plot_matrices(matrices: dict, snrs: list) -> None:
    vmax = max(np.nanmax(np.abs(v)) for v, _ in matrices.values())
    fig, axes = plt.subplots(1, len(snrs), figsize=(4.6 * len(snrs), 4.8))
    for k, (ax, snr) in enumerate(zip(np.atleast_1d(axes), snrs)):
        values, verdicts = matrices[snr]
        im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for i in range(len(NOISE_TYPES)):
            for j in range(len(NOISE_TYPES)):
                if i == j:
                    ax.text(j, i, "-", ha="center", va="center", color="grey")
                    continue
                mark = {"significant": "*", "disputed": "†"}.get(verdicts[i][j], "")
                ax.text(j, i, f"{values[i, j]:+.2f}{mark}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(NOISE_TYPES)))
        ax.set_yticks(range(len(NOISE_TYPES)))
        ax.set_xticklabels(NOISE_TYPES)
        ax.set_yticklabels(NOISE_TYPES)
        ax.set_xlabel("minus column (B)")
        if k == 0:
            ax.set_ylabel("row (A)")
        ax.set_title(f"{snr} dB:  WER(A) - WER(B), pp")
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.8, label="pp (red = row worse)")
    fig.suptitle("Every noise against every other   (* significant in both tests, "
                 "† tests disagree)", y=1.02)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "contrast_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {PLOTS_DIR / 'contrast_matrix.png'}")


def plot_contrast_group(rows: list, kind_filter, title: str, name: str) -> None:
    """Grouped bars with 95% bootstrap CIs, one group per SNR plus pooled."""
    contrasts = [f"{a} - {b}" for a, b, k in CONTRASTS if k in kind_filter]
    groups = [g for g in [r["snr_db"] for r in rows if r["contrast"] == contrasts[0]]]
    lookup = {(r["contrast"], r["snr_db"]): r for r in rows}

    width = 0.8 / len(contrasts)
    palette = ["#8c564b", "#2ca02c", "#17becf", "#1f77b4", "#9467bd", "#e377c2"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, name_c in enumerate(contrasts):
        xs = [g + (i - (len(contrasts) - 1) / 2) * width for g in range(len(groups))]
        recs = [lookup.get((name_c, g)) for g in groups]
        vals = [r["diff_pp"] if r else np.nan for r in recs]
        err = np.array([[r["diff_pp"] - r["ci_low"], r["ci_high"] - r["diff_pp"]] if r else [0, 0]
                        for r in recs]).T
        ax.bar(xs, vals, width, yerr=err, capsize=3, label=name_c,
               color=palette[i % len(palette)])
        bar_labels(ax, xs, vals)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g} dB" if g != "pooled" else "pooled" for g in groups])
    ax.set_ylabel("WER difference (percentage points)")
    ax.set_title(title)
    ax.legend()
    save(fig, name)


def plot_informational_by_reference(rows: list, snrs: list) -> None:
    """
    cafe - X split into informational (cafe - ssn) and energetic (ssn - X), for
    each stationary reference X. The identity is exact, so the two parts sum to
    the total bar.
    """
    lookup = {(r["contrast"], r["snr_db"]): r for r in rows}
    groups = snrs + ["pooled"]
    refs = STATIONARY
    fig, axes = plt.subplots(1, len(refs), figsize=(6 * len(refs), 5.5), sharey=True)
    width = 0.27
    for ax, ref in zip(axes, refs):
        total = [lookup.get((f"{STRUCTURED} - {ref}", g)) for g in groups]
        info = [lookup.get((f"{STRUCTURED} - {CONTROL}", g)) for g in groups]
        if ref == CONTROL:
            energetic = [None] * len(groups)          # identically zero
        else:
            energetic = [lookup.get((f"{CONTROL} - {ref}", g)) for g in groups]

        x = np.arange(len(groups))
        for offset, recs, label, colour in (
            (-width, total, f"total  ({STRUCTURED} - {ref})", TOTAL_COLOUR),
            (0.0, info, f"informational  ({STRUCTURED} - {CONTROL})", INFORMATIONAL_COLOUR),
            (width, energetic, f"energetic  ({CONTROL} - {ref})", ENERGETIC_COLOUR),
        ):
            if all(r is None for r in recs):
                continue
            vals = [r["diff_pp"] if r else np.nan for r in recs]
            err = np.array([[r["diff_pp"] - r["ci_low"], r["ci_high"] - r["diff_pp"]]
                            if r else [0, 0] for r in recs]).T
            ax.bar(x + offset, vals, width, yerr=err, capsize=3, label=label, color=colour)
            bar_labels(ax, x + offset, vals)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g} dB" if g != "pooled" else "pooled" for g in groups])
        ax.set_title(f"reference = {ref}" + ("   (energetic part = 0 by construction)"
                                             if ref == CONTROL else ""))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("WER difference (percentage points)")
    fig.suptitle(f"{STRUCTURED} against each stationary noise:  "
                 f"{STRUCTURED} - X = ({STRUCTURED} - {CONTROL}) + ({CONTROL} - X)", y=1.02)
    save(fig, "informational_by_reference.png")


def plot_acoustics(acoustics: dict, snrs: list, threshold_db: float) -> None:
    fig, (ax_sii, ax_gp) = plt.subplots(1, 2, figsize=(13, 5))
    for noise in NOISE_TYPES:
        sii = [acoustics[(noise, s)]["sii"] if (noise, s) in acoustics else np.nan for s in snrs]
        gp = [acoustics[(noise, s)]["gp"] if (noise, s) in acoustics else np.nan for s in snrs]
        style = "--" if noise == CONTROL else "-"
        ax_sii.plot(snrs, sii, style, marker="o", label=noise, color=NOISE_COLOURS.get(noise))
        ax_gp.plot(snrs, gp, style, marker="o", label=noise, color=NOISE_COLOURS.get(noise))
        for s, v in zip(snrs, sii):
            ax_sii.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color=NOISE_COLOURS.get(noise))
        for s, v in zip(snrs, gp):
            ax_gp.annotate(f"{v:.1f}", (s, v), textcoords="offset points", xytext=(0, 6),
                           ha="center", fontsize=7, color=NOISE_COLOURS.get(noise))
    for ax in (ax_sii, ax_gp):
        ax.set_xticks(snrs)
        ax.invert_xaxis()
        ax.set_xlabel("SNR (dB)")
        ax.legend()
    ax_sii.set_ylabel("SII (0-1, higher = more audible)")
    ax_sii.set_title("Speech Intelligibility Index\n(spectrum only: cafe and ssn must coincide)")
    ax_gp.set_ylabel(f"Glimpse proportion (%), local SNR > {threshold_db:g} dB")
    ax_gp.set_title("Glimpse proportion\n(sees temporal dips: cafe may sit above ssn)")
    save(fig, "acoustic_predictors.png")


def plot_model(model: dict, fits: dict, baseline: float, snrs: list) -> None:
    metrics = list(model)
    fig, axes = plt.subplots(len(metrics), 2, figsize=(16, 5.5 * len(metrics)),
                             gridspec_kw={"width_ratios": [1, 1.6]})
    axes = np.atleast_2d(axes)
    labels = {"sii": "SII", "gp": "Glimpse proportion (%)"}

    for row_axes, metric in zip(axes, metrics):
        ax_fit, ax_bar = row_axes
        rows, fit = model[metric], fits[metric]

        # Left: observed WER against the metric, with the stationary-only fit.
        for noise in NOISE_TYPES:
            pts = [r for r in rows if r["noise_type"] == noise]
            ax_fit.scatter([r["metric_value"] for r in pts], [r["observed_wer"] for r in pts],
                           color=NOISE_COLOURS.get(noise), s=45,
                           marker="o" if noise in STATIONARY else "D",
                           label=noise + ("" if noise in STATIONARY else "  (not in fit)"))
            for r in pts:
                ax_fit.annotate(f"{r['snr_db']}", (r["metric_value"], r["observed_wer"]),
                                textcoords="offset points", xytext=(5, 3), fontsize=7)
        xs = np.linspace(min(r["metric_value"] for r in rows),
                         max(r["metric_value"] for r in rows), 50)
        ax_fit.plot(xs, fit["intercept"] + fit["slope"] * xs, color="black",
                    label=f"energetic fit (white/pink/ssn)  R² = {fit['r2']:.2f}")
        ax_fit.fill_between(xs, fit["intercept"] + fit["slope"] * xs - fit["resid_sd"],
                            fit["intercept"] + fit["slope"] * xs + fit["resid_sd"],
                            color="black", alpha=0.08, label=f"± residual SD ({fit['resid_sd']:.2f} pp)")
        ax_fit.axhline(baseline, color="black", linestyle="--", linewidth=0.8,
                       label=f"clean = {baseline:.2f}%")
        ax_fit.set_xlabel(labels[metric])
        ax_fit.set_ylabel("Global WER %")
        ax_fit.set_title(f"WER vs {labels[metric]}  (labels = SNR dB)")
        ax_fit.legend(fontsize=8)

        # Right: every noise x SNR split into energetic + non-energetic.
        cells = [(n, s) for n in NOISE_TYPES for s in snrs]
        lookup = {(r["noise_type"], r["snr_db"]): r for r in rows}
        cells = [c for c in cells if c in lookup]
        x = np.arange(len(cells))
        width = 0.4
        en = [lookup[c]["energetic_pp"] for c in cells]
        ne = [lookup[c]["non_energetic_pp"] for c in cells]
        ax_bar.bar(x - width / 2, en, width, color=ENERGETIC_COLOUR,
                   label="energetic = predicted - clean")
        ax_bar.bar(x + width / 2, ne, width, color=INFORMATIONAL_COLOUR,
                   yerr=fit["resid_sd"], capsize=2,
                   label="non-energetic = observed - predicted  (± fit SD)")
        bar_labels(ax_bar, x - width / 2, en, fontsize=6)
        bar_labels(ax_bar, x + width / 2, ne, fontsize=6)
        ax_bar.axhline(0, color="black", linewidth=0.8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"{n}\n{s}dB" for n, s in cells], fontsize=7)
        for i in range(1, len(NOISE_TYPES)):
            ax_bar.axvline(i * len(snrs) - 0.5, color="grey", linewidth=0.5)
        ax_bar.set_ylabel("WER contribution (pp)")
        ax_bar.set_title(f"Reference-free decomposition via {labels[metric]}")
        ax_bar.legend(fontsize=8)

    save(fig, "model_decomposition.png")


def _draw_reference_panel(ax, lookup: dict, ref: str, groups: list, has_ci: bool,
                          label_size: float) -> None:
    """Every other noise X against reference `ref`: total / energetic / informational."""
    width = 0.27
    others = [n for n in NOISE_TYPES if n != ref]
    positions, labels = [], []
    for k, noise in enumerate(others):
        for g, snr in enumerate(groups):
            x = k * (len(groups) + 1) + g
            r = lookup.get((ref, noise, snr))
            positions.append(x)
            labels.append(f"{snr} dB" if snr != "pooled" else "pooled")
            if r is None:
                continue

            for offset, key, colour in ((-width, "total", TOTAL_COLOUR),
                                        (0.0, "energetic", ENERGETIC_COLOUR),
                                        (width, "informational", INFORMATIONAL_COLOUR)):
                value = r[f"{key}_pp"]
                if r.get(f"{key}_verdict") == "zero by construction":
                    ax.text(x + offset, 0, "=0", ha="center", va="bottom",
                            fontsize=label_size, color=colour, fontweight="bold")
                    continue

                err = None
                if key != "total" and has_ci:
                    err = [[value - r[f"{key}_ci_low"]], [r[f"{key}_ci_high"] - value]]
                elif key == "informational" and not has_ci:
                    err = r["informational_band_pp"]
                ax.bar(x + offset, value, width, color=colour, yerr=err, capsize=2,
                       error_kw={"linewidth": 0.8})
                mark = "*" if r.get(f"{key}_verdict") == "significant" else ""
                ax.text(x + offset, value, f"{value:+.2f}{mark}", ha="center",
                        va="bottom" if value >= 0 else "top", fontsize=label_size, rotation=90)

        centre = k * (len(groups) + 1) + (len(groups) - 1) / 2
        ax.text(centre, 1.02, f"{noise}  vs  {ref}", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=10, color=NOISE_COLOURS.get(noise), fontweight="bold")
        if k:
            ax.axvline(k * (len(groups) + 1) - 1, color="grey", linewidth=0.6)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("WER difference (pp)")
    ax.set_title(f"reference = {ref}", loc="left", fontsize=12, fontweight="bold",
                 color=NOISE_COLOURS.get(ref), pad=18)


def _legend(fig, y: float) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (TOTAL_COLOUR, ENERGETIC_COLOUR, INFORMATIONAL_COLOUR)]
    fig.legend(handles, ["total = WER(X) - WER(ref)", "energetic", "informational"],
               loc="upper center", bbox_to_anchor=(0.5, y), ncol=3, fontsize=10)


def plot_by_reference_compact(rows: list, snr, name: str) -> None:
    """
    Paper-sized version of plot_by_reference: one SNR, a 2x2 grid with one
    panel per base noise. Each panel shows, for the other three noises, the
    energetic and informational parts with 95 % CIs and the total as a marker.
    Accepts rows straight from the computation or from masking_by_reference.csv.
    """
    def num(r, k):
        return float(r[k])

    lookup = {(r["reference"], r["noise"], str(r["snr_db"])): r for r in rows
              if r.get("method", "control subtraction") == "control subtraction"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharey=True)
    width = 0.36
    for ax, ref in zip(axes.flat, NOISE_TYPES):
        others = [n for n in NOISE_TYPES if n != ref]
        for i, noise in enumerate(others):
            r = lookup.get((ref, noise, str(snr)))
            if r is None:
                continue
            for offset, key, colour in ((-width / 2, "energetic", ENERGETIC_COLOUR),
                                        (width / 2, "informational", INFORMATIONAL_COLOUR)):
                if r[f"{key}_verdict"] == "zero by construction":
                    ax.text(i + offset, 0.1, "=0", ha="center", va="bottom", fontsize=11,
                            color=colour, fontweight="bold")
                    continue
                value = num(r, f"{key}_pp")
                lo, hi = num(r, f"{key}_ci_low"), num(r, f"{key}_ci_high")
                ax.bar(i + offset, value, width, color=colour,
                       yerr=[[value - lo], [hi - value]], capsize=3, error_kw={"linewidth": 0.9})
                # Label beyond the end of the error bar so the two never overlap.
                mark = "*" if r[f"{key}_verdict"] == "significant" else ""
                ax.text(i + offset, hi + 0.2 if value >= 0 else lo - 0.2,
                        f"{value:+.2f}{mark}", ha="center",
                        va="bottom" if value >= 0 else "top", fontsize=11)
            ax.plot(i, num(r, "total_pp"), marker="D", color="black", markersize=7,
                    zorder=5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlim(-0.6, len(others) - 0.4)
        ax.set_ylim(-7, 7)
        ax.set_xticks(range(len(others)))
        ax.set_xticklabels([f"{NOISE_DISPLAY[n]}\nvs {NOISE_DISPLAY[ref].lower()}" for n in others],
                           fontsize=12)
        ax.set_title(f"Base: {NOISE_DISPLAY[ref].lower()}", fontsize=13, fontweight="bold",
                     color=NOISE_COLOURS.get(ref))
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("WER difference (pp)", fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ENERGETIC_COLOUR),
               plt.Rectangle((0, 0), 1, 1, color=INFORMATIONAL_COLOUR),
               plt.Line2D([], [], marker="D", color="black", linestyle="none")]
    fig.legend(handles, ["Energetic", "Informational", "Total difference"],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=12)
    # Reserve the top strip for the legend; save() would re-run a plain
    # tight_layout and push the panel titles back under it.
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {PLOTS_DIR / name}")


def plot_by_reference(rows: list, snrs: list, title: str, name: str) -> None:
    """
    Energetic and informational masking with each noise as the reference.

    Writes one combined figure (`name`, one row per reference) AND one figure per
    reference (`name` with _<ref> appended), so each baseline can be read or
    placed in a report on its own.

    A term that is zero BY CONSTRUCTION (not measured) is marked "=0" rather
    than drawn as a zero-height bar, which would read as "measured, no effect".
    """
    groups = snrs + ["pooled"]
    lookup = {(r["reference"], r["noise"], r["snr_db"]): r for r in rows}
    has_ci = "energetic_ci_low" in rows[0]

    fig, axes = plt.subplots(len(NOISE_TYPES), 1, figsize=(20, 5.2 * len(NOISE_TYPES)))
    for ax, ref in zip(np.atleast_1d(axes), NOISE_TYPES):
        _draw_reference_panel(ax, lookup, ref, groups, has_ci, label_size=5.5)
    _legend(fig, 1.0)
    fig.suptitle(title, fontsize=13, y=1.015)
    save(fig, name)

    stem = name[:-len(".png")] if name.endswith(".png") else name
    for ref in NOISE_TYPES:
        fig, ax = plt.subplots(figsize=(17, 6))
        _draw_reference_panel(ax, lookup, ref, groups, has_ci, label_size=7)
        _legend(fig, 1.02)
        fig.suptitle(f"{title}\nbase noise = {ref}", fontsize=12, y=1.10)
        save(fig, f"{stem}_{ref}.png")



# =============================================================================
# All 22 conditions on one audibility scale (Axes A, B and C)
# =============================================================================

BC_CONDITIONS = (
    [(AXIS_B_CONDITION, "B", AXIS_B_DIR, None)]
    + [(stretch_condition(r), "C", AXIS_C_DIR, None) for r in STRETCH_RATES]
    + [(silence_condition(p, d), "C", AXIS_C_DIR, (p, d)) for p, d in SILENCE_CONFIGS]
)


def measure_bc_sii(uids: list) -> dict:
    """Mean distortion SII per Axis B/C condition over the files on disk."""
    values = defaultdict(list)
    for uid in uids:
        clean_path = DATA_DIR / f"{uid}.wav"
        if not clean_path.exists():
            continue
        clean, sr = sf.read(str(clean_path), dtype="float64")
        for cond, _axis, folder, silence in BC_CONDITIONS:
            path = folder / f"{uid}_{cond}.wav"
            if not path.exists():
                continue
            deg, _ = sf.read(str(path), dtype="float64")
            if silence:
                deg = remove_inserted_silence(deg, sr, silence[1], silence[0], len(clean))
            values[cond].append(distortion_sii(clean, deg, sr))
    return {cond: float(np.mean(v)) for cond, v in values.items()}


def all_condition_rows(axis_a, bc_rows, baseline_rows, acoustics, bc_sii, fit, snrs) -> list:
    """Audibility vs non-audibility split, relative to clean, for every condition."""
    clean_raw, clean_rob = global_wer(baseline_rows), robust_global_wer(baseline_rows)
    b = fit["slope"]
    entries = [(f"{n} {s}dB", "A", cell(axis_a, n, s), acoustics.get((n, s), {}).get("sii"))
               for n in NOISE_TYPES for s in snrs]
    by_cond = defaultdict(list)
    for r in bc_rows:
        by_cond[r["condition"]].append(r)
    entries += [(cond, axis, by_cond.get(cond, []), bc_sii.get(cond))
                for cond, axis, _f, _s in BC_CONDITIONS]

    out = []
    for name, axis, rows, sii in entries:
        if not rows or sii is None:
            continue
        audibility = b * (sii - 1.0)
        raw, rob = global_wer(rows), robust_global_wer(rows)
        out.append({
            "condition": name, "axis": axis, "sii": round(sii, 4),
            "total_pp": round(raw - clean_raw, 4),
            "energetic_pp": round(audibility, 4),
            "non_energetic_pp": round(raw - clean_raw - audibility, 4),
            "total_robust_pp": round(rob - clean_rob, 4),
            "non_energetic_robust_pp": round(rob - clean_rob - audibility, 4),
            "fit_band_pp": round(fit["resid_sd"], 4),
        })
    return out


def plot_all_conditions(rows: list, name: str, title: str) -> None:
    """Two panels (raw / hallucination-robust WER): total, energetic, non-energetic."""
    labels = [display_condition(r["condition"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.27
    boundaries = [i for i in range(1, len(rows)) if rows[i]["axis"] != rows[i - 1]["axis"]]
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 0.6 * len(rows) + 1), 7.2), sharex=True)
    for ax, suffix, panel in ((axes[0], "", "(a) Word error rate as measured"),
                              (axes[1], "_robust", "(b) Hallucinated insertions removed")):
        total = [r[f"total{suffix}_pp"] for r in rows]
        energetic = [r["energetic_pp"] for r in rows]
        other = [r[f"non_energetic{suffix}_pp"] for r in rows]
        ax.bar(x - width, total, width, color=TOTAL_COLOUR, label="Total change vs clean")
        ax.bar(x, energetic, width, color=ENERGETIC_COLOUR,
               label="Energetic (explained by audibility)")
        ax.bar(x + width, other, width, color=INFORMATIONAL_COLOUR,
               yerr=rows[0]["fit_band_pp"], capsize=2, error_kw={"linewidth": 0.8},
               label="Non-energetic")
        # No per-bar value labels: 66 of them overlap on the small bars, and the
        # values are in results/masking_all_conditions.csv and the report tables.
        ax.axhline(0, color="black", linewidth=0.8)
        for bnd in boundaries:
            ax.axvline(bnd - 0.5, color="grey", linewidth=0.8)
        ax.set_ylabel("WER change vs clean (pp)")
        ax.set_title(panel, loc="left")
        ax.legend(fontsize=10, loc="best", framealpha=0.9)
    starts = [0] + boundaries + [len(rows)]
    for lo, hi in zip(starts[:-1], starts[1:]):
        axes[0].text((lo + hi - 1) / 2, 1.02, f"Axis {rows[lo]['axis']}",
                     transform=axes[0].get_xaxis_transform(), ha="center", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    save(fig, name)


# =============================================================================
# Main
# =============================================================================

def main(n_utterances, threshold_db: float) -> None:
    try:
        baseline_rows = read_rows(RESULTS_DIR / "baseline_results.csv")
        axis_a = read_rows(RESULTS_DIR / "axis_a_results.csv")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    snrs = sorted(SNR_LEVELS, reverse=True)
    baseline = global_wer(baseline_rows)
    wer = {(n, s): global_wer(cell(axis_a, n, s)) for n in NOISE_TYPES for s in snrs
           if cell(axis_a, n, s)}

    print("=" * 78)
    print("MASKING CONTRASTS ACROSS EVERY NOISE")
    print("=" * 78)
    print(f"Clean baseline: {baseline:.2f}%\n")
    print(f"{'SNR':>6}" + "".join(f"{n:>9}" for n in NOISE_TYPES))
    for s in snrs:
        print(f"{s:5d}dB" + "".join(f"{wer.get((n, s), np.nan):9.2f}" for n in NOISE_TYPES))

    # ---- 1. pairwise contrasts ----
    print("\n1. PAIRED CONTRASTS  (diff = WER(A) - WER(B), pp; 95% bootstrap CI)")
    rows = contrast_rows(axis_a, snrs)
    print(f"{'contrast':<15}{'kind':<15}{'SNR':>7}{'diff':>8}{'95% CI':>19}{'wilcoxon p':>12}  verdict")
    for r in rows:
        snr_txt = f"{r['snr_db']}dB" if r["snr_db"] != "pooled" else "pooled"
        print(f"{r['contrast']:<15}{r['kind']:<15}{snr_txt:>7}{r['diff_pp']:+8.2f}"
              f" [{r['ci_low']:+6.2f},{r['ci_high']:+6.2f}]{r['wilcoxon_p']:12.5f}  {r['verdict']}")
    write_rows(OUT_DIR / "masking_contrasts.csv", rows)

    matrices = {s: contrast_matrix(axis_a, s) for s in snrs}
    write_rows(OUT_DIR / "masking_contrast_matrix.csv", [
        {"snr_db": s, "row_A": NOISE_TYPES[i], "col_B": NOISE_TYPES[j],
         "diff_pp": round(float(v[i, j]), 4), "verdict": vd[i][j]}
        for s, (v, vd) in matrices.items()
        for i in range(len(NOISE_TYPES)) for j in range(len(NOISE_TYPES)) if i != j
    ])

    # ---- 2. acoustic pass ----
    print(f"\n2. ACOUSTIC PASS  (glimpse threshold {threshold_db:g} dB)")
    acoustic = measure_acoustics(axis_a, n_utterances, threshold_db)
    if not acoustic:
        print("[ERROR] No degraded audio found under data/degraded/axis_a. Run stage 2.")
        sys.exit(1)
    acoustics = mean_by_cell(acoustic)

    worst = max(abs(r["measured_snr_db"] - r["snr_db"]) for r in acoustic)
    print(f"  measured vs nominal SNR: max |error| = {worst:.3f} dB over {len(acoustic)} files")

    print(f"\n{'noise':<8}{'SNR':>6}{'SII':>8}{'GP %':>8}")
    for n in NOISE_TYPES:
        for s in snrs:
            if (n, s) in acoustics:
                a = acoustics[(n, s)]
                print(f"{n:<8}{s:5d}dB{a['sii']:8.3f}{a['gp']:8.1f}")
    if all((k, s) in acoustics for k in (STRUCTURED, CONTROL) for s in snrs):
        gap = max(abs(acoustics[(STRUCTURED, s)]["sii"] - acoustics[(CONTROL, s)]["sii"])
                  for s in snrs)
        print(f"\n  validity: max |SII({STRUCTURED}) - SII({CONTROL})| = {gap:.4f} "
              f"({'PASS' if gap < 0.01 else 'CHECK'}: equal spectra => equal SII)")
    write_rows(OUT_DIR / "masking_acoustics.csv", [
        {"noise_type": n, "snr_db": s, **{k: round(v, 4) for k, v in acoustics[(n, s)].items()}}
        for n in NOISE_TYPES for s in snrs if (n, s) in acoustics
    ])

    # ---- 3. reference-free decomposition ----
    print("\n3. REFERENCE-FREE DECOMPOSITION  (fit on white/pink/ssn only)")
    model, fits, all_model_rows = {}, {}, []
    for metric in ("sii", "gp"):
        fit, mrows = model_rows(wer, acoustics, baseline, metric, snrs)
        model[metric], fits[metric] = mrows, fit
        all_model_rows += mrows
        sign_ok = fit["slope"] < 0   # more audibility must mean fewer errors
        print(f"\n  {metric.upper()}: WER = {fit['intercept']:.3f} + {fit['slope']:.4f} * {metric}"
              f"   R² = {fit['r2']:.3f}   residual SD = {fit['resid_sd']:.2f} pp"
              + ("" if sign_ok else "   [WARN] slope >= 0: metric does not predict WER here"))
        print(f"  {'noise':<8}{'SNR':>6}{'obs':>8}{'pred':>8}{'energ.':>9}{'non-en.':>9}")
        for r in mrows:
            print(f"  {r['noise_type']:<8}{r['snr_db']:5d}dB{r['observed_wer']:8.2f}"
                  f"{r['predicted_wer']:8.2f}{r['energetic_pp']:+9.2f}{r['non_energetic_pp']:+9.2f}")
    write_rows(OUT_DIR / "masking_model.csv", all_model_rows)

    # ---- 4. every noise as the reference ----
    print("\n4. EVERY NOISE AS THE REFERENCE  (energetic / informational, pp)")
    ref_sub = by_reference_subtraction(axis_a, snrs)
    ref_sii = by_reference_model(model["sii"], fits["sii"], snrs, "sii")
    sii_lookup = {(r["reference"], r["noise"], r["snr_db"]): r for r in ref_sii}
    print(f"{'ref':<7}{'noise':<7}{'SNR':>7}{'total':>8}"
          f"{'E (subtr.)':>12}{'I (subtr.)':>12}{'E (SII)':>10}{'I (SII)':>10}")
    for r in ref_sub:
        m = sii_lookup.get((r["reference"], r["noise"], r["snr_db"]), {})
        snr_txt = f"{r['snr_db']}dB" if r["snr_db"] != "pooled" else "pooled"
        def fmt(row, key):
            if row.get(f"{key}_verdict") == "zero by construction":
                return "=0"
            return f"{row[f'{key}_pp']:+.2f}" + ("*" if row.get(f"{key}_verdict") == "significant" else "")
        print(f"{r['reference']:<7}{r['noise']:<7}{snr_txt:>7}{r['total_pp']:+8.2f}"
              f"{fmt(r, 'energetic'):>12}{fmt(r, 'informational'):>12}"
              f"{m.get('energetic_pp', float('nan')):+10.2f}{m.get('informational_pp', float('nan')):+10.2f}")
    print("  * = significant in both bootstrap and Wilcoxon;  =0 = zero by construction")
    write_rows(OUT_DIR / "masking_by_reference.csv", [
        {k: r.get(k, "") for k in (
            "method", "reference", "noise", "snr_db", "total_pp",
            "energetic_pp", "energetic_ci_low", "energetic_ci_high", "energetic_verdict",
            "informational_pp", "informational_ci_low", "informational_ci_high",
            "informational_verdict", "informational_band_pp")}
        for r in ref_sub + ref_sii
    ])

    # ---- 5. all three axes on one audibility scale ----
    print("\n5. ALL 22 CONDITIONS: AUDIBILITY-EXPLAINED (energetic) vs NOT (pp vs clean)")
    bc_rows = []
    for f in ("axis_b_results.csv", "axis_c_results.csv"):
        path = RESULTS_DIR / f
        if path.exists():
            bc_rows += read_rows(path)
    uids = sorted({r["utterance_id"] for r in axis_a})
    if n_utterances:
        uids = uids[:n_utterances]
    bc_sii = measure_bc_sii(uids)
    all_rows = all_condition_rows(axis_a, bc_rows, baseline_rows, acoustics, bc_sii,
                                  fits["sii"], snrs)
    print(f"{'condition':<18}{'SII':>7}{'total':>8}{'energ.':>8}{'non-en.':>9}"
          f"{'total(rob)':>12}{'non-en.(rob)':>13}")
    for r in all_rows:
        print(f"{r['condition']:<18}{r['sii']:7.3f}{r['total_pp']:+8.2f}{r['energetic_pp']:+8.2f}"
              f"{r['non_energetic_pp']:+9.2f}{r['total_robust_pp']:+12.2f}"
              f"{r['non_energetic_robust_pp']:+13.2f}")
    print(f"  (fit band +/- {fits['sii']['resid_sd']:.2f} pp; SII of clean = 1 by definition)")
    write_rows(OUT_DIR / "masking_all_conditions.csv", all_rows)

    # ---- plots ----
    print("\n4. PLOTS")
    plot_matrices(matrices, snrs)
    plot_contrast_group(rows, {"energetic"},
                        "Energetic contrasts: stationary, content-free maskers that differ "
                        "only in spectral shape", "energetic_contrasts.png")
    plot_contrast_group(rows, {"informational", "mixed"},
                        f"{STRUCTURED} against every other noise", "cafe_contrasts.png")
    plot_informational_by_reference(rows, snrs)
    plot_acoustics(acoustics, snrs, threshold_db)
    plot_model(model, fits, baseline, snrs)
    plot_by_reference(ref_sub, snrs,
                      "Energetic and informational masking with every noise as the reference  "
                      "(control subtraction; 95% bootstrap CIs; * significant)",
                      "masking_by_reference.png")
    plot_by_reference(ref_sii, snrs,
                      "Energetic and informational masking with every noise as the reference  "
                      "(SII model; informational measured for every noise, ± fit band)",
                      "masking_by_reference_sii.png")
    plot_by_reference_compact(ref_sub, min(SNR_LEVELS), "masking_by_reference_5db.png")

    if all_rows:
        plot_all_conditions(all_rows, "all_conditions_audibility.png",
                            "Every condition, all three axes: damage explained by lost audible "
                            "speech (energetic) vs not")
        for axis in ("A", "B", "C"):
            part = [r for r in all_rows if r["axis"] == axis]
            if part:
                plot_all_conditions(part, f"axis_{axis.lower()}_audibility.png",
                                    f"Axis {axis}: energetic (audibility-explained) vs "
                                    f"non-energetic damage, relative to clean")

    print(f"\nCSVs in {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Masking contrasts across every noise type")
    parser.add_argument("--utterances", type=int, default=None,
                        help="Limit the acoustic pass to the first N utterances (default: all)")
    parser.add_argument("--glimpse-threshold", type=float, default=3.0,
                        help="Local SNR (dB) a time-frequency cell must exceed to count as a "
                             "glimpse (default 3; Cooke 2006 found good fits from about -5 to 8)")
    args = parser.parse_args()
    main(args.utterances, args.glimpse_threshold)
