"""
Stage 5 — Statistical analysis and summary visualisations.

Aggregates the result CSVs into corpus-level WER, draws the summary figures, and
tests whether feature-space deviation predicts WER.

Reported numbers are GLOBAL (corpus) WER from src.analysis.metrics — total
errors over total reference words — not the mean of the per-utterance `wer`
column. See that module for why the two differ.

NO PANDAS, NO SEABORN. Both were used here until the AppLocker policy on the
target machine began blocking pandas' compiled extensions
("DLL load failed while importing timestamps"), which took this stage down
entirely. That is the same policy that keeps librosa out of the DSP modules, so
the constraint is permanent rather than incidental: this stage now uses the
stdlib csv module, numpy and bare matplotlib, matching what the inference stages
already do. Do not reintroduce either dependency.

Every axis gets the same three views, so the axes can be read against each other:
  1. WER against the clean baseline
  2. a severity curve, wherever the axis has a severity parameter
  3. error composition (substitutions / insertions / deletions)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display needed, and avoids a Tk dependency
import matplotlib.pyplot as plt

# Text sized for print. The report shrinks single-column figures to ~3.5 in, so
# figures are drawn near that size with 10-12 pt text (~7-8 pt once printed).
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})
from scipy.stats import pearsonr, spearmanr

from config import (
    RESULTS_DIR, SNR_LEVELS, NOISE_TYPES, NOISE_DISPLAY, display_condition,
    STRETCH_RATES, SILENCE_CONFIGS, stretch_condition, silence_condition,
)
from src.analysis.masking import decompose_masking
from src.analysis.metrics import global_wer
from src.results_io import read_rows, write_rows
from src.analysis.hallucination import hallucination_rate, robust_global_wer
from src.logger import setup_logger

logger = setup_logger("scripts.05_run_analysis")

# Consistent colour per noise type across every figure, so a reader can carry
# the mapping from one plot to the next.
NOISE_COLOURS = {
    "white": "#d62728",
    "pink": "#ff7f0e",
    "cafe": "#1f77b4",
    "ssn": "#2ca02c",
}
ERROR_COLOURS = {"substitutions": "#4c72b0", "insertions": "#dd8452", "deletions": "#55a868"}
BASELINE_COLOUR = "black"


# =============================================================================
# CSV helpers (replacing the pandas layer)
# =============================================================================

def group_by(rows: list, key) -> dict:
    out = defaultdict(list)
    for row in rows:
        out[key(row)].append(row)
    return dict(out)


def error_totals(rows: list) -> dict:
    return {
        field: sum(int(r[field]) for r in rows)
        for field in ("substitutions", "insertions", "deletions")
    }


def axis_a_cell(rows: list, noise: str, snr: int) -> list:
    return [r for r in rows
            if r["noise_type"] == noise and int(float(r["snr_db"])) == snr]


# =============================================================================
# Shared plot primitives — one implementation per figure TYPE, so Axis A, B and
# C are drawn by the same code and are therefore directly comparable.
# =============================================================================

def plot_condition_bars(labels, values, baseline_wer, title, out_path, colours=None):
    """
    WER per condition, with the clean baseline as a reference line.

    Bars carry their value as text. Several conditions sit within 0.2 pp of the
    baseline, which is not readable off the axis alone.
    """
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels) + 2), 6))
    bars = ax.bar(range(len(labels)), values,
                  color=colours or ["#4c72b0"] * len(labels))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(baseline_wer, color=BASELINE_COLOUR, linestyle="--",
               label=f"Clean baseline = {baseline_wer:.2f}%")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Global Word Error Rate (WER) %")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_composition(labels, totals, title, out_path):
    """
    Stacked S/I/D composition, row-normalised to 100%.

    Normalising factors OUT magnitude and isolates the SHAPE of the errors, which
    is what distinguishes failure modes: a condition that pushes the model into
    inserting words is failing differently from one that makes it confuse them.

    The insertion bar is easy to misread. Cafe has the lowest insertion share of
    any masker, and that is not because it is gentle — it is the only condition
    that suppresses Whisper's trailing "thank you" hallucination completely, by
    filling the trailing silence with speech-shaped energy. White and pink leave
    that silence intact and so inherit the clean baseline's insertions. Cafe's
    damage is substitutions and deletions. See src/analysis/hallucination.py.
    """
    fields = ("substitutions", "insertions", "deletions")
    pct = {f: [] for f in fields}
    for t in totals:
        total = sum(t[f] for f in fields) or 1
        for f in fields:
            pct[f].append(t[f] / total * 100)

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels) + 2), 6))
    bottom = np.zeros(len(labels))
    for f in fields:
        ax.bar(range(len(labels)), pct[f], bottom=bottom,
               label=f.capitalize(), color=ERROR_COLOURS[f])
        # Percentage inside each segment. Segments under 4% are left unlabelled;
        # there is no room for the text.
        for i, (value, base) in enumerate(zip(pct[f], bottom)):
            if value >= 4.0:
                ax.text(i, base + value / 2, f"{value:.1f}", ha="center",
                        va="center", fontsize=8, color="white")
        bottom += np.array(pct[f])

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Percentage of total errors (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_severity_curve(ax, x, series, baseline_wer, xlabel, title, invert_x=False,
                        annotate=True):
    """
    WER against a severity parameter. One line per family.

    Each series supplies its OWN x values. Sharing one x grid across families
    and padding the gaps with NaN breaks the line wherever a family was not
    measured — the silence panel showed the middle-insertion curve detached from
    its own 0 s anchor for exactly that reason.

    Ticks are pinned to the values actually measured; matplotlib's default
    spacing invented labels at 18/16/14/12 dB, implying data points that do not
    exist.
    """
    for name, entry in series.items():
        xs, values, colour = entry if len(entry) == 3 else (x, entry[0], entry[1])
        ax.plot(xs, values, marker="o", label=name, color=colour)
        # Curves that bunch together (Axis A at high SNR) get no value labels:
        # they would overlap, and the values are tabulated in the report.
        for xi, value in zip(xs, values if annotate else []):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            ax.annotate(f"{value:.2f}", (xi, value), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=9, color=colour)
    ax.axhline(baseline_wer, color=BASELINE_COLOUR, linestyle="--",
               label=f"Clean baseline = {baseline_wer:.2f}%")
    ax.set_xticks(x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Word error rate (%)")
    if title:
        ax.set_title(title, loc="left")
    if invert_x:
        ax.invert_xaxis()  # severity increases left to right
    ax.legend()


# =============================================================================
# MAIN
# =============================================================================

def run_analysis(test_mode: bool):
    logger.info("=" * 50)
    logger.info("Running Statistical & Visual Analysis Pipeline")
    logger.info("=" * 50)

    out_dir = RESULTS_DIR / "test_mode" if test_mode else RESULTS_DIR

    # Plots are written under out_dir so a --test-mode smoke run on one
    # utterance cannot overwrite the published figures.
    plots_dir = out_dir / "plots" / "summary"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        base = read_rows(out_dir / "baseline_results.csv")
        rows_a = read_rows(out_dir / "axis_a_results.csv")
        rows_b = read_rows(out_dir / "axis_b_results.csv")
        rows_c = read_rows(out_dir / "axis_c_results.csv")
        feat = read_rows(out_dir / "feature_deviation.csv")
    except FileNotFoundError as e:
        logger.error(f"Could not find result CSVs: {e}")
        sys.exit(1)

    baseline_wer = global_wer(base)
    logger.info(f"Baseline Global WER: {baseline_wer:.2f}%")

    # ---- Hallucination audit (secondary metric) ----
    logger.info("=== END-OF-AUDIO HALLUCINATION AUDIT ===")
    base_robust = robust_global_wer(base)
    logger.info(
        f"Baseline: raw {baseline_wer:.2f}% | robust {base_robust:.2f}% | "
        f"affected {hallucination_rate(base):.1f}% of utterances"
    )

    halluc_rows = [{
        "condition": "baseline", "raw_wer": round(baseline_wer, 4),
        "robust_wer": round(base_robust, 4),
        "hallucination_pct": round(hallucination_rate(base), 4),
        "delta_vs_baseline_robust": 0.0,
    }]

    def audit(label, rows):
        raw, robust = global_wer(rows), robust_global_wer(rows)
        rate = hallucination_rate(rows)
        logger.info(f"{label:<22} raw {raw:6.2f}% | robust {robust:6.2f}% | "
                    f"hallucinated {rate:5.1f}% | vs clean {robust - base_robust:+6.2f}pp")
        halluc_rows.append({
            "condition": label, "raw_wer": round(raw, 4),
            "robust_wer": round(robust, 4), "hallucination_pct": round(rate, 4),
            "delta_vs_baseline_robust": round(robust - base_robust, 4),
        })

    for noise in NOISE_TYPES:
        for snr in sorted(SNR_LEVELS, reverse=True):
            cell = axis_a_cell(rows_a, noise, snr)
            if cell:
                audit(f"{noise} {snr}dB", cell)
    for rows in (rows_b, rows_c):
        for cond, group in sorted(group_by(rows, lambda r: r["condition"]).items()):
            audit(str(cond), group)

    write_rows(out_dir / "hallucination_audit.csv", halluc_rows)
    logger.info(f"Saved hallucination audit to {out_dir / 'hallucination_audit.csv'}")

    still_below = [r["condition"] for r in halluc_rows[1:]
                   if r["delta_vs_baseline_robust"] < 0]
    if still_below:
        logger.info(
            f"Below clean baseline even after correction: {', '.join(still_below)} "
            f"— check these against the paired confidence intervals before "
            f"reporting them as improvements."
        )

    # =========================================================================
    # AXIS A — additive noise
    # =========================================================================
    snrs_desc = sorted(SNR_LEVELS, reverse=True)

    # ---- A1. Degradation curves (WER vs SNR) ----
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    series = {}
    for noise in NOISE_TYPES:
        values = [global_wer(axis_a_cell(rows_a, noise, s)) for s in snrs_desc]
        series[NOISE_DISPLAY[noise]] = (snrs_desc, values, NOISE_COLOURS.get(noise))
    plot_severity_curve(ax, snrs_desc, series, baseline_wer,
                        "SNR (dB)", "", invert_x=True, annotate=False)
    fig.tight_layout()
    fig.savefig(plots_dir / "degradation_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- A2. Error composition at the worst SNR ----
    worst_snr = min(SNR_LEVELS)
    labels_a = [n for n in NOISE_TYPES if axis_a_cell(rows_a, n, worst_snr)]
    plot_error_composition(
        labels_a,
        [error_totals(axis_a_cell(rows_a, n, worst_snr)) for n in labels_a],
        f"Axis A — error composition by noise type ({worst_snr} dB SNR)",
        plots_dir / f"error_distribution_{worst_snr}db.png",
    )

    # ---- A3. Clean vs Axis A (grouped bars over SNR) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.8 / len(NOISE_TYPES)
    for i, noise in enumerate(NOISE_TYPES):
        values = [global_wer(axis_a_cell(rows_a, noise, s)) for s in snrs_desc]
        offsets = [x + (i - (len(NOISE_TYPES) - 1) / 2) * width for x in range(len(snrs_desc))]
        ax.bar(offsets, values, width, label=noise, color=NOISE_COLOURS.get(noise))
        for xi, value in zip(offsets, values):
            ax.text(xi, value, f"{value:.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)
    ax.axhline(baseline_wer, color=BASELINE_COLOUR, linestyle="--",
               label=f"Clean baseline = {baseline_wer:.2f}%")
    ax.set_xticks(range(len(snrs_desc)))
    ax.set_xticklabels([f"{s} dB" for s in snrs_desc])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Global Word Error Rate (WER) %")
    ax.set_title("Clean vs Axis A (Additive Noise)")
    ax.legend(title="Noise type")
    fig.tight_layout()
    fig.savefig(plots_dir / "clean_vs_axis_a.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- A4. Error composition for every Axis A cell ----
    # All 16 noise x SNR cells, where A2 covers only the worst SNR. Shows how the
    # error shape moves down the SNR range: cafe's insertion share stays flat,
    # white's collapses.
    cells = [(n, s) for n in NOISE_TYPES for s in snrs_desc if axis_a_cell(rows_a, n, s)]
    plot_error_composition(
        [f"{n} {s}dB" for n, s in cells],
        [error_totals(axis_a_cell(rows_a, n, s)) for n, s in cells],
        "Axis A — error composition, every noise type and SNR",
        plots_dir / "error_distribution_axis_a_all.png",
    )

    # =========================================================================
    # AXIS B — channel
    # =========================================================================
    groups_b = group_by(rows_b, lambda r: r["condition"])
    labels_b = sorted(groups_b)

    plot_condition_bars(
        ["Clean baseline"] + labels_b,
        [baseline_wer] + [global_wer(groups_b[c]) for c in labels_b],
        baseline_wer, "Clean vs Axis B (Channel)",
        plots_dir / "clean_vs_axis_b.png",
        colours=["#8c8c8c"] + ["#4c72b0"] * len(labels_b),
    )

    # Axis B has a single condition, so a severity curve would be a single point.
    # The comparison that carries information is against the clean baseline, so
    # the composition plot includes it as the reference column.
    plot_error_composition(
        ["clean baseline"] + labels_b,
        [error_totals(base)] + [error_totals(groups_b[c]) for c in labels_b],
        "Axis B — error composition vs clean",
        plots_dir / "error_distribution_axis_b.png",
    )

    # =========================================================================
    # AXIS C — temporal. Two INDEPENDENT severity parameters, so two curves.
    # The single flat bar chart used previously hid that stretch rate and
    # silence duration are separate sweeps and not one ordered scale.
    # =========================================================================
    groups_c = group_by(rows_c, lambda r: r["condition"])
    labels_c = sorted(groups_c)

    plot_condition_bars(
        ["Clean baseline"] + labels_c,
        [baseline_wer] + [global_wer(groups_c[c]) for c in labels_c],
        baseline_wer, "Clean vs Axis C (Temporal Manipulations)",
        plots_dir / "clean_vs_axis_c.png",
        colours=["#8c8c8c"] + ["#dd8452"] * len(labels_c),
    )

    fig, (ax_rate, ax_sil) = plt.subplots(2, 1, figsize=(6.4, 7.6))

    # Rate sweep — include 1.0x (the clean baseline) as the natural anchor.
    rate_x = [1.0] + list(STRETCH_RATES)
    rate_y = [baseline_wer] + [
        global_wer(groups_c.get(stretch_condition(r), [])) if groups_c.get(stretch_condition(r)) else np.nan
        for r in STRETCH_RATES
    ]
    plot_severity_curve(ax_rate, rate_x, {"Speaking rate": (rate_x, rate_y, "#d62728")},
                        baseline_wer, "Speaking rate (\u00d7)",
                        "(a) Speaking rate")

    # Silence sweep — 0 s is the clean baseline. Start and middle insertions are
    # different manipulations, so they are separate lines rather than one curve.
    by_position = defaultdict(lambda: {0.0: baseline_wer})
    for pos, dur in SILENCE_CONFIGS:
        group = groups_c.get(silence_condition(pos, dur))
        if group:
            by_position[pos][dur] = global_wer(group)

    all_durations = sorted({d for m in by_position.values() for d in m})
    sil_series = {}
    for i, (pos, mapping) in enumerate(sorted(by_position.items())):
        colour = ["#1f77b4", "#2ca02c"][i % 2]
        durations = sorted(mapping)
        sil_series[{"start": "Onset insertion", "middle": "Mid-utterance insertion"}.get(pos, pos)] = (
            durations, [mapping[d] for d in durations], colour
        )
    plot_severity_curve(ax_sil, all_durations, sil_series, baseline_wer,
                        "Inserted silence (s)", "(b) Inserted silence")

    fig.tight_layout()
    fig.savefig(plots_dir / "degradation_curves_axis_c.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot_error_composition(
        ["clean baseline"] + labels_c,
        [error_totals(base)] + [error_totals(groups_c[c]) for c in labels_c],
        "Axis C — error composition vs clean",
        plots_dir / "error_distribution_axis_c.png",
    )

    # =========================================================================
    # Hallucination audit figure
    # =========================================================================
    labels = [display_condition(r["condition"]) for r in halluc_rows]
    positions = range(len(labels))

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 6.4))
    ax_top.bar(positions, [r["hallucination_pct"] for r in halluc_rows], color="#4c72b0")
    ax_top.axhline(halluc_rows[0]["hallucination_pct"], color="black", linestyle="--",
                   label=f"Clean baseline = {halluc_rows[0]['hallucination_pct']:.1f}%")
    ax_top.set_ylabel("Utterances (%)")
    ax_top.set_title("(a) Utterances with a hallucinated insertion", loc="left")
    ax_top.set_xticks(list(positions))
    ax_top.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax_top.legend()

    width = 0.4
    ax_bottom.bar([p - width / 2 for p in positions], [r["raw_wer"] for r in halluc_rows],
                  width, label="As measured", color="#b0b8c4")
    ax_bottom.bar([p + width / 2 for p in positions], [r["robust_wer"] for r in halluc_rows],
                  width, label="Hallucinations removed", color="#2f3b47")
    ax_bottom.axhline(base_robust, color="crimson", linestyle="--",
                      label=f"Clean baseline, hallucinations removed = {base_robust:.2f}%")
    ax_bottom.set_ylabel("Word error rate (%)")
    ax_bottom.set_title("(b) Word error rate with and without hallucinated insertions", loc="left")
    ax_bottom.set_xticks(list(positions))
    ax_bottom.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax_bottom.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "hallucination_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Masking decomposition figure
    #
    # Drawn here rather than in tools/masking_analysis.py so it regenerates with
    # the pipeline. Skipped when the SSN control is absent -- the decomposition
    # is undefined without it, and the remaining figures should still be built.
    # =========================================================================
    if "ssn" in NOISE_TYPES and any(axis_a_cell(rows_a, "ssn", s) for s in SNR_LEVELS):
        decomp = []
        for snr in snrs_desc:
            wer_by_noise = {
                n: global_wer(axis_a_cell(rows_a, n, snr))
                for n in NOISE_TYPES if axis_a_cell(rows_a, n, snr)
            }
            decomp.append((snr, decompose_masking(wer_by_noise, baseline_wer), wer_by_noise))

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: the four maskers as curves, so the reader can see that cafe and
        # SSN separate only at the bottom of the SNR range.
        for noise in NOISE_TYPES:
            ax_left.plot(snrs_desc,
                         [d[2].get(noise, np.nan) for d in decomp],
                         marker="o", label=noise, color=NOISE_COLOURS.get(noise))
        ax_left.axhline(baseline_wer, color=BASELINE_COLOUR, linestyle="--",
                        label=f"Clean baseline = {baseline_wer:.2f}%")
        ax_left.set_xticks(snrs_desc)
        ax_left.invert_xaxis()
        ax_left.set_xlabel("SNR (dB)")
        ax_left.set_ylabel("Global WER %")
        ax_left.set_title("Cafe and SSN are spectrally matched\nand separate only at low SNR")
        ax_left.legend()

        # Right: the decomposition itself, as signed bars.
        x = np.arange(len(snrs_desc))
        width = 0.38
        # np.nan, not 0.0, for a missing term: `or 0.0` would draw a missing
        # measurement as a zero-height bar, which reads as "measured, no effect".
        energetic = [np.nan if d[1]["energetic_wer"] is None else d[1]["energetic_wer"]
                     for d in decomp]
        informational = [np.nan if d[1]["informational_wer"] is None else d[1]["informational_wer"]
                         for d in decomp]
        ax_right.bar(x - width / 2, energetic, width,
                     label="Energetic (SSN \u2212 white)", color="#2ca02c")
        ax_right.bar(x + width / 2, informational, width,
                     label="Informational (cafe babble \u2212 SSN)", color="#1f77b4")
        ax_right.axhline(0, color="black", linewidth=0.8)
        ax_right.set_xticks(x)
        ax_right.set_xticklabels([f"{s} dB" for s in snrs_desc])
        ax_right.set_xlabel("SNR (dB)")
        ax_right.set_ylabel("WER contribution (percentage points)")
        ax_right.set_title("Masking decomposition\n(positive = adds errors)")
        ax_right.legend()
        for offset, values in ((-width / 2, energetic), (width / 2, informational)):
            for xi, value in zip(x + offset, values):
                if np.isnan(value):
                    continue
                ax_right.text(xi, value, f"{value:+.2f}", ha="center",
                              va="bottom" if value >= 0 else "top", fontsize=8)

        fig.tight_layout()
        fig.savefig(plots_dir / "masking_decomposition.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Bars only, no curve panel.
        #
        # The two-panel figure above works standalone, on a slide. In the paper
        # its left panel repeats the degradation-curve figure exactly -- same
        # four maskers, same axes -- so the paper uses this version instead.
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.bar(x - width / 2, energetic, width,
               label="Energetic (SSN \u2212 white)", color="#2ca02c")
        ax.bar(x + width / 2, informational, width,
               label="Informational (cafe babble \u2212 SSN)", color="#1f77b4")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s} dB" for s in snrs_desc])
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Contribution to WER (pp)")
        ax.legend()
        for offset, values in ((-width / 2, energetic), (width / 2, informational)):
            for xi, value in zip(x + offset, values):
                if np.isnan(value):
                    continue
                ax.text(xi, value, f"{value:+.2f}", ha="center",
                        va="bottom" if value >= 0 else "top", fontsize=10)
        fig.tight_layout()
        fig.savefig(plots_dir / "masking_decomposition_bars.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        write_rows(out_dir / "masking_decomposition.csv", [{
            "snr_db": snr,
            "baseline_wer": round(baseline_wer, 4),
            **{f"wer_{n}": round(wers[n], 4) for n in NOISE_TYPES if n in wers},
            **{k: (round(v, 4) if v is not None else "")
               for k, v in d.items() if k.endswith("_wer") or k.endswith("_share")},
        } for snr, d, wers in decomp])

        logger.info("=== MASKING DECOMPOSITION ===")
        for snr, d, _ in decomp:
            logger.info(
                f"{snr:5d}dB  energetic {d['energetic_wer']:+6.2f}pp   "
                f"informational {d['informational_wer']:+6.2f}pp"
            )
    else:
        logger.info("SSN control absent — skipping the masking decomposition figure.")

    # =========================================================================
    # Every condition on one scale
    #
    # The per-axis figures each have their own y-scale, so conditions cannot be
    # compared across them. This plots all 22 conditions plus the baseline
    # together, raw against hallucination-robust.
    # =========================================================================
    ordered = sorted(halluc_rows[1:], key=lambda r: r["raw_wer"])
    labels = [r["condition"] for r in ordered]
    positions = np.arange(len(labels))
    bar_w = 0.4

    fig, ax = plt.subplots(figsize=(max(10, 0.62 * len(labels) + 3), 7))
    raw_bars = ax.bar(positions - bar_w / 2, [r["raw_wer"] for r in ordered],
                      bar_w, label="WER as measured", color="#b0b8c4")
    rob_bars = ax.bar(positions + bar_w / 2, [r["robust_wer"] for r in ordered],
                      bar_w, label="WER, hallucinations removed", color="#2f3b47")
    ax.axhline(baseline_wer, color=BASELINE_COLOUR, linestyle="--",
               label=f"clean baseline (raw) = {baseline_wer:.2f}%")
    ax.axhline(base_robust, color="crimson", linestyle=":",
               label=f"clean baseline (robust) = {base_robust:.2f}%")
    for bars in (raw_bars, rob_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Global Word Error Rate (WER) %")
    ax.set_title("Every condition, ordered by measured WER")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "all_conditions_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Statistical validation
    # =========================================================================
    logger.info("=== STATISTICAL VALIDATION (PEARSON P-VALUES) ===")

    # Inner join on Axis A only: feature_deviation.csv carries no SNR for Axis
    # B/C rows, and those use the other deviation branch, so they are not on a
    # comparable scale (see scripts/04).
    feat_index = {
        (r["utterance_id"], r["noise_type"], str(int(float(r["snr_db"])))): r
        for r in feat if r["snr_db"] not in ("N/A", "")
    }

    merged_x = defaultdict(list)
    merged_y = []
    for row in rows_a:
        key = (row["utterance_id"], row["noise_type"], str(int(float(row["snr_db"]))))
        match = feat_index.get(key)
        if match:
            merged_y.append(float(row["wer"]) * 100)
            for metric in ("mfcc_mse", "ste_mse", "zcr_mse"):
                merged_x[metric].append(float(match[metric]))

    logger.info(f"Merged {len(merged_y)} Axis A rows with feature deviations.")

    if not merged_y:
        # Previously this reported "insufficient data" and exited 0, so a schema
        # drift that collapsed the join looked like a successful run.
        logger.error(
            "Feature/WER merge produced ZERO rows. axis_a_results.csv and "
            "feature_deviation.csv do not share keys — re-run stage 04."
        )
        sys.exit(1)

    if len(merged_y) <= 2:
        logger.warning(f"Only {len(merged_y)} merged rows — too few for a correlation.")
    else:
        # Pearson AND Spearman. Reporting Pearson alone was misleading: it tests
        # a LINEAR relationship, and the relationship here is monotonic but not
        # linear, so Pearson reads ~0 while Spearman is significant.
        logger.info(f"{'metric':<12}{'pearson r':>11}{'p':>11}{'spearman':>11}{'p':>11}")
        for metric in ("mfcc_mse", "ste_mse", "zcr_mse"):
            r_p, p_p = pearsonr(merged_x[metric], merged_y)
            r_s, p_s = spearmanr(merged_x[metric], merged_y)
            logger.info(f"{metric:<12}{r_p:11.3f}{p_p:11.3g}{r_s:11.3f}{p_s:11.3g}")

        # The pooled figure hides a sign reversal. Within a fixed noise type the
        # correlation is POSITIVE (more spectral distortion -> more errors);
        # BETWEEN noise types it is negative, because white noise produces
        # enormous MFCC deviation for moderate WER while cafe produces very
        # little for the worst WER in the study. Pooling cancels the two, which
        # is a Simpson's paradox and is the actual finding — a spectral-distance
        # feature is informative within a masker and misleading across maskers.
        logger.info("--- within each noise type (removes the between-noise reversal) ---")
        logger.info(f"{'noise':<10}{'n':>6}{'mfcc r':>10}{'p':>11}{'ste r':>10}{'p':>11}")
        for noise in NOISE_TYPES:
            xs_mfcc, xs_ste, ys = [], [], []
            for row in rows_a:
                if row["noise_type"] != noise:
                    continue
                key = (row["utterance_id"], noise, str(int(float(row["snr_db"]))))
                match = feat_index.get(key)
                if match:
                    xs_mfcc.append(float(match["mfcc_mse"]))
                    xs_ste.append(float(match["ste_mse"]))
                    ys.append(float(row["wer"]) * 100)
            if len(ys) > 2:
                r1, p1 = pearsonr(xs_mfcc, ys)
                r2, p2 = pearsonr(xs_ste, ys)
                logger.info(f"{noise:<10}{len(ys):>6}{r1:10.3f}{p1:11.3g}{r2:10.3f}{p2:11.3g}")

        # Mean deviation against WER per condition makes the reversal concrete.
        logger.info("--- mean MFCC deviation vs WER, worst SNR + Axis B ---")
        cond_means = defaultdict(list)
        for r in feat:
            cond_means[r["condition"]].append(float(r["mfcc_mse"]))
        pairs = []
        for noise in NOISE_TYPES:
            cell = axis_a_cell(rows_a, noise, worst_snr)
            key = f"axis_a_{noise}_{worst_snr}"
            if cell and key in cond_means:
                pairs.append((f"{noise} {worst_snr}dB",
                              float(np.mean(cond_means[key])), global_wer(cell)))
        if rows_b and "axis_b_telephonic" in cond_means:
            pairs.append(("telephonic",
                          float(np.mean(cond_means["axis_b_telephonic"])), global_wer(rows_b)))
        for name, mse, wer in sorted(pairs, key=lambda t: -t[1]):
            logger.info(f"   {name:<16} mfcc_mse {mse:8.1f}   WER {wer:5.2f}%")

    logger.info(f"Visualizations saved to {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    run_analysis(args.test_mode)
