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
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: no display needed, and avoids a Tk dependency
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from config import (
    RESULTS_DIR, SNR_LEVELS, NOISE_TYPES, AXIS_B_CONDITION,
    STRETCH_RATES, SILENCE_CONFIGS, stretch_condition, silence_condition,
)
from src.analysis.metrics import global_wer
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

def read_csv(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    """WER per condition with the clean baseline drawn as a reference line."""
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels) + 2), 6))
    ax.bar(range(len(labels)), values,
           color=colours or ["#4c72b0"] * len(labels))
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


def plot_severity_curve(ax, x, series, baseline_wer, xlabel, title, invert_x=False):
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
    ax.axhline(baseline_wer, color=BASELINE_COLOUR, linestyle="--",
               label=f"Clean baseline = {baseline_wer:.2f}%")
    ax.set_xticks(x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Global WER %")
    ax.set_title(title)
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
        base = read_csv(out_dir / "baseline_results.csv")
        rows_a = read_csv(out_dir / "axis_a_results.csv")
        rows_b = read_csv(out_dir / "axis_b_results.csv")
        rows_c = read_csv(out_dir / "axis_c_results.csv")
        feat = read_csv(out_dir / "feature_deviation.csv")
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

    write_csv(out_dir / "hallucination_audit.csv", halluc_rows)
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
    fig, ax = plt.subplots(figsize=(10, 6))
    series = {}
    for noise in NOISE_TYPES:
        values = [global_wer(axis_a_cell(rows_a, noise, s)) for s in snrs_desc]
        series[noise] = (snrs_desc, values, NOISE_COLOURS.get(noise))
    plot_severity_curve(ax, snrs_desc, series, baseline_wer,
                        "SNR (dB)", "Axis A — Global WER vs SNR", invert_x=True)
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

    fig, (ax_rate, ax_sil) = plt.subplots(1, 2, figsize=(14, 6))

    # Rate sweep — include 1.0x (the clean baseline) as the natural anchor.
    rate_x = [1.0] + list(STRETCH_RATES)
    rate_y = [baseline_wer] + [
        global_wer(groups_c.get(stretch_condition(r), [])) if groups_c.get(stretch_condition(r)) else np.nan
        for r in STRETCH_RATES
    ]
    plot_severity_curve(ax_rate, rate_x, {"speaking rate": (rate_x, rate_y, "#d62728")},
                        baseline_wer, "Speaking rate (x)",
                        "Axis C — WER vs speaking rate")

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
        sil_series[f"{pos} insertion"] = (
            durations, [mapping[d] for d in durations], colour
        )
    plot_severity_curve(ax_sil, all_durations, sil_series, baseline_wer,
                        "Inserted silence (s)", "Axis C — WER vs silence duration")

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
    labels = [r["condition"] for r in halluc_rows]
    positions = range(len(labels))

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(15, 9))
    ax_top.bar(positions, [r["hallucination_pct"] for r in halluc_rows], color="#4c72b0")
    ax_top.axhline(halluc_rows[0]["hallucination_pct"], color="black", linestyle="--",
                   label=f"clean baseline = {halluc_rows[0]['hallucination_pct']:.1f}%")
    ax_top.set_ylabel("Utterances with a trailing\nhallucination (%)")
    ax_top.set_title("Trailing-hallucination rate by condition")
    ax_top.set_xticks(list(positions))
    ax_top.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_top.legend()

    width = 0.4
    ax_bottom.bar([p - width / 2 for p in positions], [r["raw_wer"] for r in halluc_rows],
                  width, label="WER as measured", color="#b0b8c4")
    ax_bottom.bar([p + width / 2 for p in positions], [r["robust_wer"] for r in halluc_rows],
                  width, label="WER with hallucinations removed", color="#2f3b47")
    ax_bottom.axhline(base_robust, color="crimson", linestyle="--",
                      label=f"corrected clean baseline = {base_robust:.2f}%")
    ax_bottom.set_ylabel("Global WER (%)")
    ax_bottom.set_title("Removing the trailing phrase lifts every condition that exhibits it")
    ax_bottom.set_xticks(list(positions))
    ax_bottom.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_bottom.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "hallucination_analysis.png", dpi=300, bbox_inches="tight")
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
        for metric in ("mfcc_mse", "ste_mse", "zcr_mse"):
            r, p = pearsonr(merged_x[metric], merged_y)
            sig = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
            logger.info(f"{metric.upper()} vs WER: r = {r:.3f}, p = {p:.3e} ({sig})")

    logger.info(f"Visualizations saved to {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    run_analysis(args.test_mode)
