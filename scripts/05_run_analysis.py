"""
Stage 5 — Statistical analysis and summary visualisations.

Aggregates the result CSVs into corpus-level WER, draws the five summary
figures, and tests whether feature-space deviation predicts WER.

Reported numbers are GLOBAL (corpus) WER from src.analysis.metrics — total
errors over total reference words — not the mean of the per-utterance `wer`
column. See that module for why the two differ.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import warnings

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: no display needed, and avoids a Tk dependency
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from config import RESULTS_DIR, SNR_LEVELS
from src.analysis.metrics import global_wer_df
from src.logger import setup_logger

logger = setup_logger("scripts.05_run_analysis")

# Targeted rather than blanket suppression, so genuine numerical warnings
# (e.g. scipy's ConstantInputWarning from pearsonr) still surface.
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


def run_analysis(test_mode: bool):
    logger.info("=" * 50)
    logger.info("Running Statistical & Visual Analysis Pipeline")
    logger.info("=" * 50)

    out_dir = RESULTS_DIR / "test_mode" if test_mode else RESULTS_DIR

    # Plots are written under out_dir so a --test-mode smoke run on one
    # utterance cannot overwrite the published figures, which is what happened
    # when plots_dir was hardcoded to the production directory.
    plots_dir = out_dir / "plots" / "summary"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        df_base = pd.read_csv(out_dir / "baseline_results.csv")
        df_a = pd.read_csv(out_dir / "axis_a_results.csv")
        df_b = pd.read_csv(out_dir / "axis_b_results.csv")
        df_c = pd.read_csv(out_dir / "axis_c_results.csv")
        df_feat = pd.read_csv(out_dir / "feature_deviation.csv")
    except FileNotFoundError as e:
        logger.error(f"Could not find result CSVs: {e}")
        sys.exit(1)

    baseline_wer = global_wer_df(df_base)
    logger.info(f"Baseline Global WER: {baseline_wer:.2f}%")

    for df in (df_a, df_b, df_c):
        df["wer_pct"] = df["wer"] * 100

    # Global WER per (noise_type, snr) cell
    axis_a_global = [
        {"noise_type": noise, "snr_db": snr, "wer_pct": global_wer_df(group)}
        for (noise, snr), group in df_a.groupby(["noise_type", "snr_db"])
    ]
    df_a_global = pd.DataFrame(axis_a_global)

    # ---- 1. Degradation curves (WER vs SNR) ----
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_a_global, x="snr_db", y="wer_pct", hue="noise_type", marker="o")
    plt.axhline(baseline_wer, color="black", linestyle="--", label="Clean Baseline")
    plt.title("Degradation Curves: Global WER vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Global Word Error Rate (WER) %")
    plt.gca().invert_xaxis()  # severity increases left to right
    plt.legend()
    plt.savefig(plots_dir / "degradation_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- 2. Error composition at the worst SNR, row-normalized to 100% ----
    worst_snr = min(SNR_LEVELS)
    df_worst = df_a[df_a["snr_db"] == worst_snr]
    error_sums = df_worst.groupby("noise_type")[["substitutions", "insertions", "deletions"]].sum()
    # Normalizing each row factors OUT magnitude, isolating the SHAPE of the
    # errors — which is what separates cafe's insertion-heavy (hallucination)
    # signature from white/pink's substitution-heavy one.
    error_pct = error_sums.div(error_sums.sum(axis=1), axis=0) * 100
    error_pct.plot(kind="bar", stacked=True, figsize=(8, 6), colormap="viridis")
    plt.title(f"Constituent Error Distribution by Noise Type ({worst_snr}dB SNR)")
    plt.ylabel("Percentage of Total Errors (%)")
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(plots_dir / f"error_distribution_{worst_snr}db.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- 3. Clean vs Axis B ----
    b_global = [
        {"condition": cond, "wer_pct": global_wer_df(group)}
        for cond, group in df_b.groupby("condition")
    ]
    mean_b = pd.DataFrame([{"condition": "Clean Baseline", "wer_pct": baseline_wer}] + b_global)

    plt.figure(figsize=(6, 6))
    sns.barplot(data=mean_b, x="condition", y="wer_pct", hue="condition",
                palette="Blues", legend=False)
    plt.title("Clean vs Axis B (Channel)")
    plt.ylabel("Global Word Error Rate (WER) %")
    plt.savefig(plots_dir / "clean_vs_axis_b.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- 4. Clean vs Axis C ----
    c_global = [
        {"condition": cond, "wer_pct": global_wer_df(group)}
        for cond, group in df_c.groupby("condition")
    ]
    mean_c = pd.DataFrame([{"condition": "Clean Baseline", "wer_pct": baseline_wer}] + c_global)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=mean_c, x="condition", y="wer_pct", hue="condition",
                palette="Set2", legend=False)
    plt.title("Clean vs Axis C (Temporal Manipulations)")
    plt.ylabel("Global Word Error Rate (WER) %")
    plt.xticks(rotation=45)
    plt.savefig(plots_dir / "clean_vs_axis_c.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- 5. Clean vs Axis A (grouped bars) ----
    a_bars = axis_a_global + [{"noise_type": "Baseline", "snr_db": "Clean", "wer_pct": baseline_wer}]
    df_a_bars = pd.DataFrame(a_bars)
    df_a_bars["snr_db"] = pd.Categorical(
        df_a_bars["snr_db"],
        categories=["Clean"] + sorted(SNR_LEVELS, reverse=True),
        ordered=True,
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_a_bars, x="snr_db", y="wer_pct", hue="noise_type", palette="magma")
    plt.title("Clean vs Axis A (Additive Noise)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Global Word Error Rate (WER) %")
    plt.legend(title="Noise Type")
    plt.savefig(plots_dir / "clean_vs_axis_a.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Statistical validation ----
    logger.info("=== STATISTICAL VALIDATION (PEARSON P-VALUES) ===")

    # Inner join keeps Axis A only: feature_deviation.csv carries NaN snr_db for
    # Axis B/C rows, and those rows use the other deviation branch, so they are
    # not on a comparable scale (see scripts/04).
    df_merged = pd.merge(df_a, df_feat, on=["utterance_id", "noise_type", "snr_db"], how="inner")
    logger.info(f"Merged {len(df_merged)} Axis A rows with feature deviations.")

    if len(df_merged) == 0:
        # Previously this reported "insufficient data" and exited 0, so a schema
        # drift that collapsed the join looked like a successful run.
        logger.error(
            "Feature/WER merge produced ZERO rows. axis_a_results.csv and "
            "feature_deviation.csv do not share keys — re-run stage 04."
        )
        sys.exit(1)

    if len(df_merged) <= 2:
        logger.warning(f"Only {len(df_merged)} merged rows — too few for a meaningful correlation.")
    else:
        for metric in ["mfcc_mse", "ste_mse", "zcr_mse"]:
            r, p = pearsonr(df_merged[metric], df_merged["wer_pct"])
            sig = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
            logger.info(f"{metric.upper()} vs WER: r = {r:.3f}, p = {p:.3e} ({sig})")

    logger.info(f"Visualizations saved to {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    run_analysis(args.test_mode)