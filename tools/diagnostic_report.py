"""
Diagnostic dump of the result CSVs.

An interactive investigation tool, not a pipeline stage. Prints (rather than
logs) because it is meant to be read at a terminal.

Two things it is specifically useful for:
  * Printing mean-of-ratios WER alongside corpus-level WER for every condition,
    which makes the gap between the two definitions concrete.
  * Showing the jiwer alignment for a few rows, so the substitution / insertion
    / deletion attribution can be eyeballed rather than trusted.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import glob
import time
from collections import defaultdict

import jiwer

from config import RESULTS_DIR
from src.analysis.metrics import global_wer, mean_utterance_wer
from src.data.text_normalizer import normalize_text
from src.results_io import read_rows


def summarize(label, rows):
    print(f"{label} | count: {len(rows):>5} | "
          f"mean WER: {mean_utterance_wer(rows):6.2f}% | "
          f"global WER: {global_wer(rows):6.2f}%")


def main():
    print("--- 1. RAW DATA DUMP ---")

    baseline_path = RESULTS_DIR / "baseline_results.csv"
    axis_a_path = RESULTS_DIR / "axis_a_results.csv"
    axis_b_path = RESULTS_DIR / "axis_b_results.csv"
    axis_c_path = RESULTS_DIR / "axis_c_results.csv"

    for path in (baseline_path, axis_a_path, axis_b_path, axis_c_path):
        if not path.exists():
            print(f"[ERROR] Missing {path}. Run the pipeline first.")
            sys.exit(1)
        print(f"{path.name:<24} last modified: {time.ctime(os.path.getmtime(path))}")

    print("\n--- 2. WER BY CONDITION (mean-of-ratios vs corpus-level) ---")

    base_data = read_rows(baseline_path)
    summarize("BASELINE                ", base_data)

    a_data = read_rows(axis_a_path)
    a_grouped = defaultdict(list)
    for r in a_data:
        a_grouped[(r["noise_type"], int(float(r["snr_db"])))].append(r)
    for (noise, snr) in sorted(a_grouped):
        summarize(f"AXIS A {noise:>5} {snr:>2}dB      ", a_grouped[(noise, snr)])

    for path in (axis_b_path, axis_c_path):
        grouped = defaultdict(list)
        for r in read_rows(path):
            grouped[r["condition"]].append(r)
        for cond in sorted(grouped):
            summarize(f"{cond:<24}", grouped[cond])

    print("\n--- 3. VERIFY ERROR COMPUTATION LOGIC ---")
    worst = [r for r in a_data
             if r["noise_type"] == "cafe" and int(float(r["snr_db"])) == 5][:3]
    for row in worst:
        print(f"\nUtterance: {row['utterance_id']}")
        gt = normalize_text(row["ground_truth"])
        pt = normalize_text(row["predicted_text"])
        print(f"Ground Truth (norm): {gt}")
        print(f"Predicted (norm):    {pt}")
        out = jiwer.process_words(gt, pt)
        print(jiwer.visualize_alignment(out))
        print(f"S: {out.substitutions}  I: {out.insertions}  "
              f"D: {out.deletions}  WER: {out.wer * 100:.2f}%")

    print("\n--- 4. CHECK FOR STRAY RESULT FILES ---")
    # A duplicate axis_a_results.csv elsewhere in the tree would mean the
    # analysis and the inference stages were reading different data.
    stray = [p for p in glob.glob("**/axis_a_results.csv", recursive=True)
             if "venv" not in p]
    print(f"Found {len(stray)} copy/copies: {stray}")
    if len(stray) > 1:
        print("[WARN] More than one results file exists — check which one the "
              "analysis stage is actually reading.")


if __name__ == "__main__":
    main()