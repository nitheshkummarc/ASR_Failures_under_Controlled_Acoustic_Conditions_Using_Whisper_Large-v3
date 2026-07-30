"""
06_verify_results.py — Post-experiment verification for Phase 1 & Phase 2.
==========================================================================
Validates baseline and Axis A results for correctness, completeness,
and core hypothesis confirmation before proceeding to further analysis.

Run this AFTER:
  - 03_baseline_inference.py  → results/baseline_results.csv
  - 05_axis_a_inference_and_eval.py → results/axis_a_results.csv
"""

import sys
import os

import pandas as pd
import numpy as np

from config import RESULTS_DIR, AXIS_A_DIR, NUM_UTTERANCES, NOISE_TYPES, SNR_LEVELS


def section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def warn(msg: str):
    """Print a clearly visible warning."""
    print(f"\n  ⚠ WARNING: {msg}")


def ok(msg: str):
    """Print a success message."""
    print(f"  ✓ {msg}")


# =============================================================================
# CHECK 0: Verify degraded WAV file count
# =============================================================================
def check_degraded_file_count():
    section("CHECK 0: Degraded WAV File Count")

    if not AXIS_A_DIR.exists():
        warn(f"Directory not found: {AXIS_A_DIR}")
        return False

    wav_files = list(AXIS_A_DIR.glob("*.wav"))
    n_files = len(wav_files)
    expected = NUM_UTTERANCES * len(NOISE_TYPES) * len(SNR_LEVELS)  # 150 × 3 × 4 = 1800

    print(f"\n  Directory: {AXIS_A_DIR}")
    print(f"  WAV files found: {n_files}")
    print(f"  Expected:        {expected}")

    if n_files != expected:
        warn(f"Expected {expected} WAV files, found {n_files}")
        return False

    ok(f"Exactly {expected} degraded WAV files present")
    return True


# =============================================================================
# CHECK 1: Baseline results — 150 rows, corpus WER < 5%
# =============================================================================
def check_baseline():
    section("CHECK 1: Baseline Results (Clean Speech)")

    baseline_path = RESULTS_DIR / "baseline_results.csv"
    if not baseline_path.exists():
        warn(f"File not found: {baseline_path}")
        return False

    df = pd.read_csv(baseline_path)
    passed = True

    # Row count
    print(f"\n  Rows: {len(df)}  (expected: {NUM_UTTERANCES})")
    if len(df) != NUM_UTTERANCES:
        warn(f"Expected {NUM_UTTERANCES} rows, got {len(df)}")
        passed = False
    else:
        ok(f"Exactly {NUM_UTTERANCES} rows")

    # Corpus-level WER
    total_errors = df["substitutions"].sum() + df["insertions"].sum() + df["deletions"].sum()
    total_ref_words = df["word_count_ref"].sum()
    corpus_wer = total_errors / total_ref_words if total_ref_words > 0 else 0

    print(f"  Corpus WER: {corpus_wer:.4f} ({corpus_wer * 100:.2f}%)")
    if corpus_wer >= 0.07:
        warn(f"Corpus WER = {corpus_wer:.2%} — exceeds 7% threshold! "
             "Check normalization alignment or model output.")
        passed = False
    else:
        ok(f"Corpus WER = {corpus_wer:.2%} — below 7% threshold")

    # Spot-check 3 random rows
    print(f"\n  {'─' * 66}")
    print("  Spot-check: 3 random baseline utterances")
    print(f"  {'─' * 66}")
    sample = df.sample(3, random_state=42)
    for _, row in sample.iterrows():
        print(f"\n  [{row['utterance_id']}]  WER={row['wer']:.4f}  "
              f"S={row['substitutions']} I={row['insertions']} D={row['deletions']}")
        gt = row['ground_truth']
        pred = row['predicted_text']
        print(f"    GT:   {gt[:90]}{'...' if len(str(gt)) > 90 else ''}")
        print(f"    PRED: {pred[:90]}{'...' if len(str(pred)) > 90 else ''}")
        if row['wer'] == 0.0:
            print(f"    → Perfect transcription ✓")
        elif row['wer'] < 0.1:
            print(f"    → Minor errors (WER < 10%) — acceptable for clean speech")
        else:
            print(f"    → Notable errors — worth manual inspection")

    return passed


# =============================================================================
# CHECK 2: Axis A results — 1800 rows, no nulls
# =============================================================================
def check_axis_a_completeness():
    section("CHECK 2: Axis A Completeness")

    results_path = RESULTS_DIR / "axis_a_results.csv"
    if not results_path.exists():
        warn(f"File not found: {results_path}")
        return None

    df = pd.read_csv(results_path)
    expected = NUM_UTTERANCES * len(NOISE_TYPES) * len(SNR_LEVELS)
    passed = True

    # Row count
    print(f"\n  Rows: {len(df)}  (expected: {expected})")
    if len(df) != expected:
        warn(f"Expected {expected} rows, got {len(df)}")
        passed = False
    else:
        ok(f"Exactly {expected} rows")

    # Null checks
    wer_nulls = df["wer"].isna().sum()
    cer_nulls = df["cer"].isna().sum()
    print(f"  WER nulls: {wer_nulls}  |  CER nulls: {cer_nulls}")
    if wer_nulls > 0 or cer_nulls > 0:
        warn(f"Found null values — WER: {wer_nulls}, CER: {cer_nulls}")
        passed = False
    else:
        ok("No null values in WER or CER columns")

    # Verify all expected conditions are present
    for nt in NOISE_TYPES:
        for snr in SNR_LEVELS:
            count = len(df[(df["noise_type"] == nt) & (df["snr_db"] == snr)])
            if count != NUM_UTTERANCES:
                warn(f"{nt}/{snr}dB: expected {NUM_UTTERANCES} rows, got {count}")
                passed = False

    if passed:
        ok("All noise_type × snr_db combinations have correct row counts")

    return df if passed or len(df) > 0 else None


# =============================================================================
# CHECK 3: WER summary table (noise_type × snr_db)
# =============================================================================
def print_wer_summary(df: pd.DataFrame):
    section("CHECK 3: Average WER by Noise Type × SNR Level")

    print(f"\n  {'Noise Type':<12} {'20 dB':>8} {'15 dB':>8} {'10 dB':>8} {'5 dB':>8}")
    print(f"  {'─' * 48}")

    for noise_type in NOISE_TYPES:
        row_str = f"  {noise_type:<12}"
        for snr in SNR_LEVELS:
            subset = df[(df["noise_type"] == noise_type) & (df["snr_db"] == snr)]
            if len(subset) > 0:
                # Corpus-level WER (total errors / total words)
                total_err = subset["subs"].sum() + subset["ins"].sum() + subset["dels"].sum()
                total_ref = subset["word_count_ref"].sum()
                corpus_wer = total_err / total_ref if total_ref > 0 else 0
                row_str += f" {corpus_wer:>7.2%}"
            else:
                row_str += f" {'N/A':>7}"
        print(row_str)

    print(f"  {'─' * 48}")

    # Also print mean per-utterance WER for comparison
    print(f"\n  (Mean per-utterance WER for reference:)")
    print(f"  {'Noise Type':<12} {'20 dB':>8} {'15 dB':>8} {'10 dB':>8} {'5 dB':>8}")
    print(f"  {'─' * 48}")

    for noise_type in NOISE_TYPES:
        row_str = f"  {noise_type:<12}"
        for snr in SNR_LEVELS:
            subset = df[(df["noise_type"] == noise_type) & (df["snr_db"] == snr)]
            if len(subset) > 0:
                row_str += f" {subset['wer'].mean():>7.2%}"
            else:
                row_str += f" {'N/A':>7}"
        print(row_str)

    print(f"  {'─' * 48}")


# =============================================================================
# CHECK 4: Core hypothesis — WER(5dB) > WER(20dB) for each noise type
# =============================================================================
def check_degradation_hypothesis(df: pd.DataFrame):
    section("CHECK 4: Core Hypothesis — WER Increases with Lower SNR")
    print("\n  Hypothesis: For each noise type, avg WER at 5 dB > avg WER at 20 dB")
    print(f"  {'─' * 62}")

    all_passed = True

    for noise_type in NOISE_TYPES:
        # Corpus-level WER at 5 dB
        sub_5 = df[(df["noise_type"] == noise_type) & (df["snr_db"] == 5)]
        err_5 = sub_5["subs"].sum() + sub_5["ins"].sum() + sub_5["dels"].sum()
        ref_5 = sub_5["word_count_ref"].sum()
        wer_5 = err_5 / ref_5 if ref_5 > 0 else 0

        # Corpus-level WER at 20 dB
        sub_20 = df[(df["noise_type"] == noise_type) & (df["snr_db"] == 20)]
        err_20 = sub_20["subs"].sum() + sub_20["ins"].sum() + sub_20["dels"].sum()
        ref_20 = sub_20["word_count_ref"].sum()
        wer_20 = err_20 / ref_20 if ref_20 > 0 else 0

        delta = wer_5 - wer_20

        if wer_5 > wer_20:
            print(f"  ✓ {noise_type:<8}  WER@5dB={wer_5:.2%}  >  "
                  f"WER@20dB={wer_20:.2%}  (Δ = +{delta:.2%})")
        else:
            warn(f"{noise_type}: WER@5dB={wer_5:.2%} is NOT greater than "
                 f"WER@20dB={wer_20:.2%} — hypothesis FAILED for this noise type!")
            all_passed = False

    print()
    if all_passed:
        ok("Core hypothesis confirmed: WER strictly increases from 20 dB → 5 dB "
           "for all noise types")
    else:
        warn("Core hypothesis FAILED for one or more noise types — investigate!")

    return all_passed


# =============================================================================
# CHECK 5: S/I/D error profile at 5 dB per noise type
# =============================================================================
def print_error_profiles(df: pd.DataFrame):
    section("CHECK 5: Error Profiles at 5 dB (S/I/D Breakdown)")
    print("\n  Do different noise types produce different error patterns?")
    print("  (e.g., more insertions in café babble vs. more deletions in white)")
    print(f"\n  {'Noise Type':<12} {'Subs':>8} {'Ins':>8} {'Dels':>8} "
          f"{'S%':>7} {'I%':>7} {'D%':>7} {'Total':>8}")
    print(f"  {'─' * 72}")

    for noise_type in NOISE_TYPES:
        subset = df[(df["noise_type"] == noise_type) & (df["snr_db"] == 5)]
        if len(subset) == 0:
            print(f"  {noise_type:<12} {'(no data)':>8}")
            continue

        s = int(subset["subs"].sum())
        i = int(subset["ins"].sum())
        d = int(subset["dels"].sum())
        total = s + i + d

        if total > 0:
            s_pct = s / total * 100
            i_pct = i / total * 100
            d_pct = d / total * 100
        else:
            s_pct = i_pct = d_pct = 0

        print(f"  {noise_type:<12} {s:>8} {i:>8} {d:>8} "
              f"{s_pct:>6.1f}% {i_pct:>6.1f}% {d_pct:>6.1f}% {total:>8}")

    print(f"  {'─' * 72}")

    # Interpretation hints (academic framing per proposal §8)
    print(f"\n  Interpretation notes (phrase as observations, not causal claims):")
    print(f"  • If Café shows higher insertion % → 'associated with' decoder")
    print(f"    hallucinating on multi-talker background")
    print(f"  • If White shows higher deletion % → 'associated with' masking")
    print(f"    of speech segments by broadband noise")
    print(f"  • If Pink shows higher substitution % → 'associated with' spectral")
    print(f"    overlap in lower frequencies where formant transitions occur")


# =============================================================================
# BONUS: Compare same utterance at 5 dB vs 20 dB
# =============================================================================
def spot_check_snr_comparison(df: pd.DataFrame):
    section("BONUS: Same Utterance at 5 dB vs 20 dB (Spot-Check)")

    # Pick one utterance present in all conditions
    utterance_ids = df["utterance_id"].unique()
    sample_uid = utterance_ids[0]

    for noise_type in NOISE_TYPES:
        row_20 = df[(df["utterance_id"] == sample_uid) &
                     (df["noise_type"] == noise_type) &
                     (df["snr_db"] == 20)]
        row_5 = df[(df["utterance_id"] == sample_uid) &
                    (df["noise_type"] == noise_type) &
                    (df["snr_db"] == 5)]

        if len(row_20) == 0 or len(row_5) == 0:
            continue

        r20 = row_20.iloc[0]
        r5 = row_5.iloc[0]

        print(f"\n  [{sample_uid}] — {noise_type.upper()} noise:")
        print(f"    GT:       {str(r20['ground_truth'])[:80]}...")
        print(f"    @20dB:    {str(r20['predicted_text'])[:80]}...")
        print(f"              WER={r20['wer']:.4f}  S={r20['subs']} I={r20['ins']} D={r20['dels']}")
        print(f"    @5dB:     {str(r5['predicted_text'])[:80]}...")
        print(f"              WER={r5['wer']:.4f}  S={r5['subs']} I={r5['ins']} D={r5['dels']}")

        wer_diff = r5['wer'] - r20['wer']
        if wer_diff > 0:
            print(f"    → WER increased by {wer_diff:.4f} (expected ✓)")
        elif wer_diff == 0:
            print(f"    → WER unchanged (model equally robust at both SNRs)")
        else:
            print(f"    → WER decreased at lower SNR — unusual, worth investigating")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("╔" + "═" * 68 + "╗")
    print("║  06_verify_results.py                                             ║")
    print("║  Post-Experiment Verification: Phase 1 (Baseline) + Phase 2 (A)   ║")
    print("╚" + "═" * 68 + "╝")

    all_ok = True

    # Check 0: Degraded file count
    if not check_degraded_file_count():
        all_ok = False

    # Check 1: Baseline
    if not check_baseline():
        all_ok = False

    # Check 2: Axis A completeness
    axis_a_df = check_axis_a_completeness()
    if axis_a_df is None:
        all_ok = False

    if axis_a_df is not None:
        # Check 3: WER summary table
        print_wer_summary(axis_a_df)

        # Check 4: Core hypothesis
        if not check_degradation_hypothesis(axis_a_df):
            all_ok = False

        # Check 5: Error profiles at 5 dB
        print_error_profiles(axis_a_df)

        # Bonus: Spot-check comparison
        spot_check_snr_comparison(axis_a_df)

    # Final verdict
    section("FINAL VERDICT")
    if all_ok:
        print("\n  ✅ ALL CHECKS PASSED")
        print("  Phase 1 and Phase 2 (Axis A) results are verified and consistent.")
        print("  You are ready to proceed to Phase 3 (Axis B / feature extraction).")
    else:
        print("\n  ❌ SOME CHECKS FAILED")
        print("  Review the warnings above and debug before proceeding.")

    print()


if __name__ == "__main__":
    main()
