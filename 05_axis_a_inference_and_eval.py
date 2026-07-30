"""
05_axis_a_inference_and_eval.py — Axis A: Whisper Inference & Evaluation
=========================================================================
This script:
  1. Loads axis_a_manifest.csv (1,800 degraded utterances).
  2. Runs faster-whisper (large-v3, CUDA/float16) on every degraded file.
  3. Applies the EXACT SAME normalize_text() pipeline from Phase 1 to
     both ground truth and predictions (identical to baseline).
  4. Computes per-utterance WER, CER, Substitution/Insertion/Deletion
     counts using jiwer.
  5. Saves results/axis_a_results.csv.
  6. Prints per-noise-type × per-SNR summary tables for immediate analysis.

Build order (proposal §9): Results are printed White → Pink → Café so
White noise can be fully validated before inspecting the others.
"""

import sys
import time
import importlib
import gc

import pandas as pd
import numpy as np
import jiwer
from tqdm import tqdm

from config import (
    METADATA_DIR, RESULTS_DIR,
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE, WHISPER_LANGUAGE,
    NOISE_TYPES, SNR_LEVELS,
)

# Import the shared normalization function (same pipeline as Phase 1 baseline)
_norm_module = importlib.import_module("02_normalize_text")
normalize_text = _norm_module.normalize_text


# =============================================================================
# WHISPER MODEL INITIALIZATION (identical to 03_baseline_inference.py)
# =============================================================================

def load_whisper_model():
    """
    Initialize faster-whisper with OOM fallback chain:
      1. CUDA / float16  (primary — RTX 3050)
      2. CUDA / int8      (if float16 OOMs)
      3. CPU  / int8      (last resort)
    """
    from faster_whisper import WhisperModel

    print(f"  [INIT] Loading faster-whisper: {WHISPER_MODEL}")
    print(f"         Device: {WHISPER_DEVICE}, Compute: {WHISPER_COMPUTE_TYPE}")

    try:
        model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("  [OK]   Model loaded on CUDA/float16")
        return model

    except Exception as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "oom" in error_msg or "cuda" in error_msg:
            print(f"\n  [WARN] CUDA/float16 failed: {e}")
            print("  [FALL] Retrying with compute_type='int8'...")
            try:
                model = WhisperModel(
                    WHISPER_MODEL,
                    device=WHISPER_DEVICE,
                    compute_type="int8",
                )
                print("  [OK]   Model loaded on CUDA/int8")
                return model
            except Exception:
                print("  [FALL] Final fallback: CPU/int8...")
                model = WhisperModel(
                    WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8",
                )
                print("  [OK]   Model loaded on CPU/int8 (slower)")
                return model
        else:
            raise


# =============================================================================
# TRANSCRIPTION & METRICS (same logic as baseline)
# =============================================================================

def transcribe_utterance(model, wav_path: str) -> str:
    """Transcribe a single WAV file and return predicted text."""
    segments, _info = model.transcribe(
        wav_path,
        beam_size=WHISPER_BEAM_SIZE,
        language=WHISPER_LANGUAGE,
    )
    predicted = " ".join(seg.text for seg in segments)
    return predicted.strip()


def compute_metrics(reference: str, hypothesis: str) -> dict:
    """
    Compute WER, CER, and S/I/D counts via jiwer.

    Both strings must already be normalized (lowercase, no punctuation,
    numbers as words).
    """
    if not reference:
        return {
            "wer": 1.0 if hypothesis else 0.0,
            "cer": 1.0 if hypothesis else 0.0,
            "subs": 0,
            "ins": len(hypothesis.split()) if hypothesis else 0,
            "dels": 0,
            "hits": 0,
            "word_count_ref": 0,
        }

    word_output = jiwer.process_words(reference, hypothesis)
    cer_value = jiwer.cer(reference, hypothesis)

    return {
        "wer": word_output.wer,
        "cer": cer_value,
        "subs": word_output.substitutions,
        "ins": word_output.insertions,
        "dels": word_output.deletions,
        "hits": word_output.hits,
        "word_count_ref": len(reference.split()),
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("  05_axis_a_inference_and_eval.py")
    print("  Whisper inference on Axis A degraded utterances")
    print("=" * 70)

    # --- Load Axis A manifest ---
    manifest_path = METADATA_DIR / "axis_a_manifest.csv"
    if not manifest_path.exists():
        print(f"\n  [ERROR] Axis A manifest not found: {manifest_path}")
        print("  Run 04_axis_a_noise_augmentation.py first.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    print(f"\n  [LOAD] {len(df)} degraded utterances from {manifest_path.name}")
    print(f"         Noise types: {df['noise_type'].unique().tolist()}")
    print(f"         SNR levels:  {sorted(df['snr_db'].unique().tolist(), reverse=True)}")

    # --- Check for existing partial results (resume support) ---
    results_path = RESULTS_DIR / "axis_a_results.csv"
    completed_ids = set()
    existing_results = []

    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        # Create a composite key for each completed evaluation
        for _, row in existing_df.iterrows():
            key = f"{row['utterance_id']}_{row['noise_type']}_{row['snr_db']}"
            completed_ids.add(key)
        existing_results = existing_df.to_dict("records")
        print(f"  [RESUME] Found {len(completed_ids)} already-evaluated entries")

    # Filter to only unevaluated entries
    pending_indices = []
    for idx, row in df.iterrows():
        key = f"{row['utterance_id']}_{row['noise_type']}_{row['snr_db']}"
        if key not in completed_ids:
            pending_indices.append(idx)

    pending_df = df.loc[pending_indices]
    print(f"  [PLAN] {len(pending_df)} utterances to evaluate "
          f"({len(completed_ids)} already done)")

    if len(pending_df) == 0:
        print("\n  All utterances already evaluated. Loading existing results...")
        results_df = pd.read_csv(results_path)
    else:
        # --- Load Whisper model ---
        print()
        model = load_whisper_model()

        # --- Transcribe & evaluate ---
        print(f"\n  [RUN]  Evaluating {len(pending_df)} degraded utterances...\n")
        results = list(existing_results)  # Start with any existing results
        start_time = time.time()

        for _, row in tqdm(pending_df.iterrows(), total=len(pending_df),
                           desc="  Evaluating"):
            utterance_id = row["utterance_id"]
            wav_path = row["degraded_wav_path"]
            reference = row["ground_truth_normalized"]

            # Transcribe the degraded audio (with OOM protection)
            try:
                raw_prediction = transcribe_utterance(model, wav_path)
            except Exception as e:
                print(f"\n  [WARN] Failed to transcribe {utterance_id} (OOM?): {e}")
                # Try to recover VRAM
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                gc.collect()
                raw_prediction = ""

            # Apply the SAME normalization to the prediction
            normalized_prediction = normalize_text(raw_prediction)

            # Compute WER, CER, S/I/D
            metrics = compute_metrics(reference, normalized_prediction)

            results.append({
                "utterance_id": utterance_id,
                "speaker_id": row["speaker_id"],
                "noise_type": row["noise_type"],
                "snr_db": row["snr_db"],
                "duration_s": row["duration_s"],
                "ground_truth": reference,
                "predicted_text": normalized_prediction,
                "predicted_text_raw": raw_prediction,
                "wer": round(metrics["wer"], 6),
                "cer": round(metrics["cer"], 6),
                "subs": metrics["subs"],
                "ins": metrics["ins"],
                "dels": metrics["dels"],
                "hits": metrics["hits"],
                "word_count_ref": metrics["word_count_ref"],
            })

            # Periodic save every 100 utterances (crash recovery)
            if len(results) % 100 == 0:
                pd.DataFrame(results).to_csv(results_path, index=False)

            # Clear CUDA cache every 200 utterances to manage RTX 3050 VRAM.
            # Whisper's CTranslate2 backend can accumulate intermediate tensors;
            # periodic cleanup prevents gradual VRAM exhaustion during the
            # 1,800-file Axis A run.
            if len(results) % 200 == 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                gc.collect()

        # Final CUDA cleanup after all inference
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

        elapsed = time.time() - start_time
        print(f"\n  [TIME] {elapsed:.1f}s total "
              f"({elapsed / max(1, len(pending_df)):.2f}s per utterance)")

        # --- Save final results ---
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_path, index=False)
        print(f"  [SAVED] {results_path} ({len(results_df)} rows)")

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("  AXIS A RESULTS SUMMARY")
    print(f"{'=' * 70}")

    # --- WER × Noise Type × SNR Table ---
    # This is the core table from proposal §2 — populated with real measurements
    print(f"\n  WER by Noise Type × SNR Level:")
    print(f"  {'─' * 56}")
    print(f"  {'Noise Type':<12} {'20 dB':>8} {'15 dB':>8} {'10 dB':>8} {'5 dB':>8}")
    print(f"  {'─' * 56}")

    for noise_type in NOISE_TYPES:
        row_str = f"  {noise_type:<12}"
        for snr in SNR_LEVELS:
            subset = results_df[
                (results_df["noise_type"] == noise_type) &
                (results_df["snr_db"] == snr)
            ]
            if len(subset) > 0:
                # Corpus-level WER for this condition
                total_errors = subset["subs"].sum() + subset["ins"].sum() + subset["dels"].sum()
                total_words = subset["word_count_ref"].sum()
                corpus_wer = total_errors / total_words if total_words > 0 else 0
                row_str += f" {corpus_wer:>7.2%}"
            else:
                row_str += f" {'N/A':>7}"
        print(row_str)

    print(f"  {'─' * 56}")

    # --- CER Table ---
    print(f"\n  CER by Noise Type × SNR Level:")
    print(f"  {'─' * 56}")
    print(f"  {'Noise Type':<12} {'20 dB':>8} {'15 dB':>8} {'10 dB':>8} {'5 dB':>8}")
    print(f"  {'─' * 56}")

    for noise_type in NOISE_TYPES:
        row_str = f"  {noise_type:<12}"
        for snr in SNR_LEVELS:
            subset = results_df[
                (results_df["noise_type"] == noise_type) &
                (results_df["snr_db"] == snr)
            ]
            if len(subset) > 0:
                mean_cer = subset["cer"].mean()
                row_str += f" {mean_cer:>7.2%}"
            else:
                row_str += f" {'N/A':>7}"
        print(row_str)

    print(f"  {'─' * 56}")

    # --- Error Type Breakdown (S/I/D) per noise type ---
    print(f"\n  Error Type Breakdown (total S / I / D per condition):")
    print(f"  {'─' * 72}")
    print(f"  {'Noise':<8} {'SNR':>5} {'Subs':>7} {'Ins':>7} {'Dels':>7} "
          f"{'Total':>7} {'S%':>6} {'I%':>6} {'D%':>6}")
    print(f"  {'─' * 72}")

    for noise_type in NOISE_TYPES:
        for snr in SNR_LEVELS:
            subset = results_df[
                (results_df["noise_type"] == noise_type) &
                (results_df["snr_db"] == snr)
            ]
            if len(subset) == 0:
                continue
            s = subset["subs"].sum()
            i = subset["ins"].sum()
            d = subset["dels"].sum()
            total = s + i + d
            if total > 0:
                s_pct = s / total * 100
                i_pct = i / total * 100
                d_pct = d / total * 100
            else:
                s_pct = i_pct = d_pct = 0
            print(f"  {noise_type:<8} {snr:>3}dB {s:>7} {i:>7} {d:>7} "
                  f"{total:>7} {s_pct:>5.1f}% {i_pct:>5.1f}% {d_pct:>5.1f}%")

    print(f"  {'─' * 72}")

    # --- Per-noise-type WER trend (for proposal §9: validate White first) ---
    print(f"\n  WER Degradation Trend (visual):")
    for noise_type in NOISE_TYPES:
        print(f"\n    {noise_type.upper()} NOISE:")
        for snr in SNR_LEVELS:
            subset = results_df[
                (results_df["noise_type"] == noise_type) &
                (results_df["snr_db"] == snr)
            ]
            if len(subset) == 0:
                continue
            total_errors = subset["subs"].sum() + subset["ins"].sum() + subset["dels"].sum()
            total_words = subset["word_count_ref"].sum()
            wer = total_errors / total_words if total_words > 0 else 0
            bar_len = int(wer * 100)  # 1 char per percentage point
            bar = "█" * min(bar_len, 60)
            print(f"      {snr:>2}dB │ {wer:>6.2%} {bar}")

    # --- Worst utterances per noise type (for spot-checking) ---
    print(f"\n  {'─' * 66}")
    print("  Top 3 worst-performing utterances per noise type (5 dB):")
    print(f"  {'─' * 66}")

    for noise_type in NOISE_TYPES:
        subset = results_df[
            (results_df["noise_type"] == noise_type) &
            (results_df["snr_db"] == 5)
        ]
        if len(subset) == 0:
            continue
        worst = subset.nlargest(3, "wer")
        print(f"\n    {noise_type.upper()} @ 5dB:")
        for _, row in worst.iterrows():
            print(f"      [{row['utterance_id']}] WER={row['wer']:.4f} "
                  f"S={row['subs']} I={row['ins']} D={row['dels']}")
            print(f"        REF: {str(row['ground_truth'])[:80]}...")
            print(f"        HYP: {str(row['predicted_text'])[:80]}...")

    # --- Final validation ---
    expected_total = len(df)
    assert len(results_df) == expected_total, (
        f"Expected {expected_total} results, got {len(results_df)}"
    )
    assert results_df["wer"].notna().all(), "Found NaN WER values"

    print(f"\n  ✓ All {len(results_df)} degraded utterances evaluated")
    print(f"  ✓ Results saved to: {results_path}")
    print(f"\n  Ready for analysis and visualization.\n")


if __name__ == "__main__":
    main()
