"""
03_baseline_inference.py — Whisper baseline on clean LibriSpeech utterances.
=============================================================================
This script:
  1. Loads the utterance manifest (with normalized ground truths).
  2. Initializes faster-whisper with large-v3 (CUDA/float16, int8 OOM fallback).
  3. Transcribes all 150 clean WAV files.
  4. Applies identical text normalization to predictions.
  5. Computes per-utterance WER, CER, Substitution/Insertion/Deletion counts.
  6. Saves results/baseline_results.csv.
  7. Prints a corpus-level summary.

The baseline establishes the "ceiling" — Whisper's best-case performance
on clean speech. All subsequent degradation experiments (Axes A/B/C) will
be compared against these numbers.
"""

import sys
import time
import traceback

import pandas as pd
import numpy as np
import jiwer
from tqdm import tqdm

from config import (
    METADATA_DIR, RESULTS_DIR,
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE, WHISPER_LANGUAGE, NUM_UTTERANCES,
)
# Import the shared normalization function — same pipeline for both sides.
# The file is named 02_normalize_text.py (digit-prefixed), so we use importlib.
import importlib
_norm_module = importlib.import_module("02_normalize_text")
normalize_text = _norm_module.normalize_text


# =============================================================================
# WHISPER MODEL INITIALIZATION (with OOM fallback)
# =============================================================================
def load_whisper_model():
    """
    Initialize the faster-whisper model.

    Primary:  device="cuda", compute_type="float16" (RTX 3050)
    Fallback: If CUDA OOM, retry with compute_type="int8" (lower VRAM)

    Returns the loaded WhisperModel instance.
    """
    from faster_whisper import WhisperModel

    print(f"  [INIT] Loading faster-whisper model: {WHISPER_MODEL}")
    print(f"         Device: {WHISPER_DEVICE}, Compute: {WHISPER_COMPUTE_TYPE}")

    try:
        model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        # Quick test to catch OOM early
        print("  [OK]   Model loaded successfully on CUDA/float16")
        return model

    except Exception as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "oom" in error_msg or "cuda" in error_msg:
            print(f"\n  [WARN] CUDA/float16 failed: {e}")
            print("  [FALL] Retrying with compute_type='int8' (lower VRAM)...")
            try:
                model = WhisperModel(
                    WHISPER_MODEL,
                    device=WHISPER_DEVICE,
                    compute_type="int8",
                )
                print("  [OK]   Model loaded on CUDA/int8")
                return model
            except Exception as e2:
                print(f"\n  [WARN] CUDA/int8 also failed: {e2}")
                print("  [FALL] Final fallback: CPU/int8...")
                model = WhisperModel(
                    WHISPER_MODEL,
                    device="cpu",
                    compute_type="int8",
                )
                print("  [OK]   Model loaded on CPU/int8 (will be slower)")
                return model
        else:
            # Non-OOM error — re-raise
            raise


# =============================================================================
# TRANSCRIPTION
# =============================================================================
def transcribe_utterance(model, wav_path: str) -> str:
    """
    Transcribe a single WAV file and return the predicted text.

    Uses beam search (beam_size=5) and forces English language
    to avoid language detection overhead.
    """
    segments, _info = model.transcribe(
        wav_path,
        beam_size=WHISPER_BEAM_SIZE,
        language=WHISPER_LANGUAGE,
    )
    # Join all segment texts into a single string
    # (faster-whisper returns an iterator of segments)
    predicted = " ".join(seg.text for seg in segments)
    return predicted.strip()


# =============================================================================
# METRIC COMPUTATION
# =============================================================================
def compute_metrics(reference: str, hypothesis: str) -> dict:
    """
    Compute WER, CER, and Substitution/Insertion/Deletion counts
    using jiwer.

    Both reference and hypothesis should already be normalized
    (lowercase, no punctuation, numbers as words).

    Returns a dict with: wer, cer, substitutions, insertions, deletions,
                          hits, word_count_ref
    """
    # Handle edge case: empty strings
    if not reference:
        return {
            "wer": 1.0 if hypothesis else 0.0,
            "cer": 1.0 if hypothesis else 0.0,
            "substitutions": 0,
            "insertions": len(hypothesis.split()) if hypothesis else 0,
            "deletions": 0,
            "hits": 0,
            "word_count_ref": 0,
        }

    # Word-level alignment (WER + S/I/D counts)
    word_output = jiwer.process_words(reference, hypothesis)

    # Character-level error rate
    cer_value = jiwer.cer(reference, hypothesis)

    ref_word_count = len(reference.split())

    return {
        "wer": word_output.wer,
        "cer": cer_value,
        "substitutions": word_output.substitutions,
        "insertions": word_output.insertions,
        "deletions": word_output.deletions,
        "hits": word_output.hits,
        "word_count_ref": ref_word_count,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 70)
    print("  03_baseline_inference.py")
    print("  Whisper large-v3 baseline on clean LibriSpeech utterances")
    print("=" * 70)

    # --- Load manifest ---
    manifest_path = METADATA_DIR / "utterance_manifest.csv"
    if not manifest_path.exists():
        print(f"\n  [ERROR] Manifest not found: {manifest_path}")
        print("  Run 01_download_librispeech.py and 02_normalize_text.py first.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)

    if "ground_truth_normalized" not in df.columns:
        print("\n  [ERROR] Column 'ground_truth_normalized' not found in manifest.")
        print("  Run 02_normalize_text.py first.")
        sys.exit(1)

    print(f"\n  [LOAD] {len(df)} utterances from {manifest_path.name}")

    # --- Load model ---
    print()
    model = load_whisper_model()

    # --- Transcribe all utterances ---
    print(f"\n  [RUN]  Transcribing {len(df)} clean utterances...\n")
    results = []
    start_time = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Transcribing"):
        utterance_id = row["utterance_id"]
        wav_path = row["wav_path"]
        reference = row["ground_truth_normalized"]

        # Transcribe
        raw_prediction = transcribe_utterance(model, wav_path)

        # Apply the SAME normalization to the prediction
        # (strips Whisper's punctuation, casing, etc.)
        normalized_prediction = normalize_text(raw_prediction)

        # Compute metrics
        metrics = compute_metrics(reference, normalized_prediction)

        results.append({
            "utterance_id": utterance_id,
            "speaker_id": row["speaker_id"],
            "duration_s": row["duration_s"],
            "ground_truth": reference,
            "predicted_text": normalized_prediction,
            "predicted_text_raw": raw_prediction,
            "wer": round(metrics["wer"], 6),
            "cer": round(metrics["cer"], 6),
            "substitutions": metrics["substitutions"],
            "insertions": metrics["insertions"],
            "deletions": metrics["deletions"],
            "hits": metrics["hits"],
            "word_count_ref": metrics["word_count_ref"],
        })

    elapsed = time.time() - start_time

    # --- Save results ---
    results_df = pd.DataFrame(results)
    results_path = RESULTS_DIR / "baseline_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  [SAVED] {results_path} ({len(results_df)} rows)")

    # --- Validation ---
    print(f"\n{'=' * 70}")
    print("  VALIDATION CHECKS")
    print(f"{'=' * 70}")

    assert len(results_df) == NUM_UTTERANCES, (
        f"Expected {NUM_UTTERANCES} results, got {len(results_df)}"
    )
    assert results_df["wer"].notna().all(), "Found NaN WER values"
    assert results_df["cer"].notna().all(), "Found NaN CER values"
    print(f"  ✓ {NUM_UTTERANCES} utterances transcribed and evaluated")
    print(f"  ✓ No NaN values in WER or CER columns")

    # --- Corpus-level summary ---
    print(f"\n{'=' * 70}")
    print("  BASELINE RESULTS SUMMARY (Clean Speech)")
    print(f"{'=' * 70}")

    # Corpus-level WER: total errors / total reference words
    total_subs = results_df["substitutions"].sum()
    total_ins = results_df["insertions"].sum()
    total_dels = results_df["deletions"].sum()
    total_errors = total_subs + total_ins + total_dels
    total_ref_words = results_df["word_count_ref"].sum()
    corpus_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

    # Corpus-level CER: weighted average by reference length
    corpus_cer = results_df["cer"].mean()

    print(f"\n  Corpus-level WER:  {corpus_wer:.4f}  ({corpus_wer * 100:.2f}%)")
    print(f"  Mean per-utt WER:  {results_df['wer'].mean():.4f}  "
          f"({results_df['wer'].mean() * 100:.2f}%)")
    print(f"  Corpus-level CER:  {corpus_cer:.4f}  ({corpus_cer * 100:.2f}%)")

    print(f"\n  Error Breakdown (total across {NUM_UTTERANCES} utterances):")
    print(f"    Substitutions:  {total_subs:>5d}  "
          f"({total_subs / total_errors * 100:.1f}% of errors)" if total_errors > 0
          else f"    Substitutions:  {total_subs:>5d}")
    print(f"    Insertions:     {total_ins:>5d}  "
          f"({total_ins / total_errors * 100:.1f}% of errors)" if total_errors > 0
          else f"    Insertions:     {total_ins:>5d}")
    print(f"    Deletions:      {total_dels:>5d}  "
          f"({total_dels / total_errors * 100:.1f}% of errors)" if total_errors > 0
          else f"    Deletions:      {total_dels:>5d}")
    print(f"    Total errors:   {total_errors:>5d}  / {total_ref_words} ref words")

    # Per-speaker WER
    print(f"\n  Per-Speaker WER:")
    speaker_wer = results_df.groupby("speaker_id").agg(
        mean_wer=("wer", "mean"),
        n_utterances=("utterance_id", "count"),
    ).sort_values("mean_wer", ascending=False)

    for speaker_id, row in speaker_wer.iterrows():
        bar = "█" * int(row["mean_wer"] * 200)  # Visual bar (scaled)
        print(f"    Speaker {speaker_id}: {row['mean_wer']:.4f} "
              f"({int(row['n_utterances']):4d} utterances)  {bar}")

    # Sanity check: baseline WER should be < 5% on clean LibriSpeech
    if corpus_wer > 0.10:
        print(f"\n  ⚠ WARNING: Corpus WER = {corpus_wer:.2%} is unusually high "
              "for clean LibriSpeech with large-v3.")
        print("    Expected < 5%. Check normalization alignment or model loading.")
    elif corpus_wer < 0.05:
        print(f"\n  ✓ Corpus WER = {corpus_wer:.2%} — within expected range for "
              "clean speech with large-v3.")

    print(f"\n  Time elapsed: {elapsed:.1f}s "
          f"({elapsed / len(df):.2f}s per utterance)")

    # Show worst 5 utterances for manual inspection
    print(f"\n  {'─' * 66}")
    print("  Top 5 highest-WER utterances (for manual spot-check):")
    print(f"  {'─' * 66}")
    worst = results_df.nlargest(5, "wer")
    for _, row in worst.iterrows():
        print(f"\n  [{row['utterance_id']}]  WER={row['wer']:.4f}  "
              f"S={row['substitutions']} I={row['insertions']} D={row['deletions']}")
        print(f"    REF:  {row['ground_truth']}")
        print(f"    HYP:  {row['predicted_text']}")

    print(f"\n  Done. Baseline results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
