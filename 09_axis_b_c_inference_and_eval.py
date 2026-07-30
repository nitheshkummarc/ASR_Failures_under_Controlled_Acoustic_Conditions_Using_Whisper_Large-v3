"""
09_axis_b_c_inference_and_eval.py — Phase 3: Inference and Evaluation (Axis B & C)
===================================================================================
Runs Whisper large-v3 inference on the Axis B (Channel) and Axis C (Temporal) 
degraded audio. Follows the EXACT same methodology, normalizer, and OOM-fallback 
as 05_axis_a_inference_and_eval.py to ensure strictly valid cross-axis comparison.
"""

import gc
import sys
import time
from pathlib import Path
import pandas as pd
import jiwer
from tqdm import tqdm
from faster_whisper import WhisperModel

from config import (
    METADATA_DIR,
    RESULTS_DIR,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE
)
import importlib
_norm_module = importlib.import_module("02_normalize_text")
normalize_text = _norm_module.normalize_text


def compute_metrics(reference: str, hypothesis: str) -> dict:
    if not reference:
        return {"wer": 0, "cer": 0, "subs": 0, "ins": 0, "dels": 0, "hits": 0, "word_count_ref": 0}
        
    out = jiwer.process_words(reference, hypothesis)
    cer = jiwer.cer(reference, hypothesis)
    return {
        "wer": out.wer,
        "cer": cer,
        "subs": out.substitutions,
        "ins": out.insertions,
        "dels": out.deletions,
        "hits": out.hits,
        "word_count_ref": len(reference.split())
    }


def run_inference_for_axis(manifest_name: str, results_name: str, axis_desc: str):
    print(f"\n======================================================================")
    print(f"  STARTING INFERENCE: {axis_desc}")
    print(f"======================================================================")
    
    manifest_path = METADATA_DIR / manifest_name
    results_path = RESULTS_DIR / results_name
    
    if not manifest_path.exists():
        print(f"  [ERROR] Manifest not found: {manifest_path}")
        return None
        
    df = pd.read_csv(manifest_path)
    # We must join with utterance_manifest to get ground_truth
    ref_df = pd.read_csv(METADATA_DIR / "utterance_manifest.csv")
    df = df.merge(ref_df[["utterance_id", "ground_truth_normalized", "speaker_id", "duration_s"]], on="utterance_id", how="left")
    
    results = []
    
    if results_path.exists():
        print(f"  [RESUME] Found existing {results_name}. Loading...")
        results = pd.read_csv(results_path).to_dict("records")
        processed_uids = {(r["utterance_id"], r["condition"]) for r in results}
        
        pending_df = df[~df.apply(lambda row: (row["utterance_id"], row["condition"]) in processed_uids, axis=1)]
        print(f"           Skipping {len(processed_uids)} completed rows.")
        print(f"           {len(pending_df)} rows remaining to process.")
    else:
        pending_df = df
        
    if len(pending_df) == 0:
        print(f"  [DONE] All inferences complete for {axis_desc}.")
        return pd.DataFrame(results)
        
    print("\n  [INIT] Loading Whisper Model (large-v3, INT8)...")
    print(f"         Device: {WHISPER_DEVICE}, Compute Type: {WHISPER_COMPUTE_TYPE}")
    
    model = WhisperModel(
        model_size_or_path=WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type="int8"
    )
    
    print("\n  [RUN] Transcribing degraded audio...")
    start_time = time.time()
    
    for _, row in tqdm(pending_df.iterrows(), total=len(pending_df), desc=f"Infer {axis_desc}"):
        uid = row["utterance_id"]
        condition = row["condition"]
        wav_path = row["degraded_wav_path"]
        reference = str(row["ground_truth_normalized"])
        
        # Exact same OOM Fallback as Phase 2
        try:
            segments, info = model.transcribe(
                audio=wav_path,
                beam_size=WHISPER_BEAM_SIZE,
                language="en",
                condition_on_previous_text=False
            )
            raw_prediction = " ".join([segment.text for segment in segments])
        except Exception as e:
            print(f"\n    [WARNING] Exception on {uid} ({condition}): {e}")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()
            raw_prediction = ""
            
        # Normalize text
        normalized_prediction = normalize_text(raw_prediction)
        
        # Metrics
        metrics = compute_metrics(reference, normalized_prediction)
        
        results.append({
            "utterance_id": uid,
            "speaker_id": row["speaker_id"],
            "condition": condition,
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
            "word_count_ref": metrics["word_count_ref"]
        })
        
        # Save every 50 to prevent huge data loss
        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(results_path, index=False)
            
        # Periodic Cache clear
        if len(results) % 100 == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()
            
    # Final cleanup
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()
    
    elapsed = time.time() - start_time
    print(f"\n  [TIME] {elapsed:.1f}s total for {axis_desc}")
    
    # Save final
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"  [SAVED] {results_path} ({len(results_df)} rows)")
    
    return results_df

def print_summary(df, axis_desc):
    print(f"\n{'=' * 70}")
    print(f"  {axis_desc.upper()} RESULTS SUMMARY")
    print(f"{'=' * 70}")
    
    conditions = df["condition"].unique()
    
    print(f"\n  WER by Condition:")
    print(f"  {'-' * 40}")
    print(f"  {'Condition':<25} {'WER':>10}")
    print(f"  {'-' * 40}")
    
    for c in sorted(conditions):
        subset = df[df["condition"] == c]
        total_errors = subset["subs"].sum() + subset["ins"].sum() + subset["dels"].sum()
        total_words = subset["word_count_ref"].sum()
        wer = total_errors / total_words if total_words > 0 else 0
        print(f"  {c:<25} {wer:>9.2%}")
        
    print(f"  {'-' * 40}")

def main():
    print("======================================================================")
    print("  09_axis_b_c_inference_and_eval.py")
    print("  Phase 3: Axis B (Channel) and Axis C (Temporal) Inference")
    print("======================================================================")
    
    df_b = run_inference_for_axis("axis_b_manifest.csv", "axis_b_results.csv", "Axis B")
    if df_b is not None:
        print_summary(df_b, "Axis B (Channel)")
        
    df_c = run_inference_for_axis("axis_c_manifest.csv", "axis_c_results.csv", "Axis C")
    if df_c is not None:
        print_summary(df_c, "Axis C (Temporal)")

if __name__ == "__main__":
    main()
