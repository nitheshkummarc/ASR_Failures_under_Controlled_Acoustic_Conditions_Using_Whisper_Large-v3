"""
Stage 1 — Baseline inference on clean audio.

Establishes the ceiling performance of Whisper large-v3 on the 150 clean
utterances. Every degraded condition is measured relative to this number.

Also performs a one-time migration of the manifest: if `ground_truth_normalized`
is absent it is computed and written back atomically, so all later stages can
rely on it being there.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import os

from tqdm import tqdm

from config import DATA_DIR, METADATA_DIR, RESULTS_DIR
from src.data.text_normalizer import normalize_text
from src.inference.whisper_engine import WhisperEngine, TranscriptionOOMError
from src.inference.runner import (
    read_manifest, open_result_csv, transcribe_and_score, get_reference, SkipCounter,
)
from src.logger import setup_logger

logger = setup_logger("scripts.01_run_baseline")

OUTPUT_FIELDS = [
    "utterance_id", "speaker_id", "duration_s", "ground_truth",
    "predicted_text", "predicted_text_raw", "wer", "cer",
    "substitutions", "insertions", "deletions", "hits", "word_count_ref",
]


def ensure_normalized_ground_truth(manifest_path: Path) -> list:
    """
    Read the manifest, adding `ground_truth_normalized` if it is missing.

    The rewrite goes through a temp file plus os.replace(), which is atomic on
    both Windows and POSIX — the manifest is never observed half-written, even
    if the process is killed mid-write.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "ground_truth_normalized" in fieldnames:
        return rows

    logger.info("Normalizing ground truth text...")
    fieldnames.append("ground_truth_normalized")
    for row in rows:
        row["ground_truth_normalized"] = normalize_text(row["ground_truth_raw"])

    temp_path = manifest_path.with_suffix(".csv.tmp")
    with open(temp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, manifest_path)

    return rows


def run_baseline(test_mode: bool):
    logger.info("=" * 50)
    logger.info("Running Baseline Inference (Phase 1)")
    if test_mode:
        logger.info("TEST MODE: Running on a single utterance")
    logger.info("=" * 50)

    manifest_path = METADATA_DIR / "utterance_manifest.csv"
    read_manifest(manifest_path)  # existence / non-empty check with a useful message
    manifest_rows = ensure_normalized_ground_truth(manifest_path)

    if test_mode:
        manifest_rows = manifest_rows[:1]
        out_dir = RESULTS_DIR / "test_mode"
    else:
        out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "baseline_results.csv"

    processed = open_result_csv(out_csv, OUTPUT_FIELDS, lambda r: (r["utterance_id"],))

    engine = WhisperEngine()
    skips = SkipCounter("baseline")

    pbar = tqdm(total=len(manifest_rows), desc="Baseline Processing")

    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)

        for row in manifest_rows:
            uid = row["utterance_id"]
            if (uid,) in processed:
                pbar.update(1)
                continue

            reference = get_reference(row, uid)
            if reference is None:
                skips.no_reference += 1
                pbar.update(1)
                continue

            wav_path = DATA_DIR / f"{uid}.wav"

            try:
                scored = transcribe_and_score(engine, wav_path, reference)
            except TranscriptionOOMError:
                # Skip rather than record. An empty transcript would be written
                # as a genuine 100%-WER result and silently corrupt the corpus.
                skips.oom += 1
                pbar.update(1)
                continue

            writer.writerow({
                "utterance_id": uid,
                "speaker_id": row["speaker_id"],
                "duration_s": row["duration_s"],
                "ground_truth": reference,
                **scored,
            })
            f.flush()  # crash-safe: at most one row is ever lost
            pbar.update(1)

    pbar.close()
    skips.report()
    logger.info(f"Finished Baseline results in {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Run on 1 utterance")
    args = parser.parse_args()
    run_baseline(args.test_mode)