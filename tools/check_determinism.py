"""
Check that GPU decoding is reproducible.

Transcribes the same files twice within one process and compares the resulting
WERs. The project's reproducibility claim covers the seeded noise generation;
this verifies the other half — that Whisper's beam search returns identical
output for identical input on this hardware.

Note this tests determinism within a single process and model load. Cross-process
determinism (a different driver state, a different allocator) is not covered.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv

import jiwer

from config import METADATA_DIR, AXIS_B_DIR, AXIS_B_CONDITION
from src.data.text_normalizer import normalize_text
from src.inference.whisper_engine import WhisperEngine, TranscriptionOOMError


def transcribe_pass(engine, rows, label):
    print(f"\n--- {label} ---")
    results = []
    for row in rows:
        uid = row["utterance_id"]
        wav_path = AXIS_B_DIR / f"{uid}_{AXIS_B_CONDITION}.wav"
        if not wav_path.exists():
            print(f"  skip {uid}: {wav_path} not found")
            continue

        reference = normalize_text(row["ground_truth_normalized"])
        try:
            prediction = normalize_text(engine.transcribe(str(wav_path)))
        except TranscriptionOOMError:
            print(f"  skip {uid}: out of memory")
            continue

        wer = jiwer.process_words(reference, prediction).wer
        results.append((uid, wer, prediction))
        print(f"  {uid}: {wer:.4f}")
    return results


def main(n_files: int):
    manifest = METADATA_DIR / "utterance_manifest.csv"
    if not manifest.exists():
        print(f"[ERROR] Manifest not found: {manifest}")
        sys.exit(1)

    with open(manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:n_files]

    print("Initializing Whisper Engine...")
    engine = WhisperEngine()

    first = transcribe_pass(engine, rows, "RUN 1")
    second = transcribe_pass(engine, rows, "RUN 2")

    print("\n--- COMPARISON ---")
    if not first:
        print("No files were transcribed — run the pipeline first.")
        sys.exit(1)

    mismatches = 0
    for (uid, wer1, text1), (_, wer2, text2) in zip(first, second):
        if text1 != text2:
            mismatches += 1
            print(f"MISMATCH on {uid}: WER {wer1:.4f} vs {wer2:.4f}")
            print(f"  run 1: {text1}")
            print(f"  run 2: {text2}")

    if mismatches == 0:
        print(f"All {len(first)} files transcribed identically across both runs.")
    else:
        print(f"Found {mismatches} mismatch(es) — decoding is not deterministic here.")

    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check GPU decoding determinism")
    parser.add_argument("--files", type=int, default=5,
                        help="Number of utterances to test (default 5)")
    args = parser.parse_args()
    main(args.files)