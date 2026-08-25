"""
Stage 3 — Axis B (channel) and Axis C (temporal).

Per utterance:
  Axis B: telephonic 300-3400 Hz bandpass                        (1 condition)
  Axis C: time-scale modification at each rate in STRETCH_RATES  (2 conditions)
  Axis C: silence insertion per SILENCE_CONFIGS                  (3 conditions)

Condition names come from config.stretch_condition / config.silence_condition so
that this script, scripts/04, scripts/06 and verify_acoustics.py cannot drift
apart. They previously re-typed the same string literals in four places, which
fails silently (a mismatched name just finds no files).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv

import soundfile as sf
from tqdm import tqdm

from config import (
    DATA_DIR, METADATA_DIR, RESULTS_DIR, AXIS_B_DIR, AXIS_C_DIR,
    AXIS_B_CONDITION, STRETCH_RATES, SILENCE_CONFIGS,
    CONDITIONS_PER_UTTERANCE_BC, stretch_condition, silence_condition,
)
from src.augmentation.axis_b_channel import apply_telephonic_bandpass
from src.augmentation.axis_c_temporal import time_stretch_phase_vocoder, insert_silence
from src.inference.whisper_engine import WhisperEngine, TranscriptionOOMError
from src.inference.runner import (
    read_manifest, open_result_csv, transcribe_and_score, get_reference,
    write_degraded, SkipCounter,
)
from src.logger import setup_logger

logger = setup_logger("scripts.03_run_axis_b_c")

OUTPUT_FIELDS = [
    "utterance_id", "speaker_id", "condition", "duration_s",
    "ground_truth", "predicted_text", "wer", "cer", "substitutions",
    "insertions", "deletions",
]


def _resume_key(row: dict) -> tuple:
    return (row["utterance_id"], row["condition"])


def run_axis_b_c(test_mode: bool):
    logger.info("=" * 50)
    logger.info("Running Axis B (Channel) & C (Temporal) Pipeline")
    if test_mode:
        logger.info("TEST MODE: Running on a single utterance")
    logger.info("=" * 50)

    manifest_rows = read_manifest(METADATA_DIR / "utterance_manifest.csv")

    if test_mode:
        manifest_rows = manifest_rows[:1]
        out_dir = RESULTS_DIR / "test_mode"
        axis_b_dir = axis_c_dir = Path("scratch")
    else:
        out_dir = RESULTS_DIR
        axis_b_dir, axis_c_dir = AXIS_B_DIR, AXIS_C_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv_b = out_dir / "axis_b_results.csv"
    out_csv_c = out_dir / "axis_c_results.csv"

    processed_b = open_result_csv(out_csv_b, OUTPUT_FIELDS, _resume_key)
    processed_c = open_result_csv(out_csv_c, OUTPUT_FIELDS, _resume_key)

    engine = WhisperEngine()
    skips = SkipCounter("axis_b_c")

    total_iters = len(manifest_rows) * CONDITIONS_PER_UTTERANCE_BC
    pbar = tqdm(total=total_iters, desc="Axis B & C Processing")

    with open(out_csv_b, "a", encoding="utf-8", newline="") as fb, \
         open(out_csv_c, "a", encoding="utf-8", newline="") as fc:

        writer_b = csv.DictWriter(fb, fieldnames=OUTPUT_FIELDS)
        writer_c = csv.DictWriter(fc, fieldnames=OUTPUT_FIELDS)

        for row in manifest_rows:
            uid = row["utterance_id"]
            clean_wav = DATA_DIR / f"{uid}.wav"
            y, sr = sf.read(str(clean_wav))

            reference = get_reference(row, uid)
            if reference is None:
                # Checked BEFORE any degradation or inference, so nothing
                # expensive is wasted on an utterance we cannot score.
                skips.no_reference += 1
                pbar.update(CONDITIONS_PER_UTTERANCE_BC)
                continue

            # Each entry: (condition, degrade_callable, output_dir, writer, file, processed_set)
            jobs = [(
                AXIS_B_CONDITION,
                lambda sig: apply_telephonic_bandpass(sig, sr),
                axis_b_dir, writer_b, fb, processed_b,
            )]

            for rate in STRETCH_RATES:
                jobs.append((
                    stretch_condition(rate),
                    lambda sig, r=rate: time_stretch_phase_vocoder(sig, r),
                    axis_c_dir, writer_c, fc, processed_c,
                ))

            for pos, dur in SILENCE_CONFIGS:
                jobs.append((
                    silence_condition(pos, dur),
                    lambda sig, p=pos, d=dur: insert_silence(sig, sr, d, p),
                    axis_c_dir, writer_c, fc, processed_c,
                ))

            for condition, degrade, out_wav_dir, writer, handle, processed in jobs:
                if (uid, condition) in processed:
                    pbar.update(1)
                    continue

                y_deg = degrade(y)
                deg_path = out_wav_dir / f"{uid}_{condition}.wav"
                write_degraded(y_deg, deg_path, sr)

                try:
                    scored = transcribe_and_score(engine, deg_path, reference)
                except TranscriptionOOMError:
                    skips.oom += 1
                    pbar.update(1)
                    continue

                writer.writerow({
                    "utterance_id": uid,
                    "speaker_id": row["speaker_id"],
                    "condition": condition,
                    "duration_s": row["duration_s"],
                    "ground_truth": reference,
                    "predicted_text": scored["predicted_text"],
                    "wer": scored["wer"],
                    "cer": scored["cer"],
                    "substitutions": scored["substitutions"],
                    "insertions": scored["insertions"],
                    "deletions": scored["deletions"],
                })
                handle.flush()
                pbar.update(1)

    pbar.close()
    skips.report()
    logger.info("Finished Axis B & C results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    run_axis_b_c(args.test_mode)