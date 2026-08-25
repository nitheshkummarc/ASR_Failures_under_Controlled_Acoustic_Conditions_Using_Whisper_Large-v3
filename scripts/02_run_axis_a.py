"""
Stage 2 — Axis A: additive noise.

Mixes each of the 150 utterances with white, pink and cafe-babble noise at
20/15/10/5 dB SNR (1,800 degraded files), transcribes each, and scores it.

TWO REPRODUCIBILITY RULES ARE ENFORCED HERE — do not change either:

  1. LOOP ORDER IS FIXED: utterance -> noise_type -> snr. The cafe noise source
     is read through a sliding cursor and the white/pink RNGs advance per draw,
     so the iteration order determines exactly which noise each file receives.
     Reordering these loops would make a future re-run produce different audio
     than the committed results, for no benefit.

  2. NOISE IS DRAWN BEFORE THE RESUME CHECK. The RNG streams must advance once
     per (utterance, noise_type, snr) combination regardless of how many rows
     already exist. Drawing after the check — as this script used to — meant a
     resumed run consumed fewer draws than a fresh one, so every remaining file
     got different noise and the results CSV became a mixture of two
     incompatible noise realizations with nothing recording that.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv

import soundfile as sf
from tqdm import tqdm

from config import (
    DATA_DIR, METADATA_DIR, RESULTS_DIR, AXIS_A_DIR,
    TARGET_SR, SNR_LEVELS, NOISE_TYPES, RANDOM_SEED,
)
from src.augmentation.axis_a_noise import mix_at_target_snr, NoiseSource
from src.inference.whisper_engine import WhisperEngine, TranscriptionOOMError
from src.inference.runner import (
    read_manifest, open_result_csv, transcribe_and_score, get_reference,
    write_degraded, SkipCounter,
)
from src.logger import setup_logger

logger = setup_logger("scripts.02_run_axis_a")

OUTPUT_FIELDS = [
    "utterance_id", "speaker_id", "noise_type", "snr_db", "duration_s",
    "ground_truth", "predicted_text", "wer", "cer", "substitutions",
    "insertions", "deletions",
]


def _resume_key(row: dict) -> tuple:
    """
    Natural key of an Axis A row.

    int(float(...)) rather than int(...) so a value serialized as "5.0" is still
    parsed. The old code tested .isdigit() first, so "5.0" fell through as a
    string and then raised inside int() — which was swallowed, emptying the
    resume set and duplicating all 1,800 rows.
    """
    return (row["utterance_id"], row["noise_type"], int(float(row["snr_db"])))


def run_axis_a(test_mode: bool):
    logger.info("=" * 50)
    logger.info("Running Axis A (Additive Noise) Pipeline")
    if test_mode:
        logger.info("TEST MODE: Running on a single utterance")
    logger.info("=" * 50)

    manifest_rows = read_manifest(METADATA_DIR / "utterance_manifest.csv")

    if test_mode:
        manifest_rows = manifest_rows[:1]
        out_dir = RESULTS_DIR / "test_mode"
        degraded_dir = Path("scratch")
    else:
        out_dir = RESULTS_DIR
        degraded_dir = AXIS_A_DIR

    # NoiseSource owns ALL cafe babble generation, including the disjoint-speaker
    # filtering. There used to be a second generator here that built the babble
    # from these same 150 test utterances, which risked masking an utterance with
    # babble containing itself.
    all_wavs = [str(DATA_DIR / f"{u['utterance_id']}.wav") for u in manifest_rows]
    noise_source = NoiseSource(all_wavs, TARGET_SR, RANDOM_SEED)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "axis_a_results.csv"
    processed = open_result_csv(out_csv, OUTPUT_FIELDS, _resume_key)

    engine = WhisperEngine()
    skips = SkipCounter("axis_a")

    total_iters = len(manifest_rows) * len(NOISE_TYPES) * len(SNR_LEVELS)
    pbar = tqdm(total=total_iters, desc="Axis A Processing")

    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)

        # RULE 1: this nesting is fixed — utterance, then noise_type, then snr.
        for row in manifest_rows:
            uid = row["utterance_id"]
            clean_wav = DATA_DIR / f"{uid}.wav"
            y, sr = sf.read(str(clean_wav))

            for noise_type in NOISE_TYPES:
                for snr in SNR_LEVELS:
                    # RULE 2: draw noise before the resume check so the RNG
                    # streams advance identically on fresh and resumed runs.
                    raw_noise = noise_source.get_noise(noise_type, len(y))

                    if (uid, noise_type, snr) in processed:
                        pbar.update(1)
                        continue

                    reference = get_reference(row, uid)
                    if reference is None:
                        skips.no_reference += 1
                        pbar.update(1)
                        continue

                    y_deg = mix_at_target_snr(y, raw_noise, snr)

                    deg_path = degraded_dir / f"{uid}_{noise_type}_{snr}db.wav"
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
                        "noise_type": noise_type,
                        "snr_db": snr,
                        "duration_s": row["duration_s"],
                        "ground_truth": reference,
                        "predicted_text": scored["predicted_text"],
                        "wer": scored["wer"],
                        "cer": scored["cer"],
                        "substitutions": scored["substitutions"],
                        "insertions": scored["insertions"],
                        "deletions": scored["deletions"],
                    })
                    f.flush()
                    pbar.update(1)

    pbar.close()
    skips.report()
    logger.info(f"Finished Axis A results in {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    run_axis_a(args.test_mode)