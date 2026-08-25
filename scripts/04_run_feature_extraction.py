"""
Stage 4 — Feature-space deviation between clean and degraded audio.

For every degraded file produced by stages 2 and 3, extracts MFCC / STE / ZCR
from both the clean original and the degraded version and records the MSE
between them. Stage 5 then tests whether that deviation predicts WER.

IMPORTANT — THE TWO BRANCHES ARE NOT ON THE SAME SCALE:

  * Non-stretch conditions compare the MEAN feature vectors (a single
    time-average per signal), because the two signals are the same length.
  * Stretch conditions compare the full per-frame SEQUENCES after resampling
    the degraded one onto the clean frame grid, because time-averaging would
    hide exactly the temporal distortion being measured.

Time-averaging cancels sign-alternating deviations, so the first branch produces
systematically smaller numbers for equivalent distortion. Stage 5 correlates
only Axis A rows, which are all first-branch, so it is internally consistent —
but DO NOT compare mfcc_mse across axes without saying so.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import warnings

import numpy as np
import soundfile as sf
from tqdm import tqdm

from config import DATA_DIR, RESULTS_DIR, AXIS_A_DIR, AXIS_B_DIR, AXIS_C_DIR
from src.features.extractor import extract_all_features
from src.logger import setup_logger

logger = setup_logger("scripts.04_run_feature_extraction")

# scipy.signal.stft emits a padding warning for signals that do not divide
# evenly into frames. Targeted rather than a blanket filterwarnings('ignore'),
# which previously also hid genuine numerical warnings.
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.signal")

OUTPUT_FIELDS = [
    "utterance_id", "noise_type", "snr_db", "condition",
    "mfcc_mse", "ste_mse", "zcr_mse",
]


def read_csv_to_list(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_work_list(out_dir: Path, axis_a_dir: Path, axis_b_dir: Path, axis_c_dir: Path) -> list:
    """
    Reconstruct the set of (clean, degraded) pairs from the result CSVs.

    The result CSVs are used as the work-list — not the manifests — which is why
    the record count equals the number of successfully transcribed rows.
    """
    df_a = read_csv_to_list(out_dir / "axis_a_results.csv")
    df_b = read_csv_to_list(out_dir / "axis_b_results.csv")
    df_c = read_csv_to_list(out_dir / "axis_c_results.csv")

    records = []

    for row in df_a:
        uid, noise, snr = row["utterance_id"], row["noise_type"], row["snr_db"]
        records.append({
            "utterance_id": uid,
            "noise_type": noise,
            "snr_db": snr,
            "clean_path": DATA_DIR / f"{uid}.wav",
            "deg_path": axis_a_dir / f"{uid}_{noise}_{snr}db.wav",
            "condition": f"axis_a_{noise}_{snr}",
        })

    for row in df_b:
        uid, cond = row["utterance_id"], row["condition"]
        records.append({
            "utterance_id": uid,
            "noise_type": cond,
            "snr_db": "N/A",
            "clean_path": DATA_DIR / f"{uid}.wav",
            "deg_path": axis_b_dir / f"{uid}_{cond}.wav",
            "condition": f"axis_b_{cond}",
        })

    for row in df_c:
        uid, cond = row["utterance_id"], row["condition"]
        records.append({
            "utterance_id": uid,
            "noise_type": cond,
            "snr_db": "N/A",
            "clean_path": DATA_DIR / f"{uid}.wav",
            "deg_path": axis_c_dir / f"{uid}_{cond}.wav",
            "condition": f"axis_c_{cond}",
        })

    return records


def interp_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a 1-D feature sequence onto a grid of `target_len` points."""
    return np.interp(
        np.linspace(0, 1, target_len),
        np.linspace(0, 1, len(arr)),
        arr,
    )


def deviation_for_record(record: dict) -> dict:
    """Compute (mfcc_mse, ste_mse, zcr_mse) for one clean/degraded pair."""
    y_clean, sr = sf.read(str(record["clean_path"]))
    y_deg, _ = sf.read(str(record["deg_path"]))

    if "stretch" in record["condition"]:
        # Lengths differ by design. Resample the degraded feature sequences onto
        # the clean frame grid and compare frame by frame.
        feat_clean = extract_all_features(y_clean, sr)
        feat_deg = extract_all_features(y_deg, sr)

        ste_deg_interp = interp_1d(feat_deg["ste"], len(feat_clean["ste"]))
        zcr_deg_interp = interp_1d(feat_deg["zcr"], len(feat_clean["zcr"]))

        mfcc_deg_interp = np.zeros(
            (feat_deg["mfcc"].shape[0], feat_clean["mfcc"].shape[1])
        )
        for i in range(feat_deg["mfcc"].shape[0]):
            mfcc_deg_interp[i] = interp_1d(feat_deg["mfcc"][i], feat_clean["mfcc"].shape[1])

        mfcc_dev = np.mean((feat_clean["mfcc"] - mfcc_deg_interp) ** 2)
        ste_dev = np.mean((feat_clean["ste"] - ste_deg_interp) ** 2)
        zcr_dev = np.mean((feat_clean["zcr"] - zcr_deg_interp) ** 2)
    else:
        # Same-length (or silence-padded) conditions: truncate and compare the
        # time-averaged feature vectors.
        min_len = min(len(y_clean), len(y_deg))
        y_clean = y_clean[:min_len]
        y_deg = y_deg[:min_len]

        feat_clean = extract_all_features(y_clean, sr)
        feat_deg = extract_all_features(y_deg, sr)

        mfcc_dev = np.mean((feat_clean["mfcc_mean"] - feat_deg["mfcc_mean"]) ** 2)
        ste_dev = np.mean((feat_clean["ste_mean"] - feat_deg["ste_mean"]) ** 2)
        zcr_dev = np.mean((feat_clean["zcr_mean"] - feat_deg["zcr_mean"]) ** 2)

    return {
        "utterance_id": record["utterance_id"],
        "noise_type": record["noise_type"],
        "snr_db": record["snr_db"],
        "condition": record["condition"],
        "mfcc_mse": round(float(mfcc_dev), 6),
        "ste_mse": round(float(ste_dev), 6),
        "zcr_mse": round(float(zcr_dev), 6),
    }


def run_feature_extraction(test_mode: bool):
    logger.info("=" * 50)
    logger.info("Running Feature Extraction Pipeline (MFCC, STE, ZCR)")
    if test_mode:
        logger.info("TEST MODE: Running on a single utterance")
    logger.info("=" * 50)

    out_dir = RESULTS_DIR / "test_mode" if test_mode else RESULTS_DIR

    if test_mode:
        axis_a_dir = axis_b_dir = axis_c_dir = Path("scratch")
    else:
        axis_a_dir, axis_b_dir, axis_c_dir = AXIS_A_DIR, AXIS_B_DIR, AXIS_C_DIR

    try:
        records = build_work_list(out_dir, axis_a_dir, axis_b_dir, axis_c_dir)
    except FileNotFoundError as e:
        logger.error(f"Could not find result CSVs ({e}). Run stages 01, 02, 03 first.")
        sys.exit(1)

    results = []
    missing = []

    for r in tqdm(records, desc="Extracting Features"):
        # Missing files used to be skipped silently, so a wrong working
        # directory produced an empty CSV and exit code 0. Now they are counted
        # and reported, and a wholesale miss aborts the stage.
        if not r["clean_path"].exists() or not r["deg_path"].exists():
            missing.append(r["deg_path"])
            continue

        results.append(deviation_for_record(r))

    if missing:
        logger.error(f"{len(missing)} of {len(records)} audio files were missing.")
        for path in missing[:5]:
            logger.error(f"  missing: {path}")
        if len(missing) > 5:
            logger.error(f"  ... and {len(missing) - 5} more")

        if not results:
            logger.error(
                "No feature deviations could be computed at all. Are you running "
                "from the project root, and did stages 02/03 complete?"
            )
            sys.exit(1)

    out_csv = out_dir / "feature_deviation.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Saved {len(results)} feature deviations to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    run_feature_extraction(args.test_mode)