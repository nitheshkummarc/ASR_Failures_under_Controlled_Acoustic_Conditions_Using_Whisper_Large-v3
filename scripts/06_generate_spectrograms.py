"""
Stage 6 — Per-utterance spectrograms and MFCC deviation heatmaps.

Renders a clean-vs-degraded spectrogram pair and an absolute MFCC deviation
heatmap for every condition of a single showcase utterance. These are the
figures embedded in the Plots & Inferences report.

Condition names come from the config registry so this cannot drift out of sync
with the files stage 03 actually wrote.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATA_DIR, RESULTS_DIR, AXIS_A_DIR, AXIS_B_DIR, AXIS_C_DIR,
    NOISE_TYPES, SNR_LEVELS, AXIS_B_CONDITION, AXIS_C_CONDITIONS,
    SHOWCASE_UTTERANCE_ID, NUM_CEPS_LABEL_LOW, NUM_CEPS_LABEL_HIGH,
)
from src.features.extractor import extract_mel_spec_and_mfcc
from src.logger import setup_logger

logger = setup_logger("scripts.06_generate_spectrograms")


def plot_condition(y_clean, y_deg, sr, utterance_id, condition_name, plots_dir):
    """Render the spectrogram comparison and the MFCC deviation heatmap."""
    # --- 1. Spectrogram comparison ---
    f, t, Sxx_clean = scipy.signal.spectrogram(y_clean, fs=sr, nperseg=512, noverlap=256)
    f, t_deg, Sxx_deg = scipy.signal.spectrogram(y_deg, fs=sr, nperseg=512, noverlap=256)

    # +1e-10 floors the argument so silent frames give a large negative dB
    # value instead of -inf, which matplotlib cannot render.
    D_clean = 10 * np.log10(Sxx_clean + 1e-10)
    D_deg = 10 * np.log10(Sxx_deg + 1e-10)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t, f, D_clean, shading="gouraud", cmap="viridis")
    plt.title(f"Spectrogram (Clean Audio: {utterance_id})")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format="%+2.0f dB")

    plt.subplot(2, 1, 2)
    plt.pcolormesh(t_deg, f, D_deg, shading="gouraud", cmap="viridis")
    plt.title(f'Spectrogram ({condition_name.replace("_", " ").title()})')
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(plots_dir / f"spectrogram_{condition_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- 2. MFCC deviation heatmap ---
    _, mfcc_clean, _, _ = extract_mel_spec_and_mfcc(y_clean, sr)
    _, mfcc_deg, _, _ = extract_mel_spec_and_mfcc(y_deg, sr)

    min_frames = min(mfcc_clean.shape[1], mfcc_deg.shape[1])
    mfcc_diff = np.abs(mfcc_clean[:, :min_frames] - mfcc_deg[:, :min_frames])

    plt.figure(figsize=(10, 4))
    sns.heatmap(mfcc_diff, cmap="magma")
    plt.title(f'MFCC Absolute Deviation Heatmap (Clean vs {condition_name.replace("_", " ").title()})')
    plt.xlabel("Time Frames")
    # The extractor drops C0, so these are c1..c13 — not c0..c12 as the label
    # used to claim.
    plt.ylabel(f"MFCC Coefficients ({NUM_CEPS_LABEL_LOW}-{NUM_CEPS_LABEL_HIGH})")
    plt.savefig(plots_dir / f"mfcc_dev_{condition_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def run(utterance_id: str, test_mode: bool = False):
    logger.info("=" * 50)
    logger.info(f"Generating Spectrograms & Heatmaps for ALL conditions of {utterance_id}")
    logger.info("=" * 50)

    out_dir = RESULTS_DIR / "test_mode" if test_mode else RESULTS_DIR
    plots_dir = out_dir / "plots" / "utterance_breakdowns" / utterance_id
    plots_dir.mkdir(parents=True, exist_ok=True)

    if test_mode:
        axis_a_dir = axis_b_dir = axis_c_dir = Path("scratch")
    else:
        axis_a_dir, axis_b_dir, axis_c_dir = AXIS_A_DIR, AXIS_B_DIR, AXIS_C_DIR

    clean_wav = DATA_DIR / f"{utterance_id}.wav"
    if not clean_wav.exists():
        logger.error(f"Clean audio {clean_wav} missing.")
        sys.exit(1)

    y_clean, sr = sf.read(str(clean_wav))
    plotted = 0

    # --- Axis A ---
    axis_a_plots = plots_dir / "axis_a_noise"
    axis_a_plots.mkdir(parents=True, exist_ok=True)
    for noise_type in NOISE_TYPES:
        for snr in SNR_LEVELS:
            deg_wav = axis_a_dir / f"{utterance_id}_{noise_type}_{snr}db.wav"
            if not deg_wav.exists():
                logger.warning(f"Missing {deg_wav}, skipping.")
                continue
            y_deg, _ = sf.read(str(deg_wav))
            cond_name = f"axis_a_{noise_type}_{snr}db"
            plot_condition(y_clean, y_deg, sr, utterance_id, cond_name, axis_a_plots)
            logger.info(f"Plotted {cond_name}")
            plotted += 1

    # --- Axis B ---
    axis_b_plots = plots_dir / "axis_b_telephonic"
    axis_b_plots.mkdir(parents=True, exist_ok=True)
    deg_wav = axis_b_dir / f"{utterance_id}_{AXIS_B_CONDITION}.wav"
    if deg_wav.exists():
        y_deg, _ = sf.read(str(deg_wav))
        cond_name = f"axis_b_{AXIS_B_CONDITION}"
        plot_condition(y_clean, y_deg, sr, utterance_id, cond_name, axis_b_plots)
        logger.info(f"Plotted {cond_name}")
        plotted += 1
    else:
        logger.warning(f"Missing {deg_wav}, skipping.")

    # --- Axis C ---
    axis_c_plots = plots_dir / "axis_c_temporal"
    axis_c_plots.mkdir(parents=True, exist_ok=True)
    for axis_c_cond in AXIS_C_CONDITIONS:
        deg_wav = axis_c_dir / f"{utterance_id}_{axis_c_cond}.wav"
        if not deg_wav.exists():
            logger.warning(f"Missing {deg_wav}, skipping.")
            continue
        y_deg, _ = sf.read(str(deg_wav))
        # Time-scaled audio has a different length; align before plotting so the
        # two panels share a time axis.
        min_len = min(len(y_clean), len(y_deg))
        cond_name = f"axis_c_{axis_c_cond}"
        plot_condition(y_clean[:min_len], y_deg[:min_len], sr, utterance_id, cond_name, axis_c_plots)
        logger.info(f"Plotted {cond_name}")
        plotted += 1

    if plotted == 0:
        logger.error(f"No degraded audio found for {utterance_id}. Run stages 02 and 03 first.")
        sys.exit(1)

    logger.info(f"{plotted} condition(s) plotted under {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance", type=str, default=SHOWCASE_UTTERANCE_ID,
                        help="Utterance ID to break down")
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    run(args.utterance, args.test_mode)