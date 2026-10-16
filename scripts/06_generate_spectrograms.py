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

from config import (
    DATA_DIR, RESULTS_DIR, AXIS_A_DIR, AXIS_B_DIR, AXIS_C_DIR,
    NOISE_TYPES, SNR_LEVELS, AXIS_B_CONDITION, AXIS_C_CONDITIONS,
    SHOWCASE_UTTERANCE_ID, NUM_CEPS_LABEL_LOW, NUM_CEPS_LABEL_HIGH,
    STRETCH_RATES, SILENCE_CONFIGS, stretch_condition, silence_condition,
)
from src.augmentation.axis_c_temporal import remove_inserted_silence
from src.features.extractor import HOP_LENGTH, extract_mel_spec_and_mfcc, resample_frames
from src.logger import setup_logger

logger = setup_logger("scripts.06_generate_spectrograms")


# Dynamic range shown on the spectrograms: everything more than this far below
# the clean signal's peak is drawn as the floor colour.
SPECTROGRAM_RANGE_DB = 80.0

# Axis C condition name -> how to undo it, so the MFCC heatmap compares the same
# speech in both signals.
TEMPORAL_ALIGNMENT = {
    **{stretch_condition(r): ("stretch", r) for r in STRETCH_RATES},
    **{silence_condition(p, d): ("silence", p, d) for p, d in SILENCE_CONFIGS},
}


def plot_condition(y_clean, y_deg, sr, utterance_id, condition_name, plots_dir, alignment=None):
    """
    Render the spectrogram comparison and the MFCC deviation heatmap.

    Both spectrogram panels show the WHOLE signal on a shared time axis and a
    shared colour scale. They used to be truncated to the shorter signal (which
    cut the last 0.5-2 s of speech off the silence conditions) and coloured on
    independent scales (which made level differences unreadable).
    """
    pretty = condition_name.replace("_", " ").title()

    # --- 1. Spectrogram comparison ---
    spec = dict(fs=sr, window="hann", nperseg=512, noverlap=256)
    f, t, Sxx_clean = scipy.signal.spectrogram(y_clean, **spec)
    f, t_deg, Sxx_deg = scipy.signal.spectrogram(y_deg, **spec)

    # +1e-10 floors the argument so silent frames give a large negative dB
    # value instead of -inf, which matplotlib cannot render.
    D_clean = 10 * np.log10(Sxx_clean + 1e-10)
    D_deg = 10 * np.log10(Sxx_deg + 1e-10)
    vmax = float(D_clean.max())
    vmin = vmax - SPECTROGRAM_RANGE_DB
    t_end = max(len(y_clean), len(y_deg)) / sr

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, (tt, D, title) in zip(axes, ((t, D_clean, f"Clean audio: {utterance_id}"),
                                         (t_deg, D_deg, pretty))):
        mesh = ax.pcolormesh(tt, f, D, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Spectrogram ({title})")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlim(0, t_end)
    axes[1].set_xlabel("Time (s)")
    fig.colorbar(mesh, ax=list(axes), format="%+2.0f dB", label="shared scale")
    fig.savefig(plots_dir / f"spectrogram_{condition_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- 2. MFCC deviation heatmap, on time-aligned signals ---
    y_cmp = y_deg
    note = ""
    if alignment and alignment[0] == "silence":
        y_cmp = remove_inserted_silence(y_deg, sr, alignment[2], alignment[1], len(y_clean))
        note = " (inserted silence removed before comparing)"

    _, mfcc_clean, _, _ = extract_mel_spec_and_mfcc(y_clean, sr)
    _, mfcc_deg, _, _ = extract_mel_spec_and_mfcc(y_cmp, sr)

    if alignment and alignment[0] == "stretch":
        # Put the stretched frames on the clean frame grid, as stage 5 does.
        mfcc_deg = resample_frames(mfcc_deg, mfcc_clean.shape[1])
        note = " (stretched frames resampled onto the clean time grid)"

    n_frames = min(mfcc_clean.shape[1], mfcc_deg.shape[1])
    mfcc_diff = np.abs(mfcc_clean[:, :n_frames] - mfcc_deg[:, :n_frames])

    # imshow rather than seaborn.heatmap: seaborn imports pandas, whose compiled
    # extensions are blocked by the AppLocker policy on the target machine.
    # The extent puts real coefficient numbers (c1..c13, C0 is dropped by the
    # extractor) on the y axis and seconds on the x axis; bare imshow labelled
    # the rows 0-12.
    plt.figure(figsize=(10, 4))
    # |difference| is non-negative, so the scale starts at 0. An all-zero map
    # (silence conditions: the speech itself is untouched) gets a unit scale
    # instead of matplotlib's meaningless -0.1..+0.1.
    plt.imshow(mfcc_diff, cmap="magma", aspect="auto", origin="lower", interpolation="nearest",
               vmin=0.0, vmax=max(float(mfcc_diff.max()), 1.0),
               extent=[0, n_frames * HOP_LENGTH / sr,
                       NUM_CEPS_LABEL_LOW - 0.5, NUM_CEPS_LABEL_HIGH + 0.5])
    if mfcc_diff.max() < 1e-6:
        plt.text(0.5, 0.5, "speech unchanged: deviation is 0 everywhere\n"
                           "(the only change is the inserted gap itself)",
                 transform=plt.gca().transAxes, ha="center", va="center",
                 color="white", fontsize=11)
    plt.yticks(range(NUM_CEPS_LABEL_LOW, NUM_CEPS_LABEL_HIGH + 1))
    plt.colorbar(label="|MFCC difference|")
    plt.title(f"MFCC Absolute Deviation (Clean vs {pretty}){note}", fontsize=10)
    plt.xlabel("Time (s, clean signal)")
    plt.ylabel(f"MFCC coefficient (c{NUM_CEPS_LABEL_LOW}-c{NUM_CEPS_LABEL_HIGH})")
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
        # Full signals; plot_condition aligns them for the MFCC comparison.
        cond_name = f"axis_c_{axis_c_cond}"
        plot_condition(y_clean, y_deg, sr, utterance_id, cond_name, axis_c_plots,
                       alignment=TEMPORAL_ALIGNMENT.get(axis_c_cond))
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