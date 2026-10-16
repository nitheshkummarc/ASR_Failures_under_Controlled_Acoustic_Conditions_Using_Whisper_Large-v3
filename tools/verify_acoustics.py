"""
Acoustic verification — the project's real end-to-end test.

Independently re-derives SNR, bandwidth and duration from the WAV files on disk
rather than reusing pipeline code, so it genuinely validates the augmentation
rather than restating it.

Exits non-zero if any check FAILs, so it can be wired into the pipeline (it is
run as a stage by main.py) instead of scrolling past unnoticed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import soundfile as sf
from scipy.fftpack import fft


from config import (
    DATA_DIR, AXIS_A_DIR, AXIS_B_DIR, AXIS_C_DIR,
    NOISE_TYPES, SNR_LEVELS,
    TELEPHONE_LOW_HZ, TELEPHONE_HIGH_HZ,
    STRETCH_RATES, SILENCE_CONFIGS,
    stretch_condition, silence_condition,
    SHOWCASE_UTTERANCE_ID,
)
from src.results_io import read_manifest

# Tolerances
SNR_TOLERANCE_DB = 0.5
DURATION_TOLERANCE_S = 0.1
SILENCE_TOLERANCE_S = 0.05

# Maximum fraction of the original out-of-band energy allowed to survive the
# telephonic filter, measured as out_band_degraded / out_band_clean.
#
# An earlier version compared the in-band/out-of-band RATIO before and after and
# required a 10x improvement. That failed on utterances which were already
# nearly band-limited — little out-of-band energy to remove — even though the
# filter had worked perfectly. Measuring the suppression directly is independent
# of what the source happened to contain.
#
# Threshold chosen from the measured distribution over all 150 utterances:
# median 2.4%, p95 6.0%, max 11.3%. That spread is the intrinsic stopband
# roll-off of a 4th-order Butterworth bandpass (8 poles, 16th-order effective
# through filtfilt), not a
# defect. 25% sits ~2x above the worst legitimate case while still being 4x
# below the ~100% a missing or misconfigured filter would produce.
BANDPASS_MAX_RESIDUAL = 0.25


class Report:
    """Accumulates PASS/FAIL/SKIP counts so the script can set an exit code."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def check(self, ok: bool, message: str):
        if ok:
            self.passed += 1
            if self.verbose:
                print(f"[PASS] {message}")
        else:
            self.failed += 1
            self.failures.append(message)
            # Failures always print, even in quiet batch mode.
            print(f"[FAIL] {message}")

    def skip(self, message: str):
        self.skipped += 1
        if self.verbose:
            print(f"[SKIP] {message}")

    def merge(self, other: "Report"):
        self.passed += other.passed
        self.failed += other.failed
        self.skipped += other.skipped
        self.failures.extend(other.failures)


def calculate_snr(clean_signal, noisy_signal):
    """
    Back-measure the achieved SNR of a degraded file.

    Deliberately does NOT call mix_at_target_snr — independently re-deriving the
    quantity is the whole point of a verification script.

    THE CLEAN COMPONENT MAY HAVE BEEN SCALED, so a plain subtraction is wrong.
    When a mixture would exceed [-1, 1], mix_at_target_snr divides the WHOLE
    signal by its peak. That preserves the SNR exactly — both components are
    scaled identically — but it means the degraded file contains alpha*clean for
    some alpha < 1, not clean. Subtracting the unscaled clean would leave a
    clean-signal residual of (alpha - 1)*clean in the "noise" term and inflate
    the measurement. On this corpus that mis-reported 14 correctly-mixed files
    as failures, typically as ~5.8 dB against a 5 dB target.

    So recover alpha by least-squares projection of the degraded signal onto the
    clean one, then measure the residual against the scaled clean. For an
    unclipped file alpha comes out at 1.0 and this reduces to plain subtraction.
    """
    min_len = min(len(clean_signal), len(noisy_signal))
    clean_signal = clean_signal[:min_len]
    noisy_signal = noisy_signal[:min_len]

    clean_energy = float(np.dot(clean_signal, clean_signal))
    if clean_energy < 1e-20:
        # Silent reference: SNR is undefined.
        return float('nan')

    # Best-fit scaling of the clean signal inside the degraded one.
    alpha = float(np.dot(noisy_signal, clean_signal)) / clean_energy

    scaled_clean = alpha * clean_signal
    noise_only = noisy_signal - scaled_clean

    rms_clean = np.sqrt(np.mean(scaled_clean ** 2))
    rms_noise = np.sqrt(np.mean(noise_only ** 2))

    if rms_noise < 1e-10:
        # No noise was added at all — the signal is unchanged.
        return float('inf')

    return 20 * np.log10(rms_clean / rms_noise)


def verify_bandpass(signal, sr, lowcut=TELEPHONE_LOW_HZ, highcut=TELEPHONE_HIGH_HZ):
    """Return (in_band_energy, out_of_band_energy) from the magnitude spectrum."""
    N = len(signal)
    yf = fft(signal)
    # Bin k of an N-point FFT sits at k * sr / N. linspace(0, sr/2, N//2) put
    # the bins slightly too high (its step is sr/(N-2), not sr/N).
    xf = np.arange(N // 2) * sr / N

    mag = 2.0 / N * np.abs(yf[0:N // 2])

    in_band_mask = (xf >= lowcut) & (xf <= highcut)
    out_band_mask = (xf < lowcut) | (xf > highcut)

    return np.sum(mag[in_band_mask] ** 2), np.sum(mag[out_band_mask] ** 2)


def longest_silence_run_s(y, sr, threshold=1e-5):
    """
    Length of the longest contiguous near-zero run, in seconds.

    Run-length encoding via boundary detection: find where the boolean flips,
    diff those indices to get run lengths, then take alternating runs.
    """
    is_zero = np.abs(y) < threshold
    if not np.any(is_zero):
        return 0.0
    boundaries = np.where(
        np.concatenate(([is_zero[0]], is_zero[:-1] != is_zero[1:], [True]))
    )[0]
    zero_runs = np.diff(boundaries)[::2]
    return float(np.max(zero_runs)) / sr if len(zero_runs) > 0 else 0.0


def verify_acoustics(utterance_id: str, verbose: bool = True) -> Report:
    report = Report(verbose=verbose)

    if verbose:
        print("=" * 60)
        print(f"ACOUSTIC VERIFICATION REPORT FOR {utterance_id}")
        print("=" * 60)

    clean_path = DATA_DIR / f"{utterance_id}.wav"
    if not clean_path.exists():
        print(f"[ERROR] Clean file {clean_path} not found.")
        report.failed += 1
        report.failures.append(f"{utterance_id}: clean audio missing")
        return report

    y_clean, sr = sf.read(str(clean_path))
    clean_duration = len(y_clean) / sr
    if verbose:
        print(f"Clean Audio Duration: {clean_duration:.2f}s | Sample Rate: {sr}Hz\n")

    # ---------------------------------------------------------
    # 1. AXIS A — additive noise at a target SNR
    # ---------------------------------------------------------
    if verbose:
        print("--- 1. AXIS A: ADDITIVE NOISE (TARGET SNR VERIFICATION) ---")
    for noise in NOISE_TYPES:
        for target_snr in sorted(SNR_LEVELS):
            deg_file = AXIS_A_DIR / f"{utterance_id}_{noise}_{target_snr}db.wav"
            if not deg_file.exists():
                report.skip(f"{noise.upper():>5} Noise {target_snr}dB not found.")
                continue

            y_deg, _ = sf.read(str(deg_file))
            measured_snr = calculate_snr(y_clean, y_deg)

            ok = abs(measured_snr - target_snr) < SNR_TOLERANCE_DB
            report.check(
                ok,
                f"{noise.upper():>5} Noise | Target: {target_snr}dB | "
                f"Measured: {measured_snr:.2f}dB",
            )

    # ---------------------------------------------------------
    # 2. AXIS B — telephonic bandpass
    # ---------------------------------------------------------
    if verbose:
        print("\n--- 2. AXIS B: TELEPHONIC CHANNEL VERIFICATION ---")
    axis_b_file = AXIS_B_DIR / f"{utterance_id}_telephonic.wav"
    if axis_b_file.exists():
        y_b, _ = sf.read(str(axis_b_file))
        in_band_clean, out_band_clean = verify_bandpass(y_clean, sr)
        in_band_deg, out_band_deg = verify_bandpass(y_b, sr)

        # Measure what the filter actually did: how much of the original
        # out-of-band energy survives.
        residual = out_band_deg / (out_band_clean + 1e-20)
        ok = residual < BANDPASS_MAX_RESIDUAL

        report.check(
            ok,
            f"{utterance_id} Telephonic Filter | out-of-band energy retained: "
            f"{residual * 100:.2f}% (limit {BANDPASS_MAX_RESIDUAL * 100:.0f}%)",
        )
        if verbose:
            clean_ratio = in_band_clean / (out_band_clean + 1e-10)
            deg_ratio = in_band_deg / (out_band_deg + 1e-10)
            print(f"       Clean Audio    -> In-band/Out-band Energy Ratio: {clean_ratio:.2f}")
            print(f"       Degraded Audio -> In-band/Out-band Energy Ratio: {deg_ratio:.2f}")
    else:
        report.skip("Axis B Telephonic file not found.")

    # ---------------------------------------------------------
    # 3. AXIS C — temporal
    # ---------------------------------------------------------
    if verbose:
        print("\n--- 3. AXIS C: TEMPORAL VERIFICATION ---")

    # Time-scale modification.
    # rate > 1 means FASTER, so the expected duration is clean / rate.
    for rate in STRETCH_RATES:
        cond = stretch_condition(rate)
        stretch_file = AXIS_C_DIR / f"{utterance_id}_{cond}.wav"
        if not stretch_file.exists():
            report.skip(f"Axis C {cond} file not found.")
            continue

        y_stretch, _ = sf.read(str(stretch_file))
        stretch_duration = len(y_stretch) / sr
        expected_duration = clean_duration / rate

        ok = abs(stretch_duration - expected_duration) < DURATION_TOLERANCE_S
        report.check(
            ok,
            f"{cond} | Expected: {expected_duration:.2f}s | "
            f"Actual: {stretch_duration:.2f}s",
        )

    # Silence insertion — check both the total length and that real zeros exist.
    for pos, dur in SILENCE_CONFIGS:
        cond = silence_condition(pos, dur)
        silence_file = AXIS_C_DIR / f"{utterance_id}_{cond}.wav"
        if not silence_file.exists():
            report.skip(f"Axis C {cond} file not found.")
            continue

        y_sil, _ = sf.read(str(silence_file))
        sil_duration = len(y_sil) / sr
        expected_duration = clean_duration + dur
        max_zero_run_s = longest_silence_run_s(y_sil, sr)

        len_ok = abs(sil_duration - expected_duration) < SILENCE_TOLERANCE_S

        # AT LEAST `dur`, not exactly `dur`. Some LibriSpeech recordings open
        # with their own near-silence (up to ~0.4 s observed), which merges with
        # a start-position insertion into one longer run. Requiring equality
        # flagged those correctly-generated files as failures. The exact total
        # length is already asserted above, so "at least" is still sufficient to
        # catch silence that was never inserted.
        sil_ok = max_zero_run_s >= dur - SILENCE_TOLERANCE_S

        report.check(len_ok, f"{utterance_id} {cond} length | Expected: {expected_duration:.2f}s | "
                             f"Actual: {sil_duration:.2f}s")
        report.check(sil_ok, f"{utterance_id} {cond} silence | Target: >={dur:.2f}s | "
                             f"Longest true silence block: {max_zero_run_s:.2f}s")

    if verbose:
        print("\n" + "=" * 60)
        print(f"VERIFICATION COMPLETE — {report.passed} passed, "
              f"{report.failed} failed, {report.skipped} skipped")
        print("=" * 60)

    return report


def verify_all() -> Report:
    """
    Verify every utterance in the manifest.

    Quiet by default — only failures print — so a full corpus sweep produces a
    short, readable report instead of ~3,000 PASS lines.
    """
    try:
        utterance_ids = [row["utterance_id"] for row in read_manifest()]
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("=" * 60)
    print(f"BATCH ACOUSTIC VERIFICATION — {len(utterance_ids)} utterances")
    print("=" * 60)

    total = Report(verbose=False)
    for uid in utterance_ids:
        total.merge(verify_acoustics(uid, verbose=False))

    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE — {total.passed} passed, "
          f"{total.failed} failed, {total.skipped} skipped "
          f"across {len(utterance_ids)} utterances")
    print("=" * 60)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify that generated audio matches its stated acoustic parameters"
    )
    parser.add_argument("--utterance", type=str, default=SHOWCASE_UTTERANCE_ID,
                        help="Utterance ID to verify in detail")
    parser.add_argument("--all", action="store_true",
                        help="Verify every utterance in the manifest (failures only)")
    args = parser.parse_args()

    result = verify_all() if args.all else verify_acoustics(args.utterance)
    sys.exit(1 if result.failed else 0)