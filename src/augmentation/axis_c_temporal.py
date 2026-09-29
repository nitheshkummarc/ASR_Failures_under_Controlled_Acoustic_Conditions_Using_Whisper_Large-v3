"""
Axis C — Temporal Degradation
========================================================================
Two independent temporal manipulations:
  - Silence insertion (digital absolute zeros) at the start or midpoint.
  - Time-scale modification via a hand-written phase vocoder.

The phase vocoder is written directly on scipy.signal.stft/istft rather than
using librosa.effects.time_stretch. librosa depends on numba, whose JIT emits
.pyd binaries at runtime, and those are blocked by the AppLocker policy on the
target Windows machine. Do not "simplify" this back to librosa.

Driven by scripts/03_run_axis_b_c.py.
"""

import numpy as np
import scipy.signal

from config import TARGET_SR
from src.logger import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# DSP Core: Axis C (Temporal - Silence)
# =============================================================================
def insert_silence(y: np.ndarray, sr: int, duration_s: float, position: str) -> np.ndarray:
    """
    Insert exact zeros (digital absolute silence) into the waveform.

    Note that 'middle' splits at the SAMPLE midpoint, not at a word or pause
    boundary, so the gap usually lands mid-word. That is intentional — it is the
    worst case for a decoder's cross-attention alignment, which is the failure
    mode this condition probes.

    Args:
        y:          1-D signal.
        sr:         Sample rate in Hz.
        duration_s: Length of silence to insert, in seconds.
        position:   'start' or 'middle'.

    Returns:
        A new signal of length len(y) + int(duration_s * sr).

    Raises:
        ValueError: if `position` is not 'start' or 'middle'.
    """
    n_zeros = int(duration_s * sr)
    silence = np.zeros(n_zeros, dtype=y.dtype)

    if position == 'start':
        return np.concatenate((silence, y))
    elif position == 'middle':
        mid_idx = len(y) // 2
        return np.concatenate((y[:mid_idx], silence, y[mid_idx:]))
    else:
        raise ValueError("position must be 'start' or 'middle'")


# =============================================================================
# DSP Core: Axis C (Temporal - Time Scale Modification via Phase Vocoder)
# =============================================================================
def time_stretch_phase_vocoder(y: np.ndarray, rate: float, n_fft=2048, hop_length=512) -> np.ndarray:
    """
    Change the speaking rate without changing pitch.

    RATE SEMANTICS (librosa convention — read this before changing anything):

        rate > 1  ->  speech plays FASTER, output is SHORTER
                      output_duration ~= input_duration / rate
        rate < 1  ->  speech plays SLOWER, output is LONGER
        rate == 1 ->  returned unchanged

    So rate=1.5 produces audio about 0.67x the original duration. This matches
    proposal S3 ("Speaking rate: 1x / 1.2x / 1.5x"), where the parameter is the
    speaking RATE, not the duration multiplier. scripts/verify_acoustics.py
    asserts output_duration == input_duration / rate.

    DSP Theory:
      1. Compute the STFT to separate magnitude and phase.
      2. Resample the frame grid by `rate` and linearly interpolate magnitudes.
      3. Reconstruct phase by estimating each bin's instantaneous frequency from
         the inter-frame phase difference (minus the expected advance, wrapped
         to [-pi, pi]) and integrating it across the output frames. This is what
         keeps partials coherent; naively reusing the original phases produces
         the classic "phasiness" artifact.
      4. Inverse STFT back to the time domain.

    Pitch is preserved because bin centre frequencies are never altered — only
    the time axis is resampled.

    Args:
        y:          1-D signal.
        rate:       Speaking-rate multiplier (see semantics above).
        n_fft:      STFT window length.
        hop_length: STFT hop length.

    Returns:
        The time-scaled signal.

    Raises:
        ValueError: if `rate` is not positive.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")

    if rate == 1.0:
        return y

    # 1. STFT
    f, t, Zxx = scipy.signal.stft(y, fs=TARGET_SR, nperseg=n_fft, noverlap=n_fft-hop_length, window='hann')

    # 2. New time grid. Stepping by `rate` yields fewer frames when rate > 1,
    #    which is what makes rate > 1 a speed-up.
    time_steps = np.arange(0, Zxx.shape[1], rate)

    # 3. Phase accumulator. phi_advance[k] is the phase a bin is expected to
    #    advance per analysis hop: 2*pi*hop*k/n_fft, which linspace over
    #    [0, pi*hop] across n_fft/2+1 bins reproduces exactly.
    phi_advance = np.linspace(0, np.pi * hop_length, Zxx.shape[0])
    Zxx_new = np.zeros((Zxx.shape[0], len(time_steps)), dtype=complex)
    phase_acc = np.angle(Zxx[:, 0])

    # 4. Interpolate magnitudes and propagate phases
    for t_step_idx, t_step in enumerate(time_steps):
        left_idx = int(np.floor(t_step))
        right_idx = min(left_idx + 1, Zxx.shape[1] - 1)
        alpha = t_step - left_idx

        # Magnitude interpolation (linear)
        mag_left = np.abs(Zxx[:, left_idx])
        mag_right = np.abs(Zxx[:, right_idx])
        mag = (1 - alpha) * mag_left + alpha * mag_right

        # Phase propagation
        if t_step_idx > 0:
            dphi = np.angle(Zxx[:, right_idx]) - np.angle(Zxx[:, left_idx]) - phi_advance
            # Wrap to [-pi, pi] to resolve the 2*pi*n ambiguity
            dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
            true_freq = phi_advance + dphi
            phase_acc = phase_acc + true_freq

        Zxx_new[:, t_step_idx] = mag * np.exp(1j * phase_acc)

    # 5. Inverse STFT
    _, y_stretched = scipy.signal.istft(Zxx_new, fs=TARGET_SR, nperseg=n_fft, noverlap=n_fft-hop_length, window='hann')

    return y_stretched
