"""
Tests for the Axis B and Axis C degradations.

test_time_stretch_rate_direction is the important one here: a three-line
duration assertion is all it would have taken to catch the fact that the
time-scale conditions were being described as slowdowns when they are speed-ups.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from config import (
    TARGET_SR, TELEPHONE_LOW_HZ, TELEPHONE_HIGH_HZ,
    STRETCH_RATES, SILENCE_CONFIGS, stretch_condition, silence_condition,
)
from src.augmentation.axis_b_channel import apply_telephonic_bandpass
from src.augmentation.axis_c_temporal import insert_silence, time_stretch_phase_vocoder

SR = TARGET_SR


def _speechlike(duration_s=3.0):
    """A multi-harmonic signal — closer to speech than a pure tone."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    sig = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate([200, 400, 800, 1600, 3200]))
    return 0.2 * sig / np.max(np.abs(sig))


# ---------------------------------------------------------------- Axis B ----

def test_bandpass_preserves_length():
    y = _speechlike()
    assert len(apply_telephonic_bandpass(y, SR)) == len(y)


def test_bandpass_suppresses_out_of_band_energy():
    y = _speechlike()
    filtered = apply_telephonic_bandpass(y, SR)

    def band_ratio(sig):
        spec = np.abs(np.fft.rfft(sig)) ** 2
        freqs = np.fft.rfftfreq(len(sig), 1 / SR)
        inband = (freqs >= TELEPHONE_LOW_HZ) & (freqs <= TELEPHONE_HIGH_HZ)
        return spec[inband].sum() / (spec[~inband].sum() + 1e-12)

    # Same criterion verify_acoustics.py applies to the real corpus.
    assert band_ratio(filtered) > 10 * band_ratio(y)


def test_bandpass_is_zero_phase():
    """
    filtfilt must not shift the signal in time. A phase shift would introduce an
    uncontrolled second variable alongside the intended band-limiting.

    Uses a Gaussian-windowed tone burst rather than a continuous sine: a steady
    sine is periodic, so a delay of one full period is indistinguishable from no
    delay and the test could not tell them apart. The burst envelope has a
    unique peak, which pins the timing unambiguously.
    """
    t = np.linspace(0, 1, SR, endpoint=False)
    centre = 0.5
    envelope = np.exp(-((t - centre) ** 2) / (2 * 0.01 ** 2))
    y = 0.5 * envelope * np.sin(2 * np.pi * 1000 * t)  # squarely inside the passband

    filtered = apply_telephonic_bandpass(y, SR)

    # Compare the energy centroid of each envelope.
    def centroid(sig):
        energy = sig ** 2
        return float(np.sum(np.arange(len(sig)) * energy) / np.sum(energy))

    # Zero phase means the burst does not move. A causal filter of this order
    # would shift it by several samples.
    assert abs(centroid(filtered) - centroid(y)) < 1.0


# ---------------------------------------------------------------- Axis C ----

@pytest.mark.parametrize("rate", STRETCH_RATES)
def test_time_stretch_rate_direction(rate):
    """
    RATE SEMANTICS REGRESSION TEST.

    rate > 1 means the speech is delivered FASTER, so the output must be
    SHORTER by that factor. Asserting the direction explicitly is what stops
    the code and the write-up from disagreeing about which way the
    manipulation goes.
    """
    y = _speechlike(duration_s=4.0)
    out = time_stretch_phase_vocoder(y, rate)

    expected_len = len(y) / rate
    # The phase vocoder works in whole frames, so allow a few frames of slack.
    assert abs(len(out) - expected_len) < 0.05 * SR, (
        f"rate={rate} should give ~{expected_len / SR:.2f}s, "
        f"got {len(out) / SR:.2f}s (input {len(y) / SR:.2f}s)"
    )
    assert len(out) < len(y), "rate > 1 must SHORTEN the signal"


def test_time_stretch_rate_below_one_lengthens():
    y = _speechlike(duration_s=2.0)
    out = time_stretch_phase_vocoder(y, 0.5)
    assert len(out) > len(y)


def test_time_stretch_identity_returns_input():
    y = _speechlike(duration_s=1.0)
    assert time_stretch_phase_vocoder(y, 1.0) is y


def test_time_stretch_rejects_nonpositive_rate():
    y = _speechlike(duration_s=1.0)
    with pytest.raises(ValueError):
        time_stretch_phase_vocoder(y, 0.0)


def test_time_stretch_preserves_pitch():
    """Time-scale modification must not shift the spectral peak."""
    t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    out = time_stretch_phase_vocoder(y, 1.5)

    def peak_hz(sig):
        spec = np.abs(np.fft.rfft(sig))
        return np.fft.rfftfreq(len(sig), 1 / SR)[int(np.argmax(spec))]

    assert abs(peak_hz(out) - peak_hz(y)) < 15  # within a couple of FFT bins


@pytest.mark.parametrize("position,duration", SILENCE_CONFIGS)
def test_insert_silence_length_and_content(position, duration):
    y = _speechlike(duration_s=3.0)
    out = insert_silence(y, SR, duration, position)

    assert len(out) == len(y) + int(duration * SR)

    # The inserted block must be genuine digital zeros, not merely quiet.
    is_zero = np.abs(out) < 1e-9
    boundaries = np.where(np.concatenate(([is_zero[0]], is_zero[:-1] != is_zero[1:], [True])))[0]
    runs = np.diff(boundaries)[::2]
    assert np.max(runs) >= int(duration * SR) - 1


def test_insert_silence_at_start_puts_zeros_first():
    y = _speechlike(duration_s=1.0)
    out = insert_silence(y, SR, 0.5, "start")
    assert np.all(out[:int(0.5 * SR)] == 0)


def test_insert_silence_rejects_bad_position():
    y = _speechlike(duration_s=1.0)
    with pytest.raises(ValueError):
        insert_silence(y, SR, 0.5, "end")


# ------------------------------------------------------ Condition naming ----

def test_condition_names_match_the_on_disk_convention():
    """
    These exact strings are baked into filenames and into the result CSVs, so a
    change here silently orphans every file stage 03 previously wrote.
    """
    assert stretch_condition(1.2) == "stretch_1.2x"
    assert stretch_condition(1.5) == "stretch_1.5x"
    assert silence_condition("start", 0.5) == "silence_sta_0.5s"
    assert silence_condition("middle", 1.0) == "silence_mid_1.0s"
    assert silence_condition("middle", 2.0) == "silence_mid_2.0s"