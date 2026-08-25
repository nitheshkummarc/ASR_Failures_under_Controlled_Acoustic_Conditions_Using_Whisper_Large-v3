"""
Tests for the Axis A SNR mixing math.

Uses synthetic sinusoids rather than real audio: RMS is exact and deterministic,
so these cannot go flaky.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.augmentation.axis_a_noise import (
    compute_rms, mix_at_target_snr, generate_white_noise, generate_pink_noise,
)

SR = 16000


def _sines(amplitude=0.1):
    t = np.linspace(0, 1, SR)
    clean = amplitude * np.sin(2 * np.pi * 440 * t)
    noise = amplitude * np.sin(2 * np.pi * 1000 * t)
    return clean, noise


def _measured_snr(clean, mixed):
    """Back-measure SNR by recovering the noise through subtraction."""
    return 20 * np.log10(compute_rms(clean) / compute_rms(mixed - clean))


def test_compute_rms_of_silence_is_zero():
    assert compute_rms(np.zeros(1000)) == 0.0


def test_compute_rms_of_unit_sine():
    t = np.linspace(0, 1, SR, endpoint=False)
    # RMS of a unit-amplitude sine is 1/sqrt(2)
    assert np.isclose(compute_rms(np.sin(2 * np.pi * 440 * t)), 1 / np.sqrt(2), atol=1e-4)


def test_snr_math_hits_target_exactly():
    """The core guarantee: the mixture achieves the requested SNR."""
    # Amplitudes are small so the sum stays inside [-1, 1] and the anti-clip
    # branch is NOT taken — that path is tested separately below.
    clean, noise = _sines(amplitude=0.1)

    for target in (20.0, 15.0, 10.0, 5.0):
        mixed = mix_at_target_snr(clean.copy(), noise.copy(), target)
        assert np.isclose(_measured_snr(clean, mixed), target, atol=1e-5), \
            f"Expected {target} dB, got {_measured_snr(clean, mixed)} dB"


def test_anticlip_preserves_snr_ratio():
    """
    When the mixture would clip, the WHOLE signal is scaled down. Because both
    components are divided by the same peak, the SNR ratio survives exactly.

    This is the subtlest branch in the function and was previously untested.
    """
    clean, noise = _sines(amplitude=1.0)  # sum can reach 2.0 -> clipping path
    target = 10.0

    mixed = mix_at_target_snr(clean.copy(), noise.copy(), target)

    assert np.max(np.abs(mixed)) <= 1.0 + 1e-12, "anti-clip did not bound the output"

    # The clean component was scaled by the same factor, so recover the factor
    # and compare against the correspondingly scaled clean signal.
    scale = np.max(np.abs(clean + noise * (
        (compute_rms(clean) / (10 ** (target / 20))) / compute_rms(noise)
    )))
    assert np.isclose(_measured_snr(clean / scale, mixed), target, atol=1e-5)


def test_silent_noise_returns_clean_unchanged():
    clean, _ = _sines()
    mixed = mix_at_target_snr(clean.copy(), np.zeros_like(clean), 10.0)
    assert np.array_equal(mixed, clean)


def test_silent_clean_returns_clean_unchanged():
    """SNR is undefined against a silent signal, so nothing is fabricated."""
    _, noise = _sines()
    silent = np.zeros_like(noise)
    mixed = mix_at_target_snr(silent.copy(), noise.copy(), 10.0)
    assert np.array_equal(mixed, silent)


def test_mix_does_not_mutate_inputs():
    clean, noise = _sines()
    clean_before, noise_before = clean.copy(), noise.copy()
    mix_at_target_snr(clean, noise, 10.0)
    assert np.array_equal(clean, clean_before)
    assert np.array_equal(noise, noise_before)


def test_lower_snr_means_more_noise():
    """Monotonicity: 5 dB must be noisier than 20 dB."""
    clean, noise = _sines()
    noise_5 = compute_rms(mix_at_target_snr(clean.copy(), noise.copy(), 5.0) - clean)
    noise_20 = compute_rms(mix_at_target_snr(clean.copy(), noise.copy(), 20.0) - clean)
    assert noise_5 > noise_20


def test_white_noise_is_reproducible_and_spectrally_flat():
    a = generate_white_noise(8192, np.random.default_rng(42))
    b = generate_white_noise(8192, np.random.default_rng(42))
    assert np.array_equal(a, b), "same seed must give identical noise"

    spec = np.abs(np.fft.rfft(a)) ** 2
    low = spec[1:len(spec) // 2].mean()
    high = spec[len(spec) // 2:].mean()
    # Flat PSD: the two halves of the spectrum carry comparable energy.
    assert 0.5 < low / high < 2.0


def test_pink_noise_has_more_low_frequency_energy_than_white():
    n = 16384
    pink = generate_pink_noise(n, np.random.default_rng(42))
    spec = np.abs(np.fft.rfft(pink)) ** 2
    low = spec[1:len(spec) // 8].mean()
    high = spec[len(spec) // 2:].mean()
    # 1/f means low frequencies dominate by a wide margin.
    assert low > 10 * high


def test_pink_noise_has_no_dc_component():
    pink = generate_pink_noise(8192, np.random.default_rng(42))
    assert abs(np.mean(pink)) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all SNR math tests passed!")