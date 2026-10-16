"""
Tests for the speech-shaped-noise control and the masking decomposition.

The load-bearing test here is test_ssn_preserves_magnitude_spectrum. The entire
informational-masking claim rests on cafe and SSN being spectrally identical —
if phase randomization altered the magnitude spectrum even slightly, the
`cafe - ssn` difference would silently absorb an energetic component and the
headline result would be wrong in a way no other test would catch.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from config import TARGET_SR, NOISE_TYPES, RANDOM_SEED
from src.analysis.masking import (
    distortion_sii,
    speech_weighted_snr, band_snr_profile, modulation_stats, decompose_masking,
    power_spectrum,
)
from src.augmentation.axis_a_noise import (
    generate_speech_shaped_noise, generate_white_noise, generate_pink_noise,
)

SR = TARGET_SR


def _speechlike(duration_s=4.0, seed=0):
    """Amplitude-modulated harmonic stack — speech-ish spectrum and rhythm."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    carrier = sum(np.sin(2 * np.pi * f * t) / (i + 1)
                  for i, f in enumerate([150, 300, 600, 1200, 2400]))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # 4 Hz syllabic rate
    return 0.2 * envelope * carrier / np.max(np.abs(carrier)) + 1e-4 * rng.standard_normal(len(t))


# ------------------------------------------------------- SSN generation ----

def test_ssn_preserves_magnitude_spectrum():
    """
    Phase randomization must leave |FFT| untouched. If this drifts, cafe and SSN
    are no longer spectrally matched and the informational-masking estimate
    silently picks up an energetic component.
    """
    source = _speechlike()
    ssn = generate_speech_shaped_noise(source, np.random.default_rng(45))

    mag_source = np.abs(np.fft.rfft(source))
    mag_ssn = np.abs(np.fft.rfft(ssn))

    # SSN is normalized to unit std, so compare the SHAPE after rescaling.
    scale = mag_ssn.sum() / mag_source.sum()
    assert np.allclose(mag_ssn, mag_source * scale, rtol=1e-6, atol=1e-8)


def test_ssn_output_is_real_and_same_length():
    source = _speechlike(duration_s=1.0)
    ssn = generate_speech_shaped_noise(source, np.random.default_rng(45))
    assert len(ssn) == len(source)
    assert np.isrealobj(ssn)
    assert np.all(np.isfinite(ssn))


@pytest.mark.parametrize("n", [4096, 4097])
def test_ssn_handles_even_and_odd_lengths(n):
    """Odd lengths have no Nyquist bin; even lengths do and it must stay real."""
    source = _speechlike(duration_s=1.0)[:n]
    ssn = generate_speech_shaped_noise(source, np.random.default_rng(1))
    assert len(ssn) == n
    assert np.isrealobj(ssn) and np.all(np.isfinite(ssn))


def test_ssn_destroys_temporal_structure():
    """
    The point of the control: same spectrum, no rhythm. The 4 Hz modulation of
    the source must not survive into the SSN envelope.
    """
    source = _speechlike()
    ssn = generate_speech_shaped_noise(source, np.random.default_rng(45))
    assert modulation_stats(ssn, SR)["envelope_cv"] < modulation_stats(source, SR)["envelope_cv"]


def test_ssn_is_reproducible():
    source = _speechlike(duration_s=1.0)
    a = generate_speech_shaped_noise(source, np.random.default_rng(45))
    b = generate_speech_shaped_noise(source, np.random.default_rng(45))
    assert np.array_equal(a, b)


def test_ssn_differs_from_its_source():
    source = _speechlike(duration_s=1.0)
    ssn = generate_speech_shaped_noise(source, np.random.default_rng(45))
    assert not np.allclose(ssn, source / np.std(source))


def test_noise_source_survives_the_disk_round_trip_unclipped(tmp_path):
    """
    Regression: noise sources are normalized to unit std, which puts ~32% of a
    Gaussian's samples outside [-1, 1]. Writing that as PCM_16 hard-clipped them
    — the stored cafe.wav had 30% of samples at the rails and a crest factor of
    1.42 against a natural babble's 3-5. Clipping is phase-dependent, so it also
    broke the spectral match between cafe and its SSN control.
    """
    import soundfile as sf
    from src.augmentation.axis_a_noise import write_noise_source, compute_rms

    source = np.random.default_rng(3).standard_normal(SR)  # unit std, peak ~4
    path = tmp_path / "noise.wav"
    write_noise_source(path, source, SR)

    stored, _ = sf.read(str(path), dtype="float64")
    assert np.mean(np.abs(stored) >= 0.999) < 0.001, "samples are pinned to the rails"

    # Shape must survive: crest factor is preserved even though level is not.
    assert abs(np.max(np.abs(stored)) / compute_rms(stored)
               - np.max(np.abs(source)) / compute_rms(source)) < 0.1

    # And the normalized waveform matches the original to 16-bit precision.
    assert np.corrcoef(stored, source)[0, 1] > 0.9999


def test_ssn_is_registered_last_in_noise_types():
    """
    Order is load-bearing: scripts/02 advances the RNG streams and the cafe/ssn
    cursors in NOISE_TYPES order, so inserting anywhere but the end would change
    the audio every other noise type receives.
    """
    assert NOISE_TYPES[-1] == "ssn"
    assert NOISE_TYPES[:3] == ["white", "pink", "cafe"]


# ------------------------------------------------- energetic predictor ----

def test_sii_matches_for_spectrally_identical_maskers():
    """
    The validity check the whole decomposition depends on: an SII that is blind
    to structure must score a signal and its phase-randomized twin identically.
    """
    speech = _speechlike()
    babble = _speechlike(seed=7)
    ssn = generate_speech_shaped_noise(babble, np.random.default_rng(45))

    for snr in (20.0, 5.0):
        assert abs(speech_weighted_snr(speech, babble, SR, snr)
                   - speech_weighted_snr(speech, ssn, SR, snr)) < 0.01


def test_sii_increases_with_snr():
    speech = _speechlike()
    noise = generate_white_noise(len(speech), np.random.default_rng(0))
    values = [speech_weighted_snr(speech, noise, SR, s) for s in (0.0, 5.0, 10.0, 20.0)]
    assert values == sorted(values)


def test_sii_is_bounded():
    speech = _speechlike()
    noise = generate_white_noise(len(speech), np.random.default_rng(0))
    for snr in (-30.0, 0.0, 60.0):
        assert 0.0 <= speech_weighted_snr(speech, noise, SR, snr) <= 1.0


def test_pink_leaves_consonant_band_clearer_than_white():
    """
    Energetic masking, stated spectrally: pink dumps energy low, where speech is
    redundant, so it leaves the 1-4 kHz consonant band at a higher SNR than
    white does. This is why pink is the mildest masker in the results.
    """
    speech = _speechlike()
    n = len(speech)
    white = generate_white_noise(n, np.random.default_rng(0))
    pink = generate_pink_noise(n, np.random.default_rng(1))

    def consonant_snr(noise):
        return [b for b in band_snr_profile(speech, noise, SR, 5.0)
                if b["low_hz"] == 1000.0][0]["in_band_snr_db"]

    assert consonant_snr(pink) > consonant_snr(white)


def test_band_profile_covers_every_band():
    speech = _speechlike()
    noise = generate_white_noise(len(speech), np.random.default_rng(0))
    profile = band_snr_profile(speech, noise, SR, 10.0)
    assert len(profile) == 4
    assert all(np.isfinite(b["in_band_snr_db"]) for b in profile)


def test_power_spectrum_is_a_fraction_of_total_power():
    """A flat signal has nothing below the Welch resolution: fractions sum to ~1."""
    x = np.random.default_rng(0).standard_normal(SR * 4)
    assert power_spectrum(x, SR)[1].sum() == pytest.approx(1.0, abs=0.02)


def test_power_spectrum_does_not_hide_pink_infrasonic_power():
    """
    Pink noise keeps ~40 % of its power below ~15 Hz, invisible to Welch. That
    share must stay OUT of the speech bands, or pink's in-band SNR is understated.
    """
    pink = generate_pink_noise(SR * 8, np.random.default_rng(1))
    assert power_spectrum(pink, SR)[1].sum() < 0.8


# ---------------------------------------------------------- modulation ----

def test_modulation_detects_syllabic_rhythm():
    """A 4 Hz modulated signal must show more 2-8 Hz power than steady noise."""
    modulated = _speechlike()
    steady = generate_white_noise(len(modulated), np.random.default_rng(0))
    assert (modulation_stats(modulated, SR)["mod_2_8hz_pct"]
            > modulation_stats(steady, SR)["mod_2_8hz_pct"])


def test_modulation_of_steady_noise_has_no_deep_dips():
    steady = generate_white_noise(SR * 2, np.random.default_rng(0))
    assert modulation_stats(steady, SR)["dip_12db_pct"] == 0.0


def test_modulation_handles_too_short_input():
    assert modulation_stats(np.zeros(10), SR)["envelope_cv"] == 0.0


# --------------------------------------------------------- decomposition ----

def test_decomposition_splits_total_into_two_mechanisms():
    d = decompose_masking(
        {"white": 6.0, "ssn": 7.5, "cafe": 11.0}, baseline_wer=3.0
    )
    assert d["total_excess_wer"] == pytest.approx(8.0)     # 11.0 - 3.0
    assert d["energetic_wer"] == pytest.approx(1.5)        # 7.5 - 6.0
    assert d["informational_wer"] == pytest.approx(3.5)    # 11.0 - 7.5
    assert d["informational_share"] == pytest.approx(3.5 / 8.0)


def test_decomposition_reports_zero_informational_when_control_matches():
    """If babble costs exactly what its spectral twin costs, nothing is central."""
    d = decompose_masking({"white": 6.0, "ssn": 9.0, "cafe": 9.0}, baseline_wer=3.0)
    assert d["informational_wer"] == pytest.approx(0.0)
    assert d["informational_share"] == pytest.approx(0.0)


def test_decomposition_degrades_gracefully_without_the_control():
    """Before SSN audio exists the tool must still print what it can."""
    d = decompose_masking({"white": 6.0, "cafe": 11.0}, baseline_wer=3.0)
    assert d["informational_wer"] is None
    assert d["energetic_wer"] is None
    assert d["total_excess_wer"] == pytest.approx(8.0)


def test_paired_significance_detects_a_consistent_effect():
    """Every utterance worse by 2 errors — both tests must agree it is real."""
    from src.analysis.masking import paired_significance
    st = paired_significance([2] * 60, [20] * 60)
    assert st["diff_pp"] == pytest.approx(10.0)
    assert st["ci_low"] > 0 and st["ci_high"] > 0
    assert st["verdict"] == "significant"
    assert st["n_worse"] == 60 and st["n_better"] == 0


def test_paired_significance_reports_no_effect_for_balanced_noise():
    from src.analysis.masking import paired_significance
    diffs = [1, -1] * 40
    st = paired_significance(diffs, [20] * 80)
    assert st["diff_pp"] == pytest.approx(0.0)
    assert st["ci_low"] <= 0 <= st["ci_high"]
    assert st["verdict"] == "not significant"


def test_paired_significance_flags_disagreement_as_disputed():
    """
    The verdict must not silently pick whichever test is friendlier.

    Many utterances slightly better, a minority substantially worse. The signed
    ranks favour "worse" (Wilcoxon fires), but the total is dominated by a heavy
    tail, so resampling puts zero inside the interval. That is a genuine
    disagreement and must be labelled, not resolved.
    """
    from src.analysis.masking import paired_significance
    st = paired_significance([-1] * 60 + [6] * 16, [20] * 76)
    assert st["wilcoxon_significant"] and not st["bootstrap_significant"]
    assert st["verdict"] == "disputed"


def test_paired_significance_handles_empty_input():
    from src.analysis.masking import paired_significance
    st = paired_significance([], [])
    assert st["verdict"] == "not significant"
    assert st["diff_pp"] == 0.0


def test_paired_significance_is_reproducible():
    from src.analysis.masking import paired_significance
    rng = np.random.default_rng(0)
    diffs = rng.integers(-3, 4, 80).tolist()
    words = rng.integers(8, 40, 80).tolist()
    assert paired_significance(diffs, words) == paired_significance(diffs, words)


def test_decomposition_handles_zero_total_without_dividing_by_zero():
    d = decompose_masking({"white": 3.0, "ssn": 3.0, "cafe": 3.0}, baseline_wer=3.0)
    assert d["informational_share"] is None


def test_decomposition_gives_no_share_when_cafe_is_below_clean():
    """At 20 dB cafe sits below clean; dividing two negatives gave a fake '69%'."""
    d = decompose_masking({"white": 3.53, "ssn": 3.27, "cafe": 2.91}, baseline_wer=3.44)
    assert d["total_excess_wer"] < 0
    assert d["informational_share"] is None


# ------------------------------------------------------ distortion SII ----

def test_distortion_sii_of_an_unchanged_signal_is_one():
    x = _speechlike()
    assert distortion_sii(x, x.copy(), SR) == pytest.approx(1.0)


def test_distortion_sii_matches_standard_sii_for_additive_noise():
    """For additive noise the generalisation must reduce to the usual band SNR."""
    rng = np.random.default_rng(RANDOM_SEED)
    speech = rng.standard_normal(SR * 8)
    noise = generate_pink_noise(SR * 8, np.random.default_rng(1))
    noise *= np.sqrt(np.mean(speech ** 2) / np.mean(noise ** 2)) / 10 ** (5 / 20)  # 5 dB
    standard = speech_weighted_snr(speech, noise, SR, 5.0)
    assert distortion_sii(speech, speech + noise, SR) == pytest.approx(standard, abs=0.02)


def test_distortion_sii_scores_a_removed_band_as_lost():
    """Telephone band-limiting removes the 4 and 8 kHz octaves and most of 250 Hz."""
    from src.augmentation.axis_b_channel import apply_telephonic_bandpass
    x = np.random.default_rng(3).standard_normal(SR * 8)      # flat: every band populated
    sii = distortion_sii(x, apply_telephonic_bandpass(x, SR), SR)
    passband_weight = 0.1671 + 0.2373 + 0.2648                 # 500, 1000, 2000 Hz
    assert passband_weight - 0.05 < sii < passband_weight + 0.10
