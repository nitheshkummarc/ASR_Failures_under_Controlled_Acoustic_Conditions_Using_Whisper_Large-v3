"""
Energetic vs informational masking.
=============================================================================
Quantifies WHY a masker degrades recognition, not just how much.

THE TWO MECHANISMS
------------------
ENERGETIC MASKING is peripheral. Noise energy overlaps the speech energy in
frequency and swamps it, so the information never survives the front end. It is
predictable from the spectra alone: compute the SNR in each frequency band and
weight those bands by how much they matter for speech.

INFORMATIONAL MASKING is central. The masker is audible and separable from the
target, yet still interferes — because it competes for the same interpretive
machinery. Competing speech is the canonical case. It is NOT predictable from
the spectra, which is exactly what makes it hard to demonstrate.

WHY A CONTROL CONDITION IS REQUIRED
-----------------------------------
Comparing cafe babble against white noise cannot establish mechanism, because
the two differ in two ways simultaneously: babble puts its energy in the
speech bands AND carries linguistic content. Either difference could explain the
WER gap.

So the experiment adds SSN (speech-shaped noise): the cafe babble with its phase
spectrum randomized. Magnitude spectrum preserved exactly, structure destroyed.
That isolates one variable at a time:

    ssn  - white   ->  ENERGETIC    (moving energy into the speech bands)
    cafe - ssn     ->  INFORMATIONAL (same energy, now with linguistic content)

The `cafe - ssn` term is strictly "informational + temporal-modulation" masking,
since babble also fluctuates while SSN is stationary. modulation_stats() below
measures that fluctuation so the residual modulation contribution can be
reported rather than assumed away — for 15-talker babble the envelope is nearly
steady, so the term is dominated by the informational component.

THE ENERGETIC PREDICTOR
-----------------------
speech_weighted_snr() implements a simplified Speech Intelligibility Index
(ANSI S3.5-1997): band SNRs are clipped to a +/-15 dB dynamic range, mapped to a
0-1 audibility, and combined with the standard octave-band importance weights.
It deliberately omits upward spread of masking and absolute hearing threshold —
those model a human listener, and the "listener" here is a neural network.
What it provides is a principled, spectrum-only prediction of energetic masking,
which is all the decomposition needs.

Its key use is as a validity check: cafe and SSN must score the same, because
they have the same spectrum. If they do, then any WER difference between them
cannot be energetic, and the decomposition above is sound.
"""

from typing import Mapping, Sequence

import numpy as np
import scipy.signal

# ANSI S3.5-1997 octave-band importance weights. These say how much each octave
# contributes to intelligibility: the 1-4 kHz region carrying consonant place
# and manner cues dominates, while the F0 region contributes little because its
# information is largely redundant.
SII_OCTAVE_BANDS_HZ = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
SII_BAND_WEIGHTS = (0.0617, 0.1671, 0.2373, 0.2648, 0.2142, 0.0549)

# SII dynamic range: below -15 dB SNR a band contributes nothing, above +15 dB it
# contributes fully.
SII_SNR_FLOOR_DB = -15.0
SII_SNR_CEILING_DB = 15.0

# Analysis bands used for the human-readable spectral profile. Chosen to align
# with speech acoustics rather than with round numbers.
SPEECH_BANDS = (
    (50.0, 300.0, "F0 / voicing"),
    (300.0, 1000.0, "F1"),
    (1000.0, 4000.0, "F2-F3 / consonants"),
    (4000.0, 8000.0, "fricatives"),
)


def power_spectrum(x: np.ndarray, sr: int, nperseg: int = 1024):
    """
    Welch power spectral density, normalized to sum to 1.

    Normalizing makes the result a pure SHAPE — the distribution of energy over
    frequency, independent of overall level. Level is set separately by the
    target SNR, so keeping the two apart is what lets band SNRs be computed for
    any nominal SNR without regenerating audio.
    """
    freqs, psd = scipy.signal.welch(x, sr, nperseg=min(nperseg, len(x)))
    total = psd.sum()
    if total <= 0:
        return freqs, psd
    return freqs, psd / total


def band_snr_profile(
    speech: np.ndarray,
    noise: np.ndarray,
    sr: int,
    broadband_snr_db: float,
    bands: Sequence = SPEECH_BANDS,
) -> list:
    """
    Per-band SNR that results from mixing at a given BROADBAND SNR.

    This is the core of the energetic account. The experiment sets SNR on
    broadband RMS, but recognition depends on the SNR in the bands that actually
    carry phonetic information. A masker whose energy sits outside those bands is
    far less damaging at the same nominal SNR — and one shaped like speech is far
    more damaging.

    Args:
        speech:           reference speech signal.
        noise:            masker signal.
        sr:               sample rate.
        broadband_snr_db: the nominal SNR the mixture is created at.
        bands:            (low_hz, high_hz, label) triples.

    Returns:
        List of dicts with band label, edges, the speech/noise energy share in
        that band, and the resulting in-band SNR in dB.
    """
    freqs, speech_psd = power_spectrum(speech, sr)
    _, noise_psd = power_spectrum(noise, sr)

    # At a broadband SNR of S dB the total noise power is speech_power/10^(S/10).
    # Both spectra are normalized to unit total power, so scaling the noise
    # spectrum by this factor reproduces the mixture's band-by-band balance.
    noise_scale = 10.0 ** (-broadband_snr_db / 10.0)

    profile = []
    for low, high, label in bands:
        mask = (freqs >= low) & (freqs < high)
        speech_power = float(speech_psd[mask].sum())
        noise_power = float(noise_psd[mask].sum()) * noise_scale

        if noise_power <= 0:
            snr_db = float("inf")
        elif speech_power <= 0:
            snr_db = float("-inf")
        else:
            snr_db = 10.0 * np.log10(speech_power / noise_power)

        profile.append({
            "band": label,
            "low_hz": low,
            "high_hz": high,
            "speech_energy_pct": speech_power * 100.0,
            "noise_energy_pct": float(noise_psd[mask].sum()) * 100.0,
            "in_band_snr_db": snr_db,
        })
    return profile


def speech_weighted_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    sr: int,
    broadband_snr_db: float,
) -> float:
    """
    Simplified Speech Intelligibility Index — the energetic-masking predictor.

    Returns a value in [0, 1]: the fraction of speech information that survives
    the masker, judged from the spectra alone. 1.0 means every important band is
    at or above +15 dB SNR; 0.0 means every important band is buried.

    Because it is computed from the magnitude spectra only, it is blind to
    linguistic content by construction. That blindness is the point: two maskers
    with the same spectrum score identically here, so any recognition difference
    between them must come from somewhere other than energy.
    """
    freqs, speech_psd = power_spectrum(speech, sr)
    _, noise_psd = power_spectrum(noise, sr)
    noise_scale = 10.0 ** (-broadband_snr_db / 10.0)

    sii = 0.0
    for centre, weight in zip(SII_OCTAVE_BANDS_HZ, SII_BAND_WEIGHTS):
        low, high = centre / np.sqrt(2.0), centre * np.sqrt(2.0)
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            continue

        speech_power = float(speech_psd[mask].sum())
        noise_power = float(noise_psd[mask].sum()) * noise_scale
        if speech_power <= 0:
            continue
        if noise_power <= 0:
            audibility = 1.0
        else:
            snr_db = 10.0 * np.log10(speech_power / noise_power)
            clipped = np.clip(snr_db, SII_SNR_FLOOR_DB, SII_SNR_CEILING_DB)
            audibility = (clipped - SII_SNR_FLOOR_DB) / (SII_SNR_CEILING_DB - SII_SNR_FLOOR_DB)

        sii += weight * audibility

    return float(sii)


def modulation_stats(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> dict:
    """
    Temporal envelope statistics — how much the masker fluctuates.

    This measures glimpsing opportunity. A masker with deep amplitude dips lets a
    listener (or a model) catch fragments of the target in the gaps, so it is
    less damaging than its average level implies. A steady masker offers no such
    gaps.

    It also bounds the interpretation of the cafe-minus-SSN term: SSN is
    stationary by construction, so if the babble is also nearly steady, the two
    differ in linguistic content far more than in fluctuation, and the term can
    be read as predominantly informational.

    Returns:
        envelope_cv:      std/mean of the envelope. ~0 is steady, ~1 is speechlike.
        dip_6db_pct:      % of frames more than 6 dB below the median level.
        dip_12db_pct:     % of frames more than 12 dB below the median level.
        mod_2_8hz_pct:    % of envelope modulation power at the 2-8 Hz syllabic
                          rate — the signature of speech rhythm.
    """
    frame = max(1, int(sr * frame_ms / 1000.0))
    usable = len(x) // frame * frame
    if usable < frame * 4:
        return {"envelope_cv": 0.0, "dip_6db_pct": 0.0,
                "dip_12db_pct": 0.0, "mod_2_8hz_pct": 0.0}

    envelope = np.sqrt(np.mean(x[:usable].reshape(-1, frame) ** 2, axis=1))
    mean_env = float(envelope.mean())
    envelope_db = 20.0 * np.log10(envelope + 1e-12)
    median_db = float(np.median(envelope_db))

    centred = envelope - mean_env
    mod_freqs, mod_psd = scipy.signal.welch(
        centred, sr / frame, nperseg=min(512, len(centred))
    )
    total_mod = float(mod_psd.sum())
    syllabic = float(mod_psd[(mod_freqs >= 2.0) & (mod_freqs <= 8.0)].sum())

    return {
        "envelope_cv": float(envelope.std() / mean_env) if mean_env > 0 else 0.0,
        "dip_6db_pct": float((envelope_db < median_db - 6.0).mean() * 100.0),
        "dip_12db_pct": float((envelope_db < median_db - 12.0).mean() * 100.0),
        "mod_2_8hz_pct": (syllabic / total_mod * 100.0) if total_mod > 0 else 0.0,
    }


def decompose_masking(
    wer_by_noise: Mapping[str, float],
    baseline_wer: float,
    structured: str = "cafe",
    spectral_control: str = "ssn",
    reference: str = "white",
) -> dict:
    """
    Split a masker's total damage into its energetic and informational parts.

    Given WER at one SNR for the structured masker, its phase-randomized control
    and a spectrally-neutral reference:

        total         = WER(cafe)  - WER(clean)
        energetic     = WER(ssn)   - WER(white)
            what changes when energy MOVES INTO the speech bands, with content
            held absent in both terms
        informational = WER(cafe)  - WER(ssn)
            what changes when linguistic structure is ADDED, with spectrum and
            broadband SNR held identical

    Returns the three quantities plus `informational_share`, the fraction of the
    structured masker's excess-over-baseline that spectral overlap cannot
    account for. A share near 0 means the babble is just well-aimed noise; a
    share near 1 means essentially all of its damage is central.

    Returns None for any term whose inputs are absent, so a partially-complete
    results table degrades to fewer numbers instead of raising.
    """
    def get(name):
        value = wer_by_noise.get(name)
        return float(value) if value is not None else None

    structured_wer = get(structured)
    control_wer = get(spectral_control)
    reference_wer = get(reference)

    total = None if structured_wer is None else structured_wer - baseline_wer
    energetic = (
        None if (control_wer is None or reference_wer is None)
        else control_wer - reference_wer
    )
    informational = (
        None if (structured_wer is None or control_wer is None)
        else structured_wer - control_wer
    )

    share = None
    if informational is not None and total not in (None, 0.0):
        share = informational / total

    return {
        "total_excess_wer": total,
        "energetic_wer": energetic,
        "informational_wer": informational,
        "informational_share": share,
        "control_wer": control_wer,
        "structured_wer": structured_wer,
        "reference_wer": reference_wer,
    }
