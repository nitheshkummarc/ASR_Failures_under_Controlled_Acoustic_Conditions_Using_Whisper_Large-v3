"""
Axis B — Channel Degradation
========================================================================
Simulates a narrowband PSTN (telephone) channel by band-limiting the signal to
300-3400 Hz, the ITU-T G.712 passband.

Implemented with SciPy only (no librosa/numba), which keeps it runnable under
the AppLocker policy on the target Windows machine.

Driven by scripts/03_run_axis_b_c.py.
"""

import numpy as np
import scipy.signal

from config import TELEPHONE_LOW_HZ, TELEPHONE_HIGH_HZ, TELEPHONE_FILTER_ORDER
from src.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# DSP Core: Axis B (Channel)
# =============================================================================
def apply_telephonic_bandpass(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply the 300-3400 Hz bandpass that simulates a traditional PSTN channel.

    Uses a 4th-order Butterworth filter (maximally flat passband, no ripple)
    applied forward and backward via filtfilt.

    filtfilt rather than lfilter is a deliberate experimental control: filtering
    in both directions gives ZERO phase distortion. If phase were altered, it
    would become an uncontrolled second variable alongside the band-limiting
    that this axis is actually measuring. The cost: a 4th-order Butterworth
    BANDPASS is already an 8th-order filter (each lowpass pole becomes a pair),
    and filtfilt squares its magnitude response, so the effective roll-off is
    16th-order and the 300/3400 Hz edges sit at -6 dB rather than -3 dB.

    Args:
        y:  1-D signal.
        sr: Sample rate in Hz.

    Returns:
        The band-limited signal, same length as the input.

    Raises:
        ValueError: if the signal is too short for filtfilt's padding
            (roughly < 3 * filter length, i.e. a few tens of samples).
    """
    nyq = 0.5 * sr
    low = TELEPHONE_LOW_HZ / nyq
    high = TELEPHONE_HIGH_HZ / nyq
    b, a = scipy.signal.butter(TELEPHONE_FILTER_ORDER, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, y)
