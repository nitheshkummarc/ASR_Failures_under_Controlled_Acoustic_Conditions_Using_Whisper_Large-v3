"""
DSP feature extraction engine.
============================================================================
Extracts classical speech features (STE, ZCR, log-Mel spectrogram, MFCCs and
delta-MFCCs) using only NumPy and SciPy.

librosa is deliberately NOT used here. Two reasons:
  1. Academic — this is a signal-processing project, so the Mel filterbank and
     DCT need to be auditable rather than a library call.
  2. Practical — librosa pulls in numba, whose JIT emits .pyd binaries at
     runtime, and those are blocked by the AppLocker policy on the target
     Windows machine.

Callers (scripts/04_run_feature_extraction.py) use these features to compute
the MSE deviation between clean reference audio and its degraded counterparts.
"""

from functools import lru_cache

import numpy as np
import scipy.signal
import scipy.fftpack
from src.logger import setup_logger

logger = setup_logger(__name__)

# Frame geometry. These are module-level because extract_all_features() must
# frame the signal on exactly the same grid the STFT uses, and having two
# copies of 512/160 drift apart would silently misalign STE/ZCR against MFCC.
FRAME_LENGTH = 512      # 32 ms at 16 kHz
HOP_LENGTH = 160        # 10 ms at 16 kHz
N_MELS = 40
NUM_CEPS = 13
PRE_EMPHASIS = 0.97


# =============================================================================
# FEATURE EXTRACTION ENGINE (Syllabus Mapped)
# =============================================================================

def extract_ste(frames: np.ndarray) -> np.ndarray:
    """
    Topic: Time-Domain Analysis of Speech
    Syllabus Mapping: Short-Time Energy (STE)
    
    Theory:
    Speech is a non-stationary signal, so we analyze it in short frames (e.g., 20-30ms) 
    where it is quasi-stationary. The Short-Time Energy provides a measure of the 
    amplitude variation over time, which is crucial for voiced/unvoiced classification 
    and silence detection.
    
    Math: E_n = sum_{m=0}^{N-1} [x(n+m) * w(m)]^2
    Where w(m) is the windowing function applied to the frame.

    NOTE: the frames supplied by extract_all_features() are RECTANGULAR-windowed
    (w(m) = 1), so this reduces to a plain sum of squares. That is intentional —
    STE is used here only as a relative clean-vs-degraded deviation measure, and
    applying a taper would attenuate frame edges identically on both signals
    without changing the comparison.
    """
    return np.sum(frames ** 2, axis=1)


def extract_zcr(frames: np.ndarray) -> np.ndarray:
    """
    Topic: Time-Domain Analysis of Speech
    Syllabus Mapping: Zero-Crossing Rate (ZCR)
    
    Theory:
    ZCR measures the rate at which the signal changes sign within a frame. 
    It is a simple but effective spectral proxy: high ZCR is associated with 
    unvoiced speech (fricatives, noise) which have energy concentrated at high 
    frequencies, whereas low ZCR is associated with voiced speech (vowels).
    
    Math: Z_n = 1/(2N) * sum |sgn[x(m)] - sgn[x(m-1)]|

    NOTE: an exact-zero sample produces two sign changes (1 -> 0 -> -1) and is
    therefore counted twice. Exact zeros are common in 16-bit silence, so ZCR is
    marginally inflated in silent frames. This affects clean and degraded signals
    alike, so the deviation measure remains valid.
    """
    # Compute where adjacent samples have different signs
    zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    return zcr / frames.shape[1]


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def build_mel_filterbank(n_fft: int, sr: int, n_mels: int) -> np.ndarray:
    """
    Build the triangular Mel filterbank matrix, shape (n_mels, n_fft//2 + 1).

    Cached because it is a pure function of its three arguments and the pipeline
    calls it 5,400 times with identical arguments (2,700 clean/degraded pairs
    x 2 signals). Rebuilding it every time cost an O(n_mels * n_bins) Python
    double loop per call for no benefit.

    The filterbank is returned read-only so a caller cannot mutate the shared
    cached array.

    Construction: n_mels + 2 points are spaced uniformly on the Mel scale, so
    every triangle has a left and right neighbour to lean on. Each triangle
    rises linearly from bin_points[m-1] to bin_points[m] and falls to
    bin_points[m+1]. At low frequencies consecutive bin_points can coincide,
    which makes the corresponding range() empty — so the (bin[m] - bin[m-1])
    division is never evaluated and cannot divide by zero.
    """
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

    fbank.flags.writeable = False
    return fbank


def extract_mel_spec_and_mfcc(
    y: np.ndarray,
    sr: int,
    n_fft: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    num_ceps: int = NUM_CEPS,
):
    """
    Topic: Frequency-Domain Analysis & Homomorphic Speech Processing
    Syllabus Mapping: Mel-Frequency Cepstral Coefficients (MFCCs)
    
    Theory:
    1. Pre-emphasis: A high-pass filter (y[n] - 0.97*y[n-1]) balances the frequency spectrum 
       by boosting high frequencies (which naturally decay in human speech).
    2. Framing & Windowing: Signal is split into overlapping frames and multiplied by a 
       Hann window to prevent spectral leakage at the boundaries.
    3. DFT & Periodogram: Compute power spectrum to identify formants.
    4. Mel-Filterbank: Human hearing is non-linear (more discriminative at low frequencies). 
       We apply overlapping triangular filters spaced uniformly on the Mel scale.
    5. Logarithm: Take the log of Mel energies, mimicking human loudness perception and 
       separating the source (glottal pulse) from the filter (vocal tract) homomorphically.
    6. DCT (Discrete Cosine Transform): Decorrelates the overlapping Mel filterbanks,
       yielding linearly independent cepstral coefficients. The lower 13 coefficients
       represent the vocal tract spectral envelope.

    NOTE ON COEFFICIENT INDEXING: the DCT output is sliced [1 : num_ceps+1], so
    C0 (overall frame energy) is DROPPED and the returned coefficients are
    c1..c13, not c0..c12.
    """
    # Without this the short-signal case reaches scipy and fails with
    # "noverlap must be less than nperseg", which does not name the real problem.
    # frame_signal() already handles the same case. Unreachable from the 3-15 s
    # corpus; this guards callers outside the pipeline.
    if len(y) < n_fft:
        raise ValueError(
            f"Signal is {len(y)} samples; the STFT needs at least n_fft={n_fft} "
            f"({n_fft / sr:.3f}s at {sr} Hz)."
        )

    # 1. Pre-emphasis. This rebinds `y` locally, so the caller's array is
    #    untouched — which is why extract_all_features() computes STE/ZCR on the
    #    raw signal while MFCCs come from the pre-emphasized one.
    y = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1])

    # 2. Framing & Windowing via STFT
    f, t, Zxx = scipy.signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length, window='hann')
    power_spec = np.abs(Zxx) ** 2  # Shape: (n_fft/2 + 1, n_frames)

    # 3. Mel-Filterbank (cached — pure function of n_fft, sr, n_mels)
    fbank = build_mel_filterbank(n_fft, sr, n_mels)

    # Apply filterbank to power spec
    mel_spec = np.dot(fbank, power_spec)
    mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
    
    # 4. Logarithm
    log_mel_spec = 10 * np.log10(mel_spec)
    
    # 5. DCT-II for MFCCs
    mfcc = scipy.fftpack.dct(log_mel_spec, type=2, axis=0, norm='ortho')[1 : (num_ceps + 1), :]
    
    # 6. Deltas (Velocity) — regression formula d_t = sum(n*(c[t+n]-c[t-n])) / 2*sum(n^2)
    #    Vectorized over frames: build the shifted views once instead of looping
    #    per frame. Boundaries are clamped by np.clip on the index arrays.
    delta_mfcc = np.zeros_like(mfcc)
    N = 2
    n_frames_mfcc = mfcc.shape[1]
    if n_frames_mfcc > 0:
        idx = np.arange(n_frames_mfcc)
        numerator = np.zeros_like(mfcc)
        denominator = 0
        for n in range(1, N + 1):
            plus = mfcc[:, np.clip(idx + n, 0, n_frames_mfcc - 1)]
            minus = mfcc[:, np.clip(idx - n, 0, n_frames_mfcc - 1)]
            numerator += n * (plus - minus)
            denominator += n ** 2
        delta_mfcc = numerator / (2 * denominator)

    return log_mel_spec, mfcc, delta_mfcc, Zxx


def frame_signal(y: np.ndarray, frame_len: int = FRAME_LENGTH, hop: int = HOP_LENGTH) -> np.ndarray:
    """
    Split a signal into overlapping frames aligned with the STFT grid.

    Returns an array of shape (n_frames, frame_len). Signals shorter than one
    frame yield an empty (0, frame_len) array rather than raising — the previous
    arithmetic produced a negative n_frames and crashed in np.zeros with
    "negative dimensions are not allowed".
    """
    if len(y) < frame_len:
        return np.zeros((0, frame_len))

    n_frames = 1 + (len(y) - frame_len) // hop
    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        frames[i] = y[i * hop: i * hop + frame_len]
    return frames


def extract_all_features(y: np.ndarray, sr: int):
    """
    Orchestrates extraction of STE, ZCR, log-Mel spectrogram and MFCCs.

    Returns both the reduced statistics (*_mean) and the full per-frame
    sequences, so callers can choose whichever the comparison needs.
    """
    log_mel_spec, mfcc, delta_mfcc, _Zxx = extract_mel_spec_and_mfcc(y, sr)

    # STE and ZCR are computed on the RAW signal, framed on the same grid the
    # STFT used so the two feature families stay time-aligned.
    frames = frame_signal(y)

    ste = extract_ste(frames)
    zcr = extract_zcr(frames)
    
    return {
        "ste_mean": np.mean(ste),
        "zcr_mean": np.mean(zcr),
        "mel_mean": np.mean(log_mel_spec, axis=1),
        "mfcc_mean": np.mean(mfcc, axis=1),
        "delta_mean": np.mean(delta_mfcc, axis=1),
        "ste": ste,
        "zcr": zcr,
        "mfcc": mfcc
    }

