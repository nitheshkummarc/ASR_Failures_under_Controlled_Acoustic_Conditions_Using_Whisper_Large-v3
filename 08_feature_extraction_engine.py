"""
08_feature_extraction_engine.py — Phase 3: DSP Feature Extraction & Analysis
============================================================================
This module serves as the primary analytical engine for the academic report.
It extracts classical DSP speech features (STE, ZCR, Mel-Spec, MFCCs) purely
using Numpy and SciPy, bypassing black-box libraries like librosa to
ensure transparency and mapping directly to the university DSP syllabus.

It calculates the Global Statistical Deviation (MSE of feature means) between
the clean reference audio and the degraded counterparts.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal
import scipy.fftpack
from tqdm import tqdm

from config import METADATA_DIR, RESULTS_DIR, TARGET_SR

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
    """
    # Assuming frames are already windowed. We simply sum the squares.
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
    """
    # Compute where adjacent samples have different signs
    zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    return zcr / frames.shape[1]


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)


def extract_mel_spec_and_mfcc(y: np.ndarray, sr: int, n_fft=512, hop_length=160, n_mels=40, num_ceps=13):
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
    """
    # 1. Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # 2. Framing & Windowing via STFT
    f, t, Zxx = scipy.signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length, window='hann')
    power_spec = np.abs(Zxx) ** 2  # Shape: (n_fft/2 + 1, n_frames)
    
    # 3. Mel-Filterbank
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
            
    # Apply filterbank to power spec
    mel_spec = np.dot(fbank, power_spec)
    mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
    
    # 4. Logarithm
    log_mel_spec = 10 * np.log10(mel_spec)
    
    # 5. DCT-II for MFCCs
    mfcc = scipy.fftpack.dct(log_mel_spec, type=2, axis=0, norm='ortho')[1 : (num_ceps + 1), :]
    
    # 6. Deltas (Velocity)
    delta_mfcc = np.zeros_like(mfcc)
    N = 2
    for t_idx in range(mfcc.shape[1]):
        numerator = np.zeros(num_ceps)
        denominator = 0
        for n in range(1, N + 1):
            t_plus = min(t_idx + n, mfcc.shape[1] - 1)
            t_minus = max(t_idx - n, 0)
            numerator += n * (mfcc[:, t_plus] - mfcc[:, t_minus])
            denominator += n ** 2
        delta_mfcc[:, t_idx] = numerator / (2 * denominator)
        
    return log_mel_spec, mfcc, delta_mfcc, Zxx


def extract_all_features(y: np.ndarray, sr: int):
    """Orchestrates extraction of STE, ZCR, Mel-Spec, and MFCCs."""
    log_mel_spec, mfcc, delta_mfcc, Zxx = extract_mel_spec_and_mfcc(y, sr)
    
    # Reconstruct framed signal from STFT to calculate STE and ZCR
    _, frames_t = scipy.signal.istft(Zxx, fs=sr, nperseg=512, noverlap=512-160, window='hann')
    # Because istft returns a 1D array, we actually need explicit frames for STE/ZCR
    # Let's frame the time-domain signal directly to perfectly align with STFT frames
    frame_len = 512
    hop = 160
    n_frames = 1 + int((len(y) - frame_len) / hop)
    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        frames[i] = y[i * hop : i * hop + frame_len]
        
    ste = extract_ste(frames)
    zcr = extract_zcr(frames)
    
    return {
        "ste_mean": np.mean(ste),
        "zcr_mean": np.mean(zcr),
        "mel_mean": np.mean(log_mel_spec, axis=1),
        "mfcc_mean": np.mean(mfcc, axis=1),
        "delta_mean": np.mean(delta_mfcc, axis=1)
    }

# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================
def main():
    print("======================================================================")
    print("  08_feature_extraction_engine.py")
    print("  Phase 3: DSP Feature Extraction & Global Deviation")
    print("======================================================================")
    
    manifest_b = METADATA_DIR / "axis_b_manifest.csv"
    manifest_c = METADATA_DIR / "axis_c_manifest.csv"
    
    if not manifest_b.exists() or not manifest_c.exists():
        print("  [ERROR] Axis B or C manifests missing. Run 07 first.")
        sys.exit(1)
        
    df_b = pd.read_csv(manifest_b)
    df_c = pd.read_csv(manifest_c)
    
    df_all = pd.concat([df_b, df_c], ignore_index=True)
    
    deviations = []
    
    # To optimize, we cache clean features since they are re-used
    clean_cache = {}
    
    print("\n  [RUN] Extracting DSP features and computing global statistical deviation...")
    
    for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Feature Extract"):
        uid = row["utterance_id"]
        clean_path = row["clean_wav_path"]
        deg_path = row["degraded_wav_path"]
        condition = row["condition"]
        
        # Load audio
        y_deg, sr = sf.read(deg_path)
        
        # Get clean features
        if uid not in clean_cache:
            y_clean, _ = sf.read(clean_path)
            clean_cache[uid] = extract_all_features(y_clean, sr)
        
        feat_clean = clean_cache[uid]
        feat_deg = extract_all_features(y_deg, sr)
        
        # Calculate Global Statistical Deviation (MSE of means)
        # We use global statistics instead of frame-by-frame MSE because 
        # time-stretching and silence insertion fundamentally alter the temporal alignment,
        # making frame-by-frame comparison mathematically invalid.
        ste_mse = (feat_clean["ste_mean"] - feat_deg["ste_mean"]) ** 2
        zcr_mse = (feat_clean["zcr_mean"] - feat_deg["zcr_mean"]) ** 2
        mel_mse = np.mean((feat_clean["mel_mean"] - feat_deg["mel_mean"]) ** 2)
        mfcc_mse = np.mean((feat_clean["mfcc_mean"] - feat_deg["mfcc_mean"]) ** 2)
        delta_mse = np.mean((feat_clean["delta_mean"] - feat_deg["delta_mean"]) ** 2)
        
        deviations.append({
            "utterance_id": uid,
            "condition": condition,
            "ste_mse": ste_mse,
            "zcr_mse": zcr_mse,
            "mel_mse": mel_mse,
            "mfcc_mse": mfcc_mse,
            "delta_mse": delta_mse
        })
        
    df_dev = pd.DataFrame(deviations)
    out_path = RESULTS_DIR / "feature_deviation.csv"
    df_dev.to_csv(out_path, index=False)
    
    print(f"\n  [SAVED] {out_path} ({len(df_dev)} rows)")
    print("  Ready for Step 09 (Inference).")


if __name__ == "__main__":
    main()
