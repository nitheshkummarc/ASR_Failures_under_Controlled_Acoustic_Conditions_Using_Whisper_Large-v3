"""
07_axis_b_c_augmentation.py — Phase 3: Channel and Temporal Degradations
========================================================================
Applies:
  - Axis B (Channel): Telephonic bandpass filter (300-3400 Hz)
  - Axis C (Temporal): Silence insertion & Time-stretching (custom Phase Vocoder)

Designed to bypass Windows AppLocker restrictions (pure SciPy/NumPy).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal
from tqdm import tqdm

from config import (
    DATA_DIR,
    METADATA_DIR,
    AXIS_B_DIR,
    AXIS_C_DIR,
    TARGET_SR
)

# =============================================================================
# DSP Core: Axis B (Channel)
# =============================================================================
def apply_telephonic_bandpass(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Applies a strict 300-3400 Hz bandpass filter to simulate a traditional
    PSTN (Public Switched Telephone Network) channel.
    Uses a 4th-order Butterworth filter applied forward and backward (filtfilt)
    to ensure zero phase distortion.
    """
    nyq = 0.5 * sr
    low = 300.0 / nyq
    high = 3400.0 / nyq
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    y_filtered = scipy.signal.filtfilt(b, a, y)
    return y_filtered


# =============================================================================
# DSP Core: Axis C (Temporal - Silence)
# =============================================================================
def insert_silence(y: np.ndarray, sr: int, duration_s: float, position: str) -> np.ndarray:
    """
    Inserts exact zeros (digital absolute silence) into the waveform.
    position: 'start' or 'middle'
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
# DSP Core: Axis C (Temporal - Time Stretch via Phase Vocoder)
# =============================================================================
def time_stretch_phase_vocoder(y: np.ndarray, rate: float, n_fft=2048, hop_length=512) -> np.ndarray:
    """
    Pure SciPy/NumPy Phase Vocoder for time-stretching without pitch shifting.
    Written explicitly to bypass librosa/numba AppLocker restrictions on Windows.
    
    DSP Theory:
      1. Compute STFT to separate magnitude and phase.
      2. Interpolate magnitudes along the time axis according to the stretch `rate`.
      3. Compute phase advance (instantaneous frequency) and accumulate to maintain coherence.
      4. Reconstruct audio via inverse-STFT.
    """
    if rate == 1.0:
        return y
        
    # 1. STFT
    f, t, Zxx = scipy.signal.stft(y, fs=TARGET_SR, nperseg=n_fft, noverlap=n_fft-hop_length, window='hann')
    
    # Define new time grid
    time_steps = np.arange(0, Zxx.shape[1], rate)
    
    # 2. Setup Phase Accumulator
    phi_advance = np.linspace(0, np.pi * hop_length, Zxx.shape[0])
    Zxx_new = np.zeros((Zxx.shape[0], len(time_steps)), dtype=complex)
    phase_acc = np.angle(Zxx[:, 0])
    
    # 3. Interpolate and Propagate Phases
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
            # Wrap to [-pi, pi]
            dphi = dphi - 2 * np.pi * np.round(dphi / (2 * np.pi))
            true_freq = phi_advance + dphi
            phase_acc += true_freq
            
        Zxx_new[:, t_step_idx] = mag * np.exp(1j * phase_acc)
        
    # 4. Inverse STFT
    _, y_stretched = scipy.signal.istft(Zxx_new, fs=TARGET_SR, nperseg=n_fft, noverlap=n_fft-hop_length, window='hann')
    
    return y_stretched


# =============================================================================
# BATCH PROCESSING
# =============================================================================
def main():
    print("======================================================================")
    print("  07_axis_b_c_augmentation.py")
    print("  Axis B: Channel (Telephonic) | Axis C: Temporal (Silence, Stretch)")
    print("======================================================================")
    
    manifest_path = METADATA_DIR / "utterance_manifest.csv"
    if not manifest_path.exists():
        print(f"  [ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)
        
    df_clean = pd.read_csv(manifest_path)
    
    records_b = []
    records_c = []
    
    print("\n  [RUN] Generating degraded audio files...")
    
    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Augmenting"):
        uid = row["utterance_id"]
        clean_path = Path(row["wav_path"])
        
        y_clean, sr = sf.read(clean_path)
        
        # ---------------------------------------------------------------------
        # AXIS B: Channel Degradation
        # ---------------------------------------------------------------------
        b_out_path = AXIS_B_DIR / f"{uid}_telephonic.wav"
        if not b_out_path.exists():
            y_b = apply_telephonic_bandpass(y_clean, sr)
            sf.write(b_out_path, y_b, sr)
            
        records_b.append({
            "utterance_id": uid,
            "clean_wav_path": str(clean_path),
            "degraded_wav_path": str(b_out_path),
            "condition": "telephonic"
        })
        
        # ---------------------------------------------------------------------
        # AXIS C: Temporal Degradation (Silence)
        # ---------------------------------------------------------------------
        silence_configs = [
            ("silence_start_0.5s", 0.5, "start"),
            ("silence_mid_1.0s", 1.0, "middle"),
            ("silence_mid_2.0s", 2.0, "middle")
        ]
        
        for name, dur, pos in silence_configs:
            c_out_path = AXIS_C_DIR / f"{uid}_{name}.wav"
            if not c_out_path.exists():
                y_c = insert_silence(y_clean, sr, dur, pos)
                sf.write(c_out_path, y_c, sr)
                
            records_c.append({
                "utterance_id": uid,
                "clean_wav_path": str(clean_path),
                "degraded_wav_path": str(c_out_path),
                "condition": name
            })
            
        # ---------------------------------------------------------------------
        # AXIS C: Temporal Degradation (Time Stretch)
        # ---------------------------------------------------------------------
        stretch_configs = [
            ("stretch_1.2x", 1.2),
            ("stretch_1.5x", 1.5)
        ]
        
        for name, rate in stretch_configs:
            c_out_path = AXIS_C_DIR / f"{uid}_{name}.wav"
            if not c_out_path.exists():
                y_c = time_stretch_phase_vocoder(y_clean, rate)
                # Normalize audio to prevent clipping after phase-vocoder reconstruction
                if np.max(np.abs(y_c)) > 1.0:
                    y_c = y_c / np.max(np.abs(y_c)) * 0.99
                sf.write(c_out_path, y_c, sr)
                
            records_c.append({
                "utterance_id": uid,
                "clean_wav_path": str(clean_path),
                "degraded_wav_path": str(c_out_path),
                "condition": name
            })
            
    # Save Manifests
    df_b = pd.DataFrame(records_b)
    df_b.to_csv(METADATA_DIR / "axis_b_manifest.csv", index=False)
    
    df_c = pd.DataFrame(records_c)
    df_c.to_csv(METADATA_DIR / "axis_c_manifest.csv", index=False)
    
    print("\n======================================================================")
    print("  VALIDATION CHECKS")
    print("======================================================================")
    print(f"  ✓ {len(df_b)} Axis B (Channel) files generated")
    print(f"  ✓ {len(df_c)} Axis C (Temporal) files generated")
    print("\n  Ready for Step 08 (Feature Extraction).")


if __name__ == "__main__":
    main()
