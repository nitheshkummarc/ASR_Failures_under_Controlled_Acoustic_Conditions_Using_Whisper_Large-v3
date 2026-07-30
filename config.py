"""
config.py — Shared configuration for the ASR Failures project.
=============================================================================
Single source for all paths, constants, and Whisper settings.
Every script imports from here so nothing is hardcoded in multiple places.
"""

import os
from pathlib import Path

# --- CUDA DLL Path Injection for Windows ---
# ctranslate2 needs cublas64_12.dll and cudnn64_9.dll which we install via pip.
# We must add their bin directories to the PATH so they can be loaded.
_site_packages = Path(__file__).parent / "venv" / "Lib" / "site-packages"
_cublas_bin = _site_packages / "nvidia" / "cublas" / "bin"
_cudnn_bin = _site_packages / "nvidia" / "cudnn" / "bin"
if _cublas_bin.exists():
    os.environ["PATH"] += os.pathsep + str(_cublas_bin)
if _cudnn_bin.exists():
    os.environ["PATH"] += os.pathsep + str(_cudnn_bin)

# =============================================================================
# 1. PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Where the 150 selected clean WAV files live
DATA_DIR = PROJECT_ROOT / "data" / "librispeech_clean"

# Manifest CSV + selection criteria JSON
METADATA_DIR = PROJECT_ROOT / "metadata"

# All experiment results (baseline + future degradation results)
RESULTS_DIR = PROJECT_ROOT / "results"

# Temporary storage for the raw downloaded archive and extracted files
RAW_DOWNLOAD_DIR = PROJECT_ROOT / "_raw_download"

# =============================================================================
# 2. AUDIO CONSTANTS (per proposal §5)
# =============================================================================
TARGET_SR = 16_000          # 16 kHz sample rate
NUM_UTTERANCES = 150        # Number of utterances to select
DURATION_MIN = 3.0          # Minimum utterance duration (seconds)
DURATION_MAX = 15.0         # Maximum utterance duration (seconds)

# =============================================================================
# 3. WHISPER CONFIGURATION
# =============================================================================
# Model: large-v3 chosen for highest baseline accuracy (~2.7% WER on dev-clean).
# This ensures degradation-induced WER spikes are clearly attributable to the
# acoustic condition, not to model limitations.
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"             # RTX 3050
WHISPER_COMPUTE_TYPE = "float16"    # Primary; int8 is the OOM fallback
WHISPER_BEAM_SIZE = 5
WHISPER_LANGUAGE = "en"

# =============================================================================
# 4. DATASET SOURCE
# =============================================================================
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
ARCHIVE_NAME = "dev-clean.tar.gz"

# =============================================================================
# 5. RANDOM SEED (reproducibility)
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# 6. AXIS A — ADDITIVE NOISE CONFIGURATION (proposal §3)
# =============================================================================
# Noise types: White, Pink, Café  |  SNR levels: 20, 15, 10, 5 dB
NOISE_TYPES = ["white", "pink", "cafe"]
SNR_LEVELS = [20, 15, 10, 5]       # dB — from mild to severe

# Degraded audio output
DEGRADED_DIR = PROJECT_ROOT / "data" / "degraded"
AXIS_A_DIR = DEGRADED_DIR / "axis_a"
AXIS_B_DIR = DEGRADED_DIR / "axis_b"
AXIS_C_DIR = DEGRADED_DIR / "axis_c"

# Noise source files (café noise from MUSAN or synthetic babble)
NOISE_SOURCES_DIR = PROJECT_ROOT / "data" / "noise_sources"
CAFE_NOISE_PATH = NOISE_SOURCES_DIR / "cafe.wav"

# Synthetic café babble parameters (used when MUSAN not available)
CAFE_BABBLE_N_SPEAKERS = 15        # Number of overlaid speakers
CAFE_BABBLE_DURATION_S = 120       # 2 minutes of babble generated

# =============================================================================
# 7. ENSURE OUTPUT DIRECTORIES EXIST
# =============================================================================
for _dir in [DATA_DIR, METADATA_DIR, RESULTS_DIR, AXIS_A_DIR, AXIS_B_DIR, AXIS_C_DIR, NOISE_SOURCES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

