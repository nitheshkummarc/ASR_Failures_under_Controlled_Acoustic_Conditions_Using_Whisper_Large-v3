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

# Temporary storage for the raw downloaded archive and extracted files.
# This is the ONE canonical location — src/data/loader.py downloads here and
# src/augmentation/axis_a_noise.py reads the disjoint babble pool from here.
RAW_DOWNLOAD_DIR = PROJECT_ROOT / "_raw_download"
LIBRISPEECH_ROOT = RAW_DOWNLOAD_DIR / "LibriSpeech" / "dev-clean"

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
# Model: large-v3 chosen for the highest baseline accuracy available, so that
# degradation-induced WER spikes are attributable to the acoustic condition
# rather than to model limitations.
#
# NOTE: int8 is forced, not preferred. large-v3 needs ~7-8 GB in float16 and the
# target GPU is a 4 GB RTX 3050, so 8-bit quantization is the only way it fits.
# The measured clean baseline under this configuration is ~3.4% WER, slightly
# above the ~2.7% published for the unquantized model.
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"             # RTX 3050
WHISPER_COMPUTE_TYPE = "int8"    # Strict 8-bit quantization for 4GB RTX 3050
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
# Noise types: White, Pink, Café, SSN  |  SNR levels: 20, 15, 10, 5 dB
#
# ORDER IS LOAD-BEARING — APPEND ONLY. scripts/02 iterates
# utterance -> noise_type -> snr, and the cafe/ssn cursors plus the white/pink
# RNG streams advance per draw. Inserting a type anywhere but the end would
# shift every subsequent draw and silently change the audio for the types that
# follow it, invalidating the committed results.
#
# "ssn" (speech-shaped noise) is the EXPERIMENTAL CONTROL that separates the two
# masking mechanisms. It is the cafe babble with its phase spectrum randomized:
# byte-for-byte the same long-term average spectrum, but no temporal structure
# and no linguistic content. See src/analysis/masking.py.
#
#   ssn  - white  ->  energetic masking attributable to spectral overlap
#   cafe - ssn    ->  informational masking (the part energy cannot explain)
NOISE_TYPES = ["white", "pink", "cafe", "ssn"]
SNR_LEVELS = [20, 15, 10, 5]       # dB — from mild to severe

# Degraded audio output
DEGRADED_DIR = PROJECT_ROOT / "data" / "degraded"
AXIS_A_DIR = DEGRADED_DIR / "axis_a"
AXIS_B_DIR = DEGRADED_DIR / "axis_b"
AXIS_C_DIR = DEGRADED_DIR / "axis_c"

# Noise source files (café noise from MUSAN or synthetic babble)
NOISE_SOURCES_DIR = PROJECT_ROOT / "data" / "noise_sources"
CAFE_NOISE_PATH = NOISE_SOURCES_DIR / "cafe.wav"

# Speech-shaped noise: derived from CAFE_NOISE_PATH by phase randomization, so
# it is always spectrally matched to whatever cafe source is actually in use.
SSN_NOISE_PATH = NOISE_SOURCES_DIR / "ssn.wav"

# Synthetic café babble parameters (used when MUSAN not available)
CAFE_BABBLE_N_SPEAKERS = 15        # Number of overlaid speakers
CAFE_BABBLE_DURATION_S = 120       # 2 minutes of babble generated

# =============================================================================
# 7. AXIS B / C — CONDITION REGISTRY
# =============================================================================
# Single source of truth for every non-Axis-A condition name.
#
# The condition string is used to build the output filename
# ({utterance_id}_{condition}.wav) AND is written into the result CSVs, so
# scripts/03 (generation), scripts/04 (feature extraction), scripts/06
# (plots) and scripts/verify_acoustics.py must all agree on it. Import from
# here instead of re-typing the literals.

# --- Axis B: channel ---
TELEPHONE_LOW_HZ = 300.0
TELEPHONE_HIGH_HZ = 3400.0
TELEPHONE_FILTER_ORDER = 4
AXIS_B_CONDITION = "telephonic"

# --- Axis C: temporal ---
# Time-scale modification. rate > 1 means the speech is played FASTER, so the
# output is SHORTER (librosa convention — see time_stretch_phase_vocoder).
# Matches proposal §3 "Speaking rate: 1x / 1.2x / 1.5x".
STRETCH_RATES = [1.2, 1.5]

# (position, duration_seconds). `position` is 'start' or 'middle'.
SILENCE_CONFIGS = [("start", 0.5), ("middle", 1.0), ("middle", 2.0)]


def stretch_condition(rate: float) -> str:
    """Canonical condition name for a time-scale condition, e.g. 'stretch_1.2x'."""
    return f"stretch_{rate}x"


def silence_condition(position: str, duration_s: float) -> str:
    """Canonical condition name for a silence condition, e.g. 'silence_sta_0.5s'."""
    return f"silence_{position[:3]}_{duration_s}s"


# Every Axis C condition, in generation order. Used by scripts 03, 06 and
# verify_acoustics.py so a new condition only has to be added in one place.
AXIS_C_CONDITIONS = (
    [stretch_condition(r) for r in STRETCH_RATES]
    + [silence_condition(p, d) for p, d in SILENCE_CONFIGS]
)

# Conditions generated per utterance across Axes B and C (1 + 2 + 3 = 6).
CONDITIONS_PER_UTTERANCE_BC = 1 + len(AXIS_C_CONDITIONS)

# Default utterance used for the per-utterance spectrogram / heatmap breakdown.
SHOWCASE_UTTERANCE_ID = "6345-93306-0024"

# MFCC axis labels. The extractor slices the DCT output [1 : num_ceps+1], so C0
# (frame energy) is dropped and the coefficients are c1..c13 — the plots used to
# mislabel them as 0-12.
NUM_CEPS_LABEL_LOW = 1
NUM_CEPS_LABEL_HIGH = 13

# =============================================================================
# 8. ENSURE OUTPUT DIRECTORIES EXIST
# =============================================================================
for _dir in [DATA_DIR, METADATA_DIR, RESULTS_DIR, AXIS_A_DIR, AXIS_B_DIR, AXIS_C_DIR, NOISE_SOURCES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

