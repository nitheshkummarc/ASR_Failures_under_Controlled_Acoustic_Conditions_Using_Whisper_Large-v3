"""
Stage 0 — Corpus preparation.

Downloads LibriSpeech dev-clean, selects 150 utterances stratified across
speakers (seed 42), converts them to 16 kHz / 16-bit PCM mono WAV, and writes
metadata/utterance_manifest.csv.

Every later stage reads that manifest, so without this the pipeline cannot run
on a fresh clone. Idempotent — re-running is a no-op once the corpus exists.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import main as prepare_corpus

if __name__ == "__main__":
    prepare_corpus()