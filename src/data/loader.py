"""
Corpus preparation — download, slice, and organize LibriSpeech dev-clean.
=============================================================================
Driven by scripts/00_prepare_corpus.py, which is stage 0 of the pipeline.

This module:
  1. Downloads the LibriSpeech dev-clean archive (~337 MB) from OpenSLR.
  2. Extracts the archive (2,703 utterances across 40 speakers, FLAC format).
  3. Scans all utterances and filters to the 3–15 second duration range.
  4. Selects exactly 150 utterances via stratified sampling across speakers
     (maintains speaker diversity per proposal §7).
  5. Converts selected FLAC → WAV (16 kHz, 16-bit PCM, mono).
  6. Writes:
     - metadata/utterance_manifest.csv  (one row per selected utterance)
     - metadata/selection_criteria.json (documents how the 150 were chosen)

Idempotent: re-running skips completed steps (download, extraction, conversion).
"""

from src.logger import setup_logger
logger = setup_logger(__name__)

import json
import tarfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
import requests
from tqdm import tqdm

from config import (
    DATA_DIR, METADATA_DIR, RAW_DOWNLOAD_DIR, LIBRISPEECH_ROOT,
    TARGET_SR, NUM_UTTERANCES, DURATION_MIN, DURATION_MAX,
    LIBRISPEECH_URL, ARCHIVE_NAME, RANDOM_SEED,
)


# =============================================================================
# STEP 1: Download the archive
# =============================================================================
def download_archive(url: str, dest_path) -> None:
    """Download a file with a progress bar. Skips if file already exists."""
    if dest_path.exists():
        logger.info(f"  [SKIP] Archive already exists: {dest_path.name}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  [GET]  {url}")

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="  Downloading"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info(f"  [DONE] Saved → {dest_path}")


# =============================================================================
# STEP 2: Extract the archive
# =============================================================================
def extract_archive(archive_path, extract_to) -> None:
    """Extract .tar.gz archive. Uses a marker file to avoid re-extraction."""
    marker = extract_to / ".extraction_complete"
    if marker.exists():
        logger.info(f"  [SKIP] Already extracted → {extract_to}")
        return

    extract_to.mkdir(parents=True, exist_ok=True)
    logger.info(f"  [EXTRACT] {archive_path.name} → {extract_to}")

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="  Extracting", unit="files"):
            # filter="data" rejects absolute paths, ".." traversal, and special
            # file types. Available on Python 3.11.4+; fall back gracefully on
            # older interpreters (the source URL is a fixed, trusted OpenSLR
            # endpoint, so this is defence in depth rather than a live risk).
            try:
                tar.extract(member, path=extract_to, filter="data")
            except TypeError:
                tar.extract(member, path=extract_to)

    marker.touch()
    logger.info("  [DONE] Extraction complete")


# =============================================================================
# STEP 3: Parse ground-truth transcripts from .trans.txt files
# =============================================================================
def parse_transcripts(librispeech_root) -> dict:
    """
    Parse all .trans.txt files in the LibriSpeech directory tree.

    Each .trans.txt has lines like:
        1272-128104-0000 MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES

    Returns: dict mapping utterance_id → raw transcript string.
    """
    transcripts = {}
    trans_files = sorted(librispeech_root.rglob("*.trans.txt"))

    if not trans_files:
        raise FileNotFoundError(
            f"No .trans.txt files found under {librispeech_root}. "
            "Check that extraction completed correctly."
        )

    for tf in trans_files:
        with open(tf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split on first space: utterance_id REST_OF_TRANSCRIPT
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    utterance_id, text = parts
                    transcripts[utterance_id] = text

    logger.info(f"  [INFO] Parsed {len(transcripts)} transcripts from {len(trans_files)} files")
    return transcripts


# =============================================================================
# STEP 4: Scan FLAC files and collect metadata
# =============================================================================
def scan_utterances(librispeech_root, transcripts: dict) -> list:
    """
    Walk all .flac files, read their duration, and pair with transcripts.

    Returns a list of dicts, each containing:
        utterance_id, speaker_id, chapter_id, duration_s,
        flac_path, ground_truth_raw
    """
    utterances = []
    flac_files = sorted(librispeech_root.rglob("*.flac"))

    if not flac_files:
        raise FileNotFoundError(
            f"No .flac files found under {librispeech_root}."
        )

    for flac_path in tqdm(flac_files, desc="  Scanning FLACs", unit="files"):
        utterance_id = flac_path.stem  # e.g. "1272-128104-0000"
        parts = utterance_id.split("-")
        if len(parts) != 3:
            continue

        speaker_id, chapter_id, _ = parts

        # Read audio metadata without loading samples (fast)
        info = sf.info(str(flac_path))
        duration = info.duration

        transcript = transcripts.get(utterance_id)
        if transcript is None:
            # Skip utterances with no matching transcript
            continue

        utterances.append({
            "utterance_id": utterance_id,
            "speaker_id": speaker_id,
            "chapter_id": chapter_id,
            "duration_s": round(duration, 3),
            "flac_path": str(flac_path),
            "ground_truth_raw": transcript,
        })

    logger.info(f"  [INFO] Found {len(utterances)} valid utterances with transcripts")
    return utterances


# =============================================================================
# STEP 5: Stratified selection of 150 utterances
# =============================================================================
def stratified_select(utterances: list, n: int = NUM_UTTERANCES) -> list:
    """
    Select `n` utterances stratified across speakers.

    Steps:
      1. Filter to utterances within [DURATION_MIN, DURATION_MAX] seconds.
      2. Group by speaker_id.
      3. Allocate a base count per speaker + distribute the remainder.
      4. Randomly sample from each speaker's pool.

    This ensures no single speaker dominates the selected subset,
    maintaining the speaker diversity noted in proposal §7.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # Duration filter (proposal §5: 3–15 second clips)
    eligible = [u for u in utterances if DURATION_MIN <= u["duration_s"] <= DURATION_MAX]
    logger.info(f"\n  [FILTER] {len(eligible)} utterances in "
          f"[{DURATION_MIN}s, {DURATION_MAX}s] range "
          f"(out of {len(utterances)} total)")

    if len(eligible) < n:
        logger.info(f"  [WARN] Only {len(eligible)} eligible utterances — "
              f"selecting all of them (< {n} requested).")
        return eligible

    # Group by speaker
    by_speaker = defaultdict(list)
    for u in eligible:
        by_speaker[u["speaker_id"]].append(u)

    num_speakers = len(by_speaker)
    base_per_speaker = n // num_speakers
    remainder = n % num_speakers

    logger.info(f"  [STRAT] {num_speakers} speakers, "
          f"~{base_per_speaker} per speaker, "
          f"{remainder} extra distributed round-robin")

    selected = []
    # Sort speakers for reproducibility, then shuffle for fair remainder distribution
    speakers = sorted(by_speaker.keys())
    speaker_order = rng.permutation(len(speakers))

    for rank, idx in enumerate(speaker_order):
        speaker = speakers[idx]
        pool = by_speaker[speaker]
        rng.shuffle(pool)
        # Speakers ranked < remainder get one extra utterance
        take = base_per_speaker + (1 if rank < remainder else 0)
        take = min(take, len(pool))
        selected.extend(pool[:take])

    # Safety: if any speaker had too few, fill from the global pool
    if len(selected) < n:
        selected_ids = {u["utterance_id"] for u in selected}
        remaining_pool = [u for u in eligible if u["utterance_id"] not in selected_ids]
        rng.shuffle(remaining_pool)
        selected.extend(remaining_pool[: n - len(selected)])

    selected = selected[:n]

    # Summary statistics
    speaker_counts = defaultdict(int)
    for u in selected:
        speaker_counts[u["speaker_id"]] += 1

    durations = [u["duration_s"] for u in selected]
    logger.info(f"  [DONE] Selected {len(selected)} utterances from "
          f"{len(speaker_counts)} speakers")
    logger.info(f"         Duration range: {min(durations):.1f}s – {max(durations):.1f}s "
          f"(mean {np.mean(durations):.1f}s)")
    logger.info(f"         Utterances per speaker: "
          f"min={min(speaker_counts.values())}, "
          f"max={max(speaker_counts.values())}, "
          f"mean={np.mean(list(speaker_counts.values())):.1f}")

    return selected


# =============================================================================
# STEP 6: Convert FLAC → WAV (16 kHz, 16-bit PCM, mono)
# =============================================================================
def convert_to_wav(selected: list, output_dir) -> list:
    """
    Convert each selected FLAC file to WAV format.

    Target format (per proposal §5):
      - Sample rate: 16,000 Hz
      - Bit depth:   16-bit PCM
      - Channels:    Mono

    Returns list of WAV file paths (as strings).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = []

    for utt in tqdm(selected, desc="  Converting FLAC → WAV", unit="files"):
        wav_name = utt["utterance_id"] + ".wav"
        wav_path = output_dir / wav_name

        if wav_path.exists():
            wav_paths.append(str(wav_path))
            continue

        # Load FLAC with soundfile (LibriSpeech is already 16kHz mono)
        audio, sr = sf.read(utt["flac_path"])
        
        if sr != TARGET_SR:
            raise ValueError(f"Expected {TARGET_SR} Hz, got {sr} Hz for {utt['flac_path']}")

        # Save as 16-bit PCM WAV
        sf.write(str(wav_path), audio, TARGET_SR, subtype='PCM_16')
        wav_paths.append(str(wav_path))

    logger.info(f"  [DONE] {len(wav_paths)} WAV files in {output_dir}")
    return wav_paths


# =============================================================================
# STEP 7: Write manifest CSV and selection criteria JSON
# =============================================================================
def write_metadata(selected: list, wav_paths: list) -> None:
    """
    Write two metadata files:

    1. utterance_manifest.csv — one row per utterance with columns:
       utterance_id, speaker_id, chapter_id, duration_s, wav_path, ground_truth_raw

    2. selection_criteria.json — documents the selection process for
       reproducibility and academic transparency.
    """
    # convert_to_wav() appends to wav_paths in input order on BOTH the skip path
    # and the write path, so zip() below pairs correctly. Assert it rather than
    # trusting it — a mismatch would silently attach the wrong transcript to
    # every downstream WER computation.
    if len(selected) != len(wav_paths):
        raise RuntimeError(
            f"Manifest misalignment: {len(selected)} utterances but "
            f"{len(wav_paths)} WAV paths."
        )

    # Build the manifest DataFrame
    records = []
    for utt, wav_path in zip(selected, wav_paths):
        records.append({
            "utterance_id": utt["utterance_id"],
            "speaker_id": utt["speaker_id"],
            "chapter_id": utt["chapter_id"],
            "duration_s": utt["duration_s"],
            "wav_path": wav_path,
            "ground_truth_raw": utt["ground_truth_raw"],
        })

    df = pd.DataFrame(records)
    manifest_path = METADATA_DIR / "utterance_manifest.csv"
    df.to_csv(manifest_path, index=False)
    logger.info(f"  [SAVED] {manifest_path} ({len(df)} rows)")

    # Build selection criteria
    criteria = {
        "source_dataset": "LibriSpeech dev-clean",
        "source_url": LIBRISPEECH_URL,
        "total_utterances_in_dev_clean": "2703",
        "selection_count": len(selected),
        "duration_filter_s": [DURATION_MIN, DURATION_MAX],
        "sampling_strategy": "stratified by speaker_id",
        "random_seed": RANDOM_SEED,
        "output_format": "WAV 16kHz 16-bit PCM mono",
        "speakers_represented": len(set(u["speaker_id"] for u in selected)),
        "duration_stats": {
            "min": min(u["duration_s"] for u in selected),
            "max": max(u["duration_s"] for u in selected),
            "mean": round(float(np.mean([u["duration_s"] for u in selected])), 2),
        },
    }
    criteria_path = METADATA_DIR / "selection_criteria.json"
    with open(criteria_path, "w", encoding="utf-8") as f:
        json.dump(criteria, f, indent=2)
    logger.info(f"  [SAVED] {criteria_path}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    """
    Build the corpus. Idempotent at every step — safe to re-run.

    Returns early (without downloading ~337 MB) if the manifest and all its WAV
    files are already present, so this can sit at the head of the pipeline
    without cost on subsequent runs.
    """
    logger.info('=' * 70)
    logger.info('PHASE 0: LIBRISPEECH DOWNLOAD, SELECTION, AND PREPARATION')
    logger.info('=' * 70)

    manifest_path = METADATA_DIR / 'utterance_manifest.csv'
    if manifest_path.exists():
        existing_wavs = list(DATA_DIR.glob('*.wav'))
        if len(existing_wavs) >= NUM_UTTERANCES:
            logger.info(f'  [SKIP] Corpus already prepared: {manifest_path} '
                        f'with {len(existing_wavs)} WAV files.')
            logger.info('=' * 70)
            return

    # RAW_DOWNLOAD_DIR is the ONE canonical location for the archive and its
    # extraction. src/augmentation/axis_a_noise.py reads the disjoint babble
    # pool from the same place via LIBRISPEECH_ROOT.
    RAW_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DOWNLOAD_DIR / ARCHIVE_NAME

    download_archive(LIBRISPEECH_URL, archive_path)
    extract_archive(archive_path, RAW_DOWNLOAD_DIR)

    transcripts = parse_transcripts(LIBRISPEECH_ROOT)
    utterances = scan_utterances(LIBRISPEECH_ROOT, transcripts)
    selected = stratified_select(utterances)

    wav_paths = convert_to_wav(selected, DATA_DIR)
    write_metadata(selected, wav_paths)

    logger.info('=' * 70)
    logger.info('PHASE 0 COMPLETE')
    logger.info('=' * 70)

if __name__ == '__main__':
    main()

