"""
Shared plumbing for the inference stages (scripts/01, 02, 03).

Before this module existed, the "degrade -> write -> transcribe -> normalize ->
evaluate -> writerow -> flush" sequence was copy-pasted five times, and the
resume-set loader was duplicated four times with a bare `except Exception: pass`
that silently produced duplicate CSV rows.
"""

import csv
from pathlib import Path
from typing import Callable, Optional, Set

import soundfile as sf

from config import TARGET_SR
from src.data.text_normalizer import normalize_text
from src.inference.evaluator import evaluate_metrics
from src.logger import setup_logger

logger = setup_logger(__name__)


def load_resume_keys(csv_path: Path, key_fn: Callable[[dict], tuple]) -> Set[tuple]:
    """
    Rebuild the set of already-processed rows from an existing result CSV.

    Args:
        csv_path: the result CSV. Must exist.
        key_fn:   maps a CSV row to its natural key.

    Returns:
        Set of keys already present in the file.

    Raises:
        RuntimeError: if the file cannot be read or a row cannot be keyed.

    WHY THIS RAISES INSTEAD OF RETURNING WHAT IT MANAGED TO READ:
    the previous implementation wrapped this in `except Exception: pass`. A
    truncated final row (from a hard kill) would abort the read partway, leave a
    half-filled resume set, and the run would then APPEND rows the file already
    contained. Aggregate WER would silently double-count those utterances, with
    nothing in the output indicating it happened. A partial resume set is
    strictly worse than refusing to start.
    """
    keys: Set[tuple] = set()
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                keys.add(key_fn(row))
    except Exception as e:
        raise RuntimeError(
            f"Could not read existing results from {csv_path}: {e}\n"
            f"Refusing to resume from a partially-read file, because appending "
            f"would duplicate rows that are already present. Inspect the file, "
            f"then either repair it or delete it to re-run this stage."
        ) from e

    return keys


def open_result_csv(csv_path: Path, fieldnames: list, key_fn: Callable[[dict], tuple]) -> Set[tuple]:
    """
    Prepare a result CSV for appending and return the resume-key set.

    Creates the file with a header if it does not exist; otherwise reads the
    existing rows so they can be skipped.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        keys = load_resume_keys(csv_path, key_fn)
        if keys:
            logger.info(f"Found {len(keys)} existing records in {csv_path.name}. Resuming...")
        return keys

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    return set()


def write_degraded(y, path: Path, sr: int = TARGET_SR) -> None:
    """
    Write a degraded signal as 16-bit PCM WAV.

    The disk round-trip is not optional: WhisperEngine.transcribe takes a path,
    not an array. It also means every downstream measurement sees the same
    quantized signal the model saw, which is the correct thing to analyse.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")


def transcribe_and_score(engine, wav_path: Path, reference: str) -> dict:
    """
    Transcribe one file and score it against an already-normalized reference.

    Returns a dict with the shared result fields:
        predicted_text, wer, cer, substitutions, insertions, deletions

    Raises:
        TranscriptionOOMError: propagated from the engine so the caller can skip
            the row rather than record a fabricated 100%-WER result.
    """
    raw_prediction = engine.transcribe(str(wav_path))
    normalized_prediction = normalize_text(raw_prediction)
    metrics = evaluate_metrics(reference, normalized_prediction)

    return {
        "predicted_text": normalized_prediction,
        "predicted_text_raw": raw_prediction,
        "wer": round(metrics["wer"], 6),
        "cer": round(metrics["cer"], 6),
        "substitutions": metrics["substitutions"],
        "insertions": metrics["insertions"],
        "deletions": metrics["deletions"],
        "hits": metrics["hits"],
        "word_count_ref": metrics["word_count_ref"],
    }


def read_manifest(manifest_path: Path) -> list:
    """
    Read the utterance manifest, failing loudly if it is missing.

    All three inference stages need this; previously only stage 01 checked, so
    stages 02 and 03 died with a bare traceback instead of an actionable message.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run `python scripts/00_prepare_corpus.py` (or `python main.py`) first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Manifest {manifest_path} contains no utterances.")

    return rows


def get_reference(row: dict, utterance_id: str) -> Optional[str]:
    """
    Pull the normalized ground truth from a manifest row.

    Returns None (and logs) when it is missing or blank, so the caller can skip
    the utterance before doing any expensive work.
    """
    reference = (row.get("ground_truth_normalized") or "").strip()
    if not reference:
        logger.error(f"Empty ground truth for utterance {utterance_id}. Skipping.")
        return None
    return reference


class SkipCounter:
    """Tracks rows skipped for reasons that must not pass silently."""

    def __init__(self, label: str):
        self.label = label
        self.oom = 0
        self.no_reference = 0

    def report(self) -> None:
        if self.oom:
            logger.warning(
                f"[{self.label}] {self.oom} row(s) SKIPPED due to out-of-memory. "
                f"These are absent from the results CSV — re-run this stage to "
                f"fill them in. They were NOT recorded as failures."
            )
        if self.no_reference:
            logger.warning(
                f"[{self.label}] {self.no_reference} row(s) skipped: empty ground truth."
            )