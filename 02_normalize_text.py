"""
02_normalize_text.py — Text normalization for ground truth and ASR predictions.
=============================================================================
Per proposal §5:
  "lowercased, punctuation stripped, numbers normalized (e.g. '100' → 'one hundred')
   to align with ASR output formatting before WER/CER computation."

This module provides:
  - normalize_text(text): The core normalization function, applied identically
    to BOTH ground truth and Whisper predictions before jiwer comparison.
  - A CLI entry point that normalizes the ground_truth_raw column in
    utterance_manifest.csv and writes a ground_truth_normalized column.

CRITICAL DESIGN RULE:
  The same normalization function is applied to both reference and hypothesis.
  This prevents false WER/CER inflation from casing or punctuation mismatches.
"""

import re
import sys
import string

import pandas as pd
from num2words import num2words

from config import METADATA_DIR


# =============================================================================
# TEXT NORMALIZATION PIPELINE
# =============================================================================

def _convert_numbers_to_words(text: str) -> str:
    """
    Find all digit sequences in the text and replace them with English words.

    Handles:
      - Integers:    "100" → "one hundred"
      - Decimals:    "3.5" → "three point five"
      - Negatives:   "-10" → "minus ten"
      - Large nums:  "1000000" → "one million"
      - Ordinal-ish: We convert all as cardinals; ordinals like "1st" are
                     handled by stripping the suffix first.

    Edge cases:
      - Numbers embedded in words (e.g. "mp3") are left alone — the regex
        only matches standalone digit sequences.
      - Very large numbers that num2words can't handle fall back to
        spelling out individual digits.
    """
    def _replace_match(match):
        num_str = match.group(0)
        try:
            # Handle decimals
            if "." in num_str:
                integer_part, decimal_part = num_str.split(".", 1)
                result = num2words(int(integer_part)) if integer_part else "zero"
                result += " point"
                for digit in decimal_part:
                    result += " " + num2words(int(digit))
                return result
            else:
                return num2words(int(num_str))
        except (ValueError, OverflowError):
            # Fallback: spell out each digit individually
            return " ".join(num2words(int(d)) for d in num_str if d.isdigit())

    # Match standalone numbers: optional negative sign, digits, optional decimal
    # Use word boundaries to avoid matching digits inside words like "mp3"
    pattern = r"(?<!\w)-?\b\d+(?:\.\d+)?\b"
    return re.sub(pattern, _replace_match, text)


def _strip_ordinal_suffixes(text: str) -> str:
    """
    Remove ordinal suffixes so numbers can be converted as cardinals.
    E.g., "1st" → "1", "22nd" → "22", "3rd" → "3", "4th" → "4"
    """
    return re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)


def _strip_punctuation(text: str) -> str:
    """
    Remove all characters that are not alphanumeric or whitespace.
    This includes periods, commas, apostrophes, hyphens, etc.

    Note: hyphens in num2words output (e.g. "forty-two") are replaced
    with spaces so they become separate words for jiwer alignment.
    """
    # Replace hyphens with spaces first (preserves word boundaries)
    text = text.replace("-", " ")
    # Remove all remaining punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces into one and strip leading/trailing."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """
    Master normalization pipeline applied to BOTH ground truth and predictions.

    Steps (order matters):
      1. Lowercase
      2. Strip ordinal suffixes (1st → 1, before number conversion)
      3. Convert numbers to words (100 → one hundred)
      4. Strip punctuation (including hyphens → spaces)
      5. Collapse whitespace

    Args:
        text: Raw text string (ground truth or ASR prediction).

    Returns:
        Normalized text string ready for jiwer comparison.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = _strip_ordinal_suffixes(text)
    text = _convert_numbers_to_words(text)
    text = _strip_punctuation(text)
    text = _collapse_whitespace(text)

    return text


# =============================================================================
# CLI: Normalize ground truths in the manifest
# =============================================================================
def main():
    """
    Load utterance_manifest.csv, normalize the ground_truth_raw column,
    and write the result as ground_truth_normalized.
    """
    print("=" * 70)
    print("  02_normalize_text.py")
    print("  Normalize ground-truth transcripts for jiwer evaluation")
    print("=" * 70)

    manifest_path = METADATA_DIR / "utterance_manifest.csv"

    if not manifest_path.exists():
        print(f"\n  [ERROR] Manifest not found: {manifest_path}")
        print("  Run 01_download_librispeech.py first.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    print(f"\n  [LOAD] {len(df)} utterances from {manifest_path.name}")

    # Apply normalization
    print("  [NORM] Normalizing ground-truth transcripts...")
    df["ground_truth_normalized"] = df["ground_truth_raw"].apply(normalize_text)

    # Validation
    assert df["ground_truth_normalized"].notna().all(), \
        "Normalization produced NaN values"
    assert (df["ground_truth_normalized"].str.len() > 0).all(), \
        "Normalization produced empty strings"

    # Check that no numbers remain in normalized text
    has_digits = df["ground_truth_normalized"].str.contains(r"\d", regex=True)
    if has_digits.any():
        print(f"  [WARN] {has_digits.sum()} transcripts still contain digits "
              "after normalization:")
        for _, row in df[has_digits].head(5).iterrows():
            print(f"         {row['utterance_id']}: '{row['ground_truth_normalized']}'")

    # Save
    df.to_csv(manifest_path, index=False)
    print(f"  [SAVED] {manifest_path.name} (added 'ground_truth_normalized' column)")

    # Show some examples
    print(f"\n  {'─' * 66}")
    print("  Sample normalizations (first 5):")
    print(f"  {'─' * 66}")
    for _, row in df.head(5).iterrows():
        raw = row["ground_truth_raw"]
        norm = row["ground_truth_normalized"]
        uid = row["utterance_id"]
        print(f"\n  [{uid}]")
        print(f"    RAW:  {raw}")
        print(f"    NORM: {norm}")

    print(f"\n  ✓ All {len(df)} transcripts normalized successfully")
    print("  Ready for Step 03 (baseline inference).\n")


if __name__ == "__main__":
    main()
