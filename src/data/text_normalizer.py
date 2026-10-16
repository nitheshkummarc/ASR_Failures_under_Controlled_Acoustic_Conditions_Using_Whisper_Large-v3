"""
Text normalization for ground truth and ASR predictions.

LibriSpeech references are written as spoken words ("nine thousand five
hundred", "seventeen o seven", "twenty first"); Whisper writes digits ("9,500",
"1707", "21st"). Every rule below exists to turn Whisper's digits into the words
the reference uses, so WER counts recognition errors rather than formatting.
"""

import re
import string
from num2words import num2words
from src.logger import setup_logger

logger = setup_logger(__name__)

# Bare four-digit integers in this range are read as years: "1707" -> "seventeen
# o seven", matching how LibriSpeech transcribes dates. The corpus is public-
# domain books, so years in it fall here; outside it, a number is read as a
# quantity ("2500" -> "two thousand five hundred").
YEAR_MIN, YEAR_MAX = 1100, 1999


def _strip_thousands_separators(text: str) -> str:
    """'9,500' -> '9500', '1,000,000' -> '1000000'.

    Without this the comma splits the number: '9' and '500' convert separately
    and the punctuation pass then glues them into 'ninefive hundred'.
    """
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)


def _convert_ordinals(text: str) -> str:
    """'21st' -> 'twenty-first'. Stripping the suffix instead gave 'twenty one'."""
    return re.sub(
        r"\b(\d+)(st|nd|rd|th)\b",
        lambda m: num2words(int(m.group(1)), to="ordinal"),
        text, flags=re.IGNORECASE,
    )


def _year_words(n: int) -> str:
    # num2words says "seventeen oh-seven"; LibriSpeech writes "seventeen o seven".
    return num2words(n, to="year").replace("oh-", "o ")


def _convert_numbers_to_words(text: str) -> str:
    def _replace_match(match):
        num_str = match.group(0)
        try:
            if "." in num_str:
                integer_part, decimal_part = num_str.split(".", 1)
                result = num2words(int(integer_part)) if integer_part else "zero"
                result += " point"
                for digit in decimal_part:
                    result += " " + num2words(int(digit))
                return result
            n = int(num_str)
            if len(num_str) == 4 and YEAR_MIN <= n <= YEAR_MAX:
                return _year_words(n)
            return num2words(n)
        except (ValueError, OverflowError):
            return " ".join(num2words(int(d)) for d in num_str if d.isdigit())

    pattern = r"(?<!\w)-?\b\d+(?:\.\d+)?\b"
    return re.sub(pattern, _replace_match, text)

def _strip_punctuation(text: str) -> str:
    text = text.replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_text(text: str) -> str:
    """
    Master normalization pipeline applied to BOTH ground truth and predictions.
    Maps to DSP/Syllabus Concept: Lexical consistency for deterministic Word Error Rate calculation.

    Steps: Lowercase -> Strip thousands separators -> Ordinals -> Numbers/years
           -> Strip punctuation -> Collapse whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = _strip_thousands_separators(text)
    text = _convert_ordinals(text)
    text = _convert_numbers_to_words(text)
    text = _strip_punctuation(text)
    text = _collapse_whitespace(text)

    return text
