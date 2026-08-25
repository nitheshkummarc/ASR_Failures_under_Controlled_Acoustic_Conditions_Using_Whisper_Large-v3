"""
Text normalization for ground truth and ASR predictions.
"""

import re
import string
from num2words import num2words
from src.logger import setup_logger

logger = setup_logger(__name__)

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
            else:
                return num2words(int(num_str))
        except (ValueError, OverflowError):
            return " ".join(num2words(int(d)) for d in num_str if d.isdigit())

    pattern = r"(?<!\w)-?\b\d+(?:\.\d+)?\b"
    return re.sub(pattern, _replace_match, text)

def _strip_ordinal_suffixes(text: str) -> str:
    return re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)

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
    
    Steps: Lowercase -> Strip Ordinal -> Num2Words -> Strip Punctuation -> Collapse Whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = _strip_ordinal_suffixes(text)
    text = _convert_numbers_to_words(text)
    text = _strip_punctuation(text)
    text = _collapse_whitespace(text)

    return text
