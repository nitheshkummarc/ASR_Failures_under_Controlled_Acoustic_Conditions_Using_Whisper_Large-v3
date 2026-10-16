"""
Tests for text normalization and the corpus-level WER metric.

Between them these two pieces define what "correct" means for every WER number
in the project, and neither had any test coverage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.analysis.metrics import global_wer, mean_utterance_wer, count_reference_words
from src.data.text_normalizer import normalize_text
from src.inference.evaluator import evaluate_metrics


# ------------------------------------------------------- normalize_text ----

def test_lowercases_and_strips_punctuation():
    assert normalize_text("Mister Quilter's, the apostle!") == "mister quilters the apostle"


def test_hyphen_becomes_a_word_boundary():
    # Hyphens are replaced with a space BEFORE punctuation stripping, so this
    # must be two words rather than "wellknown".
    assert normalize_text("well-known") == "well known"


def test_numbers_become_words():
    assert normalize_text("60 feet") == "sixty feet"
    assert normalize_text("2500") == "two thousand five hundred"


def test_years_are_read_the_way_librispeech_writes_them():
    """Both forms below are taken from real LibriSpeech references in this corpus."""
    assert normalize_text("in 1707,") == "in seventeen o seven"
    assert normalize_text("toast, 1726.") == "toast seventeen twenty six"
    assert normalize_text("1900") == "nineteen hundred"


def test_thousands_separator_does_not_split_the_number():
    """'9,500' once became 'ninefive hundred': the comma split it in two."""
    assert normalize_text("about 9,500 feet") == "about nine thousand five hundred feet"
    assert normalize_text("1,000,000") == "one million"


def test_list_commas_are_not_mistaken_for_separators():
    assert normalize_text("3 o'clock came, 4, 5, 6, and") == "three oclock came four five six and"


def test_decimals_are_spoken_digit_by_digit():
    assert normalize_text("3.14") == "three point one four"


def test_ordinals_become_ordinal_words():
    """Stripping the suffix instead gave '21st' -> 'twenty one'."""
    assert normalize_text("1st") == "first"
    assert normalize_text("the 21st day") == "the twenty first day"
    assert normalize_text("3RD") == "third"


def test_digits_glued_to_letters_are_left_alone():
    # The (?<!\w) lookbehind prevents matching the 3 in "mp3".
    assert "three" not in normalize_text("mp3")


def test_whitespace_is_collapsed():
    assert normalize_text("  hello    world  ") == "hello world"


@pytest.mark.parametrize("value", ["", None, 123, [], {}])
def test_non_string_input_returns_empty(value):
    assert normalize_text(value) == ""


def test_normalization_is_idempotent():
    once = normalize_text("The 2nd Mister Quilter's well-known 3.5 apples!")
    assert normalize_text(once) == once


# ------------------------------------------------------ evaluate_metrics ----

def test_perfect_match_scores_zero():
    m = evaluate_metrics("the cat sat", "the cat sat")
    assert m["wer"] == 0.0
    assert m["substitutions"] == m["insertions"] == m["deletions"] == 0
    assert m["hits"] == 3
    assert m["word_count_ref"] == 3


def test_substitution_is_counted():
    m = evaluate_metrics("the cat sat", "the dog sat")
    assert m["substitutions"] == 1
    assert pytest.approx(m["wer"], abs=1e-9) == 1 / 3


def test_deletion_is_counted():
    m = evaluate_metrics("the cat sat", "the sat")
    assert m["deletions"] == 1


def test_insertion_is_counted():
    m = evaluate_metrics("the cat sat", "the big cat sat")
    assert m["insertions"] == 1


def test_empty_hypothesis_deletes_every_reference_word():
    m = evaluate_metrics("the cat sat", "")
    assert m["deletions"] == 3
    assert m["wer"] == 1.0


def test_empty_reference_with_hypothesis():
    """jiwer raises on an empty reference, so this branch is hand-written."""
    m = evaluate_metrics("", "hello world")
    assert m["wer"] == 1.0
    assert m["insertions"] == 2
    assert m["word_count_ref"] == 0


def test_empty_reference_and_empty_hypothesis():
    m = evaluate_metrics("", "")
    assert m["wer"] == 0.0
    assert m["insertions"] == 0


# ------------------------------------------------------------ global_wer ----

def _row(gt, s=0, i=0, d=0, wer=0.0):
    return {"ground_truth": gt, "substitutions": s, "insertions": i,
            "deletions": d, "wer": wer}


def test_global_wer_of_perfect_corpus_is_zero():
    assert global_wer([_row("a b c"), _row("d e")]) == 0.0


def test_global_wer_weights_by_reference_length():
    """
    The whole reason this metric exists: a long utterance must count for more
    than a short one. Here 1 error in 1 word and 1 error in 9 words give
    2/10 = 20% globally, but (100% + 11.1%)/2 = 55.6% as a mean of ratios.
    """
    rows = [
        _row("one", s=1, wer=1.0),
        _row("a b c d e f g h i", s=1, wer=1 / 9),
    ]
    assert pytest.approx(global_wer(rows), abs=1e-9) == 20.0
    assert global_wer(rows) != pytest.approx(mean_utterance_wer(rows), abs=1e-6)


def test_global_wer_sums_all_three_error_types():
    assert pytest.approx(global_wer([_row("a b c d", s=1, i=1, d=1)]), abs=1e-9) == 75.0


def test_global_wer_of_empty_corpus_is_zero():
    assert global_wer([]) == 0.0


def test_global_wer_with_no_reference_words_is_zero():
    assert global_wer([_row("", i=5)]) == 0.0


def test_count_reference_words_handles_non_strings():
    assert count_reference_words(123) == 1
    assert count_reference_words("  a   b  ") == 2