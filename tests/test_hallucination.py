"""
Tests for end-of-audio hallucination detection and the robust WER.

The correction these back is subtractive — it removes words before scoring — so
the tests are written mostly around what must NOT be stripped. An over-eager
stripper would flatter every condition and quietly manufacture the result the
analysis is supposed to test for.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.analysis.hallucination import (
    STOCK_PHRASES, trailing_hallucination, strip_trailing_hallucination,
    hallucination_rate, robust_global_wer,
)
from src.analysis.metrics import global_wer


# ------------------------------------------------------------ detection ----

def test_detects_the_canonical_thank_you():
    assert trailing_hallucination("roll into balls", "roll into balls thank you") == "thank you"


def test_strips_only_the_stock_phrase():
    assert strip_trailing_hallucination(
        "roll into balls", "roll into balls thank you"
    ) == "roll into balls"


def test_longest_stock_phrase_wins():
    """'thank you very much' must not leave 'very much' behind."""
    assert strip_trailing_hallucination(
        "the end", "the end thank you very much"
    ) == "the end"


def test_genuine_trailing_errors_are_kept():
    """
    A real mis-transcription past the reference end is a legitimate insertion and
    must survive, or the metric flatters every system.
    """
    hyp = "roll into balls and then some more words"
    assert trailing_hallucination("roll into balls", hyp) == ""
    assert strip_trailing_hallucination("roll into balls", hyp) == hyp


def test_phrase_inside_the_utterance_is_not_stripped():
    """'thank you' mid-sentence is real speech, not a closing tic."""
    hyp = "thank you she said politely"
    assert trailing_hallucination("thank you she said politely", hyp) == ""
    assert strip_trailing_hallucination("thank you she said politely", hyp) == hyp


def test_anchor_uses_last_occurrence_of_final_reference_word():
    """
    If the final reference word also appears earlier, anchoring on the FIRST
    occurrence would classify most of a correct transcript as trailing junk.
    """
    ref = "the bell rang and then the bell"
    hyp = "the bell rang and then the bell thank you"
    assert trailing_hallucination(ref, hyp) == "thank you"
    assert strip_trailing_hallucination(ref, hyp) == ref


def test_correct_transcript_is_untouched():
    assert trailing_hallucination("a b c", "a b c") == ""
    assert strip_trailing_hallucination("a b c", "a b c") == "a b c"


def test_shorter_hypothesis_is_untouched():
    assert trailing_hallucination("a b c d", "a b c") == ""


def test_empty_reference_or_hypothesis():
    assert trailing_hallucination("", "thank you") == ""
    assert trailing_hallucination("a b", "") == ""
    assert strip_trailing_hallucination("a b", "") == ""


def test_final_reference_word_absent_from_hypothesis():
    assert trailing_hallucination("alpha beta", "completely different thank you") == ""


def test_every_stock_phrase_is_detected():
    for phrase in STOCK_PHRASES:
        hyp = f"the end {phrase}"
        assert trailing_hallucination("the end", hyp) == phrase
        assert strip_trailing_hallucination("the end", hyp) == "the end"


def test_stock_phrases_are_normalized_form():
    """Text reaches this module post-normalize_text: lowercase, no punctuation."""
    for phrase in STOCK_PHRASES:
        assert phrase == phrase.lower()
        assert not any(ch in phrase for ch in ".,!?'\"")


# ----------------------------------------------------------------- rates ----

def _row(gt, hyp, s=0, i=0, d=0):
    return {"ground_truth": gt, "predicted_text": hyp,
            "substitutions": s, "insertions": i, "deletions": d}


def test_hallucination_rate_counts_affected_rows():
    rows = [
        _row("a b", "a b thank you", i=2),
        _row("c d", "c d"),
        _row("e f", "e f thank you", i=2),
        _row("g h", "g h"),
    ]
    assert hallucination_rate(rows) == pytest.approx(50.0)


def test_hallucination_rate_of_empty_corpus():
    assert hallucination_rate([]) == 0.0


# ------------------------------------------------------------ robust WER ----

def test_robust_wer_removes_hallucinated_insertions():
    """Two clean utterances, one with an appended stock phrase worth 2 insertions."""
    rows = [
        _row("a b c d e", "a b c d e thank you", i=2),
        _row("f g h i j", "f g h i j"),
    ]
    assert global_wer(rows) == pytest.approx(20.0)   # 2 errors / 10 words
    assert robust_global_wer(rows) == pytest.approx(0.0)


def test_robust_wer_leaves_genuine_errors_alone():
    rows = [_row("a b c d", "a x c d", s=1)]
    assert robust_global_wer(rows) == pytest.approx(global_wer(rows))


def test_robust_wer_never_exceeds_raw_wer():
    """Stripping can only remove insertions, so the robust figure cannot be worse."""
    rows = [
        _row("a b c", "a b c thank you", i=2),
        _row("d e f", "d x f", s=1),
        _row("g h i", "g h", d=1),
    ]
    assert robust_global_wer(rows) <= global_wer(rows) + 1e-9


def test_robust_wer_of_empty_corpus():
    assert robust_global_wer([]) == 0.0


def test_robust_wer_handles_missing_prediction_column():
    rows = [{"ground_truth": "a b", "predicted_text": None,
             "substitutions": 0, "insertions": 0, "deletions": 2}]
    assert robust_global_wer(rows) == pytest.approx(100.0)
