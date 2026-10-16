"""
Tests for hallucination detection and the robust WER.

The correction these back is subtractive — it removes words before scoring — so
the tests are written mostly around what must NOT be stripped. An over-eager
stripper would flatter every condition and quietly manufacture the result the
analysis is supposed to test for.

Every test passes an explicit bag, so none of them depends on the GPU-built
metadata/hallucination_bag.json.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pytest

from src.analysis.hallucination import (
    deloop, log10_unigram_prob, select_bag, load_bag,
    hallucination_spans, strip_hallucinations,
    hallucination_rate, robust_global_wer,
)
from src.analysis.metrics import global_wer

BAG = ("thank you", "thank you very much", "thanks for watching")


def strip(ref, hyp):
    return strip_hallucinations(ref, hyp, BAG)


# ------------------------------------------------------------ detection ----

def test_detects_the_canonical_thank_you():
    assert hallucination_spans("roll into balls", "roll into balls thank you", BAG) == [
        (3, 5, "bag")
    ]
    assert strip("roll into balls", "roll into balls thank you") == "roll into balls"


def test_longest_bag_phrase_wins():
    """'thank you very much' must not leave 'very much' behind."""
    assert strip("the end", "the end thank you very much") == "the end"


def test_genuine_inserted_errors_are_kept():
    """
    A real mis-transcription past the reference end is a legitimate insertion and
    must survive, or the metric flatters every system.
    """
    hyp = "roll into balls and then some more words"
    assert hallucination_spans("roll into balls", hyp, BAG) == []
    assert strip("roll into balls", hyp) == hyp


def test_only_the_bag_part_of_a_mixed_insertion_run_is_removed():
    assert strip("the end", "the end and thank you") == "the end and"


def test_phrase_that_matches_the_reference_is_not_stripped():
    """'thank you' that is really in the audio aligns as hits, not insertions."""
    hyp = "thank you she said politely"
    assert hallucination_spans(hyp, hyp, BAG) == []
    assert strip(hyp, hyp) == hyp


def test_reference_word_repeated_in_hypothesis_is_handled():
    ref = "the bell rang and then the bell"
    assert strip(ref, ref + " thank you") == ref


def test_hallucination_inside_an_inserted_gap_is_removed():
    """Insertions can sit mid-utterance (e.g. Axis C silence), not only at the end."""
    assert strip("a b c d", "a b thank you c d") == "a b c d"


def test_ambiguous_alignment_stays_conservative():
    """
    With a fully wrong transcript jiwer may align the bag phrase as the
    SUBSTITUTIONS and other words as the insertions. Only insertion runs are
    ever examined, so nothing is stripped -- the safe direction.
    """
    hyp = "completely different thank you"
    assert strip("alpha beta", hyp) == hyp


def test_correct_transcript_is_untouched():
    assert hallucination_spans("a b c", "a b c", BAG) == []
    assert strip("a b c", "a b c") == "a b c"


def test_shorter_hypothesis_is_untouched():
    assert strip("a b c d", "a b c") == "a b c"


def test_empty_reference_or_hypothesis():
    assert strip("", "thank you") == ""
    assert hallucination_spans("a b", "", BAG) == []
    assert strip("a b", "") == ""


def test_phrase_not_in_the_bag_is_kept():
    assert strip("the end", "the end goodbye") == "the end goodbye"


# ---------------------------------------------------------------- loops ----

def test_repetition_loop_is_removed():
    """An inserted run repeating the two words just before it is a loop."""
    assert hallucination_spans("we went home", "we went home went home", BAG) == [
        (3, 5, "loop")
    ]
    assert strip("we went home", "we went home went home") == "we went home"


def test_single_doubled_word_is_not_a_loop():
    """N >= 2: one doubled word is an ordinary insertion, and keeps its error."""
    assert strip("we went home", "we went home home") == "we went home home"


def test_deloop_collapses_repeats():
    assert deloop("thank you thank you thank you".split()) == ["thank", "you"]
    assert deloop("you you you".split()) == ["you"]
    assert deloop("a b c".split()) == ["a", "b", "c"]
    assert deloop("you you you".split(), min_words=2) == ["you", "you", "you"]


# ------------------------------------------------------------ building ----

def test_unigram_prob_ranks_common_words_as_probable():
    counts = {"the": 1000, "a": 500, "thank": 2, "you": 100}
    assert log10_unigram_prob("the", counts) > log10_unigram_prob("thank you", counts)
    assert log10_unigram_prob("watching", counts) < log10_unigram_prob("the", counts)


def test_select_bag_applies_both_filters():
    counts = {"the": 1000, "a": 500, "you": 100}     # "thank" unseen
    outputs = {"thank you": 40, "the": 9, "zebra crossing": 1}
    kept, dropped = select_bag(outputs, counts, min_count=2, max_log10_prob=-4.0)
    assert kept == ["thank you"]
    reasons = {d["phrase"]: d["reason"] for d in dropped}
    assert "ordinary English" in reasons["the"]
    assert reasons["zebra crossing"].startswith("count")


def test_load_bag_orders_longest_first(tmp_path):
    path = tmp_path / "bag.json"
    path.write_text(json.dumps({"bag": ["thank you", "thank you very much", "bye"]}))
    assert load_bag(path) == ("thank you very much", "thank you", "bye")


def test_missing_bag_names_the_builder(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_hallucination_bag"):
        load_bag(tmp_path / "absent.json")


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
    assert hallucination_rate(rows, BAG) == pytest.approx(50.0)


def test_hallucination_rate_of_empty_corpus():
    assert hallucination_rate([], BAG) == 0.0


# ------------------------------------------------------------ robust WER ----

def test_robust_wer_removes_hallucinated_insertions():
    """Two clean utterances, one with an appended bag phrase worth 2 insertions."""
    rows = [
        _row("a b c d e", "a b c d e thank you", i=2),
        _row("f g h i j", "f g h i j"),
    ]
    assert global_wer(rows) == pytest.approx(20.0)   # 2 errors / 10 words
    assert robust_global_wer(rows, BAG) == pytest.approx(0.0)


def test_robust_wer_leaves_genuine_errors_alone():
    rows = [_row("a b c d", "a x c d", s=1)]
    assert robust_global_wer(rows, BAG) == pytest.approx(global_wer(rows))


def test_robust_wer_never_exceeds_raw_wer():
    """Stripping can only remove insertions, so the robust figure cannot be worse."""
    rows = [
        _row("a b c", "a b c thank you", i=2),
        _row("d e f", "d x f", s=1),
        _row("g h i", "g h", d=1),
    ]
    assert robust_global_wer(rows, BAG) <= global_wer(rows) + 1e-9


def test_robust_wer_of_empty_corpus():
    assert robust_global_wer([], BAG) == 0.0


def test_robust_wer_handles_missing_prediction_column():
    rows = [{"ground_truth": "a b", "predicted_text": None,
             "substitutions": 0, "insertions": 0, "deletions": 2}]
    assert robust_global_wer(rows, BAG) == pytest.approx(100.0)
