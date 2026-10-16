import jiwer
from src.logger import setup_logger
logger = setup_logger(__name__)



def evaluate_metrics(reference: str, hypothesis: str) -> dict:
    """
    Compute WER, CER, and Substitution/Insertion/Deletion counts using jiwer.
    Both reference and hypothesis should already be normalized.
    """
    if not reference:
        return {
            "wer": 1.0 if hypothesis else 0.0,
            "cer": 1.0 if hypothesis else 0.0,
            "substitutions": 0,
            "insertions": len(hypothesis.split()) if hypothesis else 0,
            "deletions": 0,
            "hits": 0,
            "word_count_ref": 0,
        }

    word_output = jiwer.process_words(reference, hypothesis)
    cer_value = jiwer.cer(reference, hypothesis)
    ref_word_count = len(reference.split())

    return {
        "wer": word_output.wer,
        "cer": cer_value,
        "substitutions": word_output.substitutions,
        "insertions": word_output.insertions,
        "deletions": word_output.deletions,
        "hits": word_output.hits,
        "word_count_ref": ref_word_count,
    }


def insertion_spans(reference: str, hypothesis: str) -> list:
    """
    Hypothesis word ranges [start, end) that jiwer aligns as INSERTIONS.

    These are exactly the words WER charges as insertions -- hypothesis words
    with no reference counterpart. Uses the same alignment as evaluate_metrics,
    so what is reported here agrees with the insertion count there.
    """
    n_hyp = len(hypothesis.split()) if hypothesis else 0
    if n_hyp == 0:
        return []
    if not reference:
        return [(0, n_hyp)]

    return [
        (chunk.hyp_start_idx, chunk.hyp_end_idx)
        for chunk in jiwer.process_words(reference, hypothesis).alignments[0]
        if chunk.type == "insert"
    ]
