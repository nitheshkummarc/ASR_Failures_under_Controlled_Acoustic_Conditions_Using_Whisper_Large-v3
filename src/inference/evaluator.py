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
