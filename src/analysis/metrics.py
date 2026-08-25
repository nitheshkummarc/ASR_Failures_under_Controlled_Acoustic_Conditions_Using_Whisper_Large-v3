"""
Corpus-level evaluation metrics.

This module exists so there is exactly ONE definition of the project's headline
number. It used to be re-implemented independently in four places, which meant
four copies had to be kept in agreement by hand.

TWO WER DEFINITIONS COEXIST IN THIS PROJECT — know which one you are reading:

  * Per-utterance WER — the `wer` column in every result CSV, straight from
    jiwer. Averaging that column gives a "mean of ratios", which weights a
    5-word utterance the same as a 40-word one.

  * Global (corpus) WER — sum all errors, sum all reference words, divide once.
    Utterances are weighted by their length. This is the standard convention for
    reporting ASR benchmarks and is the number quoted everywhere in this project.
"""

from typing import Iterable, Mapping, Sequence


def count_reference_words(ground_truth: str) -> int:
    """Number of reference words in a (already normalized) transcript."""
    return len(str(ground_truth).split())


def global_wer(rows: Iterable[Mapping]) -> float:
    """
    Corpus-level Word Error Rate as a percentage.

        global_WER = 100 * sum(S + D + I) / sum(reference_words)

    Args:
        rows: any iterable of mappings (csv.DictReader rows, list-of-dicts, or
              the records of a pandas DataFrame via .to_dict('records')) that
              expose 'substitutions', 'deletions', 'insertions' and
              'ground_truth'.

    Returns:
        WER as a percentage. Returns 0.0 when there are no reference words,
        which is the only sensible value for an empty corpus.
    """
    total_errors = 0
    total_words = 0

    for row in rows:
        total_errors += (
            int(row["substitutions"]) + int(row["deletions"]) + int(row["insertions"])
        )
        total_words += count_reference_words(row["ground_truth"])

    if total_words == 0:
        return 0.0

    return (total_errors / total_words) * 100.0


def global_wer_df(df) -> float:
    """
    Same metric, but for a pandas DataFrame — avoids a per-row Python loop over
    the error columns while keeping the result identical to global_wer().
    """
    if len(df) == 0:
        return 0.0

    total_words = sum(count_reference_words(t) for t in df["ground_truth"])
    if total_words == 0:
        return 0.0

    total_errors = (
        df["substitutions"].sum() + df["deletions"].sum() + df["insertions"].sum()
    )
    return (total_errors / total_words) * 100.0


def mean_utterance_wer(rows: Sequence[Mapping]) -> float:
    """
    Mean of the per-utterance WER column, as a percentage.

    Provided for explicit comparison against global_wer() — the two disagree,
    and tools/diagnostic_report.py prints both side by side for exactly that
    reason. Do NOT use this for reported results.
    """
    rows = list(rows)
    if not rows:
        return 0.0
    return (sum(float(r["wer"]) for r in rows) / len(rows)) * 100.0