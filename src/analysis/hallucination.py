"""
End-of-audio hallucination detection and hallucination-robust WER.
=============================================================================
Whisper appends a stock closing phrase — overwhelmingly "thank you" — to audio
that ends in a low-energy region. The decoder has emitted every real word, the
remaining frames carry no speech, and the language-model prior fills the gap
before the EOS token. The phrase is absent from the reference, so WER charges
every word of it as an insertion.

WHY IT MATTERS
--------------
The effect is condition-dependent, which makes it a confound rather than a
constant offset. On this corpus it fires on 6.7% of clean utterances and supplies
roughly two thirds of the clean baseline's insertions, but cafe babble suppresses
it completely at every SNR — the masker fills the trailing silence, so there is
no low-energy region to hallucinate into.

The result is that a mildly degraded condition can post a LOWER WER than clean
audio while recognising the speech slightly worse. Without correcting for it, the
headline comparison measures "how much does this condition suppress Whisper's
closing tic" mixed in with "how much does this condition damage recognition".

SCOPE
-----
It computes a SECONDARY metric. `global_wer` in metrics.py remains the primary
reported number, because silently deleting words from a hypothesis before scoring
is not standard WER and must never masquerade as it.

Stripping is deliberately conservative — a trailing run is removed only when it
is BOTH positioned after the reference's final word AND matches the closed
STOCK_PHRASES list below. A model that genuinely mis-transcribes the last few
words keeps those errors. The list is closed and documented rather than
data-mined so the correction cannot quietly grow to fit whatever result is
wanted, and it is applied identically to every condition, clean included.

WHY NOT FIX IT IN THE DECODER INSTEAD
-------------------------------------
faster-whisper ships two options aimed at exactly this: `vad_filter=True` and
`hallucination_silence_threshold`. Both key off silence, and Axis C inserts
silence as an experimental variable. Enabling either would
strip the inserted gaps and collapse three conditions into near no-ops,
destroying that axis to clean up an artifact worth well under one WER point.
The correction belongs in scoring, not in decoding. See docs/methodology.md.
"""

from typing import Iterable, Mapping

from src.analysis.metrics import count_reference_words
from src.inference.evaluator import evaluate_metrics

# Closed list of known Whisper closing hallucinations, normalized (lowercase, no
# punctuation) to match post-normalize_text text. Ordered longest-first so the
# fullest match wins — "thank you very much" must not be left as "very much"
# after stripping "thank you".
STOCK_PHRASES = (
    "thank you very much for watching",
    "thanks for watching this video",
    "thank you very much",
    "thanks for watching",
    "thank you for watching",
    "please subscribe to my channel",
    "subscribe to my channel",
    "thank you so much",
    "thanks for listening",
    "thank you",
    "thanks",
    "bye bye",
    "goodbye",
)


def trailing_hallucination(reference: str, hypothesis: str) -> str:
    """
    Return the stock phrase the hypothesis appends past the reference, or "".

    The trailing run is located by finding the reference's final word in the
    hypothesis, searching from the END. Searching from the end matters: the last
    reference word often also occurs earlier in the sentence, and anchoring on
    the first occurrence would classify a large part of a correct transcript as
    trailing junk.

    Returns "" when the reference is empty, when the hypothesis is not longer
    than the reference, when the final reference word is absent, or when the
    trailing run is not a recognised stock phrase.
    """
    ref_words = str(reference).split()
    hyp_words = str(hypothesis).split()

    if not ref_words or len(hyp_words) <= len(ref_words):
        return ""

    last_ref = ref_words[-1]
    if last_ref not in hyp_words:
        return ""

    # Index of the LAST occurrence, so the anchor is the true sentence end.
    anchor = len(hyp_words) - 1 - hyp_words[::-1].index(last_ref)
    tail = " ".join(hyp_words[anchor + 1:])

    return tail if tail in STOCK_PHRASES else ""


def strip_trailing_hallucination(reference: str, hypothesis: str) -> str:
    """Hypothesis with a trailing stock phrase removed; unchanged if there is none."""
    tail = trailing_hallucination(reference, hypothesis)
    if not tail:
        return hypothesis

    hyp_words = str(hypothesis).split()
    return " ".join(hyp_words[: len(hyp_words) - len(tail.split())])


def hallucination_rate(rows: Iterable[Mapping]) -> float:
    """Percentage of rows whose hypothesis ends in a stock hallucination."""
    rows = list(rows)
    if not rows:
        return 0.0

    hits = sum(
        1 for r in rows
        if trailing_hallucination(r["ground_truth"], r.get("predicted_text") or "")
    )
    return hits / len(rows) * 100.0


def robust_global_wer(rows: Iterable[Mapping]) -> float:
    """
    Corpus-level WER recomputed after stripping trailing stock hallucinations.

    The stored substitution/insertion/deletion counts describe the ORIGINAL
    hypothesis, so they cannot simply be adjusted — removing words can change the
    whole alignment. Each affected row is therefore re-scored through jiwer.
    Untouched rows reuse their stored counts, so the common case stays cheap.
    """
    total_errors = 0
    total_words = 0

    for row in rows:
        reference = row["ground_truth"]
        hypothesis = row.get("predicted_text") or ""
        cleaned = strip_trailing_hallucination(reference, hypothesis)

        if cleaned == hypothesis:
            errors = (
                int(row["substitutions"]) + int(row["deletions"]) + int(row["insertions"])
            )
        else:
            m = evaluate_metrics(str(reference), cleaned)
            errors = m["substitutions"] + m["deletions"] + m["insertions"]

        total_errors += errors
        total_words += count_reference_words(reference)

    if total_words == 0:
        return 0.0

    return (total_errors / total_words) * 100.0
