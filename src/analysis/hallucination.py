"""
Hallucination detection and hallucination-robust WER.
=============================================================================
Whisper emits text that is not in the audio when part of the input carries no
speech. On this corpus the dominant case is a closing "thank you" appended to
audio that ends in a low-energy region: the decoder has emitted every real word,
the remaining frames carry no speech, and the language-model prior fills the gap
before the EOS token. WER charges every hallucinated word as an insertion.

WHY IT MATTERS
--------------
The effect is condition-dependent, which makes it a confound rather than a
constant offset. It fires on several percent of clean utterances, but cafe
babble suppresses it completely -- the masker fills the trailing silence, so
there is no low-energy region to hallucinate into. A mildly degraded condition
can therefore post a LOWER WER than clean audio while recognising the speech
slightly worse.

WHERE THE PHRASES COME FROM -- the "Bag of Hallucinations"
----------------------------------------------------------
Nothing here is a hand-typed phrase list. Following Baranski et al., "Investigation
of Whisper ASR Hallucinations Induced by Non-Speech Audio", ICASSP 2025
(arXiv:2501.11378), the bag is LEARNED from Whisper itself:

  1. Feed the same Whisper configuration audio that contains NO speech: digital
     silence, white, pink and speech-shaped noise at several levels and lengths
     (tools/build_hallucination_bag.py, grid in config.py).
  2. Whatever it transcribes is, by construction, hallucinated. De-loop each
     output and count how often each one occurs.
  3. Keep outputs that recur (>= HALLUCINATION_BAG_MIN_COUNT) AND are improbable
     as ordinary English (log10 unigram probability <= HALLUCINATION_BAG_MAX_LOG10_PROB
     under the LibriSpeech transcripts disjoint from the test set). The second
     filter is the paper's n-gram-LM criterion; it stops bag entries such as
     "the" or "a" from stripping genuine recognition errors.

The bag is derived only from non-speech audio, never from the test results, and
its parameters are fixed in config.py before it is built. That keeps the
property the old hand-written list was meant to have: the correction cannot
grow to fit whatever result is wanted.

HOW A HYPOTHESIS IS CLEANED
---------------------------
The paper removes bag phrases anywhere in a transcript, because at deployment
there is no reference. Here there is one, so the rule is stricter: only words
that jiwer ALIGNS AS INSERTIONS (no reference counterpart) can ever be removed.
Inside each insertion run, scanning left to right:

  * a bag phrase (longest first)                   -> removed  (kind "bag")
  * a repeat of the N >= HALLUCINATION_LOOP_MIN_WORDS words just before it
                                                   -> removed  (kind "loop")
  * anything else                                  -> kept, still an error

So a word that matched or substituted a reference word is never touched, and a
genuine mis-recognition keeps its errors. Because insertions can sit anywhere,
this also catches hallucinations at the start of the audio or inside a gap --
which matters for the Axis C silence conditions -- not only at the end.

SCOPE
-----
It computes a SECONDARY metric. `global_wer` in metrics.py remains the primary
reported number, because deleting words from a hypothesis before scoring is not
standard WER and must never masquerade as it.

WHY NOT FIX IT IN THE DECODER INSTEAD
-------------------------------------
faster-whisper ships two options aimed at exactly this: `vad_filter=True` and
`hallucination_silence_threshold`. Both key off silence, and Axis C inserts
silence as an experimental variable. Enabling either would strip the inserted
gaps and collapse three conditions into near no-ops, destroying that axis to
clean up an artifact worth well under one WER point. The correction belongs in
scoring, not in decoding. See docs/methodology.md.
"""

import json
import math
from functools import lru_cache
from typing import Iterable, Mapping, Optional, Sequence

from config import (
    HALLUCINATION_BAG_PATH, HALLUCINATION_BAG_MIN_COUNT,
    HALLUCINATION_BAG_MAX_LOG10_PROB, HALLUCINATION_LOOP_MIN_WORDS,
)
from src.analysis.metrics import count_reference_words
from src.inference.evaluator import evaluate_metrics, insertion_spans


# =============================================================================
# Building the bag (pure functions; the GPU part is tools/build_hallucination_bag.py)
# =============================================================================

def deloop(words: Sequence[str], min_words: int = 1) -> list:
    """
    Collapse immediate repeats: "thank you thank you thank you" -> "thank you".

    The paper's first post-processing step. Repeats of n-grams with
    n >= min_words are collapsed, longest n first.
    """
    out = list(words)
    i = 0
    while i < len(out):
        collapsed = False
        for n in range(len(out) // 2, min_words - 1, -1):
            if i >= n and out[i:i + n] == out[i - n:i]:
                del out[i:i + n]
                collapsed = True
                break
        if not collapsed:
            i += 1
    return out


def log10_unigram_prob(phrase: str, word_counts: Mapping[str, int]) -> float:
    """
    log10 P(phrase) under an add-one-smoothed unigram LM.

        P(w) = (c(w) + 1) / (N + V + 1)        N = tokens, V = vocabulary size
        log10 P(phrase) = sum over words of log10 P(w)

    The +1 in the denominator reserves mass for unseen words. A unigram model is
    crude, but the filter only has to separate everyday words ("the", "a", "you")
    from improbable sequences, which it does.
    """
    total = sum(word_counts.values())
    denom = total + len(word_counts) + 1
    return sum(math.log10((word_counts.get(w, 0) + 1) / denom) for w in phrase.split())


def select_bag(
    output_counts: Mapping[str, int],
    corpus_word_counts: Mapping[str, int],
    min_count: int = HALLUCINATION_BAG_MIN_COUNT,
    max_log10_prob: float = HALLUCINATION_BAG_MAX_LOG10_PROB,
):
    """
    Apply the two bag filters to the counted non-speech outputs.

    Returns (kept, dropped): `kept` is a list of phrases; `dropped` is a list of
    {phrase, count, log10_prob, reason} so the filtering stays auditable.
    """
    kept, dropped = [], []
    for phrase, count in sorted(output_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lp = log10_unigram_prob(phrase, corpus_word_counts)
        if count < min_count:
            reason = f"count {count} < {min_count}"
        elif lp > max_log10_prob:
            reason = f"log10 P {lp:.2f} > {max_log10_prob} (ordinary English)"
        else:
            kept.append(phrase)
            continue
        dropped.append({"phrase": phrase, "count": count,
                        "log10_prob": round(lp, 3), "reason": reason})
    return kept, dropped


# =============================================================================
# Using the bag
# =============================================================================

@lru_cache(maxsize=4)
def _load_bag_cached(path: str) -> tuple:
    with open(path, "r", encoding="utf-8") as f:
        phrases = json.load(f)["bag"]
    return tuple(sorted({p.strip() for p in phrases if p.strip()},
                        key=lambda p: (-len(p.split()), p)))


def load_bag(path=None) -> tuple:
    """
    The Bag of Hallucinations, longest phrase first (so "thank you very much" is
    tried before "thank you").

    Raises FileNotFoundError, naming the tool that builds it, if absent.
    """
    path = HALLUCINATION_BAG_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"Hallucination bag not found: {path}\n"
            f"Build it once with `python tools/build_hallucination_bag.py` (needs the GPU)."
        )
    return _load_bag_cached(str(path))


def hallucination_spans(reference: str, hypothesis: str,
                        bag: Optional[Sequence[str]] = None) -> list:
    """
    Hallucinated word ranges in the hypothesis, as (start, end, kind) with
    kind "bag" or "loop". Only ever inside jiwer insertion runs.
    """
    bag = load_bag() if bag is None else bag
    words = str(hypothesis or "").split()
    if not words:
        return []
    phrases = sorted({tuple(p.split()) for p in bag if p.strip()}, key=lambda t: -len(t))

    spans = []
    for start, end in insertion_spans(str(reference or ""), " ".join(words)):
        i = start
        while i < end:
            match = None
            for p in phrases:
                if i + len(p) <= end and tuple(words[i:i + len(p)]) == p:
                    match = (i, i + len(p), "bag")
                    break
            if match is None:
                for n in range(min(i, end - i), HALLUCINATION_LOOP_MIN_WORDS - 1, -1):
                    if words[i:i + n] == words[i - n:i]:
                        match = (i, i + n, "loop")
                        break
            if match is None:
                i += 1
            else:
                spans.append(match)
                i = match[1]
    return spans


def strip_hallucinations(reference: str, hypothesis: str,
                         bag: Optional[Sequence[str]] = None) -> str:
    """Hypothesis with every hallucinated span removed; unchanged if there are none."""
    spans = hallucination_spans(reference, hypothesis, bag)
    if not spans:
        return hypothesis
    words = str(hypothesis).split()
    drop = {i for s, e, _ in spans for i in range(s, e)}
    return " ".join(w for i, w in enumerate(words) if i not in drop)


def hallucination_rate(rows: Iterable[Mapping], bag: Optional[Sequence[str]] = None) -> float:
    """Percentage of rows whose hypothesis contains at least one hallucinated span."""
    rows = list(rows)
    if not rows:
        return 0.0
    bag = load_bag() if bag is None else bag

    hits = sum(
        1 for r in rows
        if hallucination_spans(r["ground_truth"], r.get("predicted_text") or "", bag)
    )
    return hits / len(rows) * 100.0


def robust_global_wer(rows: Iterable[Mapping], bag: Optional[Sequence[str]] = None) -> float:
    """
    Corpus-level WER recomputed after stripping hallucinated spans.

    The stored substitution/insertion/deletion counts describe the ORIGINAL
    hypothesis, so they cannot simply be adjusted — removing words can change the
    whole alignment. Each affected row is therefore re-scored through jiwer.
    Untouched rows reuse their stored counts.
    """
    bag = load_bag() if bag is None else bag
    total_errors = 0
    total_words = 0

    for row in rows:
        reference = row["ground_truth"]
        hypothesis = row.get("predicted_text") or ""
        cleaned = strip_hallucinations(reference, hypothesis, bag)

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
