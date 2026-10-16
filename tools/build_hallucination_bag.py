"""
Build the Bag of Hallucinations from Whisper's own output on non-speech audio.

Method: Baranski et al., "Investigation of Whisper ASR Hallucinations Induced by
Non-Speech Audio", ICASSP 2025 (arXiv:2501.11378). Feed the recognizer audio with
no speech in it; anything it transcribes is a hallucination by construction.
Recurring, improbable outputs become the bag that src/analysis/hallucination.py
strips (from insertions only) when computing the hallucination-robust WER.

Steps:
  1. Generate the non-speech probe grid from config.py: digital silence, and
     white / pink / speech-shaped noise at several RMS levels and lengths. The
     noise comes from the project's own NoiseSource, so it is the same family of
     signals the experiment uses. Cafe is excluded: it contains real speech.
  2. Transcribe each probe with the SAME WhisperEngine configuration as the
     experiment (large-v3, int8, beam 5, English, no previous-text conditioning).
  3. Normalize with normalize_text (as for every hypothesis), de-loop, count.
  4. Filter (src/analysis/hallucination.select_bag): minimum count, and a
     unigram-LM improbability test over the LibriSpeech dev-clean transcripts
     that are NOT among the 150 test utterances.
  5. Write metadata/hallucination_bag.json with every probe's output, so the bag
     is auditable and stage 6 never needs a GPU.

Run once from the project root (about 5-10 minutes on the RTX 3050):
    venv/Scripts/python.exe tools/build_hallucination_bag.py

Prints rather than logs, like every tool here.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import Counter

import numpy as np

# config MUST be imported before faster_whisper (via WhisperEngine): it puts the
# CUDA DLL directories on PATH.
from config import (
    TARGET_SR, RANDOM_SEED, LIBRISPEECH_ROOT,
    WHISPER_MODEL, WHISPER_COMPUTE_TYPE, WHISPER_BEAM_SIZE, WHISPER_LANGUAGE,
    HALLUCINATION_BAG_PATH, HALLUCINATION_PROBE_DIR, HALLUCINATION_PROBE_NOISES,
    HALLUCINATION_PROBE_LEVELS_DBFS, HALLUCINATION_PROBE_DURATIONS_S,
    HALLUCINATION_PROBE_REPEATS, HALLUCINATION_BAG_MIN_COUNT,
    HALLUCINATION_BAG_MAX_LOG10_PROB, HALLUCINATION_LOOP_MIN_WORDS,
)
from src.analysis.hallucination import deloop, select_bag
from src.augmentation.axis_a_noise import NoiseSource, compute_rms
from src.data.loader import parse_transcripts
from src.data.text_normalizer import normalize_text
from src.inference.runner import write_degraded
from src.inference.whisper_engine import WhisperEngine, TranscriptionOOMError
from src.results_io import read_manifest


def probe_grid() -> list:
    """(noise, level_dbfs, duration_s, repeat). Silence once per duration: it is
    deterministic, so repeats and levels would only duplicate it."""
    grid = []
    for noise in HALLUCINATION_PROBE_NOISES:
        for duration in HALLUCINATION_PROBE_DURATIONS_S:
            if noise == "silence":
                grid.append((noise, None, duration, 0))
                continue
            for level in HALLUCINATION_PROBE_LEVELS_DBFS:
                for rep in range(HALLUCINATION_PROBE_REPEATS):
                    grid.append((noise, level, duration, rep))
    return grid


def make_probe(source: NoiseSource, noise: str, level_dbfs, duration_s: float) -> np.ndarray:
    n = int(duration_s * TARGET_SR)
    if noise == "silence":
        return np.zeros(n)
    x = source.get_noise(noise, n)
    return x * (10.0 ** (level_dbfs / 20.0) / compute_rms(x))


def corpus_word_counts() -> tuple:
    """Word counts over dev-clean transcripts disjoint from the test set."""
    if not LIBRISPEECH_ROOT.exists():
        print(f"[ERROR] {LIBRISPEECH_ROOT} not found. Run scripts/00_prepare_corpus.py first.")
        sys.exit(1)
    test_ids = {row["utterance_id"] for row in read_manifest()}
    counts = Counter()
    n_utts = 0
    for uid, text in parse_transcripts(LIBRISPEECH_ROOT).items():
        if uid in test_ids:
            continue
        counts.update(normalize_text(text).split())
        n_utts += 1
    return counts, n_utts


def main() -> None:
    print("=" * 78)
    print("BUILDING THE BAG OF HALLUCINATIONS (arXiv:2501.11378)")
    print("=" * 78)

    grid = probe_grid()
    print(f"{len(grid)} non-speech probes: {HALLUCINATION_PROBE_NOISES}, "
          f"levels {HALLUCINATION_PROBE_LEVELS_DBFS} dBFS, "
          f"durations {HALLUCINATION_PROBE_DURATIONS_S} s")

    # No clean paths: the cafe source is never requested.
    source = NoiseSource([], TARGET_SR, RANDOM_SEED)
    engine = WhisperEngine()

    probes, counts, oom = [], Counter(), 0
    for k, (noise, level, duration, rep) in enumerate(grid, 1):
        tag = f"{noise}_{duration}s" if level is None else f"{noise}_{level}dbfs_{duration}s_r{rep}"
        path = HALLUCINATION_PROBE_DIR / f"{tag}.wav"
        write_degraded(make_probe(source, noise, level, duration), path)

        try:
            raw = engine.transcribe(str(path))
        except TranscriptionOOMError:
            oom += 1
            continue

        output = " ".join(deloop(normalize_text(raw).split(), min_words=1))
        if output:
            counts[output] += 1
        probes.append({"probe": tag, "noise": noise, "level_dbfs": level,
                       "duration_s": duration, "raw": raw, "output": output})
        print(f"  [{k:3d}/{len(grid)}] {tag:<26} -> {output!r}")

    word_counts, n_utts = corpus_word_counts()
    bag, dropped = select_bag(counts, word_counts)

    n_out = sum(1 for p in probes if p["output"])
    print(f"\n{n_out}/{len(probes)} probes produced text "
          f"({len(counts)} distinct outputs){f', {oom} skipped (OOM)' if oom else ''}")
    print(f"\nBAG ({len(bag)} phrases, count >= {HALLUCINATION_BAG_MIN_COUNT}, "
          f"log10 P <= {HALLUCINATION_BAG_MAX_LOG10_PROB}):")
    for phrase in bag:
        print(f"  {counts[phrase]:4d}  {phrase}")
    print(f"\nDROPPED ({len(dropped)}):")
    for d in dropped:
        print(f"  {d['count']:4d}  {d['phrase']!r:<40} {d['reason']}")

    HALLUCINATION_BAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    HALLUCINATION_BAG_PATH.write_text(json.dumps({
        "method": "Bag of Hallucinations, Baranski et al., ICASSP 2025, arXiv:2501.11378",
        "whisper": {"model": WHISPER_MODEL, "compute_type": WHISPER_COMPUTE_TYPE,
                    "beam_size": WHISPER_BEAM_SIZE, "language": WHISPER_LANGUAGE},
        "probe_grid": {"noises": HALLUCINATION_PROBE_NOISES,
                       "levels_dbfs_rms": HALLUCINATION_PROBE_LEVELS_DBFS,
                       "durations_s": HALLUCINATION_PROBE_DURATIONS_S,
                       "repeats": HALLUCINATION_PROBE_REPEATS, "seed": RANDOM_SEED},
        "filters": {"min_count": HALLUCINATION_BAG_MIN_COUNT,
                    "max_log10_unigram_prob": HALLUCINATION_BAG_MAX_LOG10_PROB,
                    "loop_min_words_when_stripping": HALLUCINATION_LOOP_MIN_WORDS},
        "language_model": {"source": "LibriSpeech dev-clean minus the 150 test utterances",
                           "utterances": n_utts, "tokens": sum(word_counts.values()),
                           "vocabulary": len(word_counts)},
        "n_probes": len(probes), "n_probes_with_output": n_out, "n_skipped_oom": oom,
        "bag": bag,
        "dropped": dropped,
        "output_counts": dict(counts.most_common()),
        "probes": probes,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {HALLUCINATION_BAG_PATH}")


if __name__ == "__main__":
    main()
