"""
main.py — Master Pipeline Orchestrator
ASR Failures under Controlled Acoustic Conditions

Runs the entire experimental pipeline end-to-end:
  0. Corpus preparation (download LibriSpeech, select 150 utterances, manifest)
  1. Baseline Whisper inference on clean audio
  2. Axis A: Additive noise degradation + inference
  3. Axis B & C: Channel distortion + temporal manipulation + inference
  4. Acoustic verification (asserts the degradations are what they claim)
  5. Feature extraction (MFCC, STE, ZCR deviation)
  6. Statistical analysis + chart generation
  7. Spectrogram & MFCC heatmap visualization

Checkpoint system:
  - A checkpoint file (results/.pipeline_checkpoint) tracks completed stages.
  - On resume, already-completed stages are skipped entirely.
  - Use --fresh to wipe the checkpoint and force a full re-run.
  - Individual scripts also have per-row resumption (skip already-processed rows).

Note: the checkpoint is deleted on full success, so running main.py twice in a
row re-executes everything. Per-row resumption makes stages 1-3 fast no-ops in
that case, but stages 5-7 fully recompute. Pass --keep-checkpoint to retain it.

Usage:
    python main.py                   # Run (or resume from last checkpoint)
    python main.py --fresh           # Wipe checkpoint and run from scratch
    python main.py --keep-checkpoint # Do not clear the checkpoint on success
"""
import subprocess
import sys
import json
import argparse
from pathlib import Path

# Project root = directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import setup_logger

logger = setup_logger("main")

CHECKPOINT_FILE = PROJECT_ROOT / "results" / ".pipeline_checkpoint"

STAGES = [
    ("00_corpus",         "scripts/00_prepare_corpus.py"),
    ("01_baseline",       "scripts/01_run_baseline.py"),
    ("02_axis_a",         "scripts/02_run_axis_a.py"),
    ("03_axis_b_c",       "scripts/03_run_axis_b_c.py"),
    ("04_verify",         "tools/verify_acoustics.py"),
    ("05_features",       "scripts/04_run_feature_extraction.py"),
    ("06_analysis",       "scripts/05_run_analysis.py"),
    ("07_spectrograms",   "scripts/06_generate_spectrograms.py"),
]


def load_checkpoint():
    """
    Load the set of completed stage names from the checkpoint file.

    Any malformed checkpoint is treated as "nothing completed". That is the safe
    direction: it costs a re-run, whereas trusting a corrupt file could skip a
    stage that never actually finished.
    """
    if not CHECKPOINT_FILE.exists():
        return set()

    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        # AttributeError covers a JSON root that is a list rather than an object,
        # TypeError a non-iterable "completed" value.
        return set(data.get("completed", []))
    except (json.JSONDecodeError, AttributeError, TypeError, OSError) as e:
        logger.warning(f"Ignoring unreadable checkpoint ({e}) — starting fresh.")
        return set()


def save_checkpoint(completed: set):
    """Save completed stage names to checkpoint file."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps({"completed": sorted(completed)}, indent=2),
        encoding="utf-8",
    )


def clear_checkpoint():
    """Remove checkpoint file for a fresh run."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Cleared pipeline checkpoint — starting fresh.")


def main():
    parser = argparse.ArgumentParser(description="ASR Failures Pipeline")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Wipe checkpoint and force a full re-run from stage 0.",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Do not clear the checkpoint after a fully successful run.",
    )
    args = parser.parse_args()

    if args.fresh:
        clear_checkpoint()

    completed = load_checkpoint()

    logger.info("=" * 60)
    logger.info("ASR Failures — Full Pipeline Execution")
    if completed:
        logger.info(f"Resuming — {len(completed)}/{len(STAGES)} stages already done")
    else:
        logger.info("Starting fresh run")
    logger.info("=" * 60)

    for i, (stage_name, script) in enumerate(STAGES, 1):
        if stage_name in completed:
            logger.info(f"[{i}/{len(STAGES)}] ⏭ Skipping {script} (already completed)")
            continue

        logger.info(f"[{i}/{len(STAGES)}] Running {script}...")
        try:
            # Stream output live so tqdm progress bars are visible
            result = subprocess.run(
                [sys.executable, "-u", script],
                cwd=PROJECT_ROOT,
                check=True,
            )
            logger.info(f"[{i}/{len(STAGES)}] ✓ Completed {script}")
            completed.add(stage_name)
            save_checkpoint(completed)
        except subprocess.CalledProcessError as e:
            logger.error(f"[{i}/{len(STAGES)}] ✗ FAILED {script} (exit code {e.returncode})")
            logger.error("")
            logger.error("Pipeline stopped. Fix the error and re-run:")
            logger.error("  python main.py          (resumes from this stage)")
            logger.error("  python main.py --fresh  (restart from scratch)")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"All {len(STAGES)} pipeline stages completed successfully!")
    logger.info("=" * 60)

    if args.keep_checkpoint:
        logger.info("Checkpoint retained (--keep-checkpoint).")
    else:
        # Clear checkpoint on full success — next run will be fresh
        clear_checkpoint()


if __name__ == "__main__":
    main()
