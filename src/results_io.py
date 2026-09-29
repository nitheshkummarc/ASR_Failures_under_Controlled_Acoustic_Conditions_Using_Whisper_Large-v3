"""
Reading and writing the project's result CSVs.

Rows are plain lists of dicts. pandas is not used anywhere in this project --
AppLocker on the target machine blocks its compiled extensions.
"""

import csv
from pathlib import Path


def read_rows(path) -> list:
    """
    Read a CSV into a list of dicts.

    A missing file raises FileNotFoundError naming the path and pointing at the
    stage table, since every caller is a pipeline stage or a CLI tool.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the pipeline stage that produces it "
            f"(see the stage table in README.md)."
        )

    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows: list) -> None:
    """
    Write a list of dicts as a CSV, header taken from the first row.

    An empty list writes nothing. A headerless file reads back as zero rows
    while looking like a successful write.
    """
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(manifest_path=None) -> list:
    """
    Read metadata/utterance_manifest.csv.

    Every stage and tool starts from the manifest, so it is read here rather
    than in each of them. Defaults to the configured path; pass one explicitly
    for a test-mode or alternative corpus.

    Raises FileNotFoundError if absent and RuntimeError if present but empty --
    an empty manifest otherwise surfaces much later as "0 utterances processed".
    """
    from config import METADATA_DIR

    path = Path(manifest_path) if manifest_path else METADATA_DIR / "utterance_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            f"Run `python scripts/00_prepare_corpus.py` (or `python main.py`) first."
        )

    rows = read_rows(path)
    if not rows:
        raise RuntimeError(f"Manifest {path} contains no utterances.")
    return rows


def word_count(text) -> int:
    """Whitespace word count, used for reference-length weighting."""
    return len(str(text).split())


def error_count(row) -> int:
    """Total errors in a result row: substitutions + insertions + deletions."""
    return int(row["substitutions"]) + int(row["insertions"]) + int(row["deletions"])
