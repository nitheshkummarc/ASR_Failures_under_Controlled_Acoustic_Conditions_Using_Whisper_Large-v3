"""
Energetic vs informational masking report.

Answers the question the WER table alone cannot: WHY does babble hurt more than
noise? It combines two independent lines of evidence.

  ACOUSTIC (from the waveforms, no recognizer involved)
      Where each masker puts its energy, the in-band SNR that results, the
      speech-weighted audibility (SII) it leaves intact, and how much its
      envelope fluctuates.

  BEHAVIOURAL (from the result CSVs)
      The WER decomposition into energetic and informational components, using
      the phase-randomized SSN control.

The validity check ties them together: cafe and SSN must score the SAME
speech-weighted SNR, because phase randomization preserves the magnitude
spectrum exactly. If they do, then the WER gap between them cannot be energetic.

Writes results/masking_decomposition.csv for the write-up; prints (rather than
logs) because it is an interactive report.

Usage:
    python tools/masking_analysis.py
    python tools/masking_analysis.py --utterances 60
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv

import numpy as np
import soundfile as sf

from config import (
    DATA_DIR, METADATA_DIR, RESULTS_DIR, TARGET_SR,
    CAFE_NOISE_PATH, SSN_NOISE_PATH, NOISE_TYPES, SNR_LEVELS, RANDOM_SEED,
)
from src.analysis.masking import (
    SPEECH_BANDS, band_snr_profile, speech_weighted_snr, modulation_stats,
    decompose_masking, paired_significance,
)
from src.analysis.metrics import global_wer
from src.augmentation.axis_a_noise import generate_white_noise, generate_pink_noise


def load_speech_reference(n_utterances: int) -> np.ndarray:
    """
    Concatenate clean utterances into one long-term speech reference.

    A single utterance is too short and too idiosyncratic for a stable long-term
    average spectrum — one speaker's voice, one sentence's phoneme balance. Tens
    of utterances across speakers converge on the corpus spectrum, which is what
    the band analysis should be measured against.
    """
    manifest = METADATA_DIR / "utterance_manifest.csv"
    if not manifest.exists():
        print(f"[ERROR] Manifest not found: {manifest}. Run the pipeline first.")
        sys.exit(1)

    with open(manifest, "r", encoding="utf-8") as f:
        uids = [row["utterance_id"] for row in csv.DictReader(f)][:n_utterances]

    chunks = []
    for uid in uids:
        wav = DATA_DIR / f"{uid}.wav"
        if wav.exists():
            chunks.append(sf.read(str(wav), dtype="float64")[0])

    if not chunks:
        print(f"[ERROR] No clean audio found in {DATA_DIR}. Run stage 0 first.")
        sys.exit(1)

    return np.concatenate(chunks)


def load_maskers(n_samples: int) -> dict:
    """
    Materialise one full-length instance of every masker.

    White and pink are regenerated from the same seeded generators the pipeline
    uses. Cafe and SSN are read from disk — they must be the exact buffers the
    degraded audio was built from, not fresh ones, or the analysis would describe
    a different experiment than the one that was run.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    maskers = {
        "white": generate_white_noise(n_samples, rng),
        "pink": generate_pink_noise(n_samples, rng),
    }

    for name, path in (("cafe", CAFE_NOISE_PATH), ("ssn", SSN_NOISE_PATH)):
        if not path.exists():
            print(f"[WARN] {name} source missing ({path}) — skipping it. "
                  f"Run scripts/02_run_axis_a.py to generate it.")
            continue
        audio, _ = sf.read(str(path), dtype="float64")
        if len(audio) < n_samples:
            audio = np.tile(audio, n_samples // len(audio) + 1)
        maskers[name] = audio[:n_samples]

    return maskers


def acoustic_report(speech: np.ndarray, maskers: dict) -> None:
    names = [n for n in NOISE_TYPES if n in maskers]

    print("\n" + "=" * 78)
    print("1. SPECTRAL OVERLAP — where each masker puts its energy")
    print("=" * 78)
    print("Speech energy is concentrated below 1 kHz, but the information that")
    print("distinguishes words lives in the 1-4 kHz consonant region.\n")
    print(f"{'band':<26} {'speech':>8} " + " ".join(f"{n:>8}" for n in names))

    profiles = {n: band_snr_profile(speech, maskers[n], TARGET_SR, 5.0) for n in names}
    for i, (low, high, label) in enumerate(SPEECH_BANDS):
        speech_pct = profiles[names[0]][i]["speech_energy_pct"]
        row = " ".join(f"{profiles[n][i]['noise_energy_pct']:8.1f}" for n in names)
        print(f"{label + f' ({low:.0f}-{high:.0f})':<26} {speech_pct:8.1f} {row}")

    print("\n" + "=" * 78)
    print("2. IN-BAND SNR at a fixed BROADBAND SNR of 5 dB")
    print("=" * 78)
    print("The experiment equalises broadband RMS, but recognition depends on the")
    print("SNR where the phonetic information actually is. LOWER = more masking.\n")
    print(f"{'band':<26} " + " ".join(f"{n:>8}" for n in names))
    for i, (low, high, label) in enumerate(SPEECH_BANDS):
        row = " ".join(f"{profiles[n][i]['in_band_snr_db']:8.2f}" for n in names)
        print(f"{label + f' ({low:.0f}-{high:.0f})':<26} {row}")

    print("\n" + "=" * 78)
    print("3. SPEECH-WEIGHTED AUDIBILITY (SII, 0-1) — the ENERGETIC prediction")
    print("=" * 78)
    print("Computed from the spectra alone, so it is blind to linguistic content.")
    print("HIGHER = more speech information survives = less energetic masking.\n")
    print(f"{'SNR':>6} " + " ".join(f"{n:>8}" for n in names))
    for snr in sorted(SNR_LEVELS, reverse=True):
        row = " ".join(
            f"{speech_weighted_snr(speech, maskers[n], TARGET_SR, snr):8.3f}" for n in names
        )
        print(f"{snr:5d}dB {row}")

    if "cafe" in maskers and "ssn" in maskers:
        gaps = [
            abs(speech_weighted_snr(speech, maskers["cafe"], TARGET_SR, s)
                - speech_weighted_snr(speech, maskers["ssn"], TARGET_SR, s))
            for s in SNR_LEVELS
        ]
        worst = max(gaps)
        verdict = "PASS" if worst < 0.01 else "FAIL"
        print(f"\n  [{verdict}] VALIDITY CHECK: max |SII(cafe) - SII(ssn)| = {worst:.4f}")
        print("         Phase randomization preserves the magnitude spectrum, so these")
        print("         must match. They do => any cafe-vs-ssn WER gap is NOT energetic.")

    print("\n" + "=" * 78)
    print("4. TEMPORAL MODULATION — glimpsing opportunity")
    print("=" * 78)
    print("A masker with deep dips lets the model catch the target in the gaps.")
    print("Real speech is included as the upper reference.\n")
    print(f"{'masker':<10} {'env CV':>9} {'dips>6dB':>10} {'dips>12dB':>11} {'2-8Hz mod':>11}")
    for name in names + ["(real speech)"]:
        signal = speech if name == "(real speech)" else maskers[name]
        st = modulation_stats(signal, TARGET_SR)
        print(f"{name:<10} {st['envelope_cv']:9.3f} {st['dip_6db_pct']:9.1f}% "
              f"{st['dip_12db_pct']:10.1f}% {st['mod_2_8hz_pct']:10.1f}%")


def behavioural_report(out_dir: Path) -> list:
    """WER decomposition per SNR. Returns rows for the CSV."""
    baseline_path = out_dir / "baseline_results.csv"
    axis_a_path = out_dir / "axis_a_results.csv"
    if not baseline_path.exists() or not axis_a_path.exists():
        print(f"\n[WARN] Result CSVs missing under {out_dir} — skipping the "
              f"behavioural decomposition. Run stages 01 and 02 first.")
        return []

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_wer = global_wer(list(csv.DictReader(f)))
    with open(axis_a_path, "r", encoding="utf-8") as f:
        axis_a = list(csv.DictReader(f))

    print("\n" + "=" * 78)
    print("5. WER DECOMPOSITION — energetic vs informational")
    print("=" * 78)
    print(f"Clean baseline global WER: {baseline_wer:.2f}%\n")
    print("  energetic     = WER(ssn)  - WER(white)   energy moved into speech bands")
    print("  informational = WER(cafe) - WER(ssn)     same spectrum, content added\n")
    print(f"{'SNR':>6} " + " ".join(f"{n:>8}" for n in NOISE_TYPES)
          + f" {'energetic':>11} {'informat.':>10} {'info share':>11}")

    rows = []
    for snr in sorted(SNR_LEVELS, reverse=True):
        wer_by_noise = {}
        for noise in NOISE_TYPES:
            cell = [r for r in axis_a
                    if r["noise_type"] == noise and int(float(r["snr_db"])) == snr]
            if cell:
                wer_by_noise[noise] = global_wer(cell)

        d = decompose_masking(wer_by_noise, baseline_wer)
        cells = " ".join(
            f"{wer_by_noise[n]:8.2f}" if n in wer_by_noise else f"{'--':>8}"
            for n in NOISE_TYPES
        )

        def fmt(v):
            return f"{v:+.2f}" if v is not None else "--"

        share = d["informational_share"]
        share_txt = f"{share * 100:.0f}%" if share is not None else "--"

        print(f"{snr:5d}dB {cells} {fmt(d['energetic_wer']):>11} "
              f"{fmt(d['informational_wer']):>10} {share_txt:>11}")

        row = {"snr_db": snr, "baseline_wer": round(baseline_wer, 4)}
        for noise in NOISE_TYPES:
            row[f"wer_{noise}"] = (round(wer_by_noise[noise], 4)
                                   if noise in wer_by_noise else "")
        for key in ("total_excess_wer", "energetic_wer",
                    "informational_wer", "informational_share"):
            row[key] = round(d[key], 4) if d[key] is not None else ""
        rows.append(row)

    if any(r["informational_wer"] != "" for r in rows):
        print("\n  A POSITIVE informational term means the babble costs more than a")
        print("  spectrally identical noise — damage that energy cannot explain.")

    # --- paired significance for the two mechanism contrasts ---
    # Both tests are printed. They weight utterances differently (see
    # paired_significance in src/analysis/masking.py) and on this data they can
    # disagree; printing a single verdict hid that.
    def paired_rows(rows_x, rows_y):
        by_x = {r["utterance_id"]: r for r in rows_x}
        by_y = {r["utterance_id"]: r for r in rows_y}
        diffs, words = [], []
        for uid in sorted(set(by_x) & set(by_y)):
            ex = sum(int(by_x[uid][f]) for f in ("substitutions", "insertions", "deletions"))
            ey = sum(int(by_y[uid][f]) for f in ("substitutions", "insertions", "deletions"))
            diffs.append(ex - ey)
            words.append(len(str(by_y[uid]["ground_truth"]).split()))
        return diffs, words

    print("\n" + "=" * 78)
    print("6. PAIRED SIGNIFICANCE OF THE TWO MECHANISMS")
    print("=" * 78)
    print("Bootstrap resamples utterances and weights by length (matching the headline")
    print("metric); Wilcoxon ranks per-utterance differences equally. Both are shown.\n")
    print(f"{'contrast':<30}{'diff':>7}{'95% CI':>18}{'wilcoxon p':>12}"
          f"{'worse':>7}{'better':>7}  verdict")

    for label, x_name, y_name in (("energetic     (ssn - white)", "ssn", "white"),
                                  ("informational (cafe - ssn)", "cafe", "ssn")):
        pooled_d, pooled_w = [], []
        for snr in sorted(SNR_LEVELS, reverse=True):
            gx = [r for r in axis_a
                  if r["noise_type"] == x_name and int(float(r["snr_db"])) == snr]
            gy = [r for r in axis_a
                  if r["noise_type"] == y_name and int(float(r["snr_db"])) == snr]
            if not gx or not gy:
                continue
            d, w = paired_rows(gx, gy)
            pooled_d += d
            pooled_w += w
            st = paired_significance(d, w)
            print(f"{label + f' @ {snr}dB':<30}{st['diff_pp']:+7.2f}"
                  f" [{st['ci_low']:+6.2f},{st['ci_high']:+6.2f}]{st['wilcoxon_p']:12.5f}"
                  f"{st['n_worse']:7d}{st['n_better']:7d}  {st['verdict']}")
        if pooled_d:
            st = paired_significance(pooled_d, pooled_w)
            print(f"{label + ' POOLED':<30}{st['diff_pp']:+7.2f}"
                  f" [{st['ci_low']:+6.2f},{st['ci_high']:+6.2f}]{st['wilcoxon_p']:12.5f}"
                  f"{st['n_worse']:7d}{st['n_better']:7d}  {st['verdict'].upper()}")
        print()

    print('  "disputed" = the two tests disagree, so the effect depends on whether')
    print("  utterances are weighted by length. Report it as such, not as a result.")
    return rows


def main(n_utterances: int, test_mode: bool) -> None:
    out_dir = RESULTS_DIR / "test_mode" if test_mode else RESULTS_DIR

    print("=" * 78)
    print("ENERGETIC vs INFORMATIONAL MASKING REPORT")
    print("=" * 78)

    speech = load_speech_reference(n_utterances)
    print(f"Speech reference: {len(speech) / TARGET_SR:.1f}s "
          f"from up to {n_utterances} utterances")

    maskers = load_maskers(len(speech))
    acoustic_report(speech, maskers)
    rows = behavioural_report(out_dir)

    if rows:
        out_csv = out_dir / "masking_decomposition.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved decomposition to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energetic vs informational masking report")
    parser.add_argument("--utterances", type=int, default=40,
                        help="Clean utterances to pool for the speech reference (default 40)")
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    main(args.utterances, args.test_mode)
