import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

def print_correlation(name, x, y, x_name, y_name, note=""):
    try:
        p_corr, p_p = pearsonr(x, y)
        s_corr, s_p = spearmanr(x, y)
        
        p_sig = "*" if p_p < 0.05 else " "
        s_sig = "*" if s_p < 0.05 else " "
        
        print(f"  [{name}] {x_name} vs {y_name}:")
        print(f"    Pearson:  {p_corr:6.3f} (p={p_p:.3e}) {p_sig}")
        print(f"    Spearman: {s_corr:6.3f} (p={s_p:.3e}) {s_sig}")
        if note:
            print(f"    Note: {note}")
        print()
    except Exception as e:
        print(f"  [{name}] Error computing correlation: {e}\n")

def main():
    print("======================================================================")
    print("  10_statistical_validation.py")
    print("  Phase 4: Statistical Validation")
    print("======================================================================\n")

    results_dir = Path("results")
    
    # Load results
    df_a = pd.read_csv(results_dir / "axis_a_results.csv")
    df_c = pd.read_csv(results_dir / "axis_c_results.csv")
    df_base = pd.read_csv(results_dir / "baseline_results.csv")
    
    # Harmonize baseline column names
    df_base = df_base.rename(columns={'insertions': 'ins', 'substitutions': 'subs', 'deletions': 'dels'})
    
    print("--- 1. AXIS A: SNR vs WER ---")
    for noise in ["white", "pink", "cafe"]:
        sub = df_a[df_a["noise_type"] == noise]
        print_correlation(f"Axis A - {noise.capitalize()}", sub["snr_db"], sub["wer"], "SNR (dB)", "WER")

    print("--- 2. AXIS C: Silence Duration vs Insertions ---")
    # i) Clean mid-position points (0s, 1s, 2s)
    base_sil = df_base.copy()
    base_sil['duration_val'] = 0.0
    
    c_mid_1 = df_c[df_c['condition'] == 'silence_mid_1.0s'].copy()
    c_mid_1['duration_val'] = 1.0
    
    c_mid_2 = df_c[df_c['condition'] == 'silence_mid_2.0s'].copy()
    c_mid_2['duration_val'] = 2.0
    
    df_sil_mid = pd.concat([base_sil, c_mid_1, c_mid_2])
    print_correlation(
        "Axis C - Silence (Mid Only)", 
        df_sil_mid['duration_val'], 
        df_sil_mid['ins'], 
        "Duration (s)", 
        "Insertions",
        "Strict apples-to-apples duration sweep (0s, 1s, 2s) all at mid position."
    )
    
    # ii) Comparison including confounded 0.5s start condition
    c_start_05 = df_c[df_c['condition'] == 'silence_start_0.5s'].copy()
    c_start_05['duration_val'] = 0.5
    
    # We will do a manual comparison of means here as requested, or correlation
    df_sil_confounded = pd.concat([df_sil_mid, c_start_05])
    print_correlation(
        "Axis C - Silence (Confounded)", 
        df_sil_confounded['duration_val'], 
        df_sil_confounded['ins'], 
        "Duration (s)", 
        "Insertions",
        "WARNING: Includes 0.5s at START position. This confounds duration with position."
    )
    
    print("--- 3. AXIS C: Time-Stretch Factor vs WER ---")
    base_stretch = df_base.copy()
    base_stretch['stretch_val'] = 1.0
    
    c_stretch_12 = df_c[df_c['condition'] == 'stretch_1.2x'].copy()
    c_stretch_12['stretch_val'] = 1.2
    
    c_stretch_15 = df_c[df_c['condition'] == 'stretch_1.5x'].copy()
    c_stretch_15['stretch_val'] = 1.5
    
    df_stretch = pd.concat([base_stretch, c_stretch_12, c_stretch_15])
    print_correlation(
        "Axis C - Stretch", 
        df_stretch['stretch_val'], 
        df_stretch['wer'], 
        "Stretch Factor", 
        "WER"
    )

    print("======================================================================")
    print("  STATISTICAL SIGNIFICANCE SUMMARY")
    print("======================================================================")
    print("  * indicates statistical significance (p < 0.05).")
    print("\n  IMPORTANT DATA CAVEAT:")
    print("  Axis A correlations are based on 4 discrete SNR levels (20, 15, 10, 5 dB).")
    print("  Axis B/C correlations are based on far fewer discrete levels (e.g. 3 levels for time-stretch).")
    print("  Therefore, while the correlations may be statistically significant across the utterances,")
    print("  the discrete x-axis variables should be interpreted as directional trends rather than")
    print("  robust, highly-sampled statistical curves.")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
