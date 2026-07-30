import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import soundfile as sf
import importlib.util

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    print("======================================================================")
    print("  11_visualization_engine.py")
    print("  Phase 4: Statistical Validation & Plots")
    print("======================================================================\n")

    results_dir = Path("results")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Load data
    df_a = pd.read_csv(results_dir / "axis_a_results.csv")
    df_c = pd.read_csv(results_dir / "axis_c_results.csv")
    df_b = pd.read_csv(results_dir / "axis_b_results.csv")
    df_base = pd.read_csv(results_dir / "baseline_results.csv")
    
    # Base WER
    base_wer = df_base['wer'].mean() * 100

    # -------------------------------------------------------------------------
    # Plot a) Line Graph: WER vs SNR
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Group by noise type and snr
    agg_a = df_a.groupby(['noise_type', 'snr_db'])['wer'].mean().reset_index()
    agg_a['wer'] *= 100  # Convert to percentage
    
    sns.lineplot(data=agg_a, x='snr_db', y='wer', hue='noise_type', marker='o', palette=['#1f77b4', '#e377c2', '#8c564b'])
    
    plt.gca().invert_xaxis()  # 20 to 5 (decreasing SNR)
    plt.title('Axis A: Word Error Rate vs. Signal-to-Noise Ratio', fontsize=14, fontweight='bold')
    plt.xlabel('SNR (dB) [Lower is noisier]', fontsize=12)
    plt.ylabel('Word Error Rate (%)', fontsize=12)
    plt.legend(title='Noise Type', title_fontsize='11', fontsize='10')
    plt.tight_layout()
    plt.savefig(plots_dir / 'axis_a_wer_vs_snr.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot b) Stacked Bar Chart: Subs, Ins, Dels at 5 dB
    # -------------------------------------------------------------------------
    df_5db = df_a[df_a['snr_db'] == 5].copy()
    
    # Calculate percentages relative to reference word count
    err_rates = []
    for noise in ['white', 'pink', 'cafe']:
        sub_df = df_5db[df_5db['noise_type'] == noise]
        total_words = sub_df['word_count_ref'].sum()
        subs = sub_df['subs'].sum() / total_words * 100
        ins = sub_df['ins'].sum() / total_words * 100
        dels = sub_df['dels'].sum() / total_words * 100
        err_rates.append({'Noise Type': noise.capitalize(), 'Substitutions': subs, 'Insertions': ins, 'Deletions': dels})
        
    df_err = pd.DataFrame(err_rates).set_index('Noise Type')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    df_err.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff', '#99ff99'], ax=ax)
    
    plt.title('Axis A (5 dB): Error Profile Breakdown', fontsize=14, fontweight='bold')
    plt.xlabel('Noise Type (at 5 dB SNR)', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Error Type')
    plt.tight_layout()
    plt.savefig(plots_dir / 'axis_a_error_profile_5db.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot c) Bar Chart: Axis B and C compared to baseline
    # -------------------------------------------------------------------------
    b_wer = df_b['wer'].mean() * 100
    
    c_agg = df_c.groupby('condition')['wer'].mean().reset_index()
    c_agg['wer'] *= 100
    
    conditions = ['baseline', 'telephonic'] + list(c_agg['condition'])
    wers = [base_wer, b_wer] + list(c_agg['wer'])
    
    plt.figure(figsize=(10, 6))
    
    # Assign colors: baseline (grey), Axis B (blue), Axis C Mid (green), Axis C Start Conf (red), Stretch (orange)
    colors = []
    for cond in conditions:
        if cond == 'baseline': colors.append('gray')
        elif cond == 'telephonic': colors.append('#1f77b4')
        elif 'silence_mid' in cond: colors.append('#2ca02c')
        elif 'silence_start' in cond: colors.append('#d62728')  # Distinct color for confound
        elif 'stretch' in cond: colors.append('#ff7f0e')
        else: colors.append('black')
        
    bars = plt.bar(conditions, wers, color=colors)
    plt.axhline(y=base_wer, color='gray', linestyle='--', alpha=0.7)
    
    plt.title('Axis B & C: WER Across Channel & Temporal Degradations', fontsize=14, fontweight='bold')
    plt.ylabel('Word Error Rate (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Annotate the confounded bar
    for i, cond in enumerate(conditions):
        if cond == 'silence_start_0.5s':
            plt.text(i, wers[i] + 0.2, "Position Confound", ha='center', color='#d62728', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(plots_dir / 'axis_b_c_wer_summary.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Plot d) Heatmap: MFCC Mean Squared Error Deviation (Clean vs 5dB Cafe)
    # -------------------------------------------------------------------------
    try:
        # Load MFCC logic from 08 script
        feat_engine = load_module("feature_engine", "08_feature_extraction_engine.py")
        
        # Pick a sample utterance (from df_a where cafe 5db)
        sample_uid = "6345-93306-0024"
        clean_path = Path("data/librispeech_clean") / f"{sample_uid}.wav"
        deg_path = Path("data/degraded/axis_a") / f"{sample_uid}_cafe_5db.wav"
        
        if clean_path.exists() and deg_path.exists():
            y_clean, sr = sf.read(clean_path)
            y_deg, _ = sf.read(deg_path)
            
            _, mfcc_clean, _, _ = feat_engine.extract_mel_spec_and_mfcc(y_clean, sr)
            _, mfcc_deg, _, _ = feat_engine.extract_mel_spec_and_mfcc(y_deg, sr)
            
            # Absolute difference
            mfcc_diff = np.abs(mfcc_clean - mfcc_deg)
            
            plt.figure(figsize=(10, 4))
            sns.heatmap(mfcc_diff, cmap='magma', cbar_kws={'label': 'Absolute Difference'})
            plt.title(f'MFCC Deviation Heatmap: Clean vs. Café 5dB (Utterance: {sample_uid})', fontsize=14, fontweight='bold')
            plt.xlabel('Time Frames', fontsize=12)
            plt.ylabel('MFCC Coefficients', fontsize=12)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(plots_dir / 'mfcc_deviation_heatmap.png', dpi=300)
            plt.close()
            print("  ✓ Heatmap generated successfully.")
        else:
            print("  [WARNING] Sample utterance files not found. Skipping heatmap.")
            
    except Exception as e:
        print(f"  [ERROR] Failed to generate heatmap: {e}")

    print(f"\n  All plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
