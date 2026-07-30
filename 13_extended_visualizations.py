import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import soundfile as sf
import scipy.signal
import importlib.util
from scipy.stats import pearsonr

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def compute_error_rates(df, condition_col, condition_val, is_axis_a=True):
    if is_axis_a:
        sub_df = df[(df['noise_type'] == condition_val[0]) & (df['snr_db'] == condition_val[1])]
    else:
        sub_df = df[df['condition'] == condition_val]
        
    total_words = sub_df['word_count_ref'].sum()
    if total_words == 0:
        return 0, 0, 0
    subs = sub_df['subs'].sum() / total_words * 100
    ins = sub_df['ins'].sum() / total_words * 100
    dels = sub_df['dels'].sum() / total_words * 100
    return subs, ins, dels

def main():
    print("======================================================================")
    print("  13_extended_visualizations.py")
    print("  Phase 4: Extended Visualizations")
    print("======================================================================\n")

    results_dir = Path("results")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Load data
    df_a = pd.read_csv(results_dir / "axis_a_results.csv")
    df_b = pd.read_csv(results_dir / "axis_b_results.csv")
    df_c = pd.read_csv(results_dir / "axis_c_results.csv")
    df_base = pd.read_csv(results_dir / "baseline_results.csv")
    df_base = df_base.rename(columns={'insertions': 'ins', 'substitutions': 'subs', 'deletions': 'dels'})
    
    # Base WER & Errors
    base_wer = df_base['wer'].mean() * 100
    b_total = df_base['word_count_ref'].sum()
    base_subs = df_base['subs'].sum() / b_total * 100
    base_ins = df_base['ins'].sum() / b_total * 100
    base_dels = df_base['dels'].sum() / b_total * 100

    # -------------------------------------------------------------------------
    # 1. Individual WER vs. SNR line plots
    # -------------------------------------------------------------------------
    sns.set_style("whitegrid")
    noises = ['white', 'pink', 'cafe']
    colors = ['#1f77b4', '#e377c2', '#8c564b']
    
    for noise, color in zip(noises, colors):
        plt.figure(figsize=(6, 4))
        agg = df_a[df_a['noise_type'] == noise].groupby('snr_db')['wer'].mean().reset_index()
        agg['wer'] *= 100
        
        sns.lineplot(data=agg, x='snr_db', y='wer', marker='o', color=color, linewidth=2)
        plt.gca().invert_xaxis()
        plt.title(f'{noise.capitalize()} Noise: WER vs SNR', fontsize=12, fontweight='bold')
        plt.xlabel('SNR (dB) [Lower is noisier]')
        plt.ylabel('Word Error Rate (%)')
        plt.tight_layout()
        plt.savefig(plots_dir / f'axis_a_wer_vs_snr_{noise}.png', dpi=300)
        plt.close()
    print("  ✓ Individual WER vs SNR plots saved.")

    # -------------------------------------------------------------------------
    # 2. Error-type stacked bar chart at EVERY SNR level (4-panel grid)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    snr_levels = [20, 15, 10, 5]
    
    for i, snr in enumerate(snr_levels):
        err_rates = []
        for noise in noises:
            s, ins, d = compute_error_rates(df_a, ('noise_type', 'snr_db'), (noise, snr))
            err_rates.append({'Noise': noise.capitalize(), 'Substitutions': s, 'Insertions': ins, 'Deletions': d})
            
        df_err = pd.DataFrame(err_rates).set_index('Noise')
        df_err.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff', '#99ff99'], ax=axes[i], legend=(i==0))
        axes[i].set_title(f'Error Profile at {snr} dB SNR')
        axes[i].set_ylabel('Error Rate (%)')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=0)
        
    plt.suptitle("Axis A: Error Profile Shifts Across Severity Levels", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'axis_a_error_profiles_all_snr.png', dpi=300)
    plt.close()
    print("  ✓ 4-panel error profile stacked bar chart saved.")

    # -------------------------------------------------------------------------
    # 3. Spectrogram comparison grid & 4. MFCC deviation heatmaps
    # -------------------------------------------------------------------------
    sample_uid = "6345-93306-0024"
    clean_path = Path("data/librispeech_clean") / f"{sample_uid}.wav"
    
    try:
        feat_engine = load_module("feature_engine", "08_feature_extraction_engine.py")
        if clean_path.exists():
            y_clean, sr = sf.read(clean_path)
            _, mfcc_clean, _, _ = feat_engine.extract_mel_spec_and_mfcc(y_clean, sr)
            
            # 3. Spectrograms
            fig_spec, axes_spec = plt.subplots(3, 3, figsize=(15, 10))
            
            def plot_spec(y, ax, title):
                ax.specgram(y, NFFT=512, Fs=sr, noverlap=256, cmap='inferno')
                ax.set_title(title)
                ax.set_ylabel('Frequency (Hz)')
                ax.set_xticks([])
                
            plot_spec(y_clean, axes_spec[0, 0], f"Clean ({sample_uid})")
            axes_spec[0, 1].axis('off')
            axes_spec[0, 2].axis('off')
            
            # 4. MFCC Heatmaps setup
            fig_mfcc, axes_mfcc = plt.subplots(3, 1, figsize=(10, 12))
            
            for row_idx, noise in enumerate(noises):
                for col_idx, snr in enumerate([20, 5]):
                    deg_path = Path("data/degraded/axis_a") / f"{sample_uid}_{noise}_{snr}db.wav"
                    if deg_path.exists():
                        y_deg, _ = sf.read(deg_path)
                        plot_spec(y_deg, axes_spec[row_idx, col_idx + 1], f"{noise.capitalize()} {snr}dB")
                        
                        # MFCC Heatmap (only for 5dB)
                        if snr == 5:
                            _, mfcc_deg, _, _ = feat_engine.extract_mel_spec_and_mfcc(y_deg, sr)
                            mfcc_diff = np.abs(mfcc_clean - mfcc_deg)
                            sns.heatmap(mfcc_diff, cmap='magma', ax=axes_mfcc[row_idx], cbar=True)
                            axes_mfcc[row_idx].set_title(f'MFCC Deviation: Clean vs {noise.capitalize()} 5dB', fontweight='bold')
                            axes_mfcc[row_idx].invert_yaxis()
                            axes_mfcc[row_idx].set_ylabel('MFCC Coeff')
            
            fig_spec.suptitle('Spectrogram Comparison: Clean vs. Noise Conditions', fontsize=16, fontweight='bold')
            fig_spec.tight_layout()
            fig_spec.savefig(plots_dir / 'spectrogram_grid.png', dpi=300)
            plt.close(fig_spec)
            print("  ✓ Spectrogram grid saved.")
            
            fig_mfcc.tight_layout()
            fig_mfcc.savefig(plots_dir / 'mfcc_deviation_all_noises_5db.png', dpi=300)
            plt.close(fig_mfcc)
            print("  ✓ MFCC deviation heatmaps saved.")
        else:
            print(f"  [WARNING] Clean file missing: {clean_path}. Skipping Spectrograms & MFCCs.")
    except Exception as e:
        print(f"  [ERROR] Spectrogram/MFCC generation failed: {e}")

    # -------------------------------------------------------------------------
    # 5. Scatter plot with regression line: SNR vs. WER per noise type
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, noise in enumerate(noises):
        sub = df_a[df_a['noise_type'] == noise].copy()
        sub['wer'] *= 100
        sns.regplot(data=sub, x='snr_db', y='wer', ax=axes[i], color=colors[i], scatter_kws={'alpha':0.3})
        axes[i].invert_xaxis()
        
        # Overlay Pearson
        r, p = pearsonr(sub['snr_db'], sub['wer'])
        sig = "*" if p < 0.05 else " "
        text = f"r = {r:.3f}\np = {p:.2e}{sig}"
        axes[i].text(0.05, 0.95, text, transform=axes[i].transAxes, fontsize=11, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[i].set_title(f'{noise.capitalize()} Noise')
        axes[i].set_xlabel('SNR (dB)')
        axes[i].set_ylabel('WER (%)' if i == 0 else '')
        
    plt.suptitle("SNR vs. WER: Regression and Pearson Correlation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'axis_a_regression_scatter.png', dpi=300)
    plt.close()
    print("  ✓ Scatter plots with regression lines saved.")

    # -------------------------------------------------------------------------
    # 6. Axis B/C bar chart breakdown by error type (Sub/Ins/Del)
    # -------------------------------------------------------------------------
    bc_conditions = ['telephonic'] + list(df_c['condition'].unique())
    
    # We add baseline to the start
    err_rates = [{'Condition': 'baseline', 'Substitutions': base_subs, 'Insertions': base_ins, 'Deletions': base_dels}]
    
    for cond in bc_conditions:
        if cond == 'telephonic':
            s, ins, d = compute_error_rates(df_b, 'condition', cond, is_axis_a=False)
        else:
            s, ins, d = compute_error_rates(df_c, 'condition', cond, is_axis_a=False)
        err_rates.append({'Condition': cond, 'Substitutions': s, 'Insertions': ins, 'Deletions': d})
        
    df_bc_err = pd.DataFrame(err_rates).set_index('Condition')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df_bc_err.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff', '#99ff99'], ax=ax)
    
    plt.title('Axis B & C: Error Profile Breakdown by Condition', fontsize=14, fontweight='bold')
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Highlight silence_start_0.5s text on x-axis
    xticklabels = ax.get_xticklabels()
    for label in xticklabels:
        if label.get_text() == 'silence_start_0.5s':
            label.set_color('#d62728')
            label.set_fontweight('bold')
    
    plt.legend(title='Error Type')
    plt.tight_layout()
    plt.savefig(plots_dir / 'axis_b_c_error_profiles.png', dpi=300)
    plt.close()
    print("  ✓ Axis B/C error profile breakdown saved.")

    print("\n  Extended visualizations complete.")

if __name__ == "__main__":
    main()
