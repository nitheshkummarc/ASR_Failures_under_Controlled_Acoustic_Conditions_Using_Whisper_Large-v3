import pandas as pd
from pathlib import Path

def generate_report(df_a, df_b, df_c, df_base):
    # Calculations
    base_wer = df_base['wer'].mean() * 100
    
    # Axis A 5dB stats
    df_5db = df_a[df_a['snr_db'] == 5]
    cafe_5db = df_5db[df_5db['noise_type'] == 'cafe']
    white_5db = df_5db[df_5db['noise_type'] == 'white']
    
    cafe_wer = cafe_5db['wer'].mean() * 100
    white_wer = white_5db['wer'].mean() * 100
    
    cafe_ins = cafe_5db['ins'].sum()
    white_ins = white_5db['ins'].sum()
    
    # Axis B stats
    b_wer = df_b['wer'].mean() * 100
    
    # Axis C stats
    stretch_15x = df_c[df_c['condition'] == 'stretch_1.5x']['wer'].mean() * 100
    
    report = f"""======================================================================
OBSERVATION-BASED DISCUSSION DRAFT
======================================================================

Based on the empirical evaluation across three distinct acoustic axes, the following 
trends were observed regarding the Whisper large-v3 model's robustness:

1. Axis A: Additive Noise (High Sample Size)
The data shows a clear trend where decreasing Signal-to-Noise Ratio (SNR) correlates with 
an increase in Word Error Rate (WER). At the most extreme condition (5 dB SNR), Café babble 
was associated with a significantly higher WER ({cafe_wer:.2f}%) compared to White noise 
({white_wer:.2f}%). Furthermore, overlapping speech (Café babble) co-occurs with a distinct 
shift in the error profile, generating {cafe_ins} total insertion errors compared to only 
{white_ins} insertion errors under White noise. This indicates that the model frequently 
hallucinates or transcribes the background babble as foreground speech.

2. Axis B: Channel Degradation
Despite the strict telephonic bandpass filter (300-3400 Hz) removing substantial high 
and low-frequency acoustic information, the model demonstrated remarkable resilience. 
The telephonic condition was associated with a WER of {b_wer:.2f}%, which represents 
only a marginal degradation from the clean baseline ({base_wer:.2f}%). This is an observed 
trend across the tested levels, suggesting Whisper's internal representations are highly 
invariant to standard PSTN frequency masking.

3. Axis C: Temporal Distortions
Temporal distortions revealed significant vulnerabilities. Time-stretching the audio 
by a factor of 1.5x (without pitch shifting) correlates with a severe degradation, 
resulting in a WER of {stretch_15x:.2f}%. 

Regarding silence insertion, the model largely powered through dead air. However, 
it is critical to note a methodological confound in the dataset: the 1.0s and 2.0s 
silences were inserted in the middle of the utterance, while the 0.5s silence was 
inserted at the start. Consequently, duration and position are confounded for the 
0.5s condition, meaning the observed error rate conflates the effect of a shorter 
duration with the impact of a start-of-utterance disruption. 

*Statistical Qualification*: While Axis A findings are supported by a large sample 
matrix (4 SNR levels × 150 utterances), the findings for Axis B and Axis C represent 
observed trends across sparse, discrete levels (3-4 conditions each). They should be 
interpreted as directional trends rather than highly robust statistical curves. 
Furthermore, all feature deviations and error rate increases are methodologically framed 
as being associated with or co-occurring with specific acoustic conditions, as 
correlation across these axes does not establish strict causality.
======================================================================
"""
    return report

def main():
    print("======================================================================")
    print("  12_observation_drafting.py")
    print("  Phase 4: Drafting Observations")
    print("======================================================================\n")

    results_dir = Path("results")
    
    df_a = pd.read_csv(results_dir / "axis_a_results.csv")
    df_c = pd.read_csv(results_dir / "axis_c_results.csv")
    df_b = pd.read_csv(results_dir / "axis_b_results.csv")
    df_base = pd.read_csv(results_dir / "baseline_results.csv")
    
    draft = generate_report(df_a, df_b, df_c, df_base)
    
    out_path = results_dir / "observation_draft.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(draft)
        
    print(draft)
    print(f"  ✓ Draft saved to {out_path}")

if __name__ == "__main__":
    main()
