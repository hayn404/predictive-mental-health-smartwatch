"""
SAE Exp4 adapter (EXTERNAL TEST) → ml/fatigue/data/fatigue_features_sae.csv

Uses the dataset's pre-computed Neurokit window features (180 s windows). We take the
DRIVING-period HRV features (suffix _Dr — they vary per window, unlike _Bl baseline),
map Neurokit names to our 12 shared features, and label by label_sleep (sleep-deprivation
fatigue). NOTE: SAE fatigue is BETWEEN-subject (each subject is one condition), unlike
MEFAR's within-subject design — disclosed in the report.
"""
import os, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEAT = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','cvRR','sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']

# our feature -> Neurokit column (driving period)
MAP = {
    'meanRR': 'HRV_MeanNN_Dr', 'sdnn': 'HRV_SDNN_Dr', 'rmssd': 'HRV_RMSSD_Dr',
    'pnn50': 'HRV_pNN50_Dr', 'pnn20': 'HRV_pNN20_Dr', 'hrMean': 'ECG_Rate_Mean_Dr',
    'sd1': 'HRV_SD1_Dr', 'sd2': 'HRV_SD2_Dr', 'sd1sd2Ratio': 'HRV_SD1SD2_Dr',
    'sampleEntropy': 'HRV_SampEn_Dr', 'dfaAlpha1': 'HRV_DFA_alpha1_Dr',
}


def main():
    f = glob.glob(os.path.join(BASE, 'data', 'sae_exp4', '**', 'features_window_180s_overlap_0.csv'), recursive=True)[0]
    df = pd.read_csv(f)
    missing = [v for v in MAP.values() if v not in df.columns]
    if missing:
        print("WARN missing cols:", missing)
    out = pd.DataFrame()
    out['subject_id'] = 'SAE_' + df['subject_id'].astype(str)
    out['label'] = df['label_sleep'].astype(int)
    for k, col in MAP.items():
        out[k] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else 0.0
    out['cvRR'] = out['sdnn'] / out['meanRR'].replace(0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out[['subject_id', 'label'] + FEAT]
    p = os.path.join(BASE, 'data', 'fatigue_features_sae.csv')
    out.to_csv(p, index=False)
    n0, n1 = int((out.label == 0).sum()), int((out.label == 1).sum())
    print(f"SAE Exp4: {len(out)} windows | rested={n0} sleep-deprived={n1} | subjects={out.subject_id.nunique()}")
    print(f"saved -> {p}")


if __name__ == '__main__':
    main()
