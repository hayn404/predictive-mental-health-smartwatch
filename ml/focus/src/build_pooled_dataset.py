"""
Build the pooled engagement dataset from CogWear + MAUS + CLARE.

Each source is brought into ONE common schema:
    dataset, subject_id, label, [14 HRV features]
label: 0 = rest / low cognitive load, 1 = high cognitive load (engaged)

- CogWear : reuse precomputed cogwear_features_samsung.csv (Galaxy Watch4 PPG)
- MAUS    : RR intervals provided (RRI_inf). rest + 0-back = low(0), 2/3-back = high(1)
- CLARE   : raw ECG -> R-peaks -> RR. baseline = low(0); experiment windows binarized
            per-subject by the self-reported 9-point load (median split)

Output: ml/focus/data/pooled_features.csv
"""
import os, sys, glob, numpy as np, pandas as pd
from scipy import signal as ss
np.seterr(all='ignore')

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(BASE, 'stress'))
from src.features import compute_time_domain, compute_nonlinear  # noqa

FEATURES = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','hrStd','hrRange','cvRR',
            'sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']
DATA = os.path.join(BASE, 'focus', 'data')


def hrv_from_rr(rr):
    rr = np.asarray(rr, float)
    rr = rr[(rr >= 300) & (rr <= 2000)]
    if len(rr) > 5:
        med = np.median(rr); rr = rr[np.abs(rr - med) / med < 0.30]
    if len(rr) < 20:
        return None
    d = {}; d.update(compute_time_domain(rr)); d.update(compute_nonlinear(rr))
    return {k: float(d.get(k, 0.0)) for k in FEATURES}


# ---------- CogWear ----------
def build_cogwear():
    df = pd.read_csv(os.path.join(DATA, 'cogwear_features_samsung.csv'))
    out = df[['subject_id', 'label'] + [f for f in FEATURES if f in df.columns]].copy()
    out.insert(0, 'dataset', 'cogwear')
    return out


# ---------- MAUS ----------
def build_maus():
    root = os.path.join(DATA, 'maus', 'MAUS', 'Data', 'IBI_sequence')
    rows = []
    for sub in sorted(os.listdir(root)):
        sdir = os.path.join(root, sub)
        if not os.path.isdir(sdir):
            continue
        for f in glob.glob(os.path.join(sdir, '*.csv')):
            name = os.path.basename(f)[:-4]
            if 'peak' in name:
                continue
            if name.startswith('rest'):
                label = 0
            elif name.startswith('trial_'):
                trial = int(name.split('_')[1])      # 1..6
                label = 0 if trial in (1, 6) else 1  # 0-back=low, 2/3-back=high
            else:
                continue
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if 'RRI_inf' not in d.columns:
                continue
            fe = hrv_from_rr(d['RRI_inf'].values)
            if fe:
                rows.append({'dataset': 'maus', 'subject_id': f'MAUS_{sub}', 'label': label, **fe})
    return pd.DataFrame(rows)


# ---------- CLARE ----------
def ecg_to_rr(sig, t):
    fs = len(t) / (t[-1] - t[0]) if t[-1] > t[0] else 0
    if fs < 50:
        return np.array([]), 0
    # CLARE ECG CAL columns are ~50% NaN (interleaved) — interpolate before filtering.
    sig = pd.Series(np.asarray(sig, float)).interpolate(limit_direction='both').values
    if np.isnan(sig).any():
        return np.array([]), fs
    nyq = fs / 2
    b, a = ss.butter(3, [5 / nyq, min(20 / nyq, 0.99)], btype='band')
    f = ss.filtfilt(b, a, sig)
    # Square (Pan-Tompkins style) so R-peaks are detected regardless of polarity.
    sq = f ** 2
    pk, _ = ss.find_peaks(sq, distance=int(0.33 * fs), height=np.percentile(sq, 75))
    if len(pk) < 3:
        return np.array([]), fs
    return t[pk], fs


def clare_windows(sig, t, win=120, step=60):
    """Yield (center_time, rr_array) for each window."""
    peak_t, fs = ecg_to_rr(sig, t)
    if len(peak_t) < 3:
        return
    start = t[0]
    while start + win <= t[-1]:
        m = (peak_t >= start) & (peak_t < start + win)
        pt = peak_t[m]
        if len(pt) >= 21:
            rr = np.diff(pt) * 1000.0
            yield (start + win / 2, rr)
        start += step


def build_clare():
    ecg_root = os.path.join(DATA, 'clare', 'ECG')
    lab_root = os.path.join(DATA, 'clare', 'Labels')
    rows = []
    for sub in sorted(os.listdir(ecg_root)):
        sdir = os.path.join(ecg_root, sub)
        if not os.path.isdir(sdir):
            continue
        try:
            labels = pd.read_csv(os.path.join(lab_root, f'{sub}.csv'))
        except Exception:
            labels = None

        # baseline sessions -> rest (low, 0)
        for n in range(4):
            bf = os.path.join(sdir, f'ecg_data_baseline_{n}.csv')
            if not os.path.exists(bf):
                continue
            d = pd.read_csv(bf)
            col = next((c for c in d.columns if 'CAL' in c and 'ECG' in c), None)
            if col is None:
                continue
            for _, rr in clare_windows(d[col].values, d['Timestamp'].values):
                fe = hrv_from_rr(rr)
                if fe:
                    rows.append({'dataset': 'clare', 'subject_id': f'CLARE_{sub}', 'label': 0, '_load': np.nan, **fe})

        # experiment sessions -> task, load from labels[level_n] (every 10s)
        exp_rows = []
        for n in range(4):
            ef = os.path.join(sdir, f'ecg_data_experiment_{n}.csv')
            if not os.path.exists(ef):
                continue
            d = pd.read_csv(ef)
            col = next((c for c in d.columns if 'CAL' in c and 'ECG' in c), None)
            if col is None:
                continue
            t0 = d['Timestamp'].values[0]
            load_seq = labels[f'level_{n}'].dropna().values if labels is not None and f'level_{n}' in labels else None
            for ctr, rr in clare_windows(d[col].values, d['Timestamp'].values):
                fe = hrv_from_rr(rr)
                if not fe:
                    continue
                load = np.nan
                if load_seq is not None and len(load_seq):
                    idx = int((ctr - t0) / 10)
                    if 0 <= idx < len(load_seq):
                        load = float(load_seq[idx])
                exp_rows.append({'dataset': 'clare', 'subject_id': f'CLARE_{sub}', '_load': load, **fe})

        # per-subject median split of reported load -> high(1)/low(0)
        loads = [r['_load'] for r in exp_rows if not np.isnan(r['_load'])]
        thr = np.median(loads) if loads else 5.0
        for r in exp_rows:
            r['label'] = 1 if (not np.isnan(r['_load']) and r['_load'] > thr) else 0
            rows.append(r)

    df = pd.DataFrame(rows)
    return df.drop(columns=['_load'], errors='ignore')


def main():
    parts = []
    for name, fn in [('cogwear', build_cogwear), ('maus', build_maus)]:  # CLARE removed (chest ECG, not a watch signal; labels d~0)
        df = fn()
        if len(df):
            n0 = int((df['label'] == 0).sum()); n1 = int((df['label'] == 1).sum())
            print(f"{name:8s}: {len(df):4d} windows | low={n0} high={n1} | subjects={df['subject_id'].nunique()}")
            parts.append(df)
    pooled = pd.concat(parts, ignore_index=True)
    cols = ['dataset', 'subject_id', 'label'] + FEATURES
    pooled = pooled[[c for c in cols if c in pooled.columns]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = os.path.join(DATA, 'pooled_features.csv')
    pooled.to_csv(out, index=False)
    print(f"\nPOOLED: {len(pooled)} windows | {pooled['subject_id'].nunique()} subjects across {pooled['dataset'].nunique()} datasets")
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
