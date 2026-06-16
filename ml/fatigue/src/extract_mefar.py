"""
MEFAR adapter (TRAIN set) → ml/fatigue/data/fatigue_features_mefar.csv

Each subject has morning + evening Empatica E4 recordings. We read IBI.csv (E4 inter-beat
intervals, seconds → ms), window by time, compute the 12 shared HRV features, and label each
session by the Chalder Fatigue Scale (>=12 = fatigued) from general_info.xlsx.
This is a WITHIN-SUBJECT design (morning vs evening), matching per-user normalization.
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stress'))
from src.features import compute_time_domain, compute_nonlinear  # noqa

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOT = os.path.join(BASE, 'data', 'mefar_raw', 'MEFAR')
FEAT = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','cvRR','sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']
WIN, STEP, MINRR, CUTOFF = 120.0, 60.0, 20, 12


def hrv(rr_ms):
    rr = np.asarray(rr_ms, float); rr = rr[(rr >= 300) & (rr <= 2000)]
    if len(rr) > 5:
        m = np.median(rr); rr = rr[np.abs(rr - m) / m < 0.30]
    if len(rr) < MINRR:
        return None
    d = {}; d.update(compute_time_domain(rr)); d.update(compute_nonlinear(rr))
    return {k: float(d.get(k, 0.0)) for k in FEAT}


def labels():
    df = pd.read_excel(os.path.join(ROOT, 'general_info.xlsx'), sheet_name='Subject List')
    mcol = [c for c in df.columns if 'morning' in c.lower() and 'fatigue' in c.lower()][0]
    ecol = [c for c in df.columns if 'evening' in c.lower() and 'fatigue' in c.lower()][0]
    out = {}
    for _, r in df.iterrows():
        sid = str(r['subjects']).strip()
        out[(sid, 'morning')] = int(float(r[mcol]) >= CUTOFF)
        out[(sid, 'evening')] = int(float(r[ecol]) >= CUTOFF)
    return out


def read_ibi(path):
    # row0 = [start_ts, "IBI"]; rows = [seconds_since_start, ibi_seconds]
    d = pd.read_csv(path, header=None, skiprows=1)
    t = d[0].to_numpy(float)
    rr_ms = d[1].to_numpy(float) * 1000.0
    return t, rr_ms


def main():
    lab = labels()
    rows = []
    for sub in sorted(os.listdir(ROOT)):
        sdir = os.path.join(ROOT, sub)
        if not os.path.isdir(sdir):
            continue
        sid = 'S' + sub.split('_')[1]
        for sess, folder in [('morning', '1.morning'), ('evening', '2.evening')]:
            ibi_path = os.path.join(sdir, folder, 'IBI.csv')
            if not os.path.exists(ibi_path) or (sid, sess) not in lab:
                continue
            try:
                t, rr = read_ibi(ibi_path)
            except Exception:
                continue
            if len(t) < MINRR:
                continue
            label = lab[(sid, sess)]
            start, tend = t[0], t[-1]
            s = start
            while s + WIN <= tend + STEP:           # allow a final partial window
                m = (t >= s) & (t < s + WIN)
                fe = hrv(rr[m])
                if fe:
                    rows.append({'subject_id': f'MEFAR_{sid}', 'session': sess, 'label': label, **fe})
                s += STEP
    df = pd.DataFrame(rows)
    out = os.path.join(BASE, 'data', 'fatigue_features_mefar.csv')
    df.to_csv(out, index=False)
    n0, n1 = int((df.label == 0).sum()), int((df.label == 1).sum())
    print(f"MEFAR: {len(df)} windows | fresh={n0} fatigued={n1} | subjects={df.subject_id.nunique()}")
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
