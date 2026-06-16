"""
Biological-age adapters → ml/bioage/data/bioage_features_{fantasia,aa}.csv
Each row: subject_id, age, [14 HRV features]   (windowed; age = subject's true/group age)

- Fantasia: read .ecg beat annotations → RR → HRV; exact age from .hea header.
- Autonomic Aging: read ECG signal (wfdb) → R-peaks → RR → HRV; age (group midpoint)
  from subject-info.csv.
HRV features reuse the shared extractor (ml/stress/src/features.py).
"""
import os, sys, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import wfdb
from scipy import signal as ss
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stress'))
from src.features import compute_time_domain, compute_nonlinear  # noqa

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(BASE, 'data')
FEAT = ['meanRR','sdnn','rmssd','pnn50','pnn20','hrMean','cvRR','sd1','sd2','sd1sd2Ratio','sampleEntropy','dfaAlpha1']
WIN, STEP, MINRR = 120.0, 60.0, 30


def hrv_from_rr(rr_ms):
    rr = np.asarray(rr_ms, float); rr = rr[(rr >= 300) & (rr <= 2000)]
    if len(rr) > 5:
        m = np.median(rr); rr = rr[np.abs(rr - m) / m < 0.30]
    if len(rr) < MINRR:
        return None
    d = {}; d.update(compute_time_domain(rr)); d.update(compute_nonlinear(rr))
    return {k: float(d.get(k, 0.0)) for k in FEAT}


def windows_from_beats(beat_times_s):
    """beat_times in seconds → yield RR(ms) arrays per WIN-second window."""
    if len(beat_times_s) < MINRR:
        return
    t0, tend = beat_times_s[0], beat_times_s[-1]
    s = t0
    while s + WIN <= tend + STEP:
        m = (beat_times_s >= s) & (beat_times_s < s + WIN)
        bt = beat_times_s[m]
        if len(bt) >= MINRR:
            yield np.diff(bt) * 1000.0
        s += STEP


def ecg_rpeaks(sig, fs):
    sig = pd.Series(np.asarray(sig, float)).interpolate(limit_direction='both').values
    if np.isnan(sig).any():
        return np.array([])
    # Downsample high-rate ECG (e.g. 1000 Hz) to ~250 Hz by slicing — instant; the
    # band-pass below provides the QRS filtering, so anti-alias decimation isn't needed.
    if fs >= 500:
        factor = int(fs // 250)
        sig = sig[::factor]
        fs = fs / factor
    nyq = fs / 2
    b, a = ss.butter(3, [5 / nyq, min(20 / nyq, 0.99)], btype='band')
    f = ss.filtfilt(b, a, sig); sq = f ** 2
    pk, _ = ss.find_peaks(sq, distance=int(0.33 * fs), height=np.percentile(sq, 90))
    return pk / fs  # seconds


# ---------- Fantasia ----------
def build_fantasia():
    root = os.path.join(DATA, 'fantasia')
    rows = []
    for hea in sorted(glob.glob(os.path.join(root, '*.hea'))):
        rec = hea[:-4]; name = os.path.basename(rec)
        txt = open(hea).read()
        age_line = [l for l in txt.splitlines() if 'Age' in l]
        if not age_line:
            continue
        age = int(age_line[0].split('Age:')[1].split()[0])
        try:
            ann = wfdb.rdann(rec, 'ecg')
            fs = wfdb.rdheader(rec).fs
            beats = ann.sample / fs
        except Exception:
            continue
        for rr in windows_from_beats(beats):
            fe = hrv_from_rr(rr)
            if fe:
                rows.append({'subject_id': f'FAN_{name}', 'age': age, **fe})
    df = pd.DataFrame(rows)
    out = os.path.join(DATA, 'bioage_features_fantasia.csv'); df.to_csv(out, index=False)
    print(f"Fantasia: {len(df)} windows | {df.subject_id.nunique()} subjects | age {df.age.min()}-{df.age.max()}")
    print(f"saved -> {out}")


# ---------- Autonomic Aging ----------
# 15 age groups → midpoint age (groups: 18-19, 20-24, 25-29, ... 80-84, 85-92)
AGE_MID = {1: 18.5, 2: 22, 3: 27, 4: 32, 5: 37, 6: 42, 7: 47, 8: 52,
           9: 57, 10: 62, 11: 67, 12: 72, 13: 77, 14: 82, 15: 88.5}


def build_aa():
    root = os.path.join(DATA, 'autonomic_aging')
    info = pd.read_csv(os.path.join(root, 'subject-info.csv'))
    age_by_id = {int(r.ID): AGE_MID.get(int(r.Age_group)) for _, r in info.dropna(subset=['Age_group']).iterrows()}
    rows = []
    heas = sorted(glob.glob(os.path.join(root, '[0-9]*.hea')))
    done = 0
    for hea in heas:
        rec = hea[:-4]; rid = int(os.path.basename(rec))
        if rid not in age_by_id or not os.path.exists(rec + '.dat'):
            continue
        age = age_by_id[rid]
        try:
            r = wfdb.rdrecord(rec)
            sig = r.p_signal[:, 0]; fs = r.fs
        except Exception:
            continue
        beats = ecg_rpeaks(sig, fs)
        for rr in windows_from_beats(beats):
            fe = hrv_from_rr(rr)
            if fe:
                rows.append({'subject_id': f'AA_{rid}', 'age': age, **fe})
        done += 1
    df = pd.DataFrame(rows)
    out = os.path.join(DATA, 'bioage_features_aa.csv'); df.to_csv(out, index=False)
    print(f"AA: {len(df)} windows | {df.subject_id.nunique()} subjects processed ({done}) | age {df.age.min():.0f}-{df.age.max():.0f}")
    print(f"saved -> {out}")


if __name__ == '__main__':
    import sys
    {'fantasia': build_fantasia, 'aa': build_aa}.get(sys.argv[1] if len(sys.argv) > 1 else 'fantasia', build_fantasia)()
