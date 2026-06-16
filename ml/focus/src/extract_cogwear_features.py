"""
Extract HRV features from the CogWear dataset for the Seren focus / cognitive-effort
model. Produces one feature row per windowed segment, labelled:
    0 = baseline (rest)        1 = cognitive_load (Stroop task)

We extract from BOTH wearables so we can compare, but the DEPLOYED model uses the
Samsung Galaxy Watch4 PPG ('samsung') because that is the exact signal the app gets
on-device. Reuses the shared HRV math from ml/stress/src/features.py so training
features match what featureEngineering.ts computes live on the watch.

Output: ml/focus/data/cogwear_features_<device>.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal as ss, interpolate

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))   # ml/
sys.path.append(os.path.join(BASE, 'stress'))
from src.features import (  # noqa: E402
    bvp_to_rr_intervals, compute_time_domain, compute_nonlinear, compute_frequency_domain,
)


def rr_enhanced(bvp: np.ndarray, fs: float, target: int = 128) -> np.ndarray:
    """
    Enhanced RR extraction for low-rate consumer PPG (e.g. Galaxy Watch4 ~23 Hz):
    bandpass to the HR band, then cubic-upsample the filtered signal to `target` Hz
    before peak detection so peak timing (hence RR / HRV) isn't quantized to the
    coarse native sample period. Lifts LOSO AUC ~0.744 -> ~0.785 vs naive peak picking.
    """
    nyq = fs / 2
    b, a = ss.butter(3, [0.5 / nyq, min(4.0 / nyq, 0.99)], btype='band')
    f = ss.filtfilt(b, a, bvp)
    n = len(f)
    t = np.arange(n) / fs
    if t[-1] <= 0:
        return np.array([])
    t2 = np.arange(0, t[-1], 1 / target)
    up = interpolate.interp1d(t, f, kind='cubic', fill_value='extrapolate')(t2)
    pk, _ = ss.find_peaks(up, distance=int(0.33 * target), height=np.percentile(up, 55))
    if len(pk) < 3:
        return np.array([])
    rr = np.diff(pk) / target * 1000
    rr = rr[(rr >= 300) & (rr <= 2000)]
    if len(rr) > 5:
        m = np.median(rr)
        rr = rr[np.abs(rr - m) / m < 0.30]
    return rr

DATA_ROOT = os.path.join(
    BASE, 'focus', 'data', 'cogwear',
    'cogwear-can-we-detect-cognitive-effort-with-consumer-grade-wearables-1.0.0',
)

# HRV features the watch can reliably reproduce from Health Connect HR (see featureEngineering.ts).
HRV_TIME = ['meanRR', 'sdnn', 'rmssd', 'pnn50', 'pnn20', 'hrMean', 'hrStd', 'hrRange', 'cvRR']
HRV_NONLIN = ['sd1', 'sd2', 'sd1sd2Ratio', 'sampleEntropy', 'dfaAlpha1']
HRV_FREQ = ['lfPower', 'hfPower', 'lfHfRatio', 'totalPower', 'lfNorm', 'hfNorm']
FEATURE_ORDER = HRV_TIME + HRV_NONLIN + HRV_FREQ

WINDOW_SEC = int(os.environ.get('COG_WINDOW', '120'))  # longer window = stabler HRV
STEP_SEC = int(os.environ.get('COG_STEP', '20'))
MIN_RR = 20            # minimum beats for a usable window
USE_ENHANCED_RR = os.environ.get('COG_RR', 'enhanced') != 'original'

DEVICE_FILES = {
    'samsung': ('samsung_bvp.csv', 'PPG GREEN'),
    'empatica': ('empatica_bvp.csv', 'bvp'),
}


def estimate_fs(t: np.ndarray) -> float:
    dur = t[-1] - t[0]
    return len(t) / dur if dur > 0 else 0.0


def window_features(bvp: np.ndarray, t: np.ndarray, fs: float) -> list:
    """Slide a WINDOW_SEC window across one recording, extract HRV features per window."""
    rows = []
    t0, tend = t[0], t[-1]
    start = t0
    while start + WINDOW_SEC <= tend:
        mask = (t >= start) & (t < start + WINDOW_SEC)
        seg = bvp[mask]
        if len(seg) >= 0.6 * WINDOW_SEC * fs:
            rr = rr_enhanced(seg, fs) if USE_ENHANCED_RR else bvp_to_rr_intervals(seg, fs=int(round(fs)))
            if len(rr) >= MIN_RR:
                feat = {}
                feat.update(compute_time_domain(rr))
                feat.update(compute_nonlinear(rr))
                feat.update(compute_frequency_domain(rr))
                rows.append(feat)
        start += STEP_SEC
    return rows


def collect_condition(folder: str, device: str, subject: str, label: int, study: str) -> list:
    fname, col = DEVICE_FILES[device]
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if col not in df.columns or 'time' not in df.columns or len(df) < 100:
        return []
    bvp = df[col].to_numpy(dtype=float)
    t = df['time'].to_numpy(dtype=float)
    fs = estimate_fs(t)
    if fs < 5:
        return []
    rows = window_features(bvp, t, fs)
    for r in rows:
        r['subject_id'] = f'COG_{subject}'
        r['study'] = study
        r['label'] = label   # 0 baseline, 1 cognitive_load
    return rows


def iter_condition_dirs():
    """Yield (folder, subject, label, study) for every baseline/cognitive_load recording."""
    # pilot: <id>/{baseline,cognitive_load}
    pilot = os.path.join(DATA_ROOT, 'pilot')
    for sid in sorted(os.listdir(pilot)):
        for cond, label in [('baseline', 0), ('cognitive_load', 1)]:
            yield os.path.join(pilot, sid, cond), sid, label, 'pilot'
    # survey_gamification: <id>/{pre,post}/{baseline,cognitive_load}
    survey = os.path.join(DATA_ROOT, 'survey_gamification')
    for sid in sorted(os.listdir(survey)):
        for session in ['pre', 'post']:
            for cond, label in [('baseline', 0), ('cognitive_load', 1)]:
                yield os.path.join(survey, sid, session, cond), sid, label, 'survey'


def build(device: str):
    all_rows = []
    for folder, sid, label, study in iter_condition_dirs():
        all_rows.extend(collect_condition(folder, device, sid, label, study))
    df = pd.DataFrame(all_rows)
    cols = ['subject_id', 'study', 'label'] + [c for c in FEATURE_ORDER if c in df.columns]
    df = df[[c for c in cols if c in df.columns]]
    out = os.path.join(BASE, 'focus', 'data', f'cogwear_features_{device}.csv')
    df.to_csv(out, index=False)
    n0 = int((df['label'] == 0).sum())
    n1 = int((df['label'] == 1).sum())
    print(f'[{device}] {len(df)} windows | rest={n0} effort={n1} | subjects={df["subject_id"].nunique()}')
    print(f'         saved -> {out}')


if __name__ == '__main__':
    for dev in ['samsung', 'empatica']:
        build(dev)
