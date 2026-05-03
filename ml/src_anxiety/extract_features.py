import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import io as sio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import extract_features, FEATURE_ORDER

DATA_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
WESAD_DIR      = os.path.join(DATA_DIR, 'wesad_wrist')
ANXIETY_2022_DIR = os.path.join(DATA_DIR, 'Anxiety Dataset 2022')
LABEL_DIR      = os.path.join(DATA_DIR, 'label_metadata')
OUTPUT_FILE    = os.path.join(DATA_DIR, 'anxiety_training_data.csv')

WINDOW_SEC   = 60    # 1-minute windows (was 5 min — too coarse, yielded ~1.8 windows/condition)
OVERLAP_FRAC = 0.5   # 50% overlap → 30 s step

# Per-subject label tables
_stai = pd.read_csv(os.path.join(LABEL_DIR, 'wesad_stai_s_scores.csv'),  index_col='subject_id')
_bai  = pd.read_csv(os.path.join(LABEL_DIR, 'anxiety2022_bai_scores.csv'), index_col='subject_id')


def stai_to_index(stai_s: float) -> float:
    """STAI-S (20–80) → anxiety index (0–100)."""
    return round(((stai_s - 20) / 60) * 100, 2)


def bai_to_index(bai: float) -> float:
    """BAI (0–63) → anxiety index (0–100)."""
    return round((bai / 63) * 100, 2)


def process_wesad():
    print("Processing WESAD Dataset...")
    rows = []
    if not os.path.exists(WESAD_DIR):
        print(f"Warning: {WESAD_DIR} not found.")
        return rows

    bvp_fs, temp_fs, acc_fs, label_fs = 64, 4, 32, 700
    win_bvp  = WINDOW_SEC * bvp_fs
    step_bvp = int(win_bvp * (1 - OVERLAP_FRAC))

    # WESAD label int → STAI-S column name
    stai_col = {1: 'baseline_stai_s', 2: 'stress_stai_s', 3: 'amusement_stai_s'}

    for pkl_file in sorted(os.listdir(WESAD_DIR)):
        if not pkl_file.endswith('.pkl'):
            continue

        subject_str = pkl_file.split('.')[0]   # e.g. "S11"
        subject_id  = f"WESAD_{subject_str}"

        if subject_str not in _stai.index:
            print(f"  {subject_id}: no STAI-S scores, skipping")
            continue

        stai_row  = _stai.loc[subject_str]
        label_map = {k: stai_to_index(stai_row[col]) for k, col in stai_col.items()}

        print(f"  {subject_id}  (baseline={label_map[1]}, stress={label_map[2]}, amusement={label_map[3]})")

        with open(os.path.join(WESAD_DIR, pkl_file), 'rb') as f:
            data = pickle.load(f)

        bvp    = data['signal']['wrist']['BVP'].flatten()
        temp   = data['signal']['wrist']['TEMP'].flatten()
        acc    = data['signal']['wrist']['ACC']
        labels = data['label'].flatten()

        n = 0
        for i in range(0, len(bvp) - win_bvp, step_bvp):
            t_start = i / bvp_fs

            bvp_win  = bvp[i: i + win_bvp]
            temp_win = temp[int(t_start * temp_fs): int(t_start * temp_fs) + WINDOW_SEC * temp_fs]
            acc_win  = acc[int(t_start * acc_fs):  int(t_start * acc_fs)  + WINDOW_SEC * acc_fs]

            label_start = int(t_start * label_fs)
            label_win   = labels[label_start: label_start + WINDOW_SEC * label_fs]

            if len(label_win) == 0:
                continue
            valid = label_win[label_win > 0]
            if len(valid) < len(label_win) * 0.8:
                continue

            values, counts = np.unique(valid, return_counts=True)
            majority = int(values[np.argmax(counts)])

            if majority not in label_map:
                continue  # skip meditation / transient

            window_dict = {'bvp': bvp_win, 'temp': temp_win, 'acc': acc_win}
            try:
                features = extract_features(window_dict)
                features['subject_id']     = subject_id
                features['dataset_source'] = 'WESAD'
                features['anxiety_index']  = label_map[majority]
                rows.append(features)
                n += 1
            except Exception:
                pass

        print(f"    -> {n} windows")

    return rows


def process_anxiety_2022():
    print("Processing Anxiety 2022 Dataset...")
    rows = []
    if not os.path.exists(ANXIETY_2022_DIR):
        print(f"Warning: {ANXIETY_2022_DIR} not found.")
        return rows

    from scipy import signal as sp_signal

    ecg_fs      = 500
    win_ecg_64  = WINDOW_SEC * 64
    step_ecg_64 = int(win_ecg_64 * (1 - OVERLAP_FRAC))

    for mat_file in sorted(os.listdir(ANXIETY_2022_DIR)):
        if not mat_file.endswith('.mat'):
            continue

        subject_str   = mat_file.split('_')[-1].replace('.mat', '')  # e.g. "A101"
        subject_id    = f"ANX_{subject_str}"

        if subject_str not in _bai.index:
            print(f"  {subject_id}: no BAI score, skipping")
            continue

        bai_val       = float(_bai.loc[subject_str, 'bai'])
        anxiety_index = bai_to_index(bai_val)
        print(f"  {subject_id}  (BAI={int(bai_val)} → index={anxiety_index})")

        mat_data = sio.loadmat(os.path.join(ANXIETY_2022_DIR, mat_file))
        ecg      = mat_data['data'][:, 0]
        ecg_64   = sp_signal.resample(ecg, int(len(ecg) * 64 / ecg_fs))

        n = 0
        for i in range(0, len(ecg_64) - win_ecg_64, step_ecg_64):
            ecg_win  = ecg_64[i: i + win_ecg_64]
            temp_win = np.full(WINDOW_SEC * 4, 32.0)
            acc_win  = np.zeros((WINDOW_SEC * 32, 3))

            window_dict = {'bvp': ecg_win, 'temp': temp_win, 'acc': acc_win}
            try:
                features = extract_features(window_dict)
                features['subject_id']     = subject_id
                features['dataset_source'] = 'ANXIETY_2022'
                features['anxiety_index']  = anxiety_index
                rows.append(features)
                n += 1
            except Exception:
                pass

        print(f"    -> {n} windows")

    return rows


if __name__ == "__main__":
    all_rows = []

    wesad_rows = process_wesad()
    all_rows.extend(wesad_rows)
    print(f"-> WESAD generated {len(wesad_rows)} windows")

    anx_rows = process_anxiety_2022()
    all_rows.extend(anx_rows)
    print(f"-> Anxiety 2022 generated {len(anx_rows)} windows")

    df = pd.DataFrame(all_rows)
    cols = ['subject_id', 'dataset_source', 'anxiety_index'] + FEATURE_ORDER
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")
