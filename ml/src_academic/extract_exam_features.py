import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import extract_features, FEATURE_ORDER

EXAM_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../data/a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0/exam_stress_pkl'
))
OUTPUT_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../data/academic_training_data.csv'
))

WINDOW_SEC   = 60
OVERLAP_FRAC = 0.5

# Grades from StudentGrades.txt — Final normalized from /200 to /100
GRADES = {
    'S1':  {'Midterm 1': 78,  'Midterm 2': 82,  'Final': 182 / 2},
    'S2':  {'Midterm 1': 82,  'Midterm 2': 85,  'Final': 180 / 2},
    'S3':  {'Midterm 1': 77,  'Midterm 2': 90,  'Final': 188 / 2},
    'S4':  {'Midterm 1': 75,  'Midterm 2': 77,  'Final': 149 / 2},
    'S5':  {'Midterm 1': 67,  'Midterm 2': 77,  'Final': 157 / 2},
    'S6':  {'Midterm 1': 71,  'Midterm 2': 64,  'Final': 175 / 2},
    'S7':  {'Midterm 1': 64,  'Midterm 2': 33,  'Final': 110 / 2},
    'S8':  {'Midterm 1': 92,  'Midterm 2': 88,  'Final': 184 / 2},
    'S9':  {'Midterm 1': 80,  'Midterm 2': 39,  'Final': 126 / 2},
    'S10': {'Midterm 1': 89,  'Midterm 2': 64,  'Final': 116 / 2},
}

# exam_performance_risk = inverse of grade (how much the student underperformed)
def grade_to_risk(grade: float) -> float:
    return round(100.0 - grade, 2)


def process_all_subjects():
    rows = []

    for pkl_file in sorted(os.listdir(EXAM_DIR)):
        if not pkl_file.endswith('.pkl'):
            continue

        subject_str = pkl_file.replace('.pkl', '')   # e.g. "S10"
        subject_id  = f"EXAM_{subject_str}"

        if subject_str not in GRADES:
            print(f"  {subject_id}: no grade entry, skipping")
            continue

        print(f"\nProcessing {subject_id}...")

        with open(os.path.join(EXAM_DIR, pkl_file), 'rb') as f:
            data = pickle.load(f)

        for session_name in ['Midterm 1', 'Midterm 2', 'Final']:
            if session_name not in data:
                print(f"  {session_name}: not in pkl, skipping")
                continue

            grade = GRADES[subject_str][session_name]
            risk  = grade_to_risk(grade)
            s     = data[session_name]

            bvp  = s['BVP']
            temp = s['TEMP']
            acc  = s['ACC']

            bvp_fs  = int(s['sr_BVP'])
            temp_fs = int(s['sr_TEMP'])
            acc_fs  = int(s['sr_ACC'])

            win_bvp  = WINDOW_SEC * bvp_fs
            step_bvp = int(win_bvp * (1 - OVERLAP_FRAC))

            n = 0
            for i in range(0, len(bvp) - win_bvp, step_bvp):
                t_start = i / bvp_fs

                bvp_win  = bvp[i: i + win_bvp]
                temp_win = temp[int(t_start * temp_fs): int(t_start * temp_fs) + WINDOW_SEC * temp_fs]
                acc_win  = acc[int(t_start * acc_fs):  int(t_start * acc_fs)  + WINDOW_SEC * acc_fs]

                if len(temp_win) < WINDOW_SEC * temp_fs * 0.8:
                    continue
                if len(acc_win) < WINDOW_SEC * acc_fs * 0.8:
                    continue

                window_dict = {'bvp': bvp_win, 'temp': temp_win, 'acc': acc_win}
                try:
                    features = extract_features(window_dict)
                    features['subject_id']           = subject_id
                    features['session_name']          = session_name
                    features['grade']                 = grade
                    features['exam_performance_risk'] = risk
                    rows.append(features)
                    n += 1
                except Exception:
                    pass

            print(f"  {session_name} (grade={grade:.1f}): {n} windows")

    return rows


if __name__ == '__main__':
    all_rows = process_all_subjects()
    df = pd.DataFrame(all_rows)
    cols = ['subject_id', 'session_name', 'grade', 'exam_performance_risk'] + FEATURE_ORDER
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")
