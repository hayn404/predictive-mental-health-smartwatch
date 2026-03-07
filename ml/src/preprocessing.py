"""
Seren ML Pipeline — Step 1: Data Preprocessing
================================================
Loads WESAD dataset (pickle files from Empatica E4 wrist sensor)
and SWELL-KW HRV dataset, normalizes them into a common format.

WESAD structure per subject (S2.pkl, S3.pkl, ...):
  {
    'signal': {
      'wrist': {
        'BVP': ndarray (N, 1) @ 64 Hz  — Blood Volume Pulse
        'EDA': ndarray (N, 1) @ 4 Hz   — Electrodermal Activity
        'TEMP': ndarray (N, 1) @ 4 Hz  — Skin Temperature
        'ACC': ndarray (N, 3) @ 32 Hz  — Accelerometer x,y,z
      },
      'chest': { ... }  — RespiBAN data (not used for smartwatch)
    },
    'label': ndarray (N,) @ 700 Hz — Labels:
        0 = not defined / transient
        1 = baseline (neutral)
        2 = stress (TSST)
        3 = amusement
        4 = meditation
  }

We only use WRIST data since it matches smartwatch capabilities.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Sampling frequencies for Empatica E4 sensors
WESAD_SAMPLING_RATES = {
    'BVP': 64,    # Blood Volume Pulse (Hz)
    'EDA': 4,     # Electrodermal Activity (Hz)
    'TEMP': 4,    # Skin Temperature (Hz)
    'ACC': 32,    # Accelerometer (Hz)
    'label': 700, # Label signal (Hz)
}

# WESAD label mapping
WESAD_LABELS = {
    0: 'transient',
    1: 'baseline',
    2: 'stress',
    3: 'amusement',
    4: 'meditation',
}

# Binary mapping for stress detection
STRESS_BINARY = {
    1: 0,  # baseline -> not stressed
    2: 1,  # stress -> stressed
    3: 0,  # amusement -> not stressed
    4: 0,  # meditation -> not stressed
}


def load_wesad_subject(filepath: str) -> Dict:
    """
    Load a single WESAD subject pickle file.

    Args:
        filepath: Path to Sx.pkl file

    Returns:
        Dict with 'wrist' sensor data and 'labels'
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    wrist = data['signal']['wrist']
    labels = data['label']

    return {
        'bvp': wrist['BVP'].flatten(),      # (N,) @ 64 Hz
        'eda': wrist['EDA'].flatten(),       # (N,) @ 4 Hz
        'temp': wrist['TEMP'].flatten(),     # (N,) @ 4 Hz
        'acc': wrist['ACC'],                 # (N, 3) @ 32 Hz
        'labels': labels.flatten(),          # (N,) @ 700 Hz
    }


def resample_labels_to_signal(labels: np.ndarray, label_rate: int,
                               target_rate: int, target_length: int) -> np.ndarray:
    """
    Resample labels from 700 Hz to match a target signal's rate and length.
    Uses nearest-neighbor (mode) downsampling per target sample.
    """
    ratio = label_rate / target_rate
    resampled = np.zeros(target_length, dtype=int)

    for i in range(target_length):
        start = int(i * ratio)
        end = int((i + 1) * ratio)
        end = min(end, len(labels))
        if start < len(labels):
            segment = labels[start:end]
            if len(segment) > 0:
                values, counts = np.unique(segment, return_counts=True)
                resampled[i] = values[np.argmax(counts)]

    return resampled


def segment_into_windows(bvp: np.ndarray, eda: np.ndarray, temp: np.ndarray,
                          acc: np.ndarray, labels: np.ndarray,
                          window_sec: int = 300,
                          overlap: float = 0.0) -> List[Dict]:
    """
    Segment all signals into fixed-length windows.

    Args:
        bvp: BVP signal @ 64 Hz
        eda: EDA signal @ 4 Hz
        temp: Temperature @ 4 Hz
        acc: Accelerometer (N, 3) @ 32 Hz
        labels: Labels resampled to 64 Hz (aligned with BVP)
        window_sec: Window duration in seconds (default 300 = 5 min)
        overlap: Overlap fraction (0-1, default 0 = no overlap)

    Returns:
        List of window dicts with signal segments and majority label
    """
    bvp_window = window_sec * WESAD_SAMPLING_RATES['BVP']
    eda_window = window_sec * WESAD_SAMPLING_RATES['EDA']
    temp_window = window_sec * WESAD_SAMPLING_RATES['TEMP']
    acc_window = window_sec * WESAD_SAMPLING_RATES['ACC']
    label_window = bvp_window  # Labels resampled to BVP rate

    step = int(bvp_window * (1 - overlap))
    windows = []

    n_windows = (len(bvp) - bvp_window) // step + 1

    for i in range(n_windows):
        bvp_start = i * step
        bvp_end = bvp_start + bvp_window

        # Scale start/end for other sampling rates
        eda_start = int(bvp_start * WESAD_SAMPLING_RATES['EDA'] / WESAD_SAMPLING_RATES['BVP'])
        eda_end = eda_start + eda_window
        temp_start = int(bvp_start * WESAD_SAMPLING_RATES['TEMP'] / WESAD_SAMPLING_RATES['BVP'])
        temp_end = temp_start + temp_window
        acc_start = int(bvp_start * WESAD_SAMPLING_RATES['ACC'] / WESAD_SAMPLING_RATES['BVP'])
        acc_end = acc_start + acc_window

        # Check bounds
        if bvp_end > len(bvp) or eda_end > len(eda) or temp_end > len(temp) or acc_end > len(acc):
            break

        # Get majority label for this window (exclude label 0 = transient)
        window_labels = labels[bvp_start:bvp_end]
        valid_labels = window_labels[window_labels > 0]
        if len(valid_labels) == 0:
            continue
        values, counts = np.unique(valid_labels, return_counts=True)
        majority_label = values[np.argmax(counts)]

        # Skip if the majority label doesn't cover at least 80% of the window
        if np.max(counts) / len(valid_labels) < 0.8:
            continue

        windows.append({
            'bvp': bvp[bvp_start:bvp_end],
            'eda': eda[eda_start:eda_end],
            'temp': temp[temp_start:temp_end],
            'acc': acc[acc_start:acc_end],
            'label': int(majority_label),
            'label_name': WESAD_LABELS.get(int(majority_label), 'unknown'),
            'stress_binary': STRESS_BINARY.get(int(majority_label), -1),
        })

    return windows


def load_all_wesad_subjects(data_dir: str, window_sec: int = 300) -> List[Dict]:
    """
    Load all WESAD subjects from directory, segment into windows.

    Args:
        data_dir: Path to WESAD data directory containing S2/, S3/, etc.
        window_sec: Window size in seconds

    Returns:
        List of all windowed segments across all subjects
    """
    data_path = Path(data_dir)
    all_windows = []

    # WESAD has subjects S2 through S17 (S12 excluded in original study)
    subject_dirs = sorted(data_path.glob('S*'))

    for subject_dir in subject_dirs:
        pkl_file = subject_dir / f'{subject_dir.name}.pkl'
        if not pkl_file.exists():
            print(f'  Skipping {subject_dir.name}: no pickle file found')
            continue

        print(f'  Loading {subject_dir.name}...')
        subject_data = load_wesad_subject(str(pkl_file))

        # Resample labels to BVP rate (64 Hz)
        labels_at_bvp_rate = resample_labels_to_signal(
            subject_data['labels'],
            WESAD_SAMPLING_RATES['label'],
            WESAD_SAMPLING_RATES['BVP'],
            len(subject_data['bvp'])
        )

        windows = segment_into_windows(
            bvp=subject_data['bvp'],
            eda=subject_data['eda'],
            temp=subject_data['temp'],
            acc=subject_data['acc'],
            labels=labels_at_bvp_rate,
            window_sec=window_sec,
        )

        # Tag with subject ID
        for w in windows:
            w['subject'] = subject_dir.name

        all_windows.extend(windows)
        print(f'    -> {len(windows)} windows ({sum(1 for w in windows if w["stress_binary"] == 1)} stress)')

    print(f'\nTotal: {len(all_windows)} windows from {len(subject_dirs)} subjects')
    return all_windows


def load_swell_hrv(filepath: str) -> pd.DataFrame:
    """
    Load the SWELL-KW HRV dataset from Kaggle (CSV format).
    This dataset already has pre-extracted HRV features.

    Expected columns include:
    - meanRR, SDNN, RMSSD, pNN50, LF, HF, LF/HF, etc.
    - condition: 0=no stress, 1=time pressure, 2=interruption

    Args:
        filepath: Path to the SWELL HRV CSV file

    Returns:
        DataFrame with features and labels
    """
    df = pd.read_csv(filepath)

    # Map condition to binary stress label
    if 'condition' in df.columns:
        df['stress_binary'] = (df['condition'] > 0).astype(int)
    elif 'Condition' in df.columns:
        df['stress_binary'] = (df['Condition'] > 0).astype(int)

    return df


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python preprocessing.py <wesad_data_dir>')
        print('  e.g.: python preprocessing.py ../data/wesad/')
        sys.exit(1)

    data_dir = sys.argv[1]
    print(f'Loading WESAD data from: {data_dir}')
    windows = load_all_wesad_subjects(data_dir, window_sec=300)

    # Save as pickle for next step
    output_path = Path(__file__).parent.parent / 'data' / 'wesad_windows.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(windows, f)
    print(f'Saved {len(windows)} windows to {output_path}')
