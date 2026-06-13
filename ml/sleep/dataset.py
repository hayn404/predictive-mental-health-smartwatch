"""
Seren — Sleep Stage Classification: Dataset Loader
====================================================
Loads DREAMT dataset, preprocesses signals, and creates
PyTorch datasets with 30-second windowed epochs.

Expected data layout:
  data/dreamt/
    S01/
      BVP.csv, ACC.csv, HR.csv, TEMP.csv, labels.csv
    S02/
      ...
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import torch

import config as cfg


# ── Signal Loading ─────────────────────────────────────────────

def load_subject_signal(subject_dir: Path, signal_name: str) -> Optional[np.ndarray]:
    """Load a single signal CSV for one subject."""
    filepath = subject_dir / f"{signal_name}.csv"
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath)
    return df.values


def load_subject_labels(subject_dir: Path) -> Optional[np.ndarray]:
    """Load sleep stage labels (one per 30-sec epoch)."""
    filepath = subject_dir / "labels.csv"
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath)
    if "stage" in df.columns:
        labels = df["stage"].values.astype(np.int64)
    elif "label" in df.columns:
        labels = df["label"].values.astype(np.int64)
    else:
        labels = df.iloc[:, -1].values.astype(np.int64)
    return labels


# ── Preprocessing ──────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, low: float, high: float,
                    fs: int, order: int = 4) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal, axis=0)


def compute_acc_magnitude(acc_data: np.ndarray) -> np.ndarray:
    """Compute acceleration magnitude from 3-axis data."""
    if acc_data.ndim == 1:
        return np.abs(acc_data)
    if acc_data.shape[1] >= 3:
        return np.sqrt(acc_data[:, 0]**2 + acc_data[:, 1]**2 + acc_data[:, 2]**2)
    return np.abs(acc_data[:, 0])


def z_normalize(signal: np.ndarray) -> np.ndarray:
    """Per-subject z-normalization."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        return signal - mean
    return (signal - mean) / std


def is_epoch_valid(bvp_epoch: np.ndarray) -> bool:
    """Reject epochs with flat/clipped BVP signal."""
    std = np.std(bvp_epoch)
    if std < cfg.BVP_CLIP_THRESHOLD:
        return False
    flat_samples = np.sum(np.abs(np.diff(bvp_epoch)) < 1e-6)
    flat_ratio = flat_samples / len(bvp_epoch)
    return flat_ratio < cfg.MAX_FLAT_RATIO


# ── Epoch Extraction ───────────────────────────────────────────

def extract_auxiliary_features(hr_epoch: np.ndarray, acc_mag_epoch: np.ndarray,
                                temp_epoch: np.ndarray) -> np.ndarray:
    """Compute 6 auxiliary features for one epoch."""
    hr_mean = np.mean(hr_epoch) if len(hr_epoch) > 0 else 0.0
    hr_std = np.std(hr_epoch) if len(hr_epoch) > 1 else 0.0
    acc_mean = np.mean(acc_mag_epoch) if len(acc_mag_epoch) > 0 else 0.0
    acc_std = np.std(acc_mag_epoch) if len(acc_mag_epoch) > 1 else 0.0
    temp_mean = np.mean(temp_epoch) if len(temp_epoch) > 0 else 0.0

    if len(temp_epoch) > 1:
        t = np.arange(len(temp_epoch), dtype=np.float64)
        temp_slope = np.polyfit(t, temp_epoch, 1)[0]
    else:
        temp_slope = 0.0

    return np.array([hr_mean, hr_std, acc_mean, acc_std, temp_mean, temp_slope],
                    dtype=np.float32)


def process_subject(subject_dir: Path, overlap: float = 0.0):
    """
    Process one subject: load signals, preprocess, window into epochs.

    Returns:
        raw_epochs: np.ndarray [N, EPOCH_SAMPLES, 2] (BVP + ACC_mag)
        aux_features: np.ndarray [N, 6]
        labels: np.ndarray [N]
    """
    # Load signals
    bvp_raw = load_subject_signal(subject_dir, "BVP")
    acc_raw = load_subject_signal(subject_dir, "ACC")
    hr_raw = load_subject_signal(subject_dir, "HR")
    temp_raw = load_subject_signal(subject_dir, "TEMP")
    labels = load_subject_labels(subject_dir)

    if bvp_raw is None or acc_raw is None or labels is None:
        return None, None, None

    # Flatten if single column
    if bvp_raw.ndim == 2 and bvp_raw.shape[1] == 1:
        bvp_raw = bvp_raw.flatten()
    elif bvp_raw.ndim == 2:
        bvp_raw = bvp_raw[:, 0]

    # Compute ACC magnitude
    acc_mag = compute_acc_magnitude(acc_raw)

    # Bandpass filter BVP
    try:
        bvp_filtered = bandpass_filter(bvp_raw, cfg.BVP_LOW_HZ, cfg.BVP_HIGH_HZ, cfg.SAMPLE_RATE)
    except Exception:
        bvp_filtered = bvp_raw

    # Z-normalize per subject
    bvp_norm = z_normalize(bvp_filtered)
    acc_norm = z_normalize(acc_mag)

    # Prepare HR and TEMP (lower rate — resample to match epochs)
    if hr_raw is not None:
        hr_flat = hr_raw.flatten()
    else:
        hr_flat = np.zeros(len(labels) * cfg.EPOCH_SEC)

    if temp_raw is not None:
        temp_flat = temp_raw.flatten()
    else:
        temp_flat = np.zeros(len(labels) * cfg.EPOCH_SEC)

    # Window into epochs
    stride = int(cfg.EPOCH_SAMPLES * (1 - overlap))
    num_epochs = min(len(labels), (len(bvp_norm) - cfg.EPOCH_SAMPLES) // stride + 1)

    raw_epochs = []
    aux_features = []
    valid_labels = []

    for i in range(num_epochs):
        start = i * stride
        end = start + cfg.EPOCH_SAMPLES

        if end > len(bvp_norm) or end > len(acc_norm):
            break

        bvp_epoch = bvp_norm[start:end]
        acc_epoch = acc_norm[start:end]

        if not is_epoch_valid(bvp_epoch):
            continue

        # Stack raw channels: [EPOCH_SAMPLES, 2]
        raw_epoch = np.stack([bvp_epoch, acc_epoch], axis=-1).astype(np.float32)
        raw_epochs.append(raw_epoch)

        # Auxiliary features from low-rate signals
        hr_start = i * cfg.EPOCH_SEC
        hr_end = hr_start + cfg.EPOCH_SEC
        hr_epoch = hr_flat[hr_start:min(hr_end, len(hr_flat))]

        temp_start = i * cfg.EPOCH_SEC
        temp_end = temp_start + cfg.EPOCH_SEC
        temp_epoch = temp_flat[temp_start:min(temp_end, len(temp_flat))]

        aux = extract_auxiliary_features(hr_epoch, acc_epoch, temp_epoch)
        aux_features.append(aux)
        valid_labels.append(labels[i])

    if len(raw_epochs) == 0:
        return None, None, None

    return (np.array(raw_epochs, dtype=np.float32),
            np.array(aux_features, dtype=np.float32),
            np.array(valid_labels, dtype=np.int64))


# ── PyTorch Dataset ────────────────────────────────────────────

class SleepStageDataset(Dataset):
    """PyTorch dataset for sleep stage classification."""

    def __init__(self, raw_epochs: np.ndarray, aux_features: np.ndarray,
                 labels: np.ndarray):
        self.raw_epochs = torch.from_numpy(raw_epochs)      # [N, 3000, 2]
        self.aux_features = torch.from_numpy(aux_features)  # [N, 6]
        self.labels = torch.from_numpy(labels)              # [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw = self.raw_epochs[idx].permute(1, 0)  # [2, 3000] for Conv1d
        aux = self.aux_features[idx]
        label = self.labels[idx]
        return raw, aux, label


# ── Dataset Builder ────────────────────────────────────────────

def get_subject_dirs() -> list[Path]:
    """Find all subject directories in the data folder."""
    if not cfg.DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {cfg.DATA_DIR}\n"
            f"Download DREAMT dataset and place it at: {cfg.DATA_DIR}"
        )
    dirs = sorted([d for d in cfg.DATA_DIR.iterdir() if d.is_dir()])
    return dirs


def build_datasets(val_ratio: float = cfg.VAL_SPLIT, seed: int = cfg.RANDOM_SEED):
    """
    Load all subjects, split into train/val by subject, return DataLoaders.

    Returns:
        train_loader, val_loader, class_weights
    """
    subject_dirs = get_subject_dirs()
    print(f"Found {len(subject_dirs)} subjects in {cfg.DATA_DIR}")

    # Split subjects into train/val
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(subject_dirs))
    val_count = max(1, int(len(subject_dirs) * val_ratio))
    val_indices = set(indices[:val_count])

    train_raw, train_aux, train_labels = [], [], []
    val_raw, val_aux, val_labels = [], [], []

    for i, sdir in enumerate(subject_dirs):
        print(f"  Processing {sdir.name}...", end=" ")
        overlap = cfg.TRAIN_OVERLAP if i not in val_indices else 0.0
        raw, aux, labels = process_subject(sdir, overlap=overlap)

        if raw is None:
            print("skipped (no valid data)")
            continue

        print(f"{len(labels)} epochs")

        if i in val_indices:
            val_raw.append(raw)
            val_aux.append(aux)
            val_labels.append(labels)
        else:
            train_raw.append(raw)
            train_aux.append(aux)
            train_labels.append(labels)

    # Concatenate
    train_raw = np.concatenate(train_raw, axis=0)
    train_aux = np.concatenate(train_aux, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    val_raw = np.concatenate(val_raw, axis=0)
    val_aux = np.concatenate(val_aux, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    print(f"\nTrain: {len(train_labels)} epochs | Val: {len(val_labels)} epochs")
    print(f"Class distribution (train): {np.bincount(train_labels, minlength=cfg.NUM_CLASSES)}")
    print(f"Class distribution (val):   {np.bincount(val_labels, minlength=cfg.NUM_CLASSES)}")

    # Compute class weights (inverse frequency)
    class_counts = np.bincount(train_labels, minlength=cfg.NUM_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * cfg.NUM_CLASSES
    class_weights = torch.from_numpy(class_weights)

    # Create datasets and loaders
    train_dataset = SleepStageDataset(train_raw, train_aux, train_labels)
    val_dataset = SleepStageDataset(val_raw, val_aux, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, class_weights
