"""
Seren — Sleep Stage Classification: Dataset Loader
====================================================
Loads the PRE-BUILT, PRE-NORMALIZED feature cache produced by
prepare_features.py and consumed verbatim by the Kaggle notebook
(seren_sleep_kaggle.ipynb) v3.2.

Per-night median/MAD normalization is baked into the cache at build time — this
module does NOT normalize. It only:
  1. loads bidsleep_features.pkl (train/val) + walch_features.pkl (held-out test)
  2. slices out the XAI-dead column (immobility_frac)
  3. appends time_of_night as feature #11
  4. builds subject-disjoint train/val split + SeqDataset sequences
  5. computes sqrt-inverse-frequency class weights

Cache pickle format: list of (subject_id, features[N,11], labels[N]) tuples.
BIDSleep subject ids are "<subject>/<recording>" (e.g. "Bidslab00/3").
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg


# ── Feature assembly (mirrors notebook cell 3) ─────────────────

def _slice_dead_features(subjects):
    """Drop XAI-confirmed dead-weight columns (immobility_frac) from cached feats."""
    return [(sid, feats[:, cfg.KEEP_CACHE_IDX].astype(np.float32), labs)
            for sid, feats, labs in subjects]


def _add_time_of_night(subjects):
    """Append time_of_night: linear position within the night scaled to [-1, +1].

    On-device equivalent: 2 * (i / max(1, n_epochs - 1)) - 1 for i in [0, n_epochs).
    """
    out = []
    for sid, feats, labs in subjects:
        n = len(feats)
        denom = max(1, n - 1)
        ton = (2.0 * np.arange(n, dtype=np.float32) / denom - 1.0).reshape(-1, 1)
        out.append((sid, np.concatenate([feats, ton], axis=1).astype(np.float32), labs))
    return out


def load_cache(data_dir):
    """Load + prepare both cache pickles. Returns (bidsleep_nights, walch_subjects)."""
    data_dir = Path(data_dir)
    walch_p = data_dir / "walch_features.pkl"
    bid_p = data_dir / "bidsleep_features.pkl"
    if not walch_p.exists() or not bid_p.exists():
        raise FileNotFoundError(
            f"Missing feature cache. Expected:\n  {bid_p}\n  {walch_p}\n"
            "Run `dvc pull` (or build via prepare_features.py) first."
        )
    with open(walch_p, "rb") as f:
        walch = pickle.load(f)
    with open(bid_p, "rb") as f:
        bid = pickle.load(f)

    walch = _add_time_of_night(_slice_dead_features(walch))
    bid = _add_time_of_night(_slice_dead_features(bid))
    fdim = walch[0][1].shape[-1]
    assert fdim == cfg.NUM_FEATURES, f"feature dim {fdim} != expected {cfg.NUM_FEATURES}"
    print(f"Loaded BIDSleep: {len(bid)} nights | Walch: {len(walch)} subjects "
          f"| feature dim = {fdim}")
    return bid, walch


# ── Sequence dataset (mirrors notebook cell 7) ─────────────────

class SeqDataset(Dataset):
    def __init__(self, subjects, seq_len, stride):
        self.items = []
        for _, feats, labs in subjects:
            n = len(labs)
            if n < seq_len:
                continue
            for s in range(0, n - seq_len + 1, stride):
                self.items.append((feats[s:s + seq_len], labs[s:s + seq_len]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, l = self.items[i]
        return torch.from_numpy(f), torch.from_numpy(l)


def build_datasets(data_dir):
    """Build train/val/test loaders + class weights, exactly like the notebook.

    Returns: (train_loader, val_loader, test_loader, class_weights, info_dict)
    """
    bid, walch = load_cache(data_dir)

    # SUBJECT-LEVEL split (not night-level) — night-level split leaks subjects.
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    subj_of = [sid.split("/")[0] for sid, _, _ in bid]
    all_subjects = sorted(set(subj_of))
    rng.shuffle(all_subjects)
    n_val_subj = max(1, int(len(all_subjects) * cfg.VAL_SPLIT))
    val_subjects = set(all_subjects[:n_val_subj])
    bid_train = [s for s, sub in zip(bid, subj_of) if sub not in val_subjects]
    bid_val = [s for s, sub in zip(bid, subj_of) if sub in val_subjects]
    print(f"Subject-disjoint split: {len(all_subjects) - n_val_subj} train subj / "
          f"{n_val_subj} val subj ({len(bid_train)} train nights / {len(bid_val)} val nights)")

    train_ds = SeqDataset(bid_train, cfg.SEQ_LEN, cfg.TRAIN_STRIDE)
    val_ds = SeqDataset(bid_val, cfg.SEQ_LEN, cfg.SEQ_LEN)         # disjoint at eval
    test_ds = SeqDataset(walch, cfg.SEQ_LEN, cfg.SEQ_LEN)          # disjoint at eval
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test(Walch) {len(test_ds)} sequences")

    # sqrt-inverse-frequency class weights (softer than full inverse)
    tr_lab = np.concatenate([l for _, _, l in bid_train])
    counts = np.bincount(tr_lab, minlength=cfg.NUM_CLASSES).astype(np.float32)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    class_weights = torch.tensor(inv_sqrt / inv_sqrt.sum() * cfg.NUM_CLASSES, dtype=torch.float32)
    print("Train class counts:", dict(zip(cfg.STAGE_NAMES, counts.astype(int).tolist())))
    print("Class weights (sqrt):", [round(w, 3) for w in class_weights.tolist()])

    pin = torch.cuda.is_available()
    nw = 2 if torch.cuda.is_available() else 0   # workers safe on Kaggle Linux/GPU
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=nw, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=nw, pin_memory=pin)

    info = {
        "num_train_nights": len(bid_train),
        "num_val_nights": len(bid_val),
        "num_test_subjects": len(walch),
    }
    return train_loader, val_loader, test_loader, class_weights, info
