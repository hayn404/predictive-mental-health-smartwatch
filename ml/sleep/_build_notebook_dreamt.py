"""Builds kaggle_train.ipynb for the 4-class DREAMT sleep stage classifier."""
import json
from pathlib import Path

CELLS = []


def md(text: str):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    })


def code(text: str):
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    })


md("""# Seren — Sleep Stage Classifier v2 (4-class, DREAMT from S3)

**Architecture:** Dilated 1D-CNN per-epoch encoder → BiLSTM sequence model over 21 consecutive epochs (seq2seq).

**Signals (4 channels @ 25 Hz):** BVP (0.5–8 Hz bandpassed), ACC magnitude, EDA, TEMP. Plus 5 aux features per epoch.

**Labels (4-class, matches app):** `W → awake (0)`, `N1+N2 → light (1)`, `N3 → deep (2)`, `R → rem (3)`. `P` skipped.

**Targets:** beat published DREAMT 4-class baselines (XGBoost 0.62 F1 / κ 0.50; U-Net 0.62 F1).

**Kaggle Secrets:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`  |  **Settings:** GPU T4 x2, Internet ON""")

code("!pip install -q onnx onnxruntime scikit-learn scipy")

code("""import boto3, io
from kaggle_secrets import UserSecretsClient

secrets    = UserSecretsClient()
aws_key    = secrets.get_secret("AWS_ACCESS_KEY_ID")
aws_secret = secrets.get_secret("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id=aws_key,
    aws_secret_access_key=aws_secret,
)

S3_BUCKET = "arn:aws:s3:us-east-1:724665945834:accesspoint/dreamt-v2-1-0-01"
S3_PREFIX = "dreamt/2.1.0/data_100Hz/"

print("S3 client ready.")""")

md("""## Step 1 — Inspect columns

Downloads only the first 8 KB of one file to read the header.""")

code("""import pandas as pd

paginator = s3.get_paginator("list_objects_v2")
subject_keys = []
for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
    for obj in page.get("Contents", []):
        if obj["Key"].endswith(".csv"):
            subject_keys.append(obj["Key"])
subject_keys = sorted(subject_keys)
print(f"Found {len(subject_keys)} subject files")

first_key = subject_keys[0]
resp      = s3.get_object(Bucket=S3_BUCKET, Key=first_key, Range="bytes=0-8191")
text      = resp["Body"].read().decode("utf-8", errors="ignore")
lines     = text.splitlines()
cols      = lines[0].split(",")
print(f"\\nFile: {first_key}")
print(f"Total columns: {len(cols)}")
for i, c in enumerate(cols):
    print(f"  [{i:3d}] {c}")
if len(lines) > 1:
    print("\\nFirst data row:")
    for c, v in zip(cols, lines[1].split(",")):
        print(f"  {c:30s} = {v}")""")

md("## Config")

code('''import json, time
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, decimate
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, confusion_matrix

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")

OUTPUT_DIR     = Path("/kaggle/working/output")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
EXPORT_DIR     = OUTPUT_DIR / "export"
for d in [OUTPUT_DIR, CHECKPOINT_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Columns (confirmed from inspect cell) ──────────────────────
COLUMN_MAP = {
    "bvp":   "BVP",
    "acc_x": "ACC_X",
    "acc_y": "ACC_Y",
    "acc_z": "ACC_Z",
    "eda":   "EDA",
    "hr":    "HR",
    "temp":  "TEMP",
    "label": "Sleep_Stage",
}

# ── 4-class mapping (matches app: awake / light / deep / rem) ──
# P is intentionally absent → epoch skipped (it's pre-sleep prep, not a stage).
STAGE_REMAP = {
    "W":  0,  # awake
    "N1": 1,  # light (merged with N2 — N1 is unreliable from wrist PPG)
    "N2": 1,  # light
    "N3": 2,  # deep
    "R":  3,  # REM
}
NUM_CLASSES   = 4
STAGE_NAMES   = ["Wake", "Light", "Deep", "REM"]
APP_STAGE_MAP = {0: "awake", 1: "light", 2: "deep", 3: "rem"}

# ── Signal config ──────────────────────────────────────────────
SOURCE_RATE       = 100
TARGET_RATE       = 25                        # decimate q=4 → integer IIR antialiasing
DECIMATE_Q        = SOURCE_RATE // TARGET_RATE
EPOCH_SEC         = 30
SAMPLES_PER_EPOCH = TARGET_RATE * EPOCH_SEC   # 750
NUM_RAW_CHANNELS  = 4                         # BVP, ACC_mag, EDA, TEMP
NUM_AUX_FEATURES  = 5                         # HR_mean, HR_std, HR_range, ACC_std, TEMP_slope

# ── Sequence-to-sequence config (L-SeqSleepNet-style temporal context) ──
SEQ_LEN       = 21                            # 21 × 30s = 10.5 min context
TRAIN_STRIDE  = SEQ_LEN // 2                  # 50% overlap → ~2× sequences for training
VAL_STRIDE    = SEQ_LEN                       # disjoint sequences for val

# ── Preprocessing ──────────────────────────────────────────────
BVP_LOW_HZ, BVP_HIGH_HZ = 0.5, 8.0
MAX_FLAT_RATIO          = 0.3
BVP_CLIP_THRESHOLD      = 0.01

# ── Model ──────────────────────────────────────────────────────
CNN_CHANNELS  = [32, 64, 128, 128]
CNN_KERNEL    = 7
CNN_DILATIONS = [1, 2, 4, 8]
EMBED_DIM     = 128
LSTM_HIDDEN   = 64
LSTM_LAYERS   = 2
DROPOUT       = 0.3

# ── Training ───────────────────────────────────────────────────
BATCH_SIZE          = 32          # 32 sequences × 21 epochs = 672 epochs per batch
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 5e-4
NUM_EPOCHS          = 60
EARLY_STOP_PATIENCE = 12
VAL_SPLIT           = 0.2
RANDOM_SEED         = 42
FOCAL_GAMMA         = 0.0   # 0 == plain α-weighted CE; sampler reverted, so no double-correction
DSU_PROB            = 0.5   # prob of applying DSU stat perturbation per batch (training only)
GRAD_CLIP           = 1.0

print(f"4-class: {STAGE_NAMES}")
print(f"Raw input per epoch: [{NUM_RAW_CHANNELS} ch × {SAMPLES_PER_EPOCH} samples @ {TARGET_RATE} Hz]")
print(f"Sequence context: {SEQ_LEN} epochs ({SEQ_LEN*EPOCH_SEC/60:.1f} min)")''')

md("""## Verify columns + label values
Downloads the first subject (~1.4 GB) and prints label distribution + column check.""")

code("""print(f"Downloading {subject_keys[0]} for verification...")
resp = s3.get_object(Bucket=S3_BUCKET, Key=subject_keys[0])
_df  = pd.read_csv(io.BytesIO(resp["Body"].read()))

print(f"Shape: {_df.shape}")
print(f"\\nAll Sleep_Stage values: {sorted(_df['Sleep_Stage'].unique())}")
print(f"Value counts:\\n{_df['Sleep_Stage'].value_counts().to_string()}")
print(f"\\nKnown in STAGE_REMAP: {[v for v in _df['Sleep_Stage'].unique() if v in STAGE_REMAP]}")
print(f"Unknown (skipped):    {[v for v in _df['Sleep_Stage'].unique() if v not in STAGE_REMAP]}")

missing = [v for v in COLUMN_MAP.values() if v not in _df.columns]
print(f"\\nMissing columns: {missing if missing else 'none — all good!'}")

del _df""")

md("## Dataset (4-channel @ 25 Hz, sequence-to-sequence)")

code('''def bandpass_filter(signal, low, high, fs_hz, order=4):
    nyq  = fs_hz / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0).astype(np.float32)


def robust_normalize(signal, mask=None):
    """Median/MAD normalization, restricted to `mask=True` samples if provided.

    Computing stats only over labeled-sleep samples (excluding P and unlabeled)
    keeps the centering consistent across subjects regardless of how much
    pre-sleep prep (P) they have in their recording.
    """
    sel  = signal[mask] if (mask is not None and mask.any()) else signal
    med  = float(np.median(sel))
    mad  = float(np.median(np.abs(sel - med)))
    # 1.4826 * MAD ≈ std for a Gaussian
    scale = 1.4826 * mad if mad > 1e-8 else 1.0
    return ((signal - med) / scale).astype(np.float32)


def decimate_signal(signal, q):
    """IIR-antialiased decimation 100 Hz → 100/q Hz, zero-phase."""
    return decimate(signal, q, ftype="iir", zero_phase=True).astype(np.float32)


def is_epoch_valid(bvp_epoch):
    if np.std(bvp_epoch) < BVP_CLIP_THRESHOLD:
        return False
    flat = np.sum(np.abs(np.diff(bvp_epoch)) < 1e-6) / max(1, len(bvp_epoch))
    return flat < MAX_FLAT_RATIO


def extract_aux(hr_window, acc_mag_window, temp_window):
    hr_mean   = float(np.mean(hr_window))                          if len(hr_window) else 0.0
    hr_std    = float(np.std(hr_window))                           if len(hr_window) > 1 else 0.0
    hr_range  = float(np.max(hr_window) - np.min(hr_window))       if len(hr_window) else 0.0
    acc_std   = float(np.std(acc_mag_window))                      if len(acc_mag_window) > 1 else 0.0
    if len(temp_window) > 1:
        x = np.arange(len(temp_window), dtype=np.float64)
        temp_slope = float(np.polyfit(x, temp_window, 1)[0])
    else:
        temp_slope = 0.0
    return np.array([hr_mean, hr_std, hr_range, acc_std, temp_slope], dtype=np.float32)


def process_subject_df(df):
    """
    Returns contiguous per-epoch arrays for one subject:
      raw: (N, 4, 750)  -- BVP, ACC_mag, EDA, TEMP @ 25 Hz, per-subject z-scored
      aux: (N, 5)
      lbl: (N,)
    Epochs with invalid BVP or unknown stage are dropped (this breaks the sequence,
    which is fine — SubjectSequenceDataset only windows over contiguous runs anyway
    because we just drop the bad epochs and slide).
    """
    col = COLUMN_MAP
    bvp_raw   = df[col["bvp"]].values.astype(np.float32)
    acc_x     = df[col["acc_x"]].values.astype(np.float32)
    acc_y     = df[col["acc_y"]].values.astype(np.float32)
    acc_z     = df[col["acc_z"]].values.astype(np.float32)
    eda_raw   = df[col["eda"]].values.astype(np.float32)
    hr_raw    = df[col["hr"]].values.astype(np.float32)
    temp_raw  = df[col["temp"]].values.astype(np.float32)
    label_col = df[col["label"]].values

    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2).astype(np.float32)

    try:
        bvp_filt = bandpass_filter(bvp_raw, BVP_LOW_HZ, BVP_HIGH_HZ, SOURCE_RATE)
    except Exception:
        bvp_filt = bvp_raw

    # Decimate 100 → 25 Hz with IIR antialiasing
    bvp_25  = decimate_signal(bvp_filt, DECIMATE_Q)
    acc_25  = decimate_signal(acc_mag,  DECIMATE_Q)
    eda_25  = decimate_signal(eda_raw,  DECIMATE_Q)
    temp_25 = decimate_signal(temp_raw, DECIMATE_Q)

    n_min    = min(len(bvp_25), len(acc_25), len(eda_25), len(temp_25))
    n_epochs = n_min // SAMPLES_PER_EPOCH

    # Build a 25 Hz mask of samples that fall inside labeled-sleep epochs
    # (W / N1 / N2 / N3 / R). Excludes P and any unlabeled tail. Used as the
    # support set for robust normalization stats.
    sleep_mask = np.zeros(n_min, dtype=bool)
    for i in range(n_epochs):
        s100 = i * EPOCH_SEC * SOURCE_RATE
        center = s100 + (EPOCH_SEC * SOURCE_RATE) // 2
        if center >= len(label_col):
            break
        if str(label_col[center]).strip() in STAGE_REMAP:
            s25 = i * SAMPLES_PER_EPOCH
            sleep_mask[s25:s25 + SAMPLES_PER_EPOCH] = True

    # Per-subject robust normalization, stats computed only over sleep samples
    bvp_25  = robust_normalize(bvp_25,  sleep_mask)
    acc_25  = robust_normalize(acc_25,  sleep_mask)
    eda_25  = robust_normalize(eda_25,  sleep_mask)
    temp_25 = robust_normalize(temp_25, sleep_mask)

    raws, auxs, lbls = [], [], []
    for i in range(n_epochs):
        s25, e25   = i * SAMPLES_PER_EPOCH, (i + 1) * SAMPLES_PER_EPOCH
        s100, e100 = i * EPOCH_SEC * SOURCE_RATE, (i + 1) * EPOCH_SEC * SOURCE_RATE
        if e100 > len(label_col):
            break

        center    = s100 + (EPOCH_SEC * SOURCE_RATE) // 2
        raw_label = str(label_col[center]).strip()
        if raw_label not in STAGE_REMAP:
            continue

        bvp_ep = bvp_25[s25:e25]
        if not is_epoch_valid(bvp_ep):
            continue

        acc_ep  = acc_25[s25:e25]
        eda_ep  = eda_25[s25:e25]
        temp_ep = temp_25[s25:e25]

        raws.append(np.stack([bvp_ep, acc_ep, eda_ep, temp_ep], axis=0))
        auxs.append(extract_aux(hr_raw[s100:e100], acc_mag[s100:e100], temp_raw[s100:e100]))
        lbls.append(STAGE_REMAP[raw_label])

    if not raws:
        return None, None, None
    return (np.stack(raws, axis=0).astype(np.float32),
            np.stack(auxs, axis=0).astype(np.float32),
            np.array(lbls, dtype=np.int64))


class SubjectSequenceDataset(Dataset):
    """Windows each subject's epoch series into SEQ_LEN sequences with given stride."""

    def __init__(self, subjects, seq_len=SEQ_LEN, stride=SEQ_LEN):
        self.items: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for raw, aux, lbl in subjects:
            n = len(lbl)
            if n < seq_len:
                continue
            for s in range(0, n - seq_len + 1, stride):
                self.items.append((raw[s:s + seq_len], aux[s:s + seq_len], lbl[s:s + seq_len]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        raw, aux, lbl = self.items[idx]
        return torch.from_numpy(raw), torch.from_numpy(aux), torch.from_numpy(lbl)


def build_datasets(val_ratio=VAL_SPLIT, seed=RANDOM_SEED):
    print(f"Found {len(subject_keys)} subjects (subject-level split)")
    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(len(subject_keys))
    n_val  = max(1, int(len(subject_keys) * val_ratio))
    val_ix = set(perm[:n_val].tolist())

    train_subjects, val_subjects = [], []
    for i, key in enumerate(subject_keys):
        name = key.split("/")[-1]
        print(f"  [{i+1}/{len(subject_keys)}] {name}...", end=" ", flush=True)
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
            df   = pd.read_csv(io.BytesIO(resp["Body"].read()))
        except Exception as ex:
            print(f"ERROR: {ex}")
            continue

        raw, aux, lbl = process_subject_df(df)
        del df
        if raw is None:
            print("skipped")
            continue
        split = "val" if i in val_ix else "train"
        print(f"{len(lbl)} epochs ({split})")
        (val_subjects if i in val_ix else train_subjects).append((raw, aux, lbl))

    train_ds = SubjectSequenceDataset(train_subjects, SEQ_LEN, TRAIN_STRIDE)
    val_ds   = SubjectSequenceDataset(val_subjects,   SEQ_LEN, VAL_STRIDE)
    print(f"\\nTrain: {len(train_ds)} sequences  |  Val: {len(val_ds)} sequences")

    all_train_labels = np.concatenate([lbl for _, _, lbl in train_subjects])
    counts = np.bincount(all_train_labels, minlength=NUM_CLASSES).astype(np.float32)
    print(f"Train class counts: {dict(zip(STAGE_NAMES, counts.astype(int).tolist()))}")
    inv     = 1.0 / np.maximum(counts, 1.0)
    weights = torch.from_numpy(inv / inv.sum() * NUM_CLASSES).float()
    print(f"Class weights (focal α): {[round(w, 3) for w in weights.tolist()]}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)
    return train_loader, val_loader, weights


print("Dataset code ready.")''')

md("""## Data audit (run this BEFORE training)

Inspects the first few subjects to rule out structural bugs that would silently
cap accuracy: dead/forward-filled channels, label run-lengths that aren't 30 s
blocks, mis-aligned epoch grid, and how many epochs survive the pipeline.""")

code('''AUDIT_N = 3
EPOCH_SAMPLES_100 = EPOCH_SEC * SOURCE_RATE   # 3000 samples = one 30 s epoch @ 100 Hz

for key in subject_keys[:AUDIT_N]:
    name = key.split("/")[-1]
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    df   = pd.read_csv(io.BytesIO(resp["Body"].read()))
    n    = len(df)
    print(f"\\n{'='*78}")
    print(f"{name}   {n:,} rows   ({n/SOURCE_RATE/3600:.2f} h @ {SOURCE_RATE} Hz)")
    print('='*78)

    # ── 1) Channel health: NaN / zero / constant fractions + range ──────────
    print(f"\\n{'channel':10s} {'NaN%':>7s} {'zero%':>7s} {'dup%':>7s} "
          f"{'min':>11s} {'max':>11s} {'mean':>11s} {'std':>11s}")
    for ch in ["bvp", "acc_x", "acc_y", "acc_z", "eda", "hr", "temp"]:
        col = COLUMN_MAP[ch]
        v   = df[col].values.astype(np.float64)
        nan_pct = 100 * np.isnan(v).mean()
        fin     = v[np.isfinite(v)]
        zero_pct = 100 * (fin == 0).mean() if len(fin) else 0.0
        # "dup%" = fraction equal to the previous sample → forward-fill indicator
        dup_pct  = 100 * np.mean(v[1:] == v[:-1]) if n > 1 else 0.0
        print(f"{col:10s} {nan_pct:7.2f} {zero_pct:7.2f} {dup_pct:7.2f} "
              f"{np.nanmin(v):11.3f} {np.nanmax(v):11.3f} "
              f"{np.nanmean(v):11.3f} {np.nanstd(v):11.3f}")
    print("  (dup% near 99 ⇒ signal is really low-rate, forward-filled to 100 Hz —")
    print("   its within-epoch std/slope features are then mostly meaningless)")

    # ── 2) Label distribution ───────────────────────────────────────────────
    labels = pd.Series(df[COLUMN_MAP["label"]].astype(str).str.strip().values)
    print("\\nLabel counts:")
    for lab, c in labels.value_counts().items():
        tag = f"class {STAGE_REMAP[lab]} ({STAGE_NAMES[STAGE_REMAP[lab]]})" if lab in STAGE_REMAP else "SKIPPED"
        print(f"  {lab:6s} {c:11,}  ({100*c/n:5.1f}%)  → {tag}")

    # ── 3) Label run-lengths: should be exact 30 s (3000-sample) blocks ──────
    lab_arr = labels.values
    changes = np.where(lab_arr[1:] != lab_arr[:-1])[0] + 1
    run_lens = np.diff(np.concatenate([[0], changes, [n]]))
    on_grid_runs  = np.mean(run_lens % EPOCH_SAMPLES_100 == 0) * 100
    on_grid_trans = (np.mean(changes % EPOCH_SAMPLES_100 == 0) * 100) if len(changes) else 100.0
    print(f"\\nLabel runs: {len(run_lens)} segments  "
          f"min={run_lens.min()}  median={int(np.median(run_lens))}  max={run_lens.max()}")
    print(f"  runs that are exact multiples of {EPOCH_SAMPLES_100} (30 s): {on_grid_runs:5.1f}%")
    print(f"  label transitions landing on the 30 s grid:               {on_grid_trans:5.1f}%")
    print("  (both should be ~100% — if not, our epoch grid is mis-aligned to PSG scoring)")

    # ── 4) Pipeline yield: how many 30 s windows survive, and why ───────────
    raw, aux, lbl = process_subject_df(df)
    n_windows = n // EPOCH_SAMPLES_100
    kept = 0 if raw is None else len(lbl)
    print(f"\\nPipeline: {n_windows} raw 30 s windows → {kept} kept "
          f"({100*kept/max(1,n_windows):.1f}%)")
    if raw is not None:
        dist = dict(zip(STAGE_NAMES, np.bincount(lbl, minlength=NUM_CLASSES).tolist()))
        print(f"  kept class dist: {dist}")
        print(f"  aux means [HR_mean,HR_std,HR_range,ACC_std,TEMP_slope]: "
              f"{np.round(aux.mean(0), 3).tolist()}")
        print(f"  aux NaN/Inf count: {int(np.isnan(aux).sum() + np.isinf(aux).sum())}")
        print(f"  raw NaN/Inf count: {int(np.isnan(raw).sum() + np.isinf(raw).sum())}")

    del df, raw, aux, lbl

print("\\nAudit complete.")''')

md("## Model (Dilated-CNN encoder + BiLSTM seq2seq)")

code('''class DSU(nn.Module):
    """Domain Shift with Uncertainty — perturbs per-channel feature statistics
    during training to make the encoder invariant to subject baseline drift.

    For each input, compute per-channel (μ, σ) over time. Estimate the
    *uncertainty* of those stats as their std across the batch. Then sample
    new stats: μ' = μ + ε₁·σ_μ, σ' = σ + ε₂·σ_σ, and re-standardize.
    At eval, identity.

    Reference: Li et al., "Uncertainty Modeling for OOD Generalization", ICLR 2022.
    Also the regularization layer in SleepPPG-Net2.
    """

    def __init__(self, p=DSU_PROB, eps=1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):                 # x: [N, C, T]
        if not self.training or self.p == 0.0 or x.size(0) < 2:
            return x
        if torch.rand(1).item() > self.p:
            return x

        N, C, _ = x.shape
        mu    = x.mean(dim=-1, keepdim=True)                                  # [N, C, 1]
        sigma = (x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt()  # [N, C, 1]

        # Per-channel uncertainty = std of (μ, σ) across the batch
        mu_unc = mu.squeeze(-1).var(dim=0, unbiased=False).clamp(min=self.eps).sqrt().view(1, C, 1)
        sg_unc = sigma.squeeze(-1).var(dim=0, unbiased=False).clamp(min=self.eps).sqrt().view(1, C, 1)

        new_mu    = mu    + torch.randn_like(mu)    * mu_unc
        new_sigma = sigma + torch.randn_like(sigma) * sg_unc

        return (x - mu) / sigma * new_sigma + new_mu


class DilatedCNNEncoder(nn.Module):
    """Per-epoch encoder: 4-channel × 750-sample input → 128-dim embedding.

    Dilations 1/2/4/8 grow the receptive field to ~120 samples (~5 s @ 25 Hz)
    without losing PPG temporal resolution — same idea as SleepPPG-Net2.
    DSU layer at input perturbs subject-baseline stats during training only.
    """

    def __init__(self, in_channels=NUM_RAW_CHANNELS,
                 channels=CNN_CHANNELS, kernel=CNN_KERNEL,
                 dilations=CNN_DILATIONS, dropout=DROPOUT):
        super().__init__()
        self.dsu = DSU(p=DSU_PROB)
        layers, c_in = [], in_channels
        for c_out, d in zip(channels, dilations):
            pad = (kernel - 1) * d // 2
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=2,
                          padding=pad, dilation=d),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            c_in = c_out
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x):                   # x: [B, C, T]
        x = self.dsu(x)
        return self.net(x).squeeze(-1)      # [B, c_out]


class SleepSeqModel(nn.Module):
    """Per-epoch CNN → concat aux → 2-layer BiLSTM over SEQ_LEN epochs → per-epoch 4-class logits.

    Inputs:
        raw: [B, S, C, T]   S=SEQ_LEN, C=4, T=750
        aux: [B, S, A]      A=5
    Output:
        logits: [B, S, NUM_CLASSES]  (one prediction per epoch)
    """

    def __init__(self):
        super().__init__()
        self.encoder = DilatedCNNEncoder()
        self.project = nn.Sequential(
            nn.Linear(self.encoder.out_dim + NUM_AUX_FEATURES, EMBED_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM, hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS, batch_first=True, bidirectional=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(LSTM_HIDDEN * 2),
            nn.Dropout(DROPOUT),
            nn.Linear(LSTM_HIDDEN * 2, NUM_CLASSES),
        )

    def forward(self, raw, aux):
        B, S, C, T = raw.shape
        per_epoch  = self.encoder(raw.reshape(B * S, C, T))    # [B*S, 128]
        per_epoch  = per_epoch.reshape(B, S, -1)
        feat       = torch.cat([per_epoch, aux], dim=-1)        # [B, S, 128+5]
        feat       = self.project(feat)                          # [B, S, EMBED_DIM]
        seq, _     = self.lstm(feat)                             # [B, S, 2H]
        return self.head(seq)                                    # [B, S, NUM_CLASSES]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_probe = SleepSeqModel().to(device)
print(f"Parameters: {_probe.count_parameters():,}  |  Device: {device}")
with torch.no_grad():
    _r = torch.randn(2, SEQ_LEN, NUM_RAW_CHANNELS, SAMPLES_PER_EPOCH, device=device)
    _a = torch.randn(2, SEQ_LEN, NUM_AUX_FEATURES, device=device)
    _o = _probe(_r, _a)
    print(f"Forward shape check: in raw {tuple(_r.shape)} aux {tuple(_a.shape)} → out {tuple(_o.shape)}")
del _probe, _r, _a, _o''')

md("""## Load data
Each subject (~1.4 GB) is streamed, decimated to 25 Hz, sliced into epochs, then freed.
Expect ~30–60 min for all 100 subjects.""")

code("train_loader, val_loader, class_weights = build_datasets()")

md("## Training (focal loss + class weights)")

code('''class FocalLoss(nn.Module):
    """α-weighted focal loss for multi-class.

    α (per-class) handles imbalance; γ down-weights easy examples. γ=2 is the standard.
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=-1)
        p     = log_p.exp()
        ce    = F.nll_loss(log_p, target, weight=self.alpha, reduction="none")
        pt    = p.gather(1, target.unsqueeze(1)).squeeze(1)
        return ((1.0 - pt).pow(self.gamma) * ce).mean()


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)

    total_loss, n_tokens = 0.0, 0
    preds_all, labels_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for raw, aux, lbl in loader:
            raw = raw.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            logits      = model(raw, aux)                  # [B, S, C]
            flat_logits = logits.reshape(-1, NUM_CLASSES)
            flat_labels = lbl.reshape(-1)
            loss        = criterion(flat_logits, flat_labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item() * flat_labels.size(0)
            n_tokens   += flat_labels.size(0)
            preds_all.append(flat_logits.argmax(-1).cpu().numpy())
            labels_all.append(flat_labels.cpu().numpy())

    preds  = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    return (total_loss / max(1, n_tokens),
            accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro", zero_division=0),
            cohen_kappa_score(labels, preds),
            preds, labels)


model     = SleepSeqModel().to(device)
print(f"Trainable parameters: {model.count_parameters():,}")

criterion = FocalLoss(alpha=class_weights.to(device), gamma=FOCAL_GAMMA)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_f1, patience_left = 0.0, EARLY_STOP_PATIENCE
history = {k: [] for k in ["train_loss", "val_loss", "train_f1", "val_f1",
                           "val_acc", "val_kappa", "lr"]}

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    tr_loss, _, tr_f1, _, _, _              = run_epoch(model, train_loader, criterion, optimizer)
    vl_loss, vl_acc, vl_f1, vl_k, vp, vl    = run_epoch(model, val_loader,   criterion)
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]

    for k, v in zip(history, [tr_loss, vl_loss, tr_f1, vl_f1, vl_acc, vl_k, lr]):
        history[k].append(v)

    print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
          f"Train L:{tr_loss:.4f} F1:{tr_f1:.3f} | "
          f"Val L:{vl_loss:.4f} Acc:{vl_acc:.3f} F1:{vl_f1:.3f} κ:{vl_k:.3f} | "
          f"LR:{lr:.6f} | {time.time()-t0:.1f}s")

    if vl_f1 > best_f1:
        best_f1, patience_left = vl_f1, EARLY_STOP_PATIENCE
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_f1": vl_f1, "val_acc": vl_acc, "val_kappa": vl_k,
        }, CHECKPOINT_DIR / "best_model.pt")

        per_class_f1 = f1_score(vl, vp, average=None,
                                labels=list(range(NUM_CLASSES)), zero_division=0)
        cm = confusion_matrix(vl, vp, labels=list(range(NUM_CLASSES)))
        print(f"  ✓ New best F1: {vl_f1:.4f}  |  per-class F1: "
              + ", ".join(f"{n}={s:.2f}" for n, s in zip(STAGE_NAMES, per_class_f1)))
        print(f"    Confusion matrix (rows=true, cols=pred):\\n{cm}")
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f"Early stop at epoch {epoch}")
            break

with open(OUTPUT_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"\\nBest val macro F1: {best_f1:.4f}")''')

md("## Export to ONNX (dynamic batch + seq_len for full-night inference)")

code('''import onnx, onnxruntime as ort

ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location="cpu", weights_only=True)
export_model = SleepSeqModel().to("cpu")
export_model.load_state_dict(ckpt["model_state_dict"])
export_model.eval()

dummy_raw = torch.randn(1, SEQ_LEN, NUM_RAW_CHANNELS, SAMPLES_PER_EPOCH)
dummy_aux = torch.randn(1, SEQ_LEN, NUM_AUX_FEATURES)

onnx_path = EXPORT_DIR / "sleep_stage_model.onnx"
torch.onnx.export(
    export_model,
    (dummy_raw, dummy_aux),
    str(onnx_path),
    input_names=["raw_signal", "aux_features"],
    output_names=["logits"],
    dynamic_axes={
        "raw_signal":   {0: "batch", 1: "seq_len"},
        "aux_features": {0: "batch", 1: "seq_len"},
        "logits":       {0: "batch", 1: "seq_len"},
    },
    opset_version=17,
    do_constant_folding=True,
)

onnx.checker.check_model(onnx.load(str(onnx_path)))
sess = ort.InferenceSession(str(onnx_path))

# Parity check
ort_out = sess.run(None, {"raw_signal": dummy_raw.numpy(),
                          "aux_features": dummy_aux.numpy()})
with torch.no_grad():
    pt_out = export_model(dummy_raw, dummy_aux).numpy()
print(f"ONNX parity check max diff: {float(np.max(np.abs(pt_out - ort_out[0]))):.6f}")

# Variable seq_len check (a full night might be ~960 epochs of 30s = 8 h)
test_raw = torch.randn(1, 33, NUM_RAW_CHANNELS, SAMPLES_PER_EPOCH)
test_aux = torch.randn(1, 33, NUM_AUX_FEATURES)
out_dyn  = sess.run(None, {"raw_signal": test_raw.numpy(),
                           "aux_features": test_aux.numpy()})
print(f"Dynamic seq_len OK: in 33 epochs → out shape {out_dyn[0].shape}")

metadata = {
    "version": "2.4.0",
    "modelType": "dilated_cnn_bilstm_seq2seq",
    "task": "sleep_stage_classification_4class",
    "numClasses": NUM_CLASSES,
    "stageNames": STAGE_NAMES,
    "stageMap": APP_STAGE_MAP,
    "input": {
        "rawSignal": {
            "shape": ["batch", "seq_len", NUM_RAW_CHANNELS, SAMPLES_PER_EPOCH],
            "channelNames": ["BVP", "ACC_magnitude", "EDA", "TEMP"],
            "sampleRateHz": TARGET_RATE,
            "sourceRateHz": SOURCE_RATE,
            "epochSeconds": EPOCH_SEC,
            "samplesPerEpoch": SAMPLES_PER_EPOCH,
            "recommendedSeqLen": SEQ_LEN,
            "minSeqLen": 1,
        },
        "auxFeatures": {
            "shape": ["batch", "seq_len", NUM_AUX_FEATURES],
            "names": ["HR_mean", "HR_std", "HR_range", "ACC_std", "TEMP_slope"],
        },
    },
    "output": {
        "logits": {
            "shape": ["batch", "seq_len", NUM_CLASSES],
            "note": "argmax along last dim, then map via stageMap",
        },
    },
    "preprocessing": {
        "decimateQ": DECIMATE_Q,
        "decimateFilter": "scipy.signal.decimate(ftype='iir', zero_phase=True)",
        "bvpBandpass": [BVP_LOW_HZ, BVP_HIGH_HZ],
        "normalization": "per_subject_median_mad_over_sleep_epochs_only",
        "channelOrder": ["BVP_bandpassed", "ACC_magnitude", "EDA", "TEMP"],
    },
    "training": {
        "loss": "focal_alpha_weighted",
        "focalGamma": FOCAL_GAMMA,
        "valF1":    float(ckpt.get("val_f1", 0)),
        "valAcc":   float(ckpt.get("val_acc", 0)),
        "valKappa": float(ckpt.get("val_kappa", 0)),
        "epoch":    int(ckpt.get("epoch", 0)),
    },
}
with open(EXPORT_DIR / "sleep_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\\nModel:    {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")
print(f"Metadata: {EXPORT_DIR / 'sleep_model_metadata.json'}")''')

md("## Training Curves")

code('''import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ep = range(1, len(history["train_loss"]) + 1)

axes[0].plot(ep, history["train_loss"], label="train")
axes[0].plot(ep, history["val_loss"],   label="val")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

axes[1].plot(ep, history["train_f1"], label="train")
axes[1].plot(ep, history["val_f1"],   label="val")
axes[1].axhline(0.62, color="gray", linestyle="--", label="DREAMT baseline (0.62)")
axes[1].set_title("Macro F1"); axes[1].set_xlabel("Epoch"); axes[1].legend()

axes[2].plot(ep, history["val_kappa"])
axes[2].axhline(0.50, color="gray", linestyle="--", label="DREAMT baseline κ (0.50)")
axes[2].set_title("Val Cohen's κ"); axes[2].set_xlabel("Epoch"); axes[2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.show()''')

md("""## Output files

| File | Drop into repo at |
|------|-------------------|
| `output/export/sleep_stage_model.onnx`    | `assets/ml/sleep_stage_model.onnx` |
| `output/export/sleep_model_metadata.json` | `assets/ml/sleep_model_metadata.json` |
| `output/checkpoints/best_model.pt`        | keep as backup |""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "kaggle_train.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB, {len(CELLS)} cells)")
