"""Builds kaggle_train.ipynb — HR + actigraphy sleep stage classifier.

Trains on BIDSleep (Apple Watch, 253 nights), tests on Walch/Sleep-Accel (held out).
Inputs are ONLY heart rate + accelerometer — the signals any Wear OS watch can provide.
"""
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


md("""# Seren — Sleep Stage Classifier v3 (HR + Actigraphy)

**Inputs:** heart rate + wrist accelerometer ONLY — the signals every Wear OS watch can provide
(via Google Health Services). No raw PPG, no EDA — those aren't available on a consumer watch.

**Datasets (both Apple Watch + EEG/PSG labels):**
- **BIDSleep** (PhysioNet `bidsleep-dataset`): 47 subjects, ~253 nights → **training**
- **Walch / Sleep-Accel** (PhysioNet `sleep-accel`): 31 subjects → **held-out test** (never seen in training)

**Features (11 per 30 s epoch, computed from HR + accel — must mirror the on-device TS code):**
HR: mean, std, min, max, range, successive-diff std (HRV proxy), Δ-vs-previous-epoch ·
Movement: activity count, immobility fraction, max movement, movement std

**Labels (4-class, matches app):** `Wake→0`, `N1+N2→1 (light)`, `N3→2 (deep)`, `REM→3`.

**Targets (beat prior work):** 4-class weighted F1 ≥ 0.72 and Deep recall > 56% (SLAMSS 2023);
3-class weighted F1 ~0.80 (SLAMSS). Walch 3-class baseline was ~0.72 acc / κ 0.3.

**Model:** temporal-conv → BiLSTM seq2seq over 21 epochs (~10 min context).""")

code("!pip install -q onnx onnxruntime scikit-learn scipy")

md("""## Step 1 — Stream from PhysioNet (no full download)

Both datasets live on PhysioNet. Instead of `wget -r` (which pulls all 5.9 GB to disk), we
fetch **one file at a time over HTTPS**, process it into the tiny feature arrays, and discard
the raw bytes — exactly how the DREAMT notebook streamed from S3.

**Sleep-Accel is open-access. BIDSleep may be credentialed** → if so, add a free PhysioNet
account's `PHYSIONET_USER` / `PHYSIONET_PASS` to Kaggle Secrets.""")

code('''import io, time as _time
import requests
import numpy as np, pandas as pd
import scipy.io as sio

WALCH_BASE    = "https://physionet.org/files/sleep-accel/1.0.0/"
BIDSLEEP_BASE = "https://physionet.org/files/bidsleep-dataset/1.0.0/"

# PhysioNet credentials (only needed for access-restricted datasets)
PN_USER = PN_PASS = None
try:
    from kaggle_secrets import UserSecretsClient
    _sec = UserSecretsClient()
    PN_USER = _sec.get_secret("PHYSIONET_USER")
    PN_PASS = _sec.get_secret("PHYSIONET_PASS")
    print("Loaded PhysioNet credentials from Kaggle Secrets.")
except Exception:
    print("No credentials found — assuming open-access (fine for Sleep-Accel).")

session = requests.Session()
if PN_USER and PN_PASS:
    session.auth = (PN_USER, PN_PASS)

NUM_WORKERS = 6   # parallel HTTP fetches per dataset loader (network-bound work)


def http_get(url, tries=4, timeout=120):
    """GET bytes with retries — the HTTPS analogue of boto3 get_object."""
    last = None
    for k in range(tries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code in (401, 403):
                raise RuntimeError(f"{r.status_code}: auth required for {url} "
                                   "— set PHYSIONET_USER/PHYSIONET_PASS in Kaggle Secrets.")
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        _time.sleep(2 * (k + 1))
    raise RuntimeError(f"Failed to fetch {url} ({last})")


def list_files(base):
    """All file paths in a PhysioNet dataset, parsed from its SHA256SUMS manifest."""
    txt = http_get(base + "SHA256SUMS.txt").decode("utf-8", "ignore")
    paths = []
    for line in txt.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            paths.append(parts[1].strip().lstrip("*"))
    return paths


print("Streaming helpers ready.")''')

md("""## Step 2 — Inspect BIDSleep format (stream a few sample files)

**Run this and paste the output before training.** BIDSleep's `.mat` key names, Dreem stage
encoding, and timestamp units are not documented — this streams one sample of each so we can
lock the parser constants in the Config cell (`BIDSLEEP_LABEL_KEY`, `BIDSLEEP_STAGE_REMAP`,
and how `hr.csv`/`motion.csv` timestamps align to the labels).""")

code('''bid_paths = list_files(BIDSLEEP_BASE)
print(f"BIDSleep: {len(bid_paths)} files\\n")
for p in bid_paths[:18]:
    print("  ", p)

mat_paths = sorted(p for p in bid_paths if p.endswith(".mat"))
print(f"\\n.mat files: {len(mat_paths)}  (≈ number of nights)")
if mat_paths:
    m = sio.loadmat(io.BytesIO(http_get(BIDSLEEP_BASE + mat_paths[0])))
    print(f"\\nKeys in {mat_paths[0]}:")
    for k, v in m.items():
        if k.startswith("__"):
            continue
        a = np.asarray(v)
        print(f"  {k:26s} shape={a.shape} dtype={a.dtype} sample={np.ravel(a)[:6]}")

for nm in ["hr.csv", "motion.csv"]:
    fs = [p for p in bid_paths if p.endswith(nm)]
    if fs:
        head = http_get(BIDSLEEP_BASE + fs[0]).decode("utf-8", "ignore").splitlines()[:4]
        print(f"\\n--- {nm}  ({fs[0]}) ---")
        print("\\n".join(head))''')

md("## Config")

code('''import json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (f1_score, accuracy_score, cohen_kappa_score,
                             confusion_matrix, recall_score)

print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")

import pickle

OUTPUT_DIR     = Path("/kaggle/working/output")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
EXPORT_DIR     = OUTPUT_DIR / "export"
CACHE_DIR      = Path("/kaggle/working/cache")     # persisted in /kaggle/working
for d in [OUTPUT_DIR, CHECKPOINT_DIR, EXPORT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def cached(name, build_fn):
    """Run build_fn() the first time, pickle the result, and return cached data thereafter.
    Delete the .pkl to force a re-stream. /kaggle/working persists if you save the notebook
    output; for cross-session reuse, save the output as a Kaggle Dataset and add it as input."""
    p = CACHE_DIR / f"{name}_features.pkl"
    if p.exists():
        with open(p, "rb") as f:
            data = pickle.load(f)
        print(f"[cache] {p.name}: loaded {len(data)} recordings ({p.stat().st_size/1e6:.1f} MB)")
        return data
    data = build_fn()
    with open(p, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"[cache] {p.name}: saved {len(data)} recordings ({p.stat().st_size/1e6:.1f} MB)")
    return data

# ── Labels: 4-class, matches app (awake/light/deep/rem) ───────────────────────
NUM_CLASSES   = 4
STAGE_NAMES   = ["Wake", "Light", "Deep", "REM"]
APP_STAGE_MAP = {0: "awake", 1: "light", 2: "deep", 3: "rem"}

# Walch labels: 0=W 1=N1 2=N2 3=N3 (4=N4 legacy) 5=REM, -1=unscored
WALCH_STAGE_REMAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}   # missing/-1 → skipped

# BIDSleep (Dreem) labels — confirmed from inspect: uint8 AASM codes 0=W 1=N1 2=N2 3=N3 4=REM.
# Any other code (e.g. 5/255 = unscored/artifact) is absent from the map → epoch skipped.
BIDSLEEP_STAGE_REMAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
BIDSLEEP_LABEL_KEY = "expert_label"  # expert-corrected (vs automated "dreem_label")
BIDSLEEP_START_KEY = "recStart"      # local wall-clock string, e.g. '2021-12-02 23:11:25'

# ── Epoch / feature config ────────────────────────────────────────────────────
EPOCH_SEC      = 30
HR_WIN_SEC     = 120     # half-width of HR window (±2 min) — tolerates sparse 0.2 Hz HR
MOVE_THRESH_G  = 0.02    # accel-magnitude-delta threshold for "immobile" (g; both datasets in g)

FEATURE_NAMES = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "hr_succdiff_std", "hr_delta_prev",
    "act_count", "immobility_frac", "act_max", "act_std",
]
NUM_FEATURES = len(FEATURE_NAMES)   # 11

# ── Sequence config ───────────────────────────────────────────────────────────
SEQ_LEN      = 21       # ~10.5 min context
TRAIN_STRIDE = SEQ_LEN // 2
EVAL_STRIDE  = SEQ_LEN

# ── Model ─────────────────────────────────────────────────────────────────────
TCN_CHANNELS = 64
TCN_KERNEL   = 5
LSTM_HIDDEN  = 64
LSTM_LAYERS  = 2
DROPOUT      = 0.3

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE          = 64
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 5e-4
NUM_EPOCHS          = 80
EARLY_STOP_PATIENCE = 15
VAL_SPLIT           = 0.15     # of BIDSleep subjects → validation
RANDOM_SEED         = 42
GRAD_CLIP           = 1.0

print(f"{NUM_FEATURES} features/epoch · seq {SEQ_LEN} epochs · classes {STAGE_NAMES}")''')

md("""## Feature extraction (shared by both datasets)

These 11 features are the **train/serve contract** — the on-device TypeScript must compute
them identically from Health Services HR + accelerometer. Keep the math here simple
(mean/std/min/max/threshold counts) so it ports exactly.""")

code('''def acc_magnitude(xyz: np.ndarray) -> np.ndarray:
    """xyz: [N,3] in g → magnitude [N]."""
    return np.sqrt((xyz.astype(np.float64) ** 2).sum(axis=1))


def extract_features_for_night(hr_t, hr_v, acc_t, acc_mag, label_t, label_raw, stage_remap):
    """Build per-epoch feature rows for one night.

    hr_t, hr_v   : 1D arrays of HR sample times (sec) and bpm values
    acc_t, acc_mag: 1D arrays of accel sample times (sec) and magnitudes (g)
    label_t      : 1D array of epoch start times (sec), 30 s apart
    label_raw    : raw stage codes aligned to label_t
    Returns (features [E,11] float32, labels [E] int64) for kept epochs only.
    """
    feats, labs = [], []
    prev_hr_mean = None

    order = np.argsort(hr_t); hr_t, hr_v = hr_t[order], hr_v[order]
    order = np.argsort(acc_t); acc_t, acc_mag = acc_t[order], acc_mag[order]

    for et, raw in zip(label_t, label_raw):
        cls = stage_remap.get(int(raw), None) if not np.isnan(raw) else None
        if cls is None:
            prev_hr_mean = None     # break trend across unscored gaps
            continue

        # HR window: epoch center ± HR_WIN_SEC
        c = et + EPOCH_SEC / 2.0
        hr_lo, hr_hi = np.searchsorted(hr_t, c - HR_WIN_SEC), np.searchsorted(hr_t, c + HR_WIN_SEC)
        hr_seg = hr_v[hr_lo:hr_hi]

        # Accel within the epoch [et, et+30)
        a_lo, a_hi = np.searchsorted(acc_t, et), np.searchsorted(acc_t, et + EPOCH_SEC)
        a_seg = acc_mag[a_lo:a_hi]

        if len(hr_seg) == 0:        # cannot form HR features → drop epoch
            prev_hr_mean = None
            continue

        hr_mean = float(np.mean(hr_seg))
        hr_std  = float(np.std(hr_seg)) if len(hr_seg) > 1 else 0.0
        hr_min  = float(np.min(hr_seg))
        hr_max  = float(np.max(hr_seg))
        hr_rng  = hr_max - hr_min
        hr_sdsd = float(np.std(np.diff(hr_seg))) if len(hr_seg) > 2 else 0.0
        hr_dlt  = (hr_mean - prev_hr_mean) if prev_hr_mean is not None else 0.0

        if len(a_seg) > 1:
            d = np.abs(np.diff(a_seg))
            act_count = float(np.sum(d))
            immob     = float(np.mean(d < MOVE_THRESH_G))
            act_max   = float(np.max(d))
            act_std   = float(np.std(a_seg))
        else:
            act_count = immob = act_max = act_std = 0.0

        feats.append([hr_mean, hr_std, hr_min, hr_max, hr_rng,
                      hr_sdsd, hr_dlt, act_count, immob, act_max, act_std])
        labs.append(cls)
        prev_hr_mean = hr_mean

    if not feats:
        return None, None
    return np.asarray(feats, dtype=np.float32), np.asarray(labs, dtype=np.int64)


def robust_normalize_features(feats: np.ndarray) -> np.ndarray:
    """Per-night median/MAD z-score + replace NaN/Inf with 0 (the median in normalized
    space). HR/motion gaps cause a tiny fraction of NaN values; this keeps training stable."""
    med = np.median(feats, axis=0)
    mad = np.median(np.abs(feats - med), axis=0)
    scale = np.where(mad > 1e-8, 1.4826 * mad, 1.0)
    out = ((feats - med) / scale).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


print("Feature code ready.")''')

md("## Parse Walch / Sleep-Accel (held-out test set) — streamed")

code('''def _loadtxt_url(url, delimiter=None):
    raw = http_get(url).decode("utf-8", "ignore")
    a = np.loadtxt(io.StringIO(raw), delimiter=delimiter)
    return a.reshape(-1, a.shape[-1]) if a.ndim == 1 else a


def _walch_subject(lp, paths_set):
    """Pull and process one Walch subject → (sid, features, labels) or None."""
    sid = lp.split("/")[-1].split("_")[0]
    hp, mp = f"heart_rate/{sid}_heartrate.txt", f"motion/{sid}_acceleration.txt"
    if hp not in paths_set or mp not in paths_set:
        return None
    lab = _loadtxt_url(WALCH_BASE + lp)                 # time stage  (whitespace)
    hr  = _loadtxt_url(WALCH_BASE + hp, delimiter=",")  # time,hr     (comma)
    mo  = _loadtxt_url(WALCH_BASE + mp)                 # time x y z  (whitespace)
    f, l = extract_features_for_night(hr[:, 0], hr[:, 1], mo[:, 0], acc_magnitude(mo[:, 1:4]),
                                      lab[:, 0], lab[:, 1], WALCH_STAGE_REMAP)
    if f is None:
        return None
    return (sid, robust_normalize_features(f), l)


def load_walch():
    """Stream Sleep-Accel from PhysioNet with parallel fetching."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    paths = set(list_files(WALCH_BASE))
    label_files = sorted(p for p in paths if p.endswith("labeled_sleep.txt"))
    out, done = [], 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(_walch_subject, lp, paths): lp for lp in label_files}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r is not None:
                    out.append(r)
            except Exception as e:
                print(f"  failed {futs[fut]}: {e}", flush=True)
            done += 1
            if done % 5 == 0 or done == len(label_files):
                print(f"  [{done}/{len(label_files)}] kept {len(out)} subjects", flush=True)
    print(f"Walch: parsed {len(out)} subjects (parallel x{NUM_WORKERS})")
    return out


walch_subjects = cached("walch", load_walch)''')

md("""## Parse BIDSleep (training set)

⚠️ Confirm `BIDSLEEP_LABEL_KEY`, `BIDSLEEP_START_KEY`, and `BIDSLEEP_STAGE_REMAP` from the
inspect cell first. The parser is written defensively but the `.mat` key names and the
label/time columns of the CSVs may need tweaking.""")

code('''_NUM_START = set(b"-.0123456789")


def _read_csv_bytes(content, value_cols):
    """BIDSleep csv bytes → (absolute_unix_sec, values). Timestamps are absolute Unix
    seconds (UTC). motion.csv has a 'Timestamp,x,y,z' header; hr.csv has none — we sniff
    the first byte and use skiprows so pandas never sees a non-numeric row (silences the
    DtypeWarning on big motion files). Label alignment happens in load_bidsleep via recStart."""
    skiprows = 0 if content[:1] in _NUM_START else 1
    df = pd.read_csv(io.BytesIO(content), header=None, skiprows=skiprows, low_memory=False)
    t = df.iloc[:, 0].to_numpy(dtype=np.float64)
    if t.max() > 1e12:                # guard: unix ms → s (this dataset is already seconds)
        t = t / 1000.0
    vals = df.iloc[:, value_cols].to_numpy(dtype=np.float64)
    return t, vals


def _bidsleep_night(mp, pset, verbose=False):
    """Pull and process one BIDSleep night → (night_id, features, labels) or None."""
    night = mp.rsplit("/", 1)[0] if "/" in mp else ""
    hr_p  = f"{night}/hr.csv" if night else "hr.csv"
    mo_p  = f"{night}/motion.csv" if night else "motion.csv"
    if hr_p not in pset or mo_p not in pset:
        return None

    m = sio.loadmat(io.BytesIO(http_get(BIDSLEEP_BASE + mp)))
    label_raw = np.ravel(np.asarray(m[BIDSLEEP_LABEL_KEY]).astype(np.float64))
    label_t   = np.arange(len(label_raw), dtype=np.float64) * EPOCH_SEC

    hr_t, hr_v = _read_csv_bytes(http_get(BIDSLEEP_BASE + hr_p), [1])
    mo_t, mo_v = _read_csv_bytes(http_get(BIDSLEEP_BASE + mo_p), [1, 2, 3])

    # Align HR/motion (absolute Unix sec, UTC) to labels (epoch 0 = recStart).
    # recStart is local wall-clock; infer the quarter-hour UTC offset from the stream so
    # alignment is correct regardless of the recording site's timezone.
    rec_utc  = pd.Timestamp(str(np.ravel(m[BIDSLEEP_START_KEY])[0]), tz="UTC").timestamp()
    first    = float(min(hr_t.min(), mo_t.min()))
    rec_unix = rec_utc + round((first - rec_utc) / 900.0) * 900.0
    hr_t, mo_t = hr_t - rec_unix, mo_t - rec_unix

    if verbose:
        print("  stage codes:", sorted(set(label_raw.astype(int).tolist())),
              "| HR rel [%.0f,%.0f]s | labels span [0,%.0f]s"
              % (hr_t.min(), hr_t.max(), label_t[-1]), flush=True)

    f, l = extract_features_for_night(hr_t, hr_v[:, 0], mo_t, acc_magnitude(mo_v),
                                      label_t, label_raw, BIDSLEEP_STAGE_REMAP)
    if f is None:
        return None
    return (night or mp, robust_normalize_features(f), l)


def load_bidsleep(limit=None):
    """Stream BIDSleep from PhysioNet with parallel fetching.
    First night runs serially as a format sanity check, the rest fan out across workers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    paths = list_files(BIDSLEEP_BASE)
    pset  = set(paths)
    mats  = sorted(p for p in paths if p.endswith(".mat"))
    if limit:
        mats = mats[:limit]
    if not mats:
        return []

    # 1) First night serial → prints the sanity check + validates format end-to-end
    out, done = [], 1
    first = _bidsleep_night(mats[0], pset, verbose=True)
    if first is not None:
        out.append(first)
    print(f"  [1/{len(mats)}] kept {len(out)} nights (warm-up)", flush=True)

    # 2) Remaining nights in parallel
    rest = mats[1:]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {ex.submit(_bidsleep_night, mp, pset): mp for mp in rest}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r is not None:
                    out.append(r)
            except Exception as e:
                print(f"  failed {futs[fut]}: {e}", flush=True)
            done += 1
            if done % 20 == 0 or done == len(mats):
                print(f"  [{done}/{len(mats)}] kept {len(out)} nights", flush=True)

    print(f"BIDSleep: parsed {len(out)} nights (parallel x{NUM_WORKERS})")
    return out


# Smoke test: bidsleep_nights = cached("bidsleep_lim10", lambda: load_bidsleep(limit=10))
bidsleep_nights = cached("bidsleep", load_bidsleep)''')

md("## Data audit")

code('''def audit(name, subjects):
    if not subjects:
        print(f"{name}: EMPTY — check parser/format!"); return
    all_lab = np.concatenate([l for _, _, l in subjects])
    all_feat = np.concatenate([f for _, f, _ in subjects])
    counts = np.bincount(all_lab, minlength=NUM_CLASSES)
    print(f"\\n{name}: {len(subjects)} recordings, {len(all_lab):,} epochs")
    print("  class dist:", dict(zip(STAGE_NAMES, counts.tolist())),
          " (Deep% = %.1f)" % (100*counts[2]/max(1, counts.sum())))
    print("  feature NaN/Inf:", int(np.isnan(all_feat).sum() + np.isinf(all_feat).sum()))
    epcs = [len(l) for _, _, l in subjects]
    print(f"  epochs/recording: min={min(epcs)} median={int(np.median(epcs))} max={max(epcs)}")

audit("BIDSleep (train)", bidsleep_nights)
audit("Walch (test)", walch_subjects)''')

md("## Build datasets (BIDSleep train/val · Walch = held-out test)")

code('''class SeqDataset(Dataset):
    def __init__(self, subjects, seq_len, stride):
        self.items = []
        for _, feats, labs in subjects:
            n = len(labs)
            if n < seq_len:
                continue
            for s in range(0, n - seq_len + 1, stride):
                self.items.append((feats[s:s+seq_len], labs[s:s+seq_len]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, l = self.items[i]
        return torch.from_numpy(f), torch.from_numpy(l)


rng = np.random.default_rng(RANDOM_SEED)
perm = rng.permutation(len(bidsleep_nights))
n_val = max(1, int(len(bidsleep_nights) * VAL_SPLIT))
val_ids = set(perm[:n_val].tolist())
bid_train = [s for i, s in enumerate(bidsleep_nights) if i not in val_ids]
bid_val   = [s for i, s in enumerate(bidsleep_nights) if i in val_ids]

train_ds = SeqDataset(bid_train, SEQ_LEN, TRAIN_STRIDE)
val_ds   = SeqDataset(bid_val,   SEQ_LEN, EVAL_STRIDE)
test_ds  = SeqDataset(walch_subjects, SEQ_LEN, EVAL_STRIDE)
print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test(Walch) {len(test_ds)} sequences")

# Class weights from training labels
tr_lab = np.concatenate([l for _, _, l in bid_train])
counts = np.bincount(tr_lab, minlength=NUM_CLASSES).astype(np.float32)
inv = 1.0 / np.maximum(counts, 1.0)
class_weights = torch.tensor(inv / inv.sum() * NUM_CLASSES, dtype=torch.float32)
print("Train class counts:", dict(zip(STAGE_NAMES, counts.astype(int).tolist())))
print("Class weights:", [round(w, 3) for w in class_weights.tolist()])

pin = torch.cuda.is_available()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=pin, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)''')

md("## Model (temporal-conv → BiLSTM seq2seq)")

code('''class SleepFeatureModel(nn.Module):
    def __init__(self, n_features=NUM_FEATURES, n_classes=NUM_CLASSES):
        super().__init__()
        self.in_norm = nn.LayerNorm(n_features)
        pad = TCN_KERNEL // 2
        self.tcn = nn.Sequential(
            nn.Conv1d(n_features, TCN_CHANNELS, TCN_KERNEL, padding=pad),
            nn.BatchNorm1d(TCN_CHANNELS), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Conv1d(TCN_CHANNELS, TCN_CHANNELS, TCN_KERNEL, padding=pad),
            nn.BatchNorm1d(TCN_CHANNELS), nn.GELU(), nn.Dropout(DROPOUT),
        )
        self.lstm = nn.LSTM(TCN_CHANNELS, LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                            batch_first=True, bidirectional=True,
                            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(LSTM_HIDDEN * 2), nn.Dropout(DROPOUT),
            nn.Linear(LSTM_HIDDEN * 2, n_classes),
        )

    def forward(self, x):                       # x: [B, S, F]
        x = self.in_norm(x)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)   # conv over time → [B, S, C]
        seq, _ = self.lstm(h)
        return self.head(seq)                   # [B, S, n_classes]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_m = SleepFeatureModel().to(device)
print(f"Parameters: {_m.count_parameters():,} | Device: {device}")
with torch.no_grad():
    _o = _m(torch.randn(2, SEQ_LEN, NUM_FEATURES, device=device))
    print("Forward check:", tuple(_o.shape))
del _m, _o''')

md("## Train on BIDSleep")

code('''def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot, n = 0.0, 0
    P, L = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for f, l in loader:
            f, l = f.to(device), l.to(device)
            logits = model(f)
            fl, ll = logits.reshape(-1, NUM_CLASSES), l.reshape(-1)
            loss = criterion(fl, ll)
            if train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step()
            tot += loss.item() * ll.size(0); n += ll.size(0)
            P.append(fl.argmax(-1).cpu().numpy()); L.append(ll.cpu().numpy())
    P, L = np.concatenate(P), np.concatenate(L)
    return (tot / max(1, n),
            f1_score(L, P, average="macro", zero_division=0),
            f1_score(L, P, average="weighted", zero_division=0),
            cohen_kappa_score(L, P), P, L)


model     = SleepFeatureModel().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))   # plain weighted CE (focal γ=0 won earlier)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_f1, patience = 0.0, EARLY_STOP_PATIENCE
history = {k: [] for k in ["tr_loss", "vl_loss", "tr_mf1", "vl_mf1", "vl_wf1", "vl_kappa"]}

for ep in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_mf1, _, _, _, _       = run_epoch(model, train_loader, criterion, optimizer)
    vl_loss, vl_mf1, vl_wf1, vl_k, _, _ = run_epoch(model, val_loader, criterion)
    scheduler.step()
    for k, v in zip(history, [tr_loss, vl_loss, tr_mf1, vl_mf1, vl_wf1, vl_k]):
        history[k].append(v)
    print(f"Ep {ep:3d}/{NUM_EPOCHS} | tr_loss {tr_loss:.3f} mF1 {tr_mf1:.3f} | "
          f"val mF1 {vl_mf1:.3f} wF1 {vl_wf1:.3f} κ {vl_k:.3f} | {time.time()-t0:.1f}s")

    if vl_mf1 > best_f1:
        best_f1, patience = vl_mf1, EARLY_STOP_PATIENCE
        torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                    "val_mf1": vl_mf1, "val_wf1": vl_wf1, "val_kappa": vl_k},
                   CHECKPOINT_DIR / "best_model.pt")
        print(f"  ✓ best val macro-F1 {vl_mf1:.4f}")
    else:
        patience -= 1
        if patience <= 0:
            print(f"Early stop @ {ep}"); break

with open(OUTPUT_DIR / "training_history.json", "w") as fp:
    json.dump(history, fp, indent=2)
print(f"\\nBest BIDSleep val macro-F1: {best_f1:.4f}")''')

md("""## Evaluate on Walch (held-out — the honest generalization number)

Walch was never seen in training. This is what the model would do on a fresh user's watch.""")

code('''ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])

_, te_mf1, te_wf1, te_k, P, L = run_epoch(model, test_loader, nn.CrossEntropyLoss())

print("="*60)
print("WALCH HELD-OUT TEST (4-class)")
print("="*60)
print(f"  macro F1    : {te_mf1:.4f}")
print(f"  weighted F1 : {te_wf1:.4f}   (SLAMSS target ≥ 0.72)")
print(f"  accuracy    : {accuracy_score(L, P):.4f}")
print(f"  Cohen κ     : {te_k:.4f}")

per_f1 = f1_score(L, P, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
per_rc = recall_score(L, P, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
print("\\n  per-class F1 / recall:")
for n, a, b in zip(STAGE_NAMES, per_f1, per_rc):
    flag = "  ← target >0.56" if n == "Deep" else ""
    print(f"    {n:6s} F1={a:.3f}  recall={b:.3f}{flag}")
print("\\n  confusion (rows=true, cols=pred):")
print(confusion_matrix(L, P, labels=list(range(NUM_CLASSES))))

# 3-class view (merge Light+Deep → NREM) for comparison to Walch/SLAMSS 3-class
to3 = {0: 0, 1: 1, 2: 1, 3: 2}
P3 = np.vectorize(to3.get)(P); L3 = np.vectorize(to3.get)(L)
print("\\n3-class (Wake/NREM/REM): "
      f"wF1 {f1_score(L3, P3, average='weighted', zero_division=0):.3f} "
      f"acc {accuracy_score(L3, P3):.3f} κ {cohen_kappa_score(L3, P3):.3f}  (SLAMSS ~0.80 wF1)")''')

md("## Export to ONNX")

code('''import onnx, onnxruntime as ort

export_model = SleepFeatureModel().to("cpu")
export_model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pt",
                                        map_location="cpu", weights_only=True)["model_state_dict"])
export_model.eval()

dummy = torch.randn(1, SEQ_LEN, NUM_FEATURES)
onnx_path = EXPORT_DIR / "sleep_stage_model.onnx"
torch.onnx.export(
    export_model, (dummy,), str(onnx_path),
    input_names=["features"], output_names=["logits"],
    dynamic_axes={"features": {0: "batch", 1: "seq_len"},
                  "logits":   {0: "batch", 1: "seq_len"}},
    opset_version=17, do_constant_folding=True,
)
onnx.checker.check_model(onnx.load(str(onnx_path)))
sess = ort.InferenceSession(str(onnx_path))
diff = np.max(np.abs(export_model(dummy).detach().numpy()
                     - sess.run(None, {"features": dummy.numpy()})[0]))
print(f"ONNX parity max diff: {diff:.6f}")

metadata = {
    "version": "3.0.0",
    "modelType": "tcn_bilstm_seq2seq_hr_actigraphy",
    "task": "sleep_stage_classification_4class",
    "trainData": "BIDSleep (Apple Watch)", "testData": "Walch/Sleep-Accel (held-out)",
    "numClasses": NUM_CLASSES, "stageNames": STAGE_NAMES, "stageMap": APP_STAGE_MAP,
    "input": {
        "features": {
            "shape": ["batch", "seq_len", NUM_FEATURES],
            "names": FEATURE_NAMES,
            "epochSeconds": EPOCH_SEC,
            "recommendedSeqLen": SEQ_LEN,
            "hrWindowSec": HR_WIN_SEC,
            "moveThresholdG": MOVE_THRESH_G,
            "normalization": "per_night_median_mad_per_feature",
        },
    },
    "output": {"logits": {"shape": ["batch", "seq_len", NUM_CLASSES],
                          "note": "argmax last dim → stageMap"}},
    "test": {"walch_macroF1": float(te_mf1), "walch_weightedF1": float(te_wf1),
             "walch_kappa": float(te_k)},
}
with open(EXPORT_DIR / "sleep_model_metadata.json", "w") as fp:
    json.dump(metadata, fp, indent=2)
print(f"Saved {onnx_path} ({onnx_path.stat().st_size/1e3:.1f} KB) + metadata")''')

md("## Training curves")

code('''import matplotlib.pyplot as plt
ep = range(1, len(history["tr_loss"]) + 1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(ep, history["tr_loss"], label="train"); ax[0].plot(ep, history["vl_loss"], label="val")
ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(ep, history["tr_mf1"], label="train macro"); ax[1].plot(ep, history["vl_mf1"], label="val macro")
ax[1].plot(ep, history["vl_wf1"], label="val weighted"); ax[1].axhline(0.72, ls="--", c="gray")
ax[1].set_title("F1"); ax[1].legend()
ax[2].plot(ep, history["vl_kappa"]); ax[2].set_title("Val κ")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150); plt.show()''')

md("""## Output files

| File | Drop into repo at |
|------|-------------------|
| `output/export/sleep_stage_model.onnx`    | `assets/ml/sleep_stage_model.onnx` |
| `output/export/sleep_model_metadata.json` | `assets/ml/sleep_model_metadata.json` |

**Remember:** the 11-feature recipe in metadata must be mirrored exactly in
`services/ai/sleepStageModel.ts` (fed from the Wear OS Health Services HR + accel stream).""")

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
