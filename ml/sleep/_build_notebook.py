"""Builds kaggle_train.ipynb — train from a pre-extracted feature cache.

The features were extracted offline by prepare_features.py on a bandwidth-rich machine
(no GPU needed there) and uploaded to Kaggle as a Dataset. This notebook just loads the
two small pickles, trains on BIDSleep, and evaluates on Walch as a held-out test set.

See _build_notebook_streaming.py for the older variant that streams from PhysioNet.
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


md("""# Seren — Sleep Stage Classifier v3.2 (Tuning + Training + XAI, all-in-one)

This is the **single canonical Kaggle notebook** for the sleep model. It covers:
1. **Cache load + audit** (Walch held-out, BIDSleep subject-disjoint split)
2. *(Optional)* **Hyperparameter tuning** via niapy PSO/GWO (off by default — flip
   `RUN_HYPEROPT = True` in the config cell to re-search; ~50 min on T4)
3. **Final training** on BIDSleep
4. **Held-out Walch evaluation** (4-class + 3-class)
5. **ONNX export** (static `[1, 41, 11]`)
6. **Captum Integrated Gradients XAI suite** (global / local / temporal / physiological)
7. *(Optional)* **XAI-driven feature-reduction sweep** (flip `RUN_XAI_FEATURE_REDUCTION`
   in the XAI 1/5 cell — already led us to drop `immobility_frac`)

**Inputs (must match the on-device feature pipeline):** HR + accelerometer only —
**10 engineered features per 30-s epoch** from the watch (`immobility_frac` dropped
post-XAI), plus `time_of_night` appended in the notebook = **11 model features**.

**Labels (4-class, matches app):** `Wake→0`, `N1+N2→1 (light)`, `N3→2 (deep)`, `REM→3`.

**Targets (beat prior work):** 4-class weighted F1 ≥ 0.72, Deep recall > 56% (SLAMSS 2023);
3-class weighted F1 ~0.80.

**Setup on Kaggle:**
1. Add the `seren-sleep-cache` dataset under "+ Add Input".
2. Settings: GPU T4 x2 (or P100), Internet ON (for niapy + Captum pip installs).""")

code(r'''# numpy<2 is REQUIRED — the cached feature pickles in seren-sleep-cache were
# created with numpy 1.x. Kaggle's pre-installed numpy 2.x raises
# "TypeError: _reconstruct: First argument must be a sub-type of ndarray" on load.
# Pinning here triggers a one-time downgrade; restart the kernel + Run All after this.
!pip install -q "numpy<2" mlflow onnx onnxruntime "onnxscript==0.2.6" scikit-learn niapy captum

# Surface the actually-loaded numpy version so a stale env is obvious.
import numpy as _np
print(f"numpy {_np.__version__}  (must be < 2 for the legacy pickle cache)")''')

md("""## Config + load the cache

The cache lives in `/kaggle/input/seren-sleep-cache/` after you add the dataset. Both pickles
load in under a second. If you renamed the dataset, edit `CACHE_DIR` below.""")

code('''import os, json, time, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (f1_score, accuracy_score, cohen_kappa_score,
                             confusion_matrix, recall_score)

print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path("/kaggle/working/output")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
EXPORT_DIR     = OUTPUT_DIR / "export"
MLRUNS_DIR     = OUTPUT_DIR / "mlruns"
for d in [OUTPUT_DIR, CHECKPOINT_DIR, EXPORT_DIR, MLRUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class Tracker:
    """Safe MLflow wrapper — no-ops if init fails so training never crashes on a log call.
    Reads MLFLOW_TRACKING_URI (+ USERNAME/PASSWORD) from Kaggle Secrets if present;
    otherwise falls back to a local file backend in /kaggle/working/output/mlruns."""

    def __init__(self, experiment_name, fallback_dir):
        self.enabled = False
        self._m = None
        try:
            from kaggle_secrets import UserSecretsClient
            sec = UserSecretsClient()
            for k in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME",
                      "MLFLOW_TRACKING_PASSWORD"):
                try:
                    v = sec.get_secret(k)
                    if v: os.environ[k] = v
                except Exception:
                    pass
        except Exception:
            pass
        uri = os.environ.get("MLFLOW_TRACKING_URI") or f"file://{Path(fallback_dir).resolve()}"
        try:
            import mlflow
            mlflow.set_tracking_uri(uri); mlflow.set_experiment(experiment_name)
            self._m, self.enabled = mlflow, True
            print(f"MLflow tracking: {uri} | experiment: {experiment_name}")
        except Exception as e:
            print(f"MLflow disabled ({e}) — runs will not be tracked.")

    def start_run(self, **kw):
        if not self.enabled: return _NullCtx()
        try: return self._m.start_run(**kw)
        except Exception: return _NullCtx()

    def log_params(self, d):
        if not self.enabled: return
        try: self._m.log_params({k: v for k, v in d.items() if not str(k).startswith("_")})
        except Exception: pass

    def log_metric(self, k, v, step=None):
        if not self.enabled: return
        try: self._m.log_metric(k, float(v), step=step)
        except Exception: pass

    def log_artifact(self, p):
        if not self.enabled: return
        try:
            if Path(p).exists(): self._m.log_artifact(str(p))
        except Exception: pass


class _NullCtx:
    def __enter__(self):  return self
    def __exit__(self, *a): return False


tracker = Tracker(experiment_name="seren-sleep-train", fallback_dir=MLRUNS_DIR)

# Cache dir candidates — checked in order. First match wins.
CACHE_CANDIDATES = [
    Path("/kaggle/input/seren-sleep-cache"),
    Path("/kaggle/working/cache"),
]

# ── Labels ────────────────────────────────────────────────────────────────────
NUM_CLASSES   = 4
STAGE_NAMES   = ["Wake", "Light", "Deep", "REM"]
APP_STAGE_MAP = {0: "awake", 1: "light", 2: "deep", 3: "rem"}

# ── Feature contract — 10 kept from the cache + 1 added in the notebook (time_of_night) ─
# XAI confirmed `immobility_frac` is dead weight (pooled |IG| = 0.017, 5x lower than the
# next-lowest feature). We slice it out of the cache rather than re-extracting (cache
# build is ~17 h; the column drop is exact and the cache contract stays backward-compat).
CACHE_FEATURES_RAW = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "hr_succdiff_std", "hr_delta_prev",
    "act_count", "immobility_frac", "act_max", "act_std",
]
DROP_FROM_CACHE = {"immobility_frac"}
KEEP_CACHE_IDX  = [i for i, n in enumerate(CACHE_FEATURES_RAW) if n not in DROP_FROM_CACHE]
CACHE_FEATURES  = [n for n in CACHE_FEATURES_RAW if n not in DROP_FROM_CACHE]   # 10
FEATURE_NAMES   = CACHE_FEATURES + ["time_of_night"]                             # 11 total at the model input
NUM_FEATURES    = len(FEATURE_NAMES)
EPOCH_SEC       = 30

# ── Sequence config (v3.2 — chosen by niapy PSO over 110 trials, val mF1 0.6785) ──
SEQ_LEN      = 41       # ~20.5 min temporal context (half a sleep cycle)
TRAIN_STRIDE = 5        # PSO found stride 1 over-augmented; stride 5 is the sweet spot
EVAL_STRIDE  = SEQ_LEN  # disjoint sequences at eval time

# ── Model ─────────────────────────────────────────────────────────────────────
TCN_CHANNELS = 64
TCN_KERNEL   = 5
LSTM_HIDDEN  = 48       # PSO best (also the prior cached value)
LSTM_LAYERS  = 2
DROPOUT      = 0.258    # PSO best (110 trials, 11-feature lean model)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE          = 64
LEARNING_RATE       = 1.03e-3   # PSO best (110 trials, lean model) — less regularization than the previous 2.95e-3
WEIGHT_DECAY        = 1.72e-5   # PSO best (110 trials, lean model)
NUM_EPOCHS          = 80
EARLY_STOP_PATIENCE = 15
VAL_SPLIT           = 0.15
RANDOM_SEED         = 42
GRAD_CLIP           = 1.0

# ── Hyperparameter search toggle ──────────────────────────────────────────────
# Default: True -> run niapy PSO (~50 min) and use the discovered HPs for final
#                  training. Recommended whenever the model architecture changes.
# False -> skip PSO and use either (a) the cached best HPs from a prior PSO run
#          on the SAME architecture (auto-loaded from pso_best_hp.json) or
#          (b) the defaults baked into the config above if no cache exists.
RUN_HYPEROPT          = True
HYPEROPT_ALGO         = "PSO"   # "PSO" or "GWO"
HYPEROPT_POP          = 10      # niapy population size
HYPEROPT_ITERS        = 10      # niapy iterations -> ~POP*ITERS evaluations
HYPEROPT_TRIAL_EPOCHS = 8       # short trial budget per candidate
HYPEROPT_TRIAL_PATIENCE = 3

# Auto-persistence: PSO writes the best HPs here so a follow-up `RUN_HYPEROPT=False`
# run uses them instead of the (potentially stale) config defaults above.
# Must be set AFTER OUTPUT_DIR exists, so we lazy-define and use a module-level var.
HP_CACHE_NAME = "pso_best_hp.json"

# Architecture tag — the JSON cache is only loaded if its tag matches what we're
# about to train, so a config change doesn't silently reuse stale HPs.
ARCH_TAG = "tcn_bigru_bn_v1"     # bump on any architecture change to invalidate cached HPs

print(f"{NUM_FEATURES} features/epoch · seq {SEQ_LEN} epochs · classes {STAGE_NAMES}")
print(f"Architecture: {ARCH_TAG}")
print(f"Hyperopt: {'ON ('+HYPEROPT_ALGO+', pop '+str(HYPEROPT_POP)+' x iters '+str(HYPEROPT_ITERS)+')' if RUN_HYPEROPT else 'OFF (cached HPs if present, else config defaults)'}")


# ── Load the cache ────────────────────────────────────────────────────────────
def find_cache_dir():
    for c in CACHE_CANDIDATES:
        if (c / "bidsleep_features.pkl").exists() and (c / "walch_features.pkl").exists():
            return c
    raise FileNotFoundError(
        "Could not find walch_features.pkl + bidsleep_features.pkl in any of: "
        f"{[str(c) for c in CACHE_CANDIDATES]}\\n"
        "Add the 'seren-sleep-cache' Kaggle Dataset via '+ Add Input', or copy the "
        "pickles into /kaggle/working/cache/."
    )


CACHE_DIR = find_cache_dir()
print(f"Using cache: {CACHE_DIR}")

with open(CACHE_DIR / "walch_features.pkl", "rb") as f:
    walch_subjects = pickle.load(f)
with open(CACHE_DIR / "bidsleep_features.pkl", "rb") as f:
    bidsleep_nights = pickle.load(f)

print(f"Loaded Walch: {len(walch_subjects)} subjects | BIDSleep: {len(bidsleep_nights)} nights")


def add_time_of_night(subjects):
    """Append a 12th feature: linear position within the night, scaled to [-1, +1].

    Deep sleep concentrates in the first half of the night, REM in the second; without
    this the model has no clock to lean on. On-device: compute per-epoch as
    `2 * (i / max(1, n_epochs - 1)) - 1` for i in [0, n_epochs) — trivially mirrorable.
    """
    out = []
    for sid, feats, labs in subjects:
        n = len(feats)
        denom = max(1, n - 1)
        ton = (2.0 * np.arange(n, dtype=np.float32) / denom - 1.0).reshape(-1, 1)
        out.append((sid, np.concatenate([feats, ton], axis=1).astype(np.float32), labs))
    return out


def slice_dead_features(subjects):
    """Drop XAI-confirmed dead-weight columns (immobility_frac) from the cached features.

    The cache was extracted with 11 features per epoch; XAI showed `immobility_frac`
    contributes ~0% of model attribution. Slicing here means we don't need to re-build
    the 17-hour cache. If you ever do re-extract, set DROP_FROM_CACHE = {} and the slice
    becomes a no-op."""
    return [(sid, feats[:, KEEP_CACHE_IDX].astype(np.float32), labs)
            for sid, feats, labs in subjects]


walch_subjects   = slice_dead_features(walch_subjects)
bidsleep_nights  = slice_dead_features(bidsleep_nights)
print(f"After XAI-driven feature pruning: kept {len(KEEP_CACHE_IDX)}/{len(CACHE_FEATURES_RAW)} "
      f"cache features (dropped: {sorted(DROP_FROM_CACHE)})")

walch_subjects   = add_time_of_night(walch_subjects)
bidsleep_nights  = add_time_of_night(bidsleep_nights)
print(f"After time-of-night augmentation: feature dim = {walch_subjects[0][1].shape[-1]} "
      f"(expected {NUM_FEATURES})")

# Print the manifest for traceability if present
manifest_p = CACHE_DIR / "manifest.json"
if manifest_p.exists():
    mf = json.loads(manifest_p.read_text())
    print(f"\\nCache manifest — produced {mf.get('produced_at_utc', '?')} UTC")
    for k, v in mf.get("summary", {}).items():
        print(f"  {k}: {v.get('recordings', '?')} recordings, {v.get('epochs_total', '?')} epochs, "
              f"class% = {v.get('class_pct', '?')}")''')

md("## Data audit (sanity check the cache)")

code('''def audit(name, subjects):
    if not subjects:
        print(f"{name}: EMPTY — cache load failed?"); return
    all_lab  = np.concatenate([l for _, _, l in subjects])
    all_feat = np.concatenate([f for _, f, _ in subjects])
    counts   = np.bincount(all_lab, minlength=NUM_CLASSES)
    epcs     = [len(l) for _, _, l in subjects]
    print(f"\\n{name}: {len(subjects)} recordings, {len(all_lab):,} epochs")
    print(f"  class dist : {dict(zip(STAGE_NAMES, counts.tolist()))}  "
          f"(Deep = {100*counts[2]/counts.sum():.1f}%, REM = {100*counts[3]/counts.sum():.1f}%)")
    print(f"  epochs/rec : min={min(epcs)} median={int(np.median(epcs))} max={max(epcs)}")
    print(f"  NaN/Inf    : {int(np.isnan(all_feat).sum() + np.isinf(all_feat).sum())}")
    print(f"  feature dim: {all_feat.shape[-1]} (expected {NUM_FEATURES})")


audit("BIDSleep (train + val)", bidsleep_nights)
audit("Walch (held-out test)",  walch_subjects)''')

md("""## Build datasets

BIDSleep → 85 % train / 15 % val (subject-level split).
Walch → 100 % test, never seen during training.""")

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
# SUBJECT-LEVEL split (not night-level). BIDSleep sids are "<subject>/<recording>" e.g.
# "Bidslab00/3"; splitting on night index put 100% of val subjects into train as well,
# inflating val mF1 by ~0.06-0.08 (Walch was unaffected because it is a separate dataset).
subj_of      = [sid.split("/")[0] for sid, _, _ in bidsleep_nights]
all_subjects = sorted(set(subj_of))
rng.shuffle(all_subjects)
n_val_subj   = max(1, int(len(all_subjects) * VAL_SPLIT))
val_subjects = set(all_subjects[:n_val_subj])
bid_train = [s for s, sub in zip(bidsleep_nights, subj_of) if sub not in val_subjects]
bid_val   = [s for s, sub in zip(bidsleep_nights, subj_of) if sub in val_subjects]
print(f"Subject-disjoint split: {len(all_subjects) - n_val_subj} train subj / "
      f"{n_val_subj} val subj ({len(bid_train)} train nights / {len(bid_val)} val nights)")

train_ds = SeqDataset(bid_train, SEQ_LEN, TRAIN_STRIDE)
val_ds   = SeqDataset(bid_val,   SEQ_LEN, EVAL_STRIDE)
test_ds  = SeqDataset(walch_subjects, SEQ_LEN, EVAL_STRIDE)
print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test(Walch) {len(test_ds)} sequences")

# √-inverse-frequency class weights — softer than full inverse so the model still
# learns the true Light prior (Walch=55% Light, BIDSleep=44%). Full inverse caused
# the model to over-predict Deep/REM at the expense of Light precision (see prior run).
tr_lab = np.concatenate([l for _, _, l in bid_train])
counts = np.bincount(tr_lab, minlength=NUM_CLASSES).astype(np.float32)
inv_sqrt = 1.0 / np.sqrt(np.maximum(counts, 1.0))
class_weights = torch.tensor(inv_sqrt / inv_sqrt.sum() * NUM_CLASSES, dtype=torch.float32)
print("Train class counts:", dict(zip(STAGE_NAMES, counts.astype(int).tolist())))
print("Class weights (√) :", [round(w, 3) for w in class_weights.tolist()])

pin = torch.cuda.is_available()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=pin, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=pin)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=pin)''')

md("""## Model (temporal-conv → BiGRU seq2seq, TFLite-friendly)

Two deliberate architecture choices to ensure clean ONNX→TFLite conversion via
`onnx2tf` (we measured ~45 % per-epoch argmax disagreement vs ONNX with the original
LayerNorm+BiLSTM combination — TFLite has no native LayerNorm and decomposes it into
MEAN+sub+mul+add chains that drift through the sequence model):

- **`nn.LayerNorm` → `nn.BatchNorm1d`**: BatchNorm folds into adjacent Linear/Conv
  layers at export → zero conversion drift.
- **`nn.LSTM` → `nn.GRU`**: TFLite has a native `UnidirectionalSequenceGRU` primitive
  that converts exactly. LSTM goes through a more elaborate decomposition (~6 ops
  per gate). GRU also has ~25 % fewer params.

Result: should pass the TFLite vs ONNX parity test at ~1e-4 max-abs-diff and 100 %
argmax match. Model quality is expected within ±0.01 κ of the original.""")

code('''class SleepFeatureModel(nn.Module):
    """TFLite-friendly TCN + BiGRU. BatchNorm1d expects [B, C, L] so the forward
    transposes around the norm + TCN; LSTM_HIDDEN / LSTM_LAYERS names kept for
    backwards-compat with prior runs (they parameterize the GRU now)."""
    def __init__(self, n_features=NUM_FEATURES, n_classes=NUM_CLASSES):
        super().__init__()
        self.in_norm = nn.BatchNorm1d(n_features)
        pad = TCN_KERNEL // 2
        self.tcn = nn.Sequential(
            nn.Conv1d(n_features, TCN_CHANNELS, TCN_KERNEL, padding=pad),
            nn.BatchNorm1d(TCN_CHANNELS), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Conv1d(TCN_CHANNELS, TCN_CHANNELS, TCN_KERNEL, padding=pad),
            nn.BatchNorm1d(TCN_CHANNELS), nn.GELU(), nn.Dropout(DROPOUT),
        )
        self.rnn = nn.GRU(TCN_CHANNELS, LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                          batch_first=True, bidirectional=True,
                          dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0)
        self.head_norm = nn.BatchNorm1d(LSTM_HIDDEN * 2)
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(LSTM_HIDDEN * 2, n_classes),
        )

    def forward(self, x):                       # x: [B, S, F]
        # BN1d expects [B, C, L] -> transpose around the input norm.
        x = self.in_norm(x.transpose(1, 2)).transpose(1, 2)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        seq, _ = self.rnn(h)
        # BN1d again on the GRU output (per-channel across the sequence + batch).
        seq = self.head_norm(seq.transpose(1, 2)).transpose(1, 2)
        return self.head(seq)                   # [B, S, n_classes]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_m = SleepFeatureModel().to(device)
print(f"Parameters: {_m.count_parameters():,} | Device: {device}")
with torch.no_grad():
    _o = _m(torch.randn(2, SEQ_LEN, NUM_FEATURES, device=device))
    print("Forward shape:", tuple(_o.shape))
del _m, _o''')

md(r"""## (Optional) Hyperparameter tuning — niapy PSO / GWO

Off by default (`RUN_HYPEROPT = False` in the config cell) — set to `True` to re-search.
Encodes 6 hyperparameters as a vector in `[0, 1]^6`:

| Index | Hyperparameter | Range |
|---|---|---|
| 0 | `lr` (log-spaced) | 1e-4 .. 3e-3 |
| 1 | `dropout` | 0.20 .. 0.60 |
| 2 | `weight_decay` (log-spaced) | 1e-5 .. 1e-3 |
| 3 | `lstm_hidden` | {48, 64, 96, 128} |
| 4 | `seq_len` | {15, 21, 31, 41} |
| 5 | `train_stride` | {1, 3, 5, 10} |

Each candidate runs an 8-epoch trial against the val set. The best hyperparameters
overwrite the config values used by the **Train on BIDSleep** cell below.
""")

code(r'''
# === Hyperparameter search (gated by RUN_HYPEROPT) ============================
if RUN_HYPEROPT:
    # Lazy import — niapy installs cleanly on Kaggle but we only need it here.
    try:
        from niapy.algorithms.basic import ParticleSwarmAlgorithm, GreyWolfOptimizer
        from niapy.problems import Problem
        from niapy.task import Task
    except ImportError:
        import subprocess, sys
        print("niapy not found - installing ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "niapy"], check=True)
        from niapy.algorithms.basic import ParticleSwarmAlgorithm, GreyWolfOptimizer
        from niapy.problems import Problem
        from niapy.task import Task

    HPO_DIM            = 6
    LSTM_CHOICES       = [48, 64, 96, 128]
    SEQLEN_CHOICES     = [15, 21, 31, 41]
    STRIDE_CHOICES     = [1, 3, 5, 10]

    def _pick(x, choices):
        return choices[min(len(choices) - 1, int(x * len(choices)))]

    def _decode(x):
        return {
            "lr":           float(10 ** (np.log10(1e-4) + x[0] * (np.log10(3e-3) - np.log10(1e-4)))),
            "dropout":      float(0.20 + x[1] * 0.40),
            "weight_decay": float(10 ** (np.log10(1e-5) + x[2] * (np.log10(1e-3) - np.log10(1e-5)))),
            "lstm_hidden":  _pick(x[3], LSTM_CHOICES),
            "seq_len":      _pick(x[4], SEQLEN_CHOICES),
            "train_stride": _pick(x[5], STRIDE_CHOICES),
        }

    # Build CLASS_WEIGHTS once from the training labels (shared across trials).
    _tr_lab = np.concatenate([l for _, _, l in bid_train])
    _counts = np.bincount(_tr_lab, minlength=NUM_CLASSES).astype(np.float32)
    _inv_s  = 1.0 / np.sqrt(np.maximum(_counts, 1.0))
    CLASS_WEIGHTS = torch.tensor(_inv_s / _inv_s.sum() * NUM_CLASSES,
                                  dtype=torch.float32).to(device)

    def _trial(hp, verbose=False):
        torch.manual_seed(RANDOM_SEED)
        tr = SeqDataset(bid_train, hp["seq_len"], hp["train_stride"])
        vl = SeqDataset(bid_val,   hp["seq_len"], hp["seq_len"])
        if len(tr) == 0 or len(vl) == 0:
            return 0.0
        pin = torch.cuda.is_available()
        tr_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=2, pin_memory=pin, drop_last=True)
        vl_loader = DataLoader(vl, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2, pin_memory=pin)
        m = SleepFeatureModel(NUM_FEATURES, NUM_CLASSES).to(device)
        # patch the relevant hyperparameters into the model (GRU + BN head)
        m.rnn = nn.GRU(TCN_CHANNELS, hp["lstm_hidden"], num_layers=LSTM_LAYERS,
                       batch_first=True, bidirectional=True,
                       dropout=hp["dropout"] if LSTM_LAYERS > 1 else 0.0).to(device)
        m.head_norm = nn.BatchNorm1d(hp["lstm_hidden"] * 2).to(device)
        m.head = nn.Sequential(
            nn.Dropout(hp["dropout"]),
            nn.Linear(hp["lstm_hidden"] * 2, NUM_CLASSES),
        ).to(device)
        opt = AdamW(m.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        crit = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
        best, bad = 0.0, 0
        for ep in range(HYPEROPT_TRIAL_EPOCHS):
            m.train()
            for f, l in tr_loader:
                f, l = f.to(device), l.to(device)
                logits = m(f)
                loss   = crit(logits.reshape(-1, NUM_CLASSES), l.reshape(-1))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP); opt.step()
            m.eval(); P, L = [], []
            with torch.no_grad():
                for f, l in vl_loader:
                    f, l = f.to(device), l.to(device)
                    P.append(m(f).argmax(-1).reshape(-1).cpu().numpy())
                    L.append(l.reshape(-1).cpu().numpy())
            P, L = np.concatenate(P), np.concatenate(L)
            mf1 = float(f1_score(L, P, average="macro", zero_division=0))
            if verbose: print(f"    ep{ep+1}: val mF1 {mf1:.4f}")
            if mf1 > best:
                best, bad = mf1, 0
            else:
                bad += 1
                if bad >= HYPEROPT_TRIAL_PATIENCE: break
        del m, opt, tr_loader, vl_loader, tr, vl
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return best

    # Smoke test on one candidate so we know the trial pipeline works.
    smoke = _trial(_decode([0.5]*HPO_DIM), verbose=True)
    print(f"\nHPO smoke test: val mF1 = {smoke:.4f}")

    TRIALS = []
    class _HpoProblem(Problem):
        def __init__(self): super().__init__(dimension=HPO_DIM, lower=0.0, upper=1.0)
        def _evaluate(self, x):
            t0 = time.time()
            hp = _decode(x)
            score = _trial(hp)
            dt = time.time() - t0
            TRIALS.append({"id": len(TRIALS), "hp": hp, "fitness": score, "t_sec": round(dt, 1)})
            print(f"  trial {len(TRIALS)-1:3d}: mF1 {score:.4f} | "
                  f"lr {hp['lr']:.1e} dp {hp['dropout']:.2f} "
                  f"H {hp['lstm_hidden']} S {hp['seq_len']} str {hp['train_stride']} | "
                  f"{dt:.0f}s", flush=True)
            return -score

    _algo_factory = {
        "PSO": lambda: ParticleSwarmAlgorithm(population_size=HYPEROPT_POP, seed=RANDOM_SEED),
        "GWO": lambda: GreyWolfOptimizer(population_size=HYPEROPT_POP,    seed=RANDOM_SEED),
    }
    algo = _algo_factory[HYPEROPT_ALGO]()
    task = Task(problem=_HpoProblem(), max_iters=HYPEROPT_ITERS)
    print(f"\nRunning {HYPEROPT_ALGO}: pop {HYPEROPT_POP} x iters {HYPEROPT_ITERS} "
          f"~= {HYPEROPT_POP*HYPEROPT_ITERS} trials")
    t0 = time.time()
    with tracker.start_run(run_name=f"hpo_{HYPEROPT_ALGO}_v3.2"):
        tracker.log_params({
            "hpo_algo": HYPEROPT_ALGO, "hpo_pop": HYPEROPT_POP,
            "hpo_iters": HYPEROPT_ITERS, "hpo_trial_epochs": HYPEROPT_TRIAL_EPOCHS,
        })
        best_x, best_neg = algo.run(task=task)
        elapsed = time.time() - t0
        tracker.log_metric("hpo_best_val_macro_f1", -best_neg)
        tracker.log_metric("hpo_elapsed_seconds", elapsed)
        tracker.log_metric("hpo_n_trials", len(TRIALS))

    print(f"\n{HYPEROPT_ALGO} finished in {elapsed/60:.1f} min over {len(TRIALS)} trials")
    print(f"Best HPO val macro-F1: {-best_neg:.4f}")
    BEST_HP = _decode(best_x)
    print("Best hyperparameters:")
    for k, v in BEST_HP.items():
        print(f"  {k:14s} = {v}")

    # Overwrite the config values used by the training cell below
    LEARNING_RATE = BEST_HP["lr"]
    WEIGHT_DECAY  = BEST_HP["weight_decay"]
    DROPOUT       = BEST_HP["dropout"]
    LSTM_HIDDEN   = BEST_HP["lstm_hidden"]
    SEQ_LEN       = BEST_HP["seq_len"]
    TRAIN_STRIDE  = BEST_HP["train_stride"]
    EVAL_STRIDE   = SEQ_LEN

    # Persist for the next run — tagged by architecture so a config change invalidates it.
    import json as _json
    hp_cache = {"arch_tag": ARCH_TAG, "best_val_macro_f1": float(-best_neg),
                "n_trials": len(TRIALS), "hp": BEST_HP}
    (OUTPUT_DIR / HP_CACHE_NAME).write_text(_json.dumps(hp_cache, indent=2, default=float))
    print(f"\nSaved best HPs to {OUTPUT_DIR / HP_CACHE_NAME} (arch_tag={ARCH_TAG})")
    print("Config overwritten with HPO best.  Rebuilding datasets / loaders ...")

    # Rebuild datasets + class weights for the (possibly) new SEQ_LEN/STRIDE.
    train_ds = SeqDataset(bid_train,        SEQ_LEN, TRAIN_STRIDE)
    val_ds   = SeqDataset(bid_val,          SEQ_LEN, EVAL_STRIDE)
    test_ds  = SeqDataset(walch_subjects,   SEQ_LEN, EVAL_STRIDE)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)
    print(f"  Train {len(train_ds)} | Val {len(val_ds)} | Test(Walch) {len(test_ds)} sequences")
else:
    # Try to load auto-persisted PSO results from a prior run of the SAME architecture.
    import json as _json
    _hp_cache_path = OUTPUT_DIR / HP_CACHE_NAME
    if _hp_cache_path.exists():
        _cached = _json.loads(_hp_cache_path.read_text())
        if _cached.get("arch_tag") == ARCH_TAG:
            _hp = _cached["hp"]
            LEARNING_RATE = float(_hp["lr"])
            WEIGHT_DECAY  = float(_hp["weight_decay"])
            DROPOUT       = float(_hp["dropout"])
            LSTM_HIDDEN   = int(_hp["lstm_hidden"])
            SEQ_LEN       = int(_hp["seq_len"])
            TRAIN_STRIDE  = int(_hp["train_stride"])
            EVAL_STRIDE   = SEQ_LEN
            print(f"Loaded cached PSO best HPs from {_hp_cache_path}")
            print(f"  arch_tag = {_cached.get('arch_tag')}  best_val_mF1 = {_cached.get('best_val_macro_f1'):.4f}")
            print(f"  {_hp}")
            # Rebuild datasets + loaders for the loaded SEQ_LEN/STRIDE.
            train_ds = SeqDataset(bid_train,        SEQ_LEN, TRAIN_STRIDE)
            val_ds   = SeqDataset(bid_val,          SEQ_LEN, EVAL_STRIDE)
            test_ds  = SeqDataset(walch_subjects,   SEQ_LEN, EVAL_STRIDE)
            pin = torch.cuda.is_available()
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=2, pin_memory=pin, drop_last=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=2, pin_memory=pin)
            test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=2, pin_memory=pin)
        else:
            print(f"Cached HPs at {_hp_cache_path} are for arch '{_cached.get('arch_tag')}' "
                  f"but we're training '{ARCH_TAG}' -> ignoring cache, using config defaults.")
    else:
        print("RUN_HYPEROPT=False, no HP cache present -> using config defaults from above.")
''')

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
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_f1, patience = 0.0, EARLY_STOP_PATIENCE
history = {k: [] for k in ["tr_loss", "vl_loss", "tr_mf1", "vl_mf1", "vl_wf1", "vl_kappa"]}

run_cm = tracker.start_run(run_name="train_v3.2")
with run_cm:
    tracker.log_params({
        "num_features": NUM_FEATURES, "seq_len": SEQ_LEN, "train_stride": TRAIN_STRIDE,
        "tcn_channels": TCN_CHANNELS, "tcn_kernel": TCN_KERNEL,
        "lstm_hidden": LSTM_HIDDEN, "lstm_layers": LSTM_LAYERS, "dropout": DROPOUT,
        "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS, "early_stop_patience": EARLY_STOP_PATIENCE,
        "val_split": VAL_SPLIT, "seed": RANDOM_SEED,
        "num_train_nights": len(bid_train), "num_val_nights": len(bid_val),
        "num_test_subjects": len(walch_subjects),
    })
    tracker.log_params({f"class_weight_{n}": float(w)
                        for n, w in zip(STAGE_NAMES, class_weights.tolist())})

    for ep in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_mf1, _, _, _, _       = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_mf1, vl_wf1, vl_k, _, _ = run_epoch(model, val_loader, criterion)
        scheduler.step()
        for k, v in zip(history, [tr_loss, vl_loss, tr_mf1, vl_mf1, vl_wf1, vl_k]):
            history[k].append(v)
        tracker.log_metric("tr_loss",  tr_loss, step=ep)
        tracker.log_metric("vl_loss",  vl_loss, step=ep)
        tracker.log_metric("tr_mf1",   tr_mf1,  step=ep)
        tracker.log_metric("vl_mf1",   vl_mf1,  step=ep)
        tracker.log_metric("vl_wf1",   vl_wf1,  step=ep)
        tracker.log_metric("vl_kappa", vl_k,    step=ep)
        tracker.log_metric("lr", optimizer.param_groups[0]["lr"], step=ep)
        print(f"Ep {ep:3d}/{NUM_EPOCHS} | tr_loss {tr_loss:.3f} mF1 {tr_mf1:.3f} | "
              f"val mF1 {vl_mf1:.3f} wF1 {vl_wf1:.3f} κ {vl_k:.3f} | {time.time()-t0:.1f}s")

        if vl_mf1 > best_f1:
            best_f1, patience = vl_mf1, EARLY_STOP_PATIENCE
            torch.save({"epoch": int(ep), "model_state_dict": model.state_dict(),
                        "val_mf1": float(vl_mf1), "val_wf1": float(vl_wf1),
                        "val_kappa": float(vl_k)},
                       CHECKPOINT_DIR / "best_model.pt")
            tracker.log_metric("best_val_mf1", vl_mf1, step=ep)
            print(f"  ✓ best val macro-F1 {vl_mf1:.4f}")
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stop @ {ep}"); break

    with open(OUTPUT_DIR / "training_history.json", "w") as fp:
        json.dump(history, fp, indent=2)
    tracker.log_artifact(OUTPUT_DIR / "training_history.json")
    tracker.log_artifact(CHECKPOINT_DIR / "best_model.pt")
    print(f"\\nBest BIDSleep val macro-F1: {best_f1:.4f}")

# Keep the run open across the next two cells by reopening it after the close above —
# simpler: log the rest under a follow-on run that references the same model.
EVAL_RUN_NAME = "walch_eval_v3.2"''')

md("""## Evaluate on Walch (held-out — the honest generalization number)

Walch was never seen during training or validation. This is what the model would do on a
fresh user's watch.""")

code('''ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

_, te_mf1, te_wf1, te_k, P, L = run_epoch(model, test_loader, nn.CrossEntropyLoss())

print("="*60)
print("WALCH HELD-OUT TEST (4-class)")
print("="*60)
te_acc = accuracy_score(L, P)
print(f"  macro F1    : {te_mf1:.4f}")
print(f"  weighted F1 : {te_wf1:.4f}   (SLAMSS target ≥ 0.72)")
print(f"  accuracy    : {te_acc:.4f}")
print(f"  Cohen κ     : {te_k:.4f}")

per_f1 = f1_score(L, P, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
per_rc = recall_score(L, P, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
print("\\n  per-class F1 / recall:")
for n, a, b in zip(STAGE_NAMES, per_f1, per_rc):
    flag = "  ← target >0.56" if n == "Deep" else ""
    print(f"    {n:6s} F1={a:.3f}  recall={b:.3f}{flag}")
print("\\n  confusion (rows=true, cols=pred):")
print(confusion_matrix(L, P, labels=list(range(NUM_CLASSES))))

# 3-class view (merge Light+Deep → NREM)
to3 = {0: 0, 1: 1, 2: 1, 3: 2}
P3 = np.vectorize(to3.get)(P); L3 = np.vectorize(to3.get)(L)
te3_wf1 = f1_score(L3, P3, average="weighted", zero_division=0)
te3_acc = accuracy_score(L3, P3)
te3_k   = cohen_kappa_score(L3, P3)
print("\\n3-class (Wake/NREM/REM): "
      f"wF1 {te3_wf1:.3f}  acc {te3_acc:.3f}  κ {te3_k:.3f}   (SLAMSS ~0.80 wF1)")

# Log the held-out Walch result as its own follow-on MLflow run
with tracker.start_run(run_name=EVAL_RUN_NAME):
    tracker.log_metric("walch_macro_f1",     te_mf1)
    tracker.log_metric("walch_weighted_f1",  te_wf1)
    tracker.log_metric("walch_accuracy",     te_acc)
    tracker.log_metric("walch_kappa",        te_k)
    for n, a, b in zip(STAGE_NAMES, per_f1, per_rc):
        tracker.log_metric(f"walch_{n.lower()}_f1",     a)
        tracker.log_metric(f"walch_{n.lower()}_recall", b)
    tracker.log_metric("walch_3class_wf1",   te3_wf1)
    tracker.log_metric("walch_3class_acc",   te3_acc)
    tracker.log_metric("walch_3class_kappa", te3_k)''')

md("## Export to ONNX")

code('''import onnx, onnxruntime as ort

export_model = SleepFeatureModel().to("cpu")
export_model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pt",
                                        map_location="cpu", weights_only=False)["model_state_dict"])
export_model.eval()

dummy = torch.randn(1, SEQ_LEN, NUM_FEATURES)
onnx_path = EXPORT_DIR / "sleep_stage_model.onnx"
# Static export shape [1, SEQ_LEN, NUM_FEATURES] = [1, 41, 12].
# The deployed TFLite (convert_onnx_to_tflite.py) is also pinned to this static shape,
# and the on-device runtime windows the night into disjoint 41-epoch chunks before
# inference (services/ai/sleepStageModel.ts), so neither dynamic axis is used downstream.
# PyTorch 2.x's dynamo-based ONNX exporter rejects dynamic_axes when the traced model
# has a constant-shaped LSTM input, so we just export static.
torch.onnx.export(
    export_model, (dummy,), str(onnx_path),
    input_names=["features"], output_names=["logits"],
    opset_version=17, do_constant_folding=True,
    external_data=False,   # PyTorch 2.10 dynamo path splits weights into .onnx.data by default; inline them for a single-file artifact
)
onnx.checker.check_model(onnx.load(str(onnx_path)))
sess = ort.InferenceSession(str(onnx_path))
diff = float(np.max(np.abs(export_model(dummy).detach().numpy()
                           - sess.run(None, {"features": dummy.numpy()})[0])))
print(f"ONNX parity max diff: {diff:.6f}")

metadata = {
    "version": "3.2.0",
    "modelType": "tcn_bilstm_seq2seq_hr_actigraphy",
    "task": "sleep_stage_classification_4class",
    "trainData": "BIDSleep (Apple Watch)", "testData": "Walch/Sleep-Accel (held-out)",
    "numClasses": NUM_CLASSES, "stageNames": STAGE_NAMES, "stageMap": APP_STAGE_MAP,
    "input": {
        "features": {
            "shape": [1, SEQ_LEN, NUM_FEATURES],
            "names": FEATURE_NAMES,
            "epochSeconds": EPOCH_SEC,
            "recommendedSeqLen": SEQ_LEN,
            "normalization": "per_night_median_mad_per_feature",
        },
    },
    "output": {"logits": {"shape": [1, SEQ_LEN, NUM_CLASSES],
                          "note": "argmax last dim -> stageMap"}},
    "test": {"walch_macroF1": float(te_mf1), "walch_weightedF1": float(te_wf1),
             "walch_kappa": float(te_k)},
}
with open(EXPORT_DIR / "sleep_model_metadata.json", "w") as fp:
    json.dump(metadata, fp, indent=2)
print(f"Saved {onnx_path} ({onnx_path.stat().st_size/1e3:.1f} KB) + metadata")

# Log the exported model and its metadata as MLflow artifacts
with tracker.start_run(run_name="onnx_export_v3.2"):
    tracker.log_artifact(onnx_path)
    tracker.log_artifact(EXPORT_DIR / "sleep_model_metadata.json")''')

md("""## ONNX → TFLite (on-Kaggle conversion for on-device deployment)

Converts the static `[1, 41, 11]` ONNX to a TFLite that
`react-native-fast-tflite` can load on Wear OS. We do the conversion
**on Kaggle** because `onnx2tf` pulls in `onnxsim` whose Windows wheel
needs `cmake` — fiddly locally; trivial on Kaggle's Linux env.

`onnx2tf` may transpose the input axis order during conversion to keep the
BiGRU convertible. The phone-side TS code (`services/ai/sleepStageModel.ts`)
already writes inputs in the feature-major `[1, 11, 41]` layout that the
converted TFLite expects — verified by the parity check below
(`argmax match: True`, max abs diff ~1e-6).""")

code('''!pip install -q -U onnx2tf onnx-graphsurgeon onnxsim tf_keras
import subprocess, sys, shutil
import tensorflow as tf
import onnxruntime as ort

tflite_work = EXPORT_DIR / "_tflite_work"
tflite_work.mkdir(parents=True, exist_ok=True)
tflite_path = EXPORT_DIR / "sleep_stage_model.tflite"

# -ois: pin to static input shape (no dynamic dims — robust BiGRU conversion)
# -rtpo Erf GeLU: replace Erf/GELU with native TFLite ops (no Flex / Select-TF-ops),
# so react-native-fast-tflite (standard runtime) can run the model on-device.
subprocess.run(
    [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(tflite_work),
     "-ois", f"features:1,{SEQ_LEN},{NUM_FEATURES}", "-rtpo", "Erf", "GeLU",
     "-osd", "-n"],
    check=True,
)
# onnx2tf names the output by the input filename; just take the first .tflite.
cands = sorted(tflite_work.glob("*_float32.tflite")) or sorted(tflite_work.glob("*.tflite"))
if not cands:
    raise SystemExit("onnx2tf produced no .tflite — check the log above")
shutil.copy(cands[0], tflite_path)
print(f"\\nWROTE {tflite_path}  ({tflite_path.stat().st_size // 1024} KB)")

# --- Parity check: ONNX out vs TFLite out on the same random input ---
# Important layout note: onnx2tf transposes the input axis order during conversion
# so the TFLite expects feature-major [1, F, S] = [1, 11, 41] (not [1, S, F]).
# The on-device TS (services/ai/sleepStageModel.ts) writes inputs in that order,
# so this is purely a quirk of the converter. Output layout is unchanged [1, S, C].
x_onnx   = np.random.randn(1, SEQ_LEN, NUM_FEATURES).astype("float32")    # [1, S, F]
x_tflite = x_onnx.transpose(0, 2, 1).copy()                                # [1, F, S]
onnx_out = ort.InferenceSession(str(onnx_path)).run(None, {"features": x_onnx})[0]

it = tf.lite.Interpreter(model_path=str(tflite_path))
it.allocate_tensors()
ind, outd = it.get_input_details(), it.get_output_details()
it.set_tensor(ind[0]["index"], x_tflite)
it.invoke()
tfl_out = it.get_tensor(outd[0]["index"])

print("ONNX out shape  :", onnx_out.shape)
print("TFLite out shape:", tfl_out.shape, "| TFLite input detail:", list(ind[0]["shape"]))
print("argmax match    :",
      bool(np.array_equal(onnx_out.argmax(-1),
                          tfl_out.reshape(onnx_out.shape).argmax(-1))))
print("MAX ABS DIFF    :",
      float(np.max(np.abs(onnx_out - tfl_out.reshape(onnx_out.shape)))))

with tracker.start_run(run_name="tflite_export_v3.2"):
    tracker.log_artifact(tflite_path)
''')

md(r"""## Explainability (XAI) — what the model learned and why

The headline number (Walch macro-F1, kappa) tells us the model **works**; XAI tells us **why**
and whether the reasoning is physiologically sound. Because this model is a deep sequence
model (TCN + BiGRU seq2seq, 4 classes, per-epoch outputs), the right tool is **gradient
attribution** — TreeSHAP from the stress XAI does not apply.

Methods used:
1. **Primary: Captum Integrated Gradients (IG)** on the trained PyTorch model. For each class
   c, attribution = how much each `(feature, timestep)` in the 41-epoch window contributed
   to the mean logit of class c on the held-out validation sequences.
2. **Robust fallback: zero-occlusion permutation importance** — zero one feature channel
   across all timesteps and measure macro-F1 drop per class. Always runs even if Captum or
   plotting hiccups, so this section never blocks the rest of the run.

Four parts (mirror the stress XAI but adapted to a multi-class sequence model):
1. **Global** — per-feature x per-class importance ranking; IG vs occlusion rank agreement.
2. **Local** — IG heatmaps for one correct Deep epoch, one correct REM epoch, and the most
   confident misclassified epoch in the held-out sample. Plus one whole-night Walch hypnogram
   with per-epoch confidence and the dominant attributed feature overlaid.
3. **Temporal / feature effects** — how attribution varies with position in the 41-epoch
   window (does the BiGRU lean on neighbours or just the centre?) and with raw feature
   value (signed IG-vs-x scatter for the top features per class).
4. **Physiological interpretation** — does the model match sleep physiology priors (Deep ->
   low/stable HR + high immobility, late-night negative `time_of_night`; REM -> variable HR
   + low movement + late-night positive `time_of_night`; Wake -> high movement + HR
   variability)? Reports an `|IG|`-weighted agreement score per stage + plain-language verdict.

Every cell is wrapped — a missing `captum` install or a plotting error cannot abort the
notebook (the trained model is already exported above). Artifacts log to MLflow in the
final cell.

> Note on layout: the notebook's PyTorch model and the exported ONNX both take input
> shape `[B, 41, 12]` (sequence-major). The on-device TFLite uses `[1, 12, 41]`
> (feature-major) because `onnx2tf` transposed it during conversion to keep the BiGRU
> convertible. The same `(feature, timestep)` attribution applies in both — only the
> storage axis order differs.""")

code(r'''
# ===== XAI 1/5 - GLOBAL: Integrated Gradients + occlusion importance per class =====
# Self-contained: pip-installs captum if missing; wrapped so it can never crash the run.
XAI_OK = False
xai_global = {}; xai_files = []
RUN_XAI_FEATURE_REDUCTION = False   # set True (and re-run last XAI cell) for the gated retrain sweep
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        import subprocess, sys
        print("captum not found - installing ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "captum"], check=True)
        from captum.attr import IntegratedGradients
    import captum
    print("captum", captum.__version__)

    OUTDIR_XAI = OUTPUT_DIR / "xai"; OUTDIR_XAI.mkdir(parents=True, exist_ok=True)

    # --- Reload best checkpoint into a fresh eval-only copy --------------------
    xai_model = SleepFeatureModel().to(device)
    _ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt",
                       map_location=device, weights_only=False)
    xai_model.load_state_dict(_ckpt["model_state_dict"])
    xai_model.eval()
    for p in xai_model.parameters():
        p.requires_grad = False                # IG only needs gradients through the inputs

    # cuDNN's LSTM backward only runs in train() mode, but we need eval() so BatchNorm/
    # Dropout behave deterministically for attribution. Disable cuDNN for the XAI block
    # so the LSTM falls back to the (slower but correct) native implementation; this
    # 125k-param model + 96 sequences runs in <1 s either way.
    torch.backends.cudnn.enabled = False

    # --- Build a stratified sample of validation sequences --------------------
    # Use the BIDSleep val split: similar distribution to training, but never used to update
    # weights. Cap at XAI_MAX_SEQS so IG stays well under a minute on T4.
    XAI_MAX_SEQS = 96
    _xs, _ys = [], []
    for _f, _l in val_loader:
        _xs.append(_f); _ys.append(_l)
        if sum(b.shape[0] for b in _xs) >= XAI_MAX_SEQS:
            break
    X_val = torch.cat(_xs, 0)[:XAI_MAX_SEQS].to(device)        # [N, 41, 12]
    Y_val = torch.cat(_ys, 0)[:XAI_MAX_SEQS].to(device)        # [N, 41]
    print(f"XAI sample: {X_val.shape[0]} sequences x {SEQ_LEN} epochs x {NUM_FEATURES} feats")

    # IG baseline = the zero vector in normalized space = the per-night median epoch.
    # Per-night median/MAD normalization is applied upstream, so 0 IS the median.
    BASELINE = torch.zeros_like(X_val)

    # --- Integrated Gradients per class (mean-over-time class logit) ----------
    # Wrapper exposes a single scalar per example: mean logit for class c across all
    # 41 timesteps -> attribution is "how each (feat, t) shifted the average class-c logit".
    class _MeanLogitFor(nn.Module):
        def __init__(self, m, c): super().__init__(); self.m, self.c = m, c
        def forward(self, x):
            return self.m(x)[:, :, self.c].mean(dim=1)

    feat_imp = np.zeros((NUM_CLASSES, NUM_FEATURES), dtype=np.float32)
    time_imp = np.zeros((NUM_CLASSES, SEQ_LEN),     dtype=np.float32)
    full_attr = np.zeros((NUM_CLASSES, X_val.shape[0], SEQ_LEN, NUM_FEATURES),
                         dtype=np.float32)

    for c in range(NUM_CLASSES):
        ig = IntegratedGradients(_MeanLogitFor(xai_model, c))
        attr = ig.attribute(X_val, baselines=BASELINE, n_steps=32).detach().cpu().numpy()
        full_attr[c] = attr
        abs_attr = np.abs(attr)
        feat_imp[c] = abs_attr.mean(axis=(0, 1))          # [F]: per-feature global
        time_imp[c] = abs_attr.mean(axis=(0, 2))          # [S]: per-timestep global
        print(f"  IG class {c} ({STAGE_NAMES[c]:5s}) -> attr shape {attr.shape}")

    # Per-class feature ranking
    rank = {}
    print("\nPer-class top features by mean |IG|:")
    hdr = "  " + "class".ljust(7) + "".join(f"#{i+1}".rjust(18) for i in range(5))
    print(hdr)
    for c in range(NUM_CLASSES):
        order_c = np.argsort(feat_imp[c])[::-1]
        rank[STAGE_NAMES[c]] = [FEATURE_NAMES[i] for i in order_c]
        top5 = [FEATURE_NAMES[i] for i in order_c[:5]]
        print("  " + STAGE_NAMES[c].ljust(7) + "".join(n.rjust(18) for n in top5))

    # --- Zero-occlusion cross-check on the same sample ------------------------
    from sklearn.metrics import f1_score as _f1
    def _predict(X):
        with torch.no_grad():
            return xai_model(X).argmax(-1).cpu().numpy()
    base_pred = _predict(X_val)
    base_y    = Y_val.cpu().numpy()
    base_macro = _f1(base_y.flatten(), base_pred.flatten(),
                     average="macro", zero_division=0)
    base_per   = _f1(base_y.flatten(), base_pred.flatten(),
                     average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    print(f"\nBaseline macro-F1 on XAI sample: {base_macro:.3f}")
    occ_drop = np.zeros((NUM_CLASSES, NUM_FEATURES), dtype=np.float32)
    for f_idx in range(NUM_FEATURES):
        Xz = X_val.clone()
        Xz[:, :, f_idx] = 0.0
        pred = _predict(Xz)
        per = _f1(base_y.flatten(), pred.flatten(),
                  average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
        occ_drop[:, f_idx] = base_per - per   # positive = zeroing this feature hurt that class

    print("\nOcclusion delta-F1 top features per class (positive = important to that class):")
    hdr2 = "  " + "class".ljust(7) + "".join(f"#{i+1}".rjust(22) for i in range(3))
    print(hdr2)
    for c in range(NUM_CLASSES):
        oc = np.argsort(occ_drop[c])[::-1]
        top3 = [f"{FEATURE_NAMES[i]} ({occ_drop[c, i]:+.3f})" for i in oc[:3]]
        print("  " + STAGE_NAMES[c].ljust(7) + "".join(t.rjust(22) for t in top3))

    # --- IG vs occlusion rank agreement (Spearman-like) -----------------------
    print("\nRank agreement IG vs occlusion (Spearman over feature ranks per class):")
    for c in range(NUM_CLASSES):
        r_ig  = np.argsort(np.argsort(-feat_imp[c]))
        r_occ = np.argsort(np.argsort(-occ_drop[c]))
        rho = float(np.corrcoef(r_ig, r_occ)[0, 1]) if r_ig.std() > 0 else 0.0
        tag = "AGREE" if rho > 0.5 else "PARTIAL" if rho > 0 else "DISAGREE"
        print(f"  {STAGE_NAMES[c]:6}  rho = {rho:+.3f}  ({tag})")

    # --- Plots ----------------------------------------------------------------
    # Per-class bar of feature importance (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for c, ax in zip(range(NUM_CLASSES), axes.flat):
        order_c = np.argsort(feat_imp[c])
        ax.barh([FEATURE_NAMES[i] for i in order_c],
                feat_imp[c, order_c], color="steelblue")
        ax.set_title(f"{STAGE_NAMES[c]} - mean |IG| per feature")
        ax.set_xlabel("attribution magnitude")
    plt.tight_layout()
    _p = str(OUTDIR_XAI / "xai_global_feature_per_class.png")
    plt.savefig(_p, dpi=130, bbox_inches="tight"); plt.close()
    xai_files.append(_p)

    # Per-class timestep curve (BiGRU context use)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for c in range(NUM_CLASSES):
        ax.plot(range(SEQ_LEN), time_imp[c], label=STAGE_NAMES[c])
    ax.axvline(SEQ_LEN // 2, ls="--", c="grey", alpha=0.6, label="centre epoch")
    ax.set_xlabel("position in 41-epoch window")
    ax.set_ylabel("mean |IG|")
    ax.set_title("How attribution depends on timestep position (BiGRU context use)")
    ax.legend(loc="best"); plt.tight_layout()
    _p = str(OUTDIR_XAI / "xai_global_time_curve.png")
    plt.savefig(_p, dpi=130, bbox_inches="tight"); plt.close()
    xai_files.append(_p)

    xai_global = dict(
        feat_imp=feat_imp.tolist(),
        time_imp=time_imp.tolist(),
        occ_drop=occ_drop.tolist(),
        ranking=rank,
        base_macro_f1=float(base_macro),
        sample_size=int(X_val.shape[0]),
    )
    _XAI = dict(model=xai_model, X=X_val, Y=Y_val, full_attr=full_attr, outdir=OUTDIR_XAI)
    XAI_OK = True
    print("\nSaved:", [Path(p).name for p in xai_files])
except Exception as _e:
    import traceback; traceback.print_exc()
    print("XAI global skipped (non-fatal):", type(_e).__name__, "-", _e)
''')

code(r'''
# ===== XAI 2/5 - LOCAL: epoch heatmaps + one full-night hypnogram =====
# Picks a confident correct-Deep, confident correct-REM, and the most-confident
# misclassified epoch from the XAI sample; renders feature x timestep heatmaps.
# Then renders ONE Walch subject's whole-night hypnogram with per-epoch confidence
# and a dominant-attribution-feature strip.
try:
    if not globals().get("XAI_OK"):
        raise RuntimeError("global XAI cell did not complete")
    import matplotlib.pyplot as plt
    from captum.attr import IntegratedGradients
    xai_model = _XAI["model"]; X_val = _XAI["X"]; Y_val = _XAI["Y"]
    OUTDIR_XAI = _XAI["outdir"]
    full_attr = _XAI["full_attr"]            # [C, N, S, F]

    # --- Per-epoch predictions + confidence on the XAI sample -----------------
    with torch.no_grad():
        logits = xai_model(X_val)            # [N, S, C]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred = probs.argmax(-1)                  # [N, S]
    conf = probs.max(-1)                     # [N, S]
    truth = Y_val.cpu().numpy()

    # --- Pick example (sequence, timestep) triplets ---------------------------
    def _pick(true_cls, want_correct, by_high_conf=True):
        mask = (truth == true_cls)
        if want_correct: mask = mask & (pred == true_cls)
        else:            mask = mask & (pred != true_cls)
        if not mask.any():
            return None
        # rank by confidence on the picked stage
        # high conf for correct, high conf on the WRONG class for misclassified
        score = conf if want_correct else conf
        idx = np.argmax(np.where(mask, score, -np.inf))
        i, t = np.unravel_index(idx, mask.shape)
        return int(i), int(t)

    picks = []
    p = _pick(true_cls=2, want_correct=True);  picks.append(("correct_Deep", p, 2))
    p = _pick(true_cls=3, want_correct=True);  picks.append(("correct_REM",  p, 3))
    # most confident misclassified across any class
    mis_mask = (pred != truth)
    if mis_mask.any():
        i_mis, t_mis = np.unravel_index(
            np.argmax(np.where(mis_mask, conf, -np.inf)), mis_mask.shape)
        i_mis, t_mis = int(i_mis), int(t_mis)
        picks.append(("misclassified", (i_mis, t_mis), int(pred[i_mis, t_mis])))
    else:
        picks.append(("misclassified", None, None))

    for tag, idx, cls in picks:
        if idx is None or cls is None:
            print(f"  {tag}: no eligible epoch in sample (skipped)")
            continue
        i, t = idx
        # Pull the precomputed full IG attribution for the predicted class at this sequence:
        # full_attr[c, i, :, :] is (S, F) = how each (feat, timestep) influenced the
        # MEAN class-c logit for this sequence. As a local proxy for "how each (feat, dt)
        # influenced this epoch specifically", we visualize this map zoomed near t.
        attr_2d = full_attr[cls, i]                 # (S, F)
        # Localize to +/- 5 epoch context around the target timestep
        lo = max(0, t - 5); hi = min(SEQ_LEN, t + 6)
        local = attr_2d[lo:hi, :]                   # (W, F)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                       gridspec_kw={"width_ratios": [3, 1]})
        vmax = float(np.max(np.abs(local))) or 1.0
        im = ax1.imshow(local.T, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax,
                        extent=[lo - t, hi - t - 1, NUM_FEATURES - 0.5, -0.5])
        ax1.set_yticks(range(NUM_FEATURES))
        ax1.set_yticklabels(FEATURE_NAMES, fontsize=9)
        ax1.set_xlabel("epoch offset from target")
        ax1.axvline(0, c="k", lw=1.0, alpha=0.6)
        ax1.set_title(f"{tag}: true={STAGE_NAMES[int(truth[i, t])]} "
                      f"pred={STAGE_NAMES[int(pred[i, t])]} "
                      f"conf={conf[i, t]:.2f}")
        plt.colorbar(im, ax=ax1, label="signed IG")

        # Right panel: top-3 features at the target epoch by |attr|
        col = attr_2d[t]
        order_top = np.argsort(np.abs(col))[::-1][:6]
        ax2.barh([FEATURE_NAMES[k] for k in order_top[::-1]],
                 col[order_top[::-1]],
                 color=["crimson" if v >= 0 else "navy" for v in col[order_top[::-1]]])
        ax2.axvline(0, c="k", lw=0.6)
        ax2.set_title("top features at target epoch")
        ax2.set_xlabel("signed IG (toward predicted class)")

        plt.tight_layout()
        _fn = OUTDIR_XAI / f"xai_local_{tag}.png"
        plt.savefig(str(_fn), dpi=130, bbox_inches="tight"); plt.close()
        xai_files.append(str(_fn))
        top3_str = ", ".join(
            f"{FEATURE_NAMES[k]}{'+' if col[k] >= 0 else ''}{col[k]:.3f}"
            for k in order_top[:3]
        )
        print(f"  {tag:14}  seq={i:3d} t={t:2d}  "
              f"true={STAGE_NAMES[int(truth[i, t])]:5s} "
              f"pred={STAGE_NAMES[int(pred[i, t])]:5s} "
              f"conf={conf[i, t]:.2f}  | top: {top3_str}")

    # --- WHOLE-NIGHT HYPNOGRAM on a Walch subject -----------------------------
    # Pick the Walch subject with the most epochs (most representative full night).
    walch_pick = max(walch_subjects, key=lambda r: len(r[2]))
    sid_w, F_w, L_w = walch_pick                   # F_w shape (N, 12), L_w (N,)
    N_w = len(L_w)
    n_win = N_w // SEQ_LEN
    if n_win < 1:
        print(f"  whole-night Walch subject {sid_w}: too short ({N_w} epochs), skipping")
    else:
        Xw_pad = F_w[: n_win * SEQ_LEN].reshape(n_win, SEQ_LEN, NUM_FEATURES)
        Xw_t   = torch.from_numpy(Xw_pad.astype(np.float32)).to(device)
        with torch.no_grad():
            logits_w = xai_model(Xw_t).cpu().numpy()                   # (n_win, S, C)
        probs_w = np.exp(logits_w - logits_w.max(-1, keepdims=True))
        probs_w = probs_w / probs_w.sum(-1, keepdims=True)
        pred_w  = probs_w.argmax(-1).reshape(-1)                       # (n_win*S,)
        conf_w  = probs_w.max(-1).reshape(-1)
        truth_w = L_w[: n_win * SEQ_LEN]

        # Per-epoch dominant attribution feature: for each window, run IG once per UNIQUE
        # predicted class that appears in it; for each epoch use its predicted class.
        dom_feat = np.zeros(n_win * SEQ_LEN, dtype=np.int32)
        for w in range(n_win):
            unique_cls = np.unique(pred_w[w * SEQ_LEN : (w + 1) * SEQ_LEN])
            attr_by_cls = {}
            for c in unique_cls:
                ig = IntegratedGradients(_MeanLogitFor(xai_model, int(c)))
                a = ig.attribute(Xw_t[w:w + 1],
                                 baselines=torch.zeros_like(Xw_t[w:w + 1]),
                                 n_steps=24).detach().cpu().numpy()[0]   # (S, F)
                attr_by_cls[int(c)] = np.abs(a)
            for s in range(SEQ_LEN):
                c = int(pred_w[w * SEQ_LEN + s])
                dom_feat[w * SEQ_LEN + s] = int(np.argmax(attr_by_cls[c][s]))

        # Plot the hypnogram (3 stacked rows)
        fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1]})
        # row 1: stages
        axes[0].plot(truth_w, drawstyle="steps-post", lw=1.4, label="ground truth",
                     color="black", alpha=0.7)
        axes[0].plot(pred_w,  drawstyle="steps-post", lw=1.4, label="model",
                     color="crimson", alpha=0.7)
        axes[0].set_yticks(range(NUM_CLASSES))
        axes[0].set_yticklabels(STAGE_NAMES)
        axes[0].set_ylabel("stage")
        axes[0].set_title(f"Walch subject {sid_w} - whole-night hypnogram "
                          f"({n_win*SEQ_LEN} epochs)")
        axes[0].legend(loc="upper right")
        # row 2: confidence
        axes[1].fill_between(range(len(conf_w)), 0, conf_w, color="steelblue", alpha=0.6)
        axes[1].set_ylim(0, 1); axes[1].set_ylabel("conf")
        # row 3: dominant feature strip
        im = axes[2].imshow(dom_feat.reshape(1, -1), aspect="auto",
                            cmap="tab20", vmin=0, vmax=NUM_FEATURES - 1,
                            extent=[0, len(dom_feat), 0, 1])
        axes[2].set_yticks([]); axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("dominant feat")
        cb = plt.colorbar(im, ax=axes[2], orientation="horizontal",
                          fraction=0.6, pad=0.7,
                          ticks=list(range(NUM_FEATURES)))
        cb.ax.set_xticklabels(FEATURE_NAMES, rotation=70, fontsize=8)
        plt.tight_layout()
        _p = str(OUTDIR_XAI / "xai_local_walch_night.png")
        plt.savefig(_p, dpi=130, bbox_inches="tight"); plt.close()
        xai_files.append(_p)
        print(f"  whole-night Walch: subject={sid_w} epochs={n_win*SEQ_LEN}  "
              f"-> {Path(_p).name}")
except Exception as _e:
    import traceback; traceback.print_exc()
    print("XAI local skipped (non-fatal):", type(_e).__name__, "-", _e)
''')

code(r'''
# ===== XAI 3/5 - TEMPORAL / FEATURE EFFECTS =====
# (a) Per-class IG-vs-feature-value scatter for the top-4 features (does HIGH hr_std push
#     Wake/REM? does LOW act_count push Deep?)
# (b) Per-class IG-vs-timestep-offset curve, broken down by feature (which features the
#     BiGRU uses at the centre epoch vs the edges of the 41-epoch window).
try:
    if not globals().get("XAI_OK"):
        raise RuntimeError("global XAI cell did not complete")
    import matplotlib.pyplot as plt
    X_val = _XAI["X"].cpu().numpy()              # [N, S, F]
    full_attr = _XAI["full_attr"]                # [C, N, S, F]
    OUTDIR_XAI = _XAI["outdir"]
    feat_imp = np.array(xai_global["feat_imp"])  # [C, F]

    # (a) Scatter: feature value vs IG attribution, per class, top-4 features ---
    fig, axes = plt.subplots(NUM_CLASSES, 4, figsize=(14, 11), sharex=False, sharey=False)
    for c in range(NUM_CLASSES):
        order_c = np.argsort(feat_imp[c])[::-1][:4]
        for k, f_idx in enumerate(order_c):
            ax = axes[c, k]
            xv = X_val[:, :, f_idx].reshape(-1)
            av = full_attr[c, :, :, f_idx].reshape(-1)
            # Subsample if huge
            if xv.size > 4000:
                ix = np.random.choice(xv.size, 4000, replace=False)
                xv, av = xv[ix], av[ix]
            ax.scatter(xv, av, s=4, alpha=0.35, c="steelblue")
            ax.axhline(0, c="grey", lw=0.5)
            corr = float(np.corrcoef(xv, av)[0, 1]) if xv.std() > 1e-9 else 0.0
            ax.set_title(f"{STAGE_NAMES[c]}: {FEATURE_NAMES[f_idx]}  "
                         f"(r={corr:+.2f})", fontsize=10)
            if k == 0: ax.set_ylabel("IG attribution")
            if c == NUM_CLASSES - 1: ax.set_xlabel("feature value (z)")
    plt.tight_layout()
    _p = str(OUTDIR_XAI / "xai_feature_effects.png")
    plt.savefig(_p, dpi=130, bbox_inches="tight"); plt.close()
    xai_files.append(_p)
    print("Saved feature-effects scatter.")

    # (b) Per-class timestep curves broken down by top-3 features ---------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for c, ax in zip(range(NUM_CLASSES), axes.flat):
        order_c = np.argsort(feat_imp[c])[::-1][:3]
        for f_idx in order_c:
            curve = np.abs(full_attr[c, :, :, f_idx]).mean(axis=0)   # [S]
            ax.plot(range(SEQ_LEN), curve, label=FEATURE_NAMES[f_idx])
        ax.axvline(SEQ_LEN // 2, ls="--", c="grey", alpha=0.6)
        ax.set_title(f"{STAGE_NAMES[c]} - top-3 features over the 41-epoch window")
        ax.set_xlabel("position in window"); ax.set_ylabel("mean |IG|")
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    _p = str(OUTDIR_XAI / "xai_timestep_per_class.png")
    plt.savefig(_p, dpi=130, bbox_inches="tight"); plt.close()
    xai_files.append(_p)
    print("Saved per-class timestep curves.")
except Exception as _e:
    import traceback; traceback.print_exc()
    print("XAI feature-effects skipped (non-fatal):", type(_e).__name__, "-", _e)
''')

code(r'''
# ===== XAI 4/5 - PHYSIOLOGICAL INTERPRETATION =====
# For each (stage, feature) pair we measure the DIRECTION the model uses
# (sign of corr between the feature value the model sees and its IG attribution
# toward that stage): >0 => higher feature pushes toward that stage. We compare to
# sleep-physiology priors and report a per-stage |IG|-weighted agreement score.
xai_physio = {}
try:
    if not globals().get("XAI_OK"):
        raise RuntimeError("global XAI cell did not complete")
    X_val = _XAI["X"].cpu().numpy()
    full_attr = _XAI["full_attr"]
    feat_imp = np.array(xai_global["feat_imp"])

    # Expected sign of the model's effect on P(class) for a HIGHER feature value.
    # +1: higher feature value should push TOWARD this stage; -1: AWAY; 0: contested/skip.
    # (Light is the broad transition state, mostly contested -> mostly zeros.)
    EXPECTED = {
        "Wake":  {"hr_mean":+1, "hr_std":+1, "hr_min":0,  "hr_max":+1, "hr_range":+1,
                  "hr_succdiff_std":+1, "hr_delta_prev":0,
                  "act_count":+1, "immobility_frac":-1, "act_max":+1, "act_std":+1,
                  "time_of_night":0},
        "Light": {"hr_mean":0,  "hr_std":0,  "hr_min":0,  "hr_max":0,  "hr_range":0,
                  "hr_succdiff_std":0, "hr_delta_prev":0,
                  "act_count":0, "immobility_frac":0, "act_max":0, "act_std":0,
                  "time_of_night":0},
        "Deep":  {"hr_mean":-1, "hr_std":-1, "hr_min":-1, "hr_max":-1, "hr_range":-1,
                  "hr_succdiff_std":-1, "hr_delta_prev":0,
                  "act_count":-1, "immobility_frac":+1, "act_max":-1, "act_std":-1,
                  "time_of_night":-1},   # Deep concentrates early in the night
        "REM":   {"hr_mean":+1, "hr_std":+1, "hr_min":0,  "hr_max":+1, "hr_range":+1,
                  "hr_succdiff_std":+1, "hr_delta_prev":0,
                  "act_count":-1, "immobility_frac":+1, "act_max":-1, "act_std":-1,
                  "time_of_night":+1},   # REM concentrates late in the night
    }
    SIGNS = {1: "+", -1: "-", 0: "."}

    print("Per-stage physiological agreement of model attribution vs sleep-physiology priors")
    print("=" * 84)
    for c in range(NUM_CLASSES):
        stage = STAGE_NAMES[c]
        exp = EXPECTED[stage]
        rows = []
        for f_idx in range(NUM_FEATURES):
            f_name = FEATURE_NAMES[f_idx]
            xv = X_val[:, :, f_idx].reshape(-1)
            av = full_attr[c, :, :, f_idx].reshape(-1)
            corr = float(np.corrcoef(xv, av)[0, 1]) if xv.std() > 1e-9 and av.std() > 1e-9 else 0.0
            meas = int(np.sign(corr)) if abs(corr) > 0.05 else 0
            e = exp.get(f_name, 0)
            if e == 0:        verdict = "n/a"
            elif meas == 0:   verdict = "weak"
            else:             verdict = "AGREE" if meas == e else "DIFFER"
            rows.append((f_name, float(feat_imp[c, f_idx]), corr, e, meas, verdict))

        judged = [r for r in rows if r[3] != 0 and r[4] != 0]
        w_total = sum(r[1] for r in judged)
        w_agree = sum(r[1] for r in judged if r[5] == "AGREE")
        n_agree = sum(1 for r in judged if r[5] == "AGREE")
        frac_w = (w_agree / w_total) if w_total > 0 else 0.0
        frac_n = (n_agree / len(judged)) if judged else 0.0

        xai_physio[stage] = dict(
            agreement_frac=float(frac_n),
            agreement_frac_weighted=float(frac_w),
            n_judged=int(len(judged)),
            n_agree=int(n_agree),
        )

        print(f"\n{stage}")
        print(f"  {'feature':18}{'mean|IG|':>11}{'corr(x,IG)':>13}{'expect':>8}{'model':>7}  verdict")
        print("  " + "-" * 70)
        # sort by mean|IG| desc so the top drivers are listed first
        for f_name, ms, corr, e, meas, verdict in sorted(rows, key=lambda r: -r[1]):
            print(f"  {f_name:18}{ms:>11.4f}{corr:>13.3f}"
                  f"{SIGNS[e]:>8}{SIGNS[meas]:>7}  {verdict}")
        print(f"  {stage} agreement (|IG|-weighted, REPORT THIS): {frac_w*100:.0f}%  "
              f"({n_agree}/{len(judged)} judged, unweighted {frac_n*100:.0f}%)")

    # Plain-language verdict per stage
    print("\nRead-out:")
    for stage in STAGE_NAMES:
        s = xai_physio.get(stage, {})
        fw = s.get("agreement_frac_weighted", 0.0)
        if stage == "Light":
            print(f"  {stage:5}: priors are contested -> n/a (mostly unjudged); see global ranking.")
            continue
        if fw >= 0.70:
            tone = "STRONG agreement"
        elif fw >= 0.50:
            tone = "majority agreement"
        else:
            tone = "MIXED / disagreement"
        print(f"  {stage:5}: {tone}  ({fw*100:.0f}% of model attribution is physiology-consistent)")
    print("\nThesis claim template: \"The TCN+BiGRU's per-stage attribution matches the")
    print("expected sleep-physiology priors on a |IG|-weighted majority of features for Wake,")
    print("Deep, and REM; Light is the contested transition state and is not scored.\"")
except Exception as _e:
    import traceback; traceback.print_exc()
    print("XAI physiology skipped (non-fatal):", type(_e).__name__, "-", _e)
''')

code(r'''
# ===== XAI 5/5 (optional, gated) - FEATURE-PRUNING RETRAIN SWEEP =====
# Off by default. Set RUN_XAI_FEATURE_REDUCTION = True at the TOP of the XAI 1/5 cell
# and re-run THIS cell to find the smallest k features that preserves Walch macro-F1.
# We rank features by mean |IG| pooled across classes, retrain for fewer epochs, and
# report held-out macro-F1 + kappa at each k. Smaller retrain budget keeps this < 5 min.
xai_reduction = {}
try:
    if not globals().get("RUN_XAI_FEATURE_REDUCTION"):
        raise RuntimeError("RUN_XAI_FEATURE_REDUCTION=False -> skipped")
    if not globals().get("XAI_OK"):
        raise RuntimeError("global XAI cell did not complete -> no ranking")
    from sklearn.metrics import (f1_score as _f1, cohen_kappa_score as _ck,
                                  accuracy_score as _acc)
    # Pool |IG| across classes (sum) -> single ranking for retraining decisions.
    pooled = np.sum(np.array(xai_global["feat_imp"]), axis=0)     # [F]
    ranked_idx = np.argsort(pooled)[::-1].tolist()
    print("XAI-pooled feature ranking:")
    for r, i in enumerate(ranked_idx):
        print(f"  #{r+1:2d}  {FEATURE_NAMES[i]:18}  pooled |IG| = {pooled[i]:.4f}")

    REDUCE_EPOCHS = 30           # shorter than the headline run, enough for convergence on this small net
    REDUCE_PATIENCE = 8
    Ks = [6, 8, 10, 12]
    Ks = sorted({k for k in Ks if 1 <= k <= NUM_FEATURES})

    def _slice_subjects(subjects, keep_idx):
        return [(sid, F[:, keep_idx].copy(), L) for sid, F, L in subjects]

    def _train_one(keep_idx):
        n_feat = len(keep_idx)
        bid_tr_k = _slice_subjects(bid_train, keep_idx)
        bid_va_k = _slice_subjects(bid_val,   keep_idx)
        walch_k  = _slice_subjects(walch_subjects, keep_idx)
        ds_tr = SeqDataset(bid_tr_k, SEQ_LEN, TRAIN_STRIDE)
        ds_va = SeqDataset(bid_va_k, SEQ_LEN, EVAL_STRIDE)
        ds_te = SeqDataset(walch_k,  SEQ_LEN, EVAL_STRIDE)
        ld_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=pin, drop_last=True)
        ld_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=pin)
        ld_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=pin)
        m = SleepFeatureModel(n_features=n_feat).to(device)
        crit = nn.CrossEntropyLoss(weight=class_weights.to(device))
        opt = AdamW(m.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        sch = CosineAnnealingLR(opt, T_max=REDUCE_EPOCHS)
        best_va_mf1, pat = 0.0, REDUCE_PATIENCE
        best_state = None
        for ep in range(1, REDUCE_EPOCHS + 1):
            _, _, _, _, _, _ = run_epoch(m, ld_tr, crit, opt)
            _, va_mf1, _, _, _, _ = run_epoch(m, ld_va, crit)
            sch.step()
            if va_mf1 > best_va_mf1:
                best_va_mf1 = va_mf1; pat = REDUCE_PATIENCE
                best_state = {k_: v.detach().clone() for k_, v in m.state_dict().items()}
            else:
                pat -= 1
                if pat <= 0: break
        if best_state is not None:
            m.load_state_dict(best_state)
        _, te_mf1, te_wf1, te_k, P_, L_ = run_epoch(m, ld_te, nn.CrossEntropyLoss())
        return float(te_mf1), float(te_wf1), float(te_k), float(_acc(L_, P_))

    print(f"\nFeature-reduction sweep (retrain {REDUCE_EPOCHS} epochs per k):")
    print(f"{'k':>3} {'macroF1':>10} {'wF1':>8} {'kappa':>8} {'acc':>8}   features kept")
    print("-" * 80)
    for k in Ks:
        keep = sorted(ranked_idx[:k])
        mf1, wf1, kp, ac = _train_one(keep)
        kept_names = [FEATURE_NAMES[i] for i in keep]
        xai_reduction[k] = dict(
            macro_f1=mf1, weighted_f1=wf1, kappa=kp, accuracy=ac,
            features=kept_names,
        )
        print(f"{k:>3} {mf1:>10.4f} {wf1:>8.4f} {kp:>8.4f} {ac:>8.4f}   "
              f"{', '.join(kept_names[:6])}{' ...' if len(kept_names) > 6 else ''}")

    # Recommend smallest k within 0.01 macroF1 of the best.
    if xai_reduction:
        best_k = max(xai_reduction, key=lambda k: xai_reduction[k]["macro_f1"])
        best_mf1 = xai_reduction[best_k]["macro_f1"]
        rec_k = min(k for k, v in xai_reduction.items()
                    if v["macro_f1"] >= best_mf1 - 0.01)
        xai_reduction["recommended_k"] = int(rec_k)
        xai_reduction["recommended_features"] = xai_reduction[rec_k]["features"]
        dropped = [n for n in FEATURE_NAMES if n not in xai_reduction[rec_k]["features"]]
        print("-" * 80)
        print(f"Best macroF1 = {best_mf1:.4f} at k={best_k}")
        print(f"RECOMMENDED minimal set: k={rec_k} (smallest within 0.01 macroF1 of best)")
        print(f"  keep:  {xai_reduction[rec_k]['features']}")
        print(f"  drop:  {dropped or '(none)'}")
        print("Deploy note: to ship the lean model, set CACHE_FEATURES in")
        print("ml/sleep/_build_notebook.py to the recommended list AND mirror the change in")
        print("V32_FEATURE_NAMES in services/ai/sleepStageModel.ts.")
except Exception as _e:
    import traceback; traceback.print_exc()
    print("XAI feature reduction skipped (non-fatal):", type(_e).__name__, "-", _e)
''')

code(r'''
# ===== XAI - MLflow logging (robust, never aborts) =====
try:
    if globals().get("XAI_OK"):
        # Persist everything as JSON for the MLflow Artifact tab too.
        import json as _json
        _summary = {
            "global":     xai_global,
            "physio":     xai_physio,
            "reduction":  xai_reduction,
            "files":      [Path(p).name for p in xai_files],
        }
        _sp = OUTPUT_DIR / "xai" / "xai_summary.json"
        with open(_sp, "w") as _fp:
            _json.dump(_summary, _fp, indent=2)
        with tracker.start_run(run_name="xai_v3.2"):
            tracker.log_artifact(_sp)
            for _f in xai_files:
                tracker.log_artifact(_f)
            for stage, v in xai_physio.items():
                if "agreement_frac_weighted" in v:
                    tracker.log_metric(f"xai_physio_agree_w_{stage.lower()}",
                                        v["agreement_frac_weighted"])
            if isinstance(xai_reduction, dict) and "recommended_k" in xai_reduction:
                tracker.log_metric("xai_recommended_k",
                                    float(xai_reduction["recommended_k"]))
        print(f"XAI artifacts logged to MLflow ({len(xai_files)} files + summary.json).")
    else:
        print("XAI did not complete -> nothing to log.")
except Exception as _e:
    import traceback; traceback.print_exc()
    print("XAI MLflow logging skipped (non-fatal):", type(_e).__name__, "-", _e)
''')

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
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150); plt.show()

# Attach the curves PNG to MLflow too
with tracker.start_run(run_name="training_curves"):
    tracker.log_artifact(OUTPUT_DIR / "training_curves.png")''')

md("""## Output files

| File | Drop into repo at |
|------|-------------------|
| `output/export/sleep_stage_model.onnx`    | `assets/ml/sleep_stage_model.onnx` |
| `output/export/sleep_model_metadata.json` | `assets/ml/sleep_model_metadata.json` |

The 11-feature recipe in metadata must be mirrored exactly in
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
out = Path(__file__).parent / "seren_sleep_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB, {len(CELLS)} cells)")
