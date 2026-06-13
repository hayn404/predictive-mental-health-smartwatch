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


md("""# Seren — Sleep Stage Classifier v3 (Train from Cache)

Reads the pre-extracted features (produced by `prepare_features.py` and uploaded as a
Kaggle Dataset), trains on **BIDSleep**, evaluates on **Walch / Sleep-Accel** as a
held-out test set. No streaming, no PhysioNet credentials.

**Inputs (must match the cache the model will run on at inference):**
HR + accelerometer only — 11 engineered per-30 s-epoch features.

**Labels (4-class, matches app):** `Wake→0`, `N1+N2→1 (light)`, `N3→2 (deep)`, `REM→3`.

**Targets (beat prior work):** 4-class weighted F1 ≥ 0.72, Deep recall > 56% (SLAMSS 2023);
3-class weighted F1 ~0.80.

**Setup on Kaggle:**
1. Add the `seren-sleep-cache` dataset under "+ Add Input".
2. Settings: GPU T4 x2 (or P100), Internet OFF (not needed).""")

code("!pip install -q mlflow onnx onnxruntime scikit-learn")

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

# ── Feature contract — 11 from the cache + 1 added in the notebook (time_of_night) ─
CACHE_FEATURES = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "hr_succdiff_std", "hr_delta_prev",
    "act_count", "immobility_frac", "act_max", "act_std",
]
FEATURE_NAMES = CACHE_FEATURES + ["time_of_night"]   # 12 total at the model input
NUM_FEATURES  = len(FEATURE_NAMES)
EPOCH_SEC     = 30

# ── Sequence config (v3.2 — chosen by niapy PSO over 110 trials, val mF1 0.6785) ──
SEQ_LEN      = 41       # ~20.5 min temporal context (half a sleep cycle)
TRAIN_STRIDE = 5        # PSO found stride 1 over-augmented; stride 5 is the sweet spot
EVAL_STRIDE  = SEQ_LEN  # disjoint sequences at eval time

# ── Model ─────────────────────────────────────────────────────────────────────
TCN_CHANNELS = 64
TCN_KERNEL   = 5
LSTM_HIDDEN  = 48       # was 64; smaller = less overfit
LSTM_LAYERS  = 2
DROPOUT      = 0.37     # PSO best (was 0.4)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE          = 64
LEARNING_RATE       = 2.95e-3   # PSO best (was 1e-3) — note: at search upper bound
WEIGHT_DECAY        = 2.5e-5    # PSO best (was 5e-4) — L2 wasn't helping
NUM_EPOCHS          = 80
EARLY_STOP_PATIENCE = 15
VAL_SPLIT           = 0.15
RANDOM_SEED         = 42
GRAD_CLIP           = 1.0

print(f"{NUM_FEATURES} features/epoch · seq {SEQ_LEN} epochs · classes {STAGE_NAMES}")


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


rng    = np.random.default_rng(RANDOM_SEED)
perm   = rng.permutation(len(bidsleep_nights))
n_val  = max(1, int(len(bidsleep_nights) * VAL_SPLIT))
val_ix = set(perm[:n_val].tolist())
bid_train = [s for i, s in enumerate(bidsleep_nights) if i not in val_ix]
bid_val   = [s for i, s in enumerate(bidsleep_nights) if i in val_ix]

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
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        seq, _ = self.lstm(h)
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
torch.onnx.export(
    export_model, (dummy,), str(onnx_path),
    input_names=["features"], output_names=["logits"],
    dynamic_axes={"features": {0: "batch", 1: "seq_len"},
                  "logits":   {0: "batch", 1: "seq_len"}},
    opset_version=17, do_constant_folding=True,
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
            "shape": ["batch", "seq_len", NUM_FEATURES],
            "names": FEATURE_NAMES,
            "epochSeconds": EPOCH_SEC,
            "recommendedSeqLen": SEQ_LEN,
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
print(f"Saved {onnx_path} ({onnx_path.stat().st_size/1e3:.1f} KB) + metadata")

# Log the exported model and its metadata as MLflow artifacts
with tracker.start_run(run_name="onnx_export_v3.2"):
    tracker.log_artifact(onnx_path)
    tracker.log_artifact(EXPORT_DIR / "sleep_model_metadata.json")''')

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
out = Path(__file__).parent / "kaggle_train.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB, {len(CELLS)} cells)")
