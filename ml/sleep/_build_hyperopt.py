"""Builds kaggle_hyperopt.ipynb — nature-inspired hyperparameter search via niapy.

Default algorithm is Particle Swarm Optimization (PSO); swap a single line to use
Grey Wolf Optimizer (GWO) or any other niapy algorithm. The search runs on the same
cached features the training notebook uses, so no streaming is needed here either.

Search space (6 dims, continuous in [0,1] → real values inside `decode`):
    lr           ∈ [1e-4, 3e-3]    log
    dropout      ∈ [0.20, 0.60]
    weight_decay ∈ [1e-5, 1e-3]    log
    lstm_hidden  ∈ {48, 64, 96, 128}
    seq_len      ∈ {15, 21, 31, 41}
    train_stride ∈ {1, 3, 5, 10}

Output:
    output/hyperopt/best_hparams.json   — copy these numbers into kaggle_train.ipynb
    output/hyperopt/trials.json         — all evaluations (for the report)
    output/hyperopt/convergence.png     — best-so-far vs trial number
"""
import json
from pathlib import Path

CELLS = []


def md(t):  CELLS.append({"cell_type": "markdown", "metadata": {},
                          "source": [l + "\n" for l in t.rstrip("\n").split("\n")]})
def code(t): CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                           "outputs": [], "source": [l + "\n" for l in t.rstrip("\n").split("\n")]})


md("""# Seren — Hyperparameter Search (niapy, switchable algorithm)

Nature-inspired hyperparameter optimization over 6 model + training knobs, using the
`niapy` library. **Algorithm is switchable** via the `SEARCH_ALGO` constant in the
config cell (`"PSO"` or `"GWO"`). Run the notebook twice — once with each — and the
final "Comparison" cell will overlay the two convergence curves for the report.

**Budget per run**: ~100 candidate evaluations (pop × iters). Each evaluation is a short
training run (max 8 epochs, early-stopped) → ~30–50 min on a Kaggle T4 for the whole
run. Bad trials prune themselves quickly via patience.

**Setup**: GPU T4, Internet OFF (uses the cached features), add `seren-sleep-cache`
Dataset. MLflow auto-logs every trial; if DagsHub creds are in Kaggle Secrets the
runs land there and you can compare PSO vs GWO live in the DagsHub Experiments tab.""")

code("!pip install -q niapy mlflow onnx onnxruntime scikit-learn")

md("## Config + load cache (mirrors kaggle_train.ipynb)")

code('''import json, time, pickle, math, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, cohen_kappa_score

print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")

import os

OUTPUT_DIR    = Path("/kaggle/working/output")
HYPEROPT_DIR  = OUTPUT_DIR / "hyperopt"
MLRUNS_DIR    = HYPEROPT_DIR / "mlruns"
HYPEROPT_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


class Tracker:
    """Safe MLflow wrapper. All methods are no-ops if MLflow init fails so the
    experiment continues regardless. Reads MLFLOW_TRACKING_URI (and the
    USERNAME/PASSWORD pair) from Kaggle Secrets if available; falls back to a
    local file backend in /kaggle/working."""

    def __init__(self, experiment_name, fallback_dir):
        self.enabled = False
        self._m = None
        # 1) Pull MLflow env vars from Kaggle Secrets if present
        try:
            from kaggle_secrets import UserSecretsClient
            sec = UserSecretsClient()
            for k in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME",
                      "MLFLOW_TRACKING_PASSWORD"):
                try:
                    v = sec.get_secret(k)
                    if v:
                        os.environ[k] = v
                except Exception:
                    pass
        except Exception:
            pass
        # 2) Resolve tracking URI: env first, else local file
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            uri = f"file://{Path(fallback_dir).resolve()}"
        # 3) Init
        try:
            import mlflow
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            self._m = mlflow
            self.enabled = True
            print(f"MLflow tracking: {uri} | experiment: {experiment_name}")
        except Exception as e:
            print(f"MLflow disabled ({e}) — runs will not be tracked.")

    def start_run(self, **kw):
        if not self.enabled:
            return _NullCtx()
        try:
            return self._m.start_run(**kw)
        except Exception as e:
            print(f"  mlflow start_run failed: {e}")
            return _NullCtx()

    def log_params(self, d):
        if not self.enabled: return
        try:
            clean = {k: v for k, v in d.items() if not str(k).startswith("_")}
            self._m.log_params(clean)
        except Exception:
            pass

    def log_metric(self, k, v, step=None):
        if not self.enabled: return
        try:
            self._m.log_metric(k, float(v), step=step)
        except Exception:
            pass

    def log_artifact(self, path):
        if not self.enabled: return
        try:
            if Path(path).exists():
                self._m.log_artifact(str(path))
        except Exception:
            pass

    def set_tag(self, k, v):
        if not self.enabled: return
        try:
            self._m.set_tag(k, v)
        except Exception:
            pass


class _NullCtx:
    def __enter__(self):  return self
    def __exit__(self, *a): return False


tracker = Tracker(experiment_name="seren-sleep-hpo", fallback_dir=MLRUNS_DIR)

CACHE_CANDIDATES = [Path("/kaggle/input/seren-sleep-cache"),
                    Path("/kaggle/working/cache")]


def find_cache_dir():
    for c in CACHE_CANDIDATES:
        if (c / "bidsleep_features.pkl").exists() and (c / "walch_features.pkl").exists():
            return c
    raise FileNotFoundError("Cache not found in: " + str([str(c) for c in CACHE_CANDIDATES]))


CACHE_DIR = find_cache_dir()
print(f"Cache: {CACHE_DIR}")

with open(CACHE_DIR / "walch_features.pkl",    "rb") as f: walch_subjects   = pickle.load(f)
with open(CACHE_DIR / "bidsleep_features.pkl", "rb") as f: bidsleep_nights  = pickle.load(f)

# Add the 12th feature (time-of-night) — identical to the training notebook
def add_time_of_night(subjects):
    out = []
    for sid, feats, labs in subjects:
        n = len(feats); denom = max(1, n - 1)
        ton = (2.0 * np.arange(n, dtype=np.float32) / denom - 1.0).reshape(-1, 1)
        out.append((sid, np.concatenate([feats, ton], axis=1).astype(np.float32), labs))
    return out

walch_subjects  = add_time_of_night(walch_subjects)
bidsleep_nights = add_time_of_night(bidsleep_nights)

NUM_CLASSES  = 4
NUM_FEATURES = walch_subjects[0][1].shape[-1]   # 12
STAGE_NAMES  = ["Wake", "Light", "Deep", "REM"]
RANDOM_SEED  = 42

print(f"BIDSleep: {len(bidsleep_nights)} nights | Walch: {len(walch_subjects)} subjects | "
      f"features: {NUM_FEATURES}")

# Subject-level split of BIDSleep — same as training notebook for consistency
rng    = np.random.default_rng(RANDOM_SEED)
perm   = rng.permutation(len(bidsleep_nights))
val_ix = set(perm[:max(1, int(len(bidsleep_nights) * 0.15))].tolist())
bid_train = [s for i, s in enumerate(bidsleep_nights) if i not in val_ix]
bid_val   = [s for i, s in enumerate(bidsleep_nights) if i in val_ix]
print(f"Train nights: {len(bid_train)} | Val nights: {len(bid_val)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Pick the algorithm here — "PSO" or "GWO". Run the notebook once with each;
# the Comparison cell at the end overlays both convergence curves.
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_ALGO = "GWO"

POPULATION_SIZE = 10
MAX_ITERS       = 10   # 10 × 10 ≈ 100 evaluations per run

print(f"\\nAlgorithm: {SEARCH_ALGO} | pop {POPULATION_SIZE} × iters {MAX_ITERS} "
      f"≈ {POPULATION_SIZE*MAX_ITERS} evaluations")''')

md("## Search space + decoder")

code('''# Each candidate is a 6-D vector in [0,1]; `decode` maps it to real hyperparameters.

DIM = 6
LSTM_CHOICES   = [48, 64, 96, 128]
SEQLEN_CHOICES = [15, 21, 31, 41]
STRIDE_CHOICES = [1, 3, 5, 10]


def _pick(x, choices):
    """Map x ∈ [0,1] to a choice in a discrete list."""
    return choices[min(len(choices) - 1, int(x * len(choices)))]


def decode(x):
    return {
        "lr":           float(10 ** (np.log10(1e-4) + x[0] * (np.log10(3e-3) - np.log10(1e-4)))),
        "dropout":      float(0.20 + x[1] * 0.40),
        "weight_decay": float(10 ** (np.log10(1e-5) + x[2] * (np.log10(1e-3) - np.log10(1e-5)))),
        "lstm_hidden":  _pick(x[3], LSTM_CHOICES),
        "seq_len":      _pick(x[4], SEQLEN_CHOICES),
        "train_stride": _pick(x[5], STRIDE_CHOICES),
    }


print("Example decode([0.5]*6):", decode([0.5]*6))''')

md("## One-trial training function (the objective)")

code('''class SeqDataset(Dataset):
    def __init__(self, subjects, seq_len, stride):
        self.items = []
        for _, feats, labs in subjects:
            n = len(labs)
            if n < seq_len:
                continue
            for s in range(0, n - seq_len + 1, stride):
                self.items.append((feats[s:s+seq_len], labs[s:s+seq_len]))

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        f, l = self.items[i]
        return torch.from_numpy(f), torch.from_numpy(l)


class SleepFeatureModel(nn.Module):
    def __init__(self, n_features, lstm_hidden, dropout, n_classes=NUM_CLASSES,
                 tcn_channels=64, tcn_kernel=5, lstm_layers=2):
        super().__init__()
        self.in_norm = nn.LayerNorm(n_features)
        pad = tcn_kernel // 2
        self.tcn = nn.Sequential(
            nn.Conv1d(n_features, tcn_channels, tcn_kernel, padding=pad),
            nn.BatchNorm1d(tcn_channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(tcn_channels, tcn_channels, tcn_kernel, padding=pad),
            nn.BatchNorm1d(tcn_channels), nn.GELU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(tcn_channels, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2), nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, n_classes),
        )

    def forward(self, x):
        x = self.in_norm(x)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        seq, _ = self.lstm(h)
        return self.head(seq)


# Pre-compute the √-inverse class weights once from the training labels
_tr_lab = np.concatenate([l for _, _, l in bid_train])
_counts = np.bincount(_tr_lab, minlength=NUM_CLASSES).astype(np.float32)
_inv_s  = 1.0 / np.sqrt(np.maximum(_counts, 1.0))
CLASS_WEIGHTS = torch.tensor(_inv_s / _inv_s.sum() * NUM_CLASSES, dtype=torch.float32).to(device)


# Trial budget — kept short so the search is feasible. Final retraining happens elsewhere.
TRIAL_MAX_EPOCHS = 8
TRIAL_PATIENCE   = 3
TRIAL_BATCH_SIZE = 64


def train_eval(hp, verbose=False):
    """Run one short training trial and return best val macro-F1 achieved."""
    torch.manual_seed(RANDOM_SEED)
    train_ds = SeqDataset(bid_train, hp["seq_len"], hp["train_stride"])
    val_ds   = SeqDataset(bid_val,   hp["seq_len"], hp["seq_len"])
    if len(train_ds) == 0 or len(val_ds) == 0:
        return 0.0

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=TRIAL_BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=TRIAL_BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)

    model = SleepFeatureModel(NUM_FEATURES, hp["lstm_hidden"], hp["dropout"]).to(device)
    optim_ = AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    crit   = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

    best, bad = 0.0, 0
    for ep in range(TRIAL_MAX_EPOCHS):
        # train
        model.train()
        for f, l in train_loader:
            f, l = f.to(device, non_blocking=True), l.to(device, non_blocking=True)
            logits = model(f)
            loss   = crit(logits.reshape(-1, NUM_CLASSES), l.reshape(-1))
            optim_.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim_.step()
        # val
        model.eval(); P, L = [], []
        with torch.no_grad():
            for f, l in val_loader:
                f = f.to(device); l = l.to(device)
                logits = model(f)
                P.append(logits.argmax(-1).reshape(-1).cpu().numpy())
                L.append(l.reshape(-1).cpu().numpy())
        P, L = np.concatenate(P), np.concatenate(L)
        mf1 = float(f1_score(L, P, average="macro", zero_division=0))
        if verbose:
            print(f"    ep{ep+1}: val mF1 {mf1:.4f}")
        if mf1 > best:
            best, bad = mf1, 0
        else:
            bad += 1
            if bad >= TRIAL_PATIENCE:
                break

    # free GPU memory between trials
    del model, optim_, train_loader, val_loader, train_ds, val_ds
    torch.cuda.empty_cache()
    return best


# Smoke test on one decoded point so we know it runs end-to-end before launching PSO
_smoke = train_eval(decode([0.5]*6), verbose=True)
print(f"Smoke test: val mF1 = {_smoke:.4f}")''')

md("## Run the search (niapy PSO)")

code('''from niapy.algorithms.basic import (
    ParticleSwarmAlgorithm,
    GreyWolfOptimizer,
    # add more nature-inspired algos here as needed (Whale, Bat, Firefly, Cuckoo, …)
)
from niapy.problems import Problem
from niapy.task import Task

_ALGO_FACTORY = {
    "PSO": lambda: ParticleSwarmAlgorithm(population_size=POPULATION_SIZE, seed=RANDOM_SEED),
    "GWO": lambda: GreyWolfOptimizer(population_size=POPULATION_SIZE, seed=RANDOM_SEED),
}
assert SEARCH_ALGO in _ALGO_FACTORY, f"Unknown algorithm: {SEARCH_ALGO}"
ALGO_NAME = SEARCH_ALGO   # kept for compatibility with the cells below

# Track every trial for plotting + the report
TRIALS = []   # list of {"id", "hp", "fitness", "t_sec"}


class HpoProblem(Problem):
    def __init__(self):
        super().__init__(dimension=DIM, lower=0.0, upper=1.0)

    def _evaluate(self, x):
        t0 = time.time()
        hp = decode(x)
        trial_id = len(TRIALS)
        # Each trial = a nested MLflow run under the parent sweep
        with tracker.start_run(nested=True, run_name=f"trial_{trial_id:03d}"):
            tracker.set_tag("trial_id", trial_id)
            tracker.set_tag("algorithm", ALGO_NAME)
            tracker.log_params(hp)
            score = train_eval(hp)
            dt = time.time() - t0
            tracker.log_metric("val_macro_f1", score)
            tracker.log_metric("trial_seconds", dt)
        TRIALS.append({"id": trial_id, "hp": hp, "fitness": score, "t_sec": round(dt, 1)})
        print(f"  trial {trial_id:3d}: mF1 {score:.4f} | "
              f"lr {hp['lr']:.1e} dp {hp['dropout']:.2f} wd {hp['weight_decay']:.1e} "
              f"H {hp['lstm_hidden']} S {hp['seq_len']} str {hp['train_stride']} | "
              f"{dt:.0f}s", flush=True)
        return -score   # niapy minimizes, we want to maximize macro-F1


task = Task(problem=HpoProblem(), max_iters=MAX_ITERS)
algo = _ALGO_FACTORY[ALGO_NAME]()
print(f"\\nLaunching {ALGO_NAME}: pop={POPULATION_SIZE} × iters={MAX_ITERS} "
      f"≈ {POPULATION_SIZE * MAX_ITERS} evaluations\\n")

# Parent run for the whole sweep
parent_cm = tracker.start_run(run_name=f"{ALGO_NAME}_sweep")
with parent_cm:
    tracker.log_params({
        "algorithm":        ALGO_NAME,
        "population_size":  POPULATION_SIZE,
        "max_iters":        MAX_ITERS,
        "seed":             RANDOM_SEED,
        "dim":              DIM,
        "trial_max_epochs": TRIAL_MAX_EPOCHS,
        "trial_patience":   TRIAL_PATIENCE,
        "num_train_nights": len(bid_train),
        "num_val_nights":   len(bid_val),
        "num_features":     NUM_FEATURES,
    })
    t_start = time.time()
    best_x, best_neg = algo.run(task=task)
    elapsed = time.time() - t_start
    tracker.log_metric("best_val_macro_f1", -best_neg)
    tracker.log_metric("elapsed_seconds",   elapsed)
    tracker.log_metric("n_trials",          len(TRIALS))

print(f"\\n{ALGO_NAME} finished in {elapsed/60:.1f} min over {len(TRIALS)} evaluations")
print(f"Best val macro-F1: {-best_neg:.4f}")''')

md("## Results")

code('''best_hp = decode(best_x)
best_hp["_val_macro_F1"] = float(-best_neg)
best_hp["_algorithm"]    = ALGO_NAME
best_hp["_n_trials"]     = len(TRIALS)
best_hp["_elapsed_min"]  = round(elapsed / 60, 1)

print("Best hyperparameters:")
for k, v in best_hp.items():
    print(f"  {k:18s} = {v}")

# Algorithm-specific filenames → PSO and GWO results coexist
suffix    = ALGO_NAME.lower()
best_path = HYPEROPT_DIR / f"best_hparams_{suffix}.json"
trial_path = HYPEROPT_DIR / f"trials_{suffix}.json"
conv_path  = HYPEROPT_DIR / f"convergence_{suffix}.png"

with open(best_path, "w")  as f: json.dump(best_hp, f, indent=2)
with open(trial_path, "w") as f: json.dump(TRIALS,  f, indent=2)
print(f"\\nSaved → {best_path}")
print(f"Saved → {trial_path}")

# Convergence plot — best-so-far vs trial index
import matplotlib.pyplot as plt
fitnesses = [t["fitness"] for t in TRIALS]
best_so_far = np.maximum.accumulate(fitnesses)
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(TRIALS) + 1), fitnesses, "o-", alpha=0.4, label="trial")
plt.plot(range(1, len(TRIALS) + 1), best_so_far, "r-", lw=2, label="best so far")
plt.xlabel("trial"); plt.ylabel("val macro F1"); plt.legend()
plt.title(f"{ALGO_NAME} convergence ({len(TRIALS)} evaluations, best = {best_so_far[-1]:.4f})")
plt.tight_layout()
plt.savefig(conv_path, dpi=150)
plt.show()

# Attach the artifacts to a fresh tracking run so they're browsable in the MLflow UI
with tracker.start_run(run_name=f"{ALGO_NAME}_summary"):
    tracker.log_params(best_hp)
    tracker.log_artifact(best_path)
    tracker.log_artifact(trial_path)
    tracker.log_artifact(conv_path)''')

md("""## Top 5 trials (for the report)""")

code('''top = sorted(TRIALS, key=lambda t: -t["fitness"])[:5]
print(f"{'rank':<5}{'mF1':<8}{'lr':<10}{'dp':<6}{'wd':<10}{'H':<5}{'S':<5}{'str':<5}{'t(s)'}")
for i, t in enumerate(top, 1):
    h = t["hp"]
    print(f"{i:<5}{t['fitness']:<8.4f}{h['lr']:<10.1e}{h['dropout']:<6.2f}"
          f"{h['weight_decay']:<10.1e}{h['lstm_hidden']:<5}{h['seq_len']:<5}"
          f"{h['train_stride']:<5}{t['t_sec']}")''')

md("""## Comparison: PSO vs GWO (renders only if both runs exist)

Run this cell after you've executed the notebook **twice** — once with `SEARCH_ALGO = "PSO"`,
once with `SEARCH_ALGO = "GWO"`. It overlays the best-so-far convergence curves and prints
the head-to-head table for the report.""")

code('''pso_p = HYPEROPT_DIR / "trials_pso.json"
gwo_p = HYPEROPT_DIR / "trials_gwo.json"

if pso_p.exists() and gwo_p.exists():
    pso_t = json.load(open(pso_p))
    gwo_t = json.load(open(gwo_p))
    pso_best = np.maximum.accumulate([t["fitness"] for t in pso_t])
    gwo_best = np.maximum.accumulate([t["fitness"] for t in gwo_t])

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(pso_best)+1), pso_best, lw=2, label=f"PSO  (best {pso_best[-1]:.4f})")
    plt.plot(range(1, len(gwo_best)+1), gwo_best, lw=2, label=f"GWO  (best {gwo_best[-1]:.4f})")
    plt.xlabel("trial"); plt.ylabel("best val macro F1 so far")
    plt.title("Nature-inspired HPO: PSO vs GWO convergence")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    cmp_path = HYPEROPT_DIR / "comparison_pso_vs_gwo.png"
    plt.savefig(cmp_path, dpi=150); plt.show()

    pso_best_hp = json.load(open(HYPEROPT_DIR / "best_hparams_pso.json"))
    gwo_best_hp = json.load(open(HYPEROPT_DIR / "best_hparams_gwo.json"))
    print(f"{'metric':<22}{'PSO':>12}{'GWO':>12}")
    print(f"{'best val macro F1':<22}{pso_best_hp['_val_macro_F1']:>12.4f}{gwo_best_hp['_val_macro_F1']:>12.4f}")
    print(f"{'n_trials':<22}{pso_best_hp['_n_trials']:>12}{gwo_best_hp['_n_trials']:>12}")
    print(f"{'elapsed_min':<22}{pso_best_hp['_elapsed_min']:>12}{gwo_best_hp['_elapsed_min']:>12}")
    for k in ["lr", "dropout", "weight_decay", "lstm_hidden", "seq_len", "train_stride"]:
        v_p, v_g = pso_best_hp[k], gwo_best_hp[k]
        v_ps = f"{v_p:.3e}" if isinstance(v_p, float) and v_p < 0.01 else f"{v_p:.4g}"
        v_gs = f"{v_g:.3e}" if isinstance(v_g, float) and v_g < 0.01 else f"{v_g:.4g}"
        print(f"{k:<22}{v_ps:>12}{v_gs:>12}")

    with tracker.start_run(run_name="pso_vs_gwo_comparison"):
        tracker.log_metric("pso_best_val_mf1", pso_best_hp["_val_macro_F1"])
        tracker.log_metric("gwo_best_val_mf1", gwo_best_hp["_val_macro_F1"])
        tracker.log_artifact(cmp_path)
else:
    have = [p.name for p in (pso_p, gwo_p) if p.exists()]
    miss = [p.name for p in (pso_p, gwo_p) if not p.exists()]
    print(f"Have: {have}\\nStill need: {miss}")
    print(f"Run the notebook again with the other SEARCH_ALGO value.")''')

md("""## Next steps

1. **Download the best_hparams_<algo>.json** for the algorithm with the higher val mF1.
   *(Also download `output/hyperopt/mlruns/` if you want the full MLflow database. View
   it locally with `mlflow ui --backend-store-uri ./mlruns` → browse to `http://localhost:5000`.)*
2. Open `kaggle_train.ipynb` and replace the hyperparameter constants in the Config cell
   with the values from `best_hparams.json` (`LEARNING_RATE`, `DROPOUT`, `WEIGHT_DECAY`,
   `LSTM_HIDDEN`, `SEQ_LEN`, `TRAIN_STRIDE`).
3. Run the training notebook end-to-end with full epochs to produce the final ONNX.
4. Include `convergence.png` and `trials.json` in the report's hyperparameter-optimization
   section.

For the report's methodology paragraph:

> "Hyperparameter optimization was performed using Particle Swarm Optimization
> (Kennedy & Eberhart, 1995) via the `niapy` library, a nature-inspired
> metaheuristic that searches the 6-dimensional space of {learning rate,
> dropout, weight decay, LSTM hidden size, sequence length, training stride}.
> The swarm consisted of 10 particles evolved over 10 iterations
> (100 candidate evaluations), with each candidate evaluated by an 8-epoch
> training run with patience-3 early stopping on BIDSleep validation macro-F1.\"""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).parent / "kaggle_hyperopt.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB, {len(CELLS)} cells)")
