"""
Seren — Sleep Stage Classification: Configuration
===================================================
Mirrors the Kaggle training notebook (seren_sleep_kaggle.ipynb) v3.2 exactly.

Key contract:
  - The model consumes PRE-NORMALIZED cached epoch features (per-night median/MAD
    normalization is applied UPSTREAM by prepare_features.py at cache-build time).
  - 11 features/epoch: 10 kept from the cache (immobility_frac dropped after XAI)
    + time_of_night appended at load time.
  - 4 classes: Wake / Light / Deep / REM.
  - Architecture: TCN (2x Conv1d) + BiGRU + BatchNorm head (tcn_bigru_bn_v1).

Hyperparameters default to the notebook's baked PSO-best values and can be
overridden from params.yaml via apply_params().
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"             # checkpoints, history, curves
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
# EXPORT_DIR is set by train.py from the --out CLI arg (assets/ml/sleep)
EXPORT_DIR = Path(__file__).resolve().parents[2] / "assets" / "ml" / "sleep"

# ── Labels (4-class) ───────────────────────────────────────────
NUM_CLASSES = 4
STAGE_NAMES = ["Wake", "Light", "Deep", "REM"]
APP_STAGE_MAP = {0: "awake", 1: "light", 2: "deep", 3: "rem"}

# ── Feature contract ───────────────────────────────────────────
# The cache was extracted with 11 columns; XAI confirmed immobility_frac is dead
# weight (pooled |IG| ~0%), so we slice it out at load time (cache build is ~17h).
CACHE_FEATURES_RAW = [
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    "hr_succdiff_std", "hr_delta_prev",
    "act_count", "immobility_frac", "act_max", "act_std",
]
DROP_FROM_CACHE = {"immobility_frac"}
KEEP_CACHE_IDX = [i for i, n in enumerate(CACHE_FEATURES_RAW) if n not in DROP_FROM_CACHE]
CACHE_FEATURES = [n for n in CACHE_FEATURES_RAW if n not in DROP_FROM_CACHE]   # 10
FEATURE_NAMES = CACHE_FEATURES + ["time_of_night"]                            # 11
NUM_FEATURES = len(FEATURE_NAMES)
EPOCH_SEC = 30

# ── Sequence config ────────────────────────────────────────────
SEQ_LEN = 41          # ~20.5 min temporal context (half a sleep cycle)
TRAIN_STRIDE = 5      # PSO found stride 1 over-augments; stride 5 is the sweet spot
# eval/test use disjoint sequences (stride == SEQ_LEN) — derived in dataset.py

# ── Model architecture (tcn_bigru_bn_v1) ───────────────────────
TCN_CHANNELS = 64
TCN_KERNEL = 5
LSTM_HIDDEN = 48      # parameterizes the BiGRU (name kept for back-compat)
LSTM_LAYERS = 2
DROPOUT = 0.258       # PSO best (110 trials, 11-feature lean model)
ARCH_TAG = "tcn_bigru_bn_v1"

# ── Training ───────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1.03e-3   # PSO best (110 trials, lean model)
WEIGHT_DECAY = 1.72e-5    # PSO best (110 trials, lean model)
NUM_EPOCHS = 80
EARLY_STOP_PATIENCE = 15
VAL_SPLIT = 0.15
RANDOM_SEED = 42
GRAD_CLIP = 1.0


def apply_params(params_path):
    """Override hyperparameters from params.yaml's `sleep:` block.
    Mutates module globals so model.py / dataset.py (which read cfg.* at
    instantiation) pick up the values. Call this BEFORE building anything.
    """
    import yaml
    global LEARNING_RATE, DROPOUT, LSTM_HIDDEN, SEQ_LEN, TRAIN_STRIDE
    global WEIGHT_DECAY, NUM_EPOCHS, ARCH_TAG, TCN_CHANNELS, TCN_KERNEL
    global LSTM_LAYERS, BATCH_SIZE, EARLY_STOP_PATIENCE, VAL_SPLIT, RANDOM_SEED

    if params_path is None or not Path(params_path).exists():
        return
    with open(params_path) as f:
        raw = yaml.safe_load(f) or {}
    s = raw.get("sleep", {}) or {}

    LEARNING_RATE = float(s.get("lr", LEARNING_RATE))
    DROPOUT = float(s.get("dropout", DROPOUT))
    LSTM_HIDDEN = int(s.get("lstm_hidden", LSTM_HIDDEN))
    SEQ_LEN = int(s.get("seq_len", SEQ_LEN))
    TRAIN_STRIDE = int(s.get("train_stride", TRAIN_STRIDE))
    WEIGHT_DECAY = float(s.get("weight_decay", WEIGHT_DECAY))
    NUM_EPOCHS = int(s.get("epochs", NUM_EPOCHS))
    ARCH_TAG = str(s.get("arch", ARCH_TAG))
    TCN_CHANNELS = int(s.get("tcn_channels", TCN_CHANNELS))
    TCN_KERNEL = int(s.get("tcn_kernel", TCN_KERNEL))
    LSTM_LAYERS = int(s.get("lstm_layers", LSTM_LAYERS))
    BATCH_SIZE = int(s.get("batch_size", BATCH_SIZE))
    EARLY_STOP_PATIENCE = int(s.get("early_stop_patience", EARLY_STOP_PATIENCE))
    VAL_SPLIT = float(s.get("val_split", VAL_SPLIT))
    RANDOM_SEED = int(s.get("seed", RANDOM_SEED))
