"""
Seren — Sleep Stage Classification: Configuration
===================================================
Central config for hyperparameters, paths, and constants.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "dreamt"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
EXPORT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "ml"

# ── Signal Config ──────────────────────────────────────────────
SAMPLE_RATE = 100          # Hz (data_100Hz variant)
EPOCH_SEC = 30             # AASM standard epoch
EPOCH_SAMPLES = SAMPLE_RATE * EPOCH_SEC  # 3000 samples per epoch
OVERLAP = 0.0              # 0% for evaluation alignment with labels
TRAIN_OVERLAP = 0.5        # 50% overlap for training augmentation

# Signals used (drop EDA — not available on Galaxy Watch)
SIGNALS_RAW = ["BVP", "ACC"]       # High-rate raw channels
SIGNALS_AUX = ["HR", "TEMP"]       # Low-rate auxiliary channels
NUM_RAW_CHANNELS = 2               # BVP + ACC magnitude
NUM_AUX_FEATURES = 6               # HR_mean, HR_std, ACC_mean, ACC_std, TEMP_mean, TEMP_slope

# BVP bandpass filter
BVP_LOW_HZ = 0.5
BVP_HIGH_HZ = 8.0

# ── Sleep Stage Labels ────────────────────────────────────────
NUM_CLASSES = 5
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # DREAMT label → model index

# Mapping from 5-class model output to app's SleepStageType
APP_STAGE_MAP = {
    0: "awake",
    1: "light",   # N1 → light
    2: "light",   # N2 → light
    3: "deep",    # N3 → deep
    4: "rem",     # REM → rem
}

# ── Model Architecture ────────────────────────────────────────
CNN_CHANNELS = [32, 64, 128]
CNN_KERNELS = [50, 10, 5]
CNN_STRIDES = [5, 2, 2]
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
CLASSIFIER_HIDDEN = 64
DROPOUT = 0.2

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
SCHEDULER = "cosine"       # CosineAnnealingLR
VAL_SPLIT = 0.2            # 20% of subjects for validation
RANDOM_SEED = 42

# ── Artifact Rejection ────────────────────────────────────────
MAX_FLAT_RATIO = 0.3       # Reject epoch if >30% of BVP is flat/clipped
BVP_CLIP_THRESHOLD = 0.01  # Std below this = flat signal
