"""
Seren ML Pipeline — Configuration Manager
============================================
Loads YAML config and provides typed access to all settings.
Supports environment variable overrides.
"""

import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

ML_ROOT = Path(__file__).parent.parent
CONFIG_DIR = ML_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


@dataclass
class DataConfig:
    wesad_dir: str = "/app/data/wesad"
    swell_csv: str = "/app/data/swell/swell_hrv.csv"
    features_csv: str = "/app/data/wesad_features.csv"
    windows_pkl: str = "/app/data/wesad_windows.pkl"
    window_sec: int = 60        # 60-s windows match watch polling cadence (notebook default)
    overlap: float = 0.0
    label_purity: float = 0.8

    # Dataset selection — mirrors notebook TRAIN_ON / EVAL_ON / USE_* flags
    train_on: str = "SIPD+PhysioStress"   # "+"-joined fusion; SIPD+PhysioStress is the thesis config
    eval_on: str = "WESAD"                # held-out cross-dataset test set
    use_physiostress: bool = True
    physiostress_dir: str = "/app/data/physiostress"

    # Feature set flags — must match notebook USE_FREQ_DOMAIN / USE_MORPHOLOGY
    normalization: str = "per_subject"    # "global" | "per_subject"
    use_freq_domain: bool = False         # 7 Welch PSD features — off = 14-feature lean model
    use_morphology: bool = False          # 8 PPG pulse-shape features — off for watch compat

    # Threshold calibration
    calibrate_threshold: bool = True
    threshold_objective: str = "prior"   # "prior" | "f1" | "youden" | "balanced"

    # Hyperparameter tuning (nature-inspired PSO/GWO, matches notebook)
    run_tuning: bool = True
    tuners: tuple = ("pso", "gwo")       # higher-CV-AUC one wins
    tune_pop: int = 8
    tune_iters: int = 12

    # Synthetic data fallback (CI smoke-test only)
    n_subjects: int = 15
    windows_per_subject: int = 40
    stress_ratio: float = 0.35
    seed: int = 42


@dataclass
class ModelConfig:
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    eval_metric: str = "logloss"
    seed: int = 42


@dataclass
class ExportConfig:
    model_dir: str = "/app/models"
    model_json: str = "stress_model.json"
    metadata_json: str = "model_metadata.json"
    app_assets_dir: str = "/app/assets/ml"


@dataclass
class MLflowConfig:
    tracking_uri: str = "http://mlflow:5000"
    experiment_name: str = "seren-stress-detection"


@dataclass
class PipelineConfig:
    project_name: str = "seren-stress-model"
    version: str = "1.0.0"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    log_level: str = "INFO"


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load config from YAML file with environment variable overrides.

    Priority: env vars > config file > defaults
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG

    raw = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {path}")
    else:
        logger.warning(f"Config file not found: {path}, using defaults")

    cfg = PipelineConfig()

    # Project
    project = raw.get("project", {})
    cfg.project_name = project.get("name", cfg.project_name)
    cfg.version = project.get("version", cfg.version)

    # Data
    data = raw.get("data", {})
    cfg.data.wesad_dir = os.environ.get("WESAD_DIR", data.get("wesad_dir", cfg.data.wesad_dir))
    cfg.data.swell_csv = os.environ.get("SWELL_CSV", data.get("swell_csv", cfg.data.swell_csv))
    cfg.data.features_csv = data.get("features_csv", cfg.data.features_csv)
    cfg.data.windows_pkl = data.get("windows_pkl", cfg.data.windows_pkl)
    cfg.data.window_sec = data.get("window_sec", cfg.data.window_sec)
    cfg.data.overlap = data.get("overlap", cfg.data.overlap)
    cfg.data.label_purity = data.get("label_purity", cfg.data.label_purity)
    cfg.data.train_on = os.environ.get("TRAIN_ON", data.get("train_on", cfg.data.train_on))
    cfg.data.eval_on = os.environ.get("EVAL_ON", data.get("final_test", cfg.data.eval_on))
    cfg.data.use_physiostress = data.get("use_physiostress", cfg.data.use_physiostress)
    cfg.data.physiostress_dir = os.environ.get("PHYSIOSTRESS_DIR", data.get("physiostress_dir", cfg.data.physiostress_dir))
    cfg.data.normalization = os.environ.get("NORMALIZATION", data.get("normalization", cfg.data.normalization))
    cfg.data.use_freq_domain = data.get("use_freq_domain", cfg.data.use_freq_domain)
    cfg.data.use_morphology = data.get("use_morphology", cfg.data.use_morphology)
    cfg.data.calibrate_threshold = data.get("calibrate_threshold", cfg.data.calibrate_threshold)
    cfg.data.threshold_objective = data.get("threshold_objective", cfg.data.threshold_objective)
    cfg.data.run_tuning = data.get("run_tuning", cfg.data.run_tuning)
    cfg.data.tune_pop = int(data.get("tune_pop", cfg.data.tune_pop))
    cfg.data.tune_iters = int(data.get("tune_iters", cfg.data.tune_iters))

    syn = data.get("synthetic", {})
    cfg.data.n_subjects = syn.get("n_subjects", cfg.data.n_subjects)
    cfg.data.windows_per_subject = syn.get("windows_per_subject", cfg.data.windows_per_subject)
    cfg.data.stress_ratio = syn.get("stress_ratio", cfg.data.stress_ratio)
    cfg.data.seed = syn.get("seed", cfg.data.seed)

    # Model
    model = raw.get("model", {})
    cfg.model.n_estimators = int(os.environ.get("N_ESTIMATORS", model.get("n_estimators", cfg.model.n_estimators)))
    cfg.model.max_depth = int(os.environ.get("MAX_DEPTH", model.get("max_depth", cfg.model.max_depth)))
    cfg.model.learning_rate = float(os.environ.get("LEARNING_RATE", model.get("learning_rate", cfg.model.learning_rate)))
    cfg.model.subsample = model.get("subsample", cfg.model.subsample)
    cfg.model.colsample_bytree = model.get("colsample_bytree", cfg.model.colsample_bytree)
    cfg.model.min_child_weight = model.get("min_child_weight", cfg.model.min_child_weight)
    cfg.model.eval_metric = model.get("eval_metric", cfg.model.eval_metric)
    cfg.model.seed = model.get("seed", cfg.model.seed)

    # Export
    export = raw.get("export", {})
    cfg.export.model_dir = os.environ.get("MODEL_DIR", export.get("model_dir", cfg.export.model_dir))
    cfg.export.model_json = export.get("model_json", cfg.export.model_json)
    cfg.export.metadata_json = export.get("metadata_json", cfg.export.metadata_json)
    cfg.export.app_assets_dir = os.environ.get("APP_ASSETS_DIR", export.get("app_assets_dir", cfg.export.app_assets_dir))

    # MLflow
    mf = raw.get("mlflow", {})
    cfg.mlflow.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mf.get("tracking_uri", cfg.mlflow.tracking_uri))
    cfg.mlflow.experiment_name = mf.get("experiment_name", cfg.mlflow.experiment_name)

    # Logging
    log = raw.get("logging", {})
    cfg.log_level = os.environ.get("LOG_LEVEL", log.get("level", cfg.log_level))

    # Resolve relative paths from ML_ROOT
    cfg.data.wesad_dir = str((ML_ROOT / cfg.data.wesad_dir).resolve())
    cfg.data.swell_csv = str((ML_ROOT / cfg.data.swell_csv).resolve())
    cfg.data.features_csv = str((ML_ROOT / cfg.data.features_csv).resolve())
    cfg.data.windows_pkl = str((ML_ROOT / cfg.data.windows_pkl).resolve())
    cfg.export.model_dir = str((ML_ROOT / cfg.export.model_dir).resolve())
    cfg.export.app_assets_dir = str((ML_ROOT / cfg.export.app_assets_dir).resolve())

    return cfg
