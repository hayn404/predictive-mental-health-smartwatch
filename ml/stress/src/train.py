"""
Seren ML Pipeline — Step 3: Model Training
=============================================
Trains an XGBoost model for stress prediction and exports it
as a JSON decision tree file for on-device TypeScript inference.

Features:
  - MLflow experiment tracking (metrics, params, artifacts)
  - Leave-One-Subject-Out CV for unbiased evaluation
  - Configurable via YAML + environment variables
  - Exports JSON tree dump for pure TypeScript inference
"""

import json
import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, precision_score, recall_score
)
from xgboost import XGBClassifier
import joblib
import mlflow
import mlflow.sklearn

from features import FEATURE_ORDER
from config import load_config, PipelineConfig

logger = logging.getLogger(__name__)


# ============================================================
# Data Loading
# ============================================================

def load_features(cfg: PipelineConfig) -> pd.DataFrame:
    """Load features from CSV or generate synthetic data."""
    features_path = Path(cfg.data.features_csv)

    if features_path.exists():
        logger.info(f"Loading features from {features_path}")
        df = pd.read_csv(features_path)
    else:
        # Try loading from WESAD raw data
        wesad_dir = Path(cfg.data.wesad_dir)
        if wesad_dir.exists() and any(wesad_dir.glob("S*")):
            logger.info("WESAD data found. Running preprocessing + feature extraction...")
            from preprocessing import load_all_wesad_subjects
            from features import extract_features_dataframe
            windows = load_all_wesad_subjects(str(wesad_dir), cfg.data.window_sec)
            df = extract_features_dataframe(windows)
            # Cache for next run
            features_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(features_path, index=False)
            logger.info(f"Cached features to {features_path}")
        else:
            logger.warning("No WESAD data found. Generating synthetic data for training.")
            from synthetic_data import generate_synthetic_dataset
            df = generate_synthetic_dataset(
                n_subjects=cfg.data.n_subjects,
                windows_per_subject=cfg.data.windows_per_subject,
                stress_ratio=cfg.data.stress_ratio,
                seed=cfg.data.seed,
            )

    # Filter to valid stress labels
    df = df[df["stress_binary"].isin([0, 1])].copy()

    # Drop rows with NaN or Inf
    feature_cols = [c for c in FEATURE_ORDER if c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)

    logger.info(f"Dataset: {len(df)} samples | "
                f"Not stressed: {(df['stress_binary']==0).sum()} | "
                f"Stressed: {(df['stress_binary']==1).sum()} | "
                f"Subjects: {df['subject'].nunique()}")

    return df


# ============================================================
# Training with LOSO-CV
# ============================================================

def train_and_evaluate(df: pd.DataFrame, cfg: PipelineConfig) -> dict:
    """
    Train XGBoost with Leave-One-Subject-Out CV and full-data final model.
    """
    feature_cols = [c for c in FEATURE_ORDER if c in df.columns]
    X = df[feature_cols].values
    y = df["stress_binary"].values
    groups = df["subject"].values

    logger.info("=== Leave-One-Subject-Out Cross-Validation ===")
    logo = LeaveOneGroupOut()

    y_pred_all = np.zeros_like(y)
    y_prob_all = np.zeros_like(y, dtype=float)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = _create_xgb_model(cfg, y_train)
        model.fit(X_train_scaled, y_train)

        y_pred_all[test_idx] = model.predict(X_test_scaled)
        y_prob_all[test_idx] = model.predict_proba(X_test_scaled)[:, 1]

        subject = np.unique(groups[test_idx])[0]
        fold_acc = accuracy_score(y_test, y_pred_all[test_idx])
        fold_metrics.append({"subject": subject, "accuracy": fold_acc})
        logger.info(f"  Fold {fold_idx+1} (test={subject}): accuracy={fold_acc:.3f}")

    # Overall CV metrics
    cv_metrics = {
        "cv_accuracy": accuracy_score(y, y_pred_all),
        "cv_f1_weighted": f1_score(y, y_pred_all, average="weighted"),
        "cv_f1_binary": f1_score(y, y_pred_all, average="binary"),
        "cv_precision": precision_score(y, y_pred_all),
        "cv_recall": recall_score(y, y_pred_all),
        "cv_auc_roc": roc_auc_score(y, y_prob_all),
    }

    logger.info(f"\nLOSO-CV Results:")
    for k, v in cv_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y, y_pred_all, target_names=['Not Stressed', 'Stressed'])}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y, y_pred_all)}")

    # ---- Train Final Model on All Data ----
    logger.info("\n=== Training Final Model on All Data ===")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    final_model = _create_xgb_model(cfg, y)
    final_model.fit(X_scaled, y)

    # Feature importance
    importances = final_model.feature_importances_
    importance_dict = dict(zip(feature_cols, importances.tolist()))
    importance_order = np.argsort(importances)[::-1]

    logger.info("\nTop 10 Feature Importances:")
    for i in importance_order[:10]:
        logger.info(f"  {feature_cols[i]:25s} {importances[i]:.4f}")

    return {
        "model": final_model,
        "scaler": final_scaler,
        "feature_cols": feature_cols,
        "importances": importance_dict,
        "fold_metrics": fold_metrics,
        **cv_metrics,
    }


def _create_xgb_model(cfg: PipelineConfig, y_train: np.ndarray) -> XGBClassifier:
    """Create XGBoost classifier with config hyperparameters."""
    return XGBClassifier(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
        subsample=cfg.model.subsample,
        colsample_bytree=cfg.model.colsample_bytree,
        min_child_weight=cfg.model.min_child_weight,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=cfg.model.seed,
        eval_metric=cfg.model.eval_metric,
        enable_categorical=False,
    )


# ============================================================
# Model Export (JSON for TypeScript)
# ============================================================

def export_model_json(model: XGBClassifier, scaler: StandardScaler,
                      feature_cols: list, metrics: dict, cfg: PipelineConfig) -> Path:
    """Export XGBoost model as JSON for TypeScript on-device inference."""
    output_path = Path(cfg.export.model_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    booster = model.get_booster()
    tree_dump = booster.get_dump(dump_format="json")
    trees = [json.loads(tree) for tree in tree_dump]

    normalization = {
        "mean": dict(zip(feature_cols, scaler.mean_.tolist())),
        "std": dict(zip(feature_cols, scaler.scale_.tolist())),
    }

    model_export = {
        "version": cfg.version,
        "modelType": "xgboost_binary_classifier",
        "task": "stress_detection",
        "features": feature_cols,
        "numFeatures": len(feature_cols),
        "numTrees": len(trees),
        "baseScore": 0.5,
        "learningRate": model.learning_rate,
        "normalization": normalization,
        "trees": trees,
        "metrics": {k: v for k, v in metrics.items()
                    if k.startswith("cv_") and isinstance(v, (int, float))},
        "importances": metrics.get("importances", {}),
        "stressLevels": {
            "low": {"min": 0, "max": 25, "label": "Low", "color": "#35e27e"},
            "moderate": {"min": 25, "max": 50, "label": "Moderate", "color": "#9B8EC4"},
            "elevated": {"min": 50, "max": 75, "label": "Elevated", "color": "#E8A87C"},
            "high": {"min": 75, "max": 100, "label": "High", "color": "#C4897B"},
        },
    }

    model_path = output_path / cfg.export.model_json
    with open(model_path, "w") as f:
        json.dump(model_export, f, indent=2)
    logger.info(f"Exported model to {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    metadata = {
        "version": model_export["version"],
        "features": feature_cols,
        "normalization": normalization,
        "metrics": model_export["metrics"],
        "importances": model_export["importances"],
        "stressLevels": model_export["stressLevels"],
    }
    metadata_path = output_path / cfg.export.metadata_json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    joblib.dump(model, output_path / "stress_model.joblib")
    joblib.dump(scaler, output_path / "scaler.joblib")
    logger.info("Exported sklearn artifacts")

    return model_path


def copy_model_to_app(cfg: PipelineConfig):
    """Copy exported model JSON to React Native app assets."""
    import shutil
    src = Path(cfg.export.model_dir) / cfg.export.model_json
    dst_dir = Path(cfg.export.app_assets_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if src.exists():
        shutil.copy2(src, dst_dir / cfg.export.model_json)
        meta_src = Path(cfg.export.model_dir) / cfg.export.metadata_json
        if meta_src.exists():
            shutil.copy2(meta_src, dst_dir / cfg.export.metadata_json)
        logger.info(f"Copied model to app assets: {dst_dir}")
    else:
        logger.error(f"Model file not found: {src}")


# ============================================================
# MLflow Integration
# ============================================================

def run_with_mlflow(cfg: PipelineConfig):
    """Execute the full pipeline with MLflow tracking."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"stress-model-v{cfg.version}") as run:
        logger.info(f"MLflow run: {run.info.run_id}")

        mlflow.log_params({
            "n_estimators": cfg.model.n_estimators,
            "max_depth": cfg.model.max_depth,
            "learning_rate": cfg.model.learning_rate,
            "subsample": cfg.model.subsample,
            "colsample_bytree": cfg.model.colsample_bytree,
            "min_child_weight": cfg.model.min_child_weight,
            "window_sec": cfg.data.window_sec,
            "cv_strategy": "LOSO",
        })

        df = load_features(cfg)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_subjects", df["subject"].nunique())

        results = train_and_evaluate(df, cfg)

        mlflow.log_metrics({k: v for k, v in results.items()
                           if k.startswith("cv_") and isinstance(v, (int, float))})

        for fm in results["fold_metrics"]:
            mlflow.log_metric(f"fold_acc_{fm['subject']}", fm["accuracy"])

        model_path = export_model_json(
            model=results["model"], scaler=results["scaler"],
            feature_cols=results["feature_cols"], metrics=results, cfg=cfg,
        )

        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(results["model"], "xgboost_model")

        copy_model_to_app(cfg)

        _print_summary(results, run.info.run_id, model_path)

    return results


def run_without_mlflow(cfg: PipelineConfig):
    """Execute training without MLflow."""
    df = load_features(cfg)
    results = train_and_evaluate(df, cfg)

    model_path = export_model_json(
        model=results["model"], scaler=results["scaler"],
        feature_cols=results["feature_cols"], metrics=results, cfg=cfg,
    )
    copy_model_to_app(cfg)
    _print_summary(results, None, model_path)

    return results


def _print_summary(results: dict, run_id: str | None, model_path: Path):
    logger.info(f"\n{'='*50}")
    logger.info(f"Training Complete!")
    logger.info(f"  LOSO-CV Accuracy:  {results['cv_accuracy']:.4f}")
    logger.info(f"  LOSO-CV AUC-ROC:   {results['cv_auc_roc']:.4f}")
    logger.info(f"  LOSO-CV F1:        {results['cv_f1_weighted']:.4f}")
    logger.info(f"  LOSO-CV Precision: {results['cv_precision']:.4f}")
    logger.info(f"  LOSO-CV Recall:    {results['cv_recall']:.4f}")
    if run_id:
        logger.info(f"  MLflow Run ID:     {run_id}")
    logger.info(f"  Model exported:    {model_path}")
    logger.info(f"{'='*50}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config(config_path)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    use_mlflow = os.environ.get("USE_MLFLOW", "true").lower() == "true"

    if use_mlflow:
        try:
            run_with_mlflow(cfg)
        except Exception as e:
            logger.warning(f"MLflow connection failed ({e}), running without tracking")
            run_without_mlflow(cfg)
    else:
        run_without_mlflow(cfg)
