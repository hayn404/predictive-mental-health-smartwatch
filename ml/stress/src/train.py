"""
Seren ML Pipeline — Stress Model Training
==========================================
Ports the Kaggle notebook (seren_stress_kaggle.ipynb) cross-dataset flow into
the DVC pipeline contract.

Protocol (matches the notebook + ml/stress/metrics.json):
  - Train on a fusion of dev datasets (default SIPD + PhysioStress).
  - Report within-train Leave-One-Subject-Out AUC (cv_auc_roc_loso).
  - Report the headline held-out cross-dataset AUC on WESAD (heldout_auc),
    using per-subject z-normalization (the thesis number).
  - Calibrate the decision threshold on out-of-fold LOSO probabilities.
  - Export a 14-feature XGBoost JSON for on-device TypeScript inference,
    trained with GLOBAL StandardScaler normalization (drop-in for the watch).

DVC contract:
  python ml/stress/src/train.py --params params.yaml --data ml/data/features/stress \
      --out assets/ml/stress --metrics ml/stress/metrics.json
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, precision_score, recall_score
)
from xgboost import XGBClassifier
import joblib

from features import FEATURE_ORDER
from config import load_config, PipelineConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ci"))
import viz  # noqa: E402
from bootstrap import bootstrap_ci  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================
# Normalization + threshold helpers (mirror the notebook)
# ============================================================

def per_subject_z(df: pd.DataFrame, cols: list, group_col: str = "subject", eps: float = 1e-8) -> pd.DataFrame:
    """Z-normalize each feature within each subject's rows (notebook per_subject_z)."""
    out = df.copy()
    gp = out.groupby(group_col)[cols]
    mu = gp.transform("mean")
    sd = gp.transform("std").replace(0, np.nan)
    out[cols] = ((out[cols] - mu) / (sd + eps)).fillna(0.0)
    return out


def pick_threshold(y_true: np.ndarray, prob: np.ndarray, objective: str = "f1") -> float:
    """Pick a decision threshold on out-of-fold probs (no test leakage)."""
    grid = np.unique(np.round(np.quantile(prob, np.linspace(0, 1, 101)), 4))
    grid = grid[(grid > 0) & (grid < 1)]
    best_t, best_s = 0.5, -1.0
    for t in grid:
        pred = (prob >= t).astype(int)
        if objective == "youden":
            tp = ((pred == 1) & (y_true == 1)).sum(); fn = ((pred == 0) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum(); tn = ((pred == 0) & (y_true == 0)).sum()
            s = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
        elif objective == "balanced":
            from sklearn.metrics import balanced_accuracy_score
            s = balanced_accuracy_score(y_true, pred)
        else:  # f1
            s = f1_score(y_true, pred, zero_division=0)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


# ============================================================
# Data Loading (per-dataset CSVs → train fusion + held-out)
# ============================================================

def _load_one(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name.lower()}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {path}")
    df = df_clean(pd.read_csv(path))
    logger.info(f"  {name}: {len(df)} windows / {df['subject'].nunique()} subjects "
                f"| balance {df['stress_binary'].value_counts().to_dict()}")
    return df


def df_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid stress labels + drop NaN/Inf rows on the 14 model features."""
    df = df[df["stress_binary"].isin([0, 1])].copy()
    feats = [c for c in FEATURE_ORDER if c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats)
    return df


def load_datasets(cfg: PipelineConfig, data_dir: str):
    """Load the train-fusion dataframe + the held-out eval dataframe."""
    data_dir = Path(data_dir)
    train_names = [n for n in cfg.data.train_on.split("+") if n]
    eval_name = cfg.data.eval_on

    logger.info(f"=== Loading datasets from {data_dir} ===")
    train_parts = [_load_one(data_dir, n) for n in train_names]
    train_df = (train_parts[0] if len(train_parts) == 1
                else pd.concat(train_parts, ignore_index=True))

    eval_df = None
    if eval_name and eval_name not in train_names:
        eval_df = _load_one(data_dir, eval_name)

    logger.info(f"TRAIN = {'+'.join(train_names)}: {len(train_df)} windows / "
                f"{train_df['subject'].nunique()} subjects")
    logger.info(f"EVAL  = {eval_name if eval_df is not None else '(none)'}"
                + (f": {len(eval_df)} windows" if eval_df is not None else ""))
    return train_df, eval_df


# ============================================================
# Training + cross-dataset evaluation
# ============================================================

def base_params(cfg: PipelineConfig) -> dict:
    """Hand-set BASE_PARAMS (matches notebook cell 10)."""
    return dict(
        n_estimators=cfg.model.n_estimators, max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate, subsample=cfg.model.subsample,
        colsample_bytree=cfg.model.colsample_bytree,
        min_child_weight=cfg.model.min_child_weight, gamma=0.0, reg_lambda=1.0,
    )


def _create_xgb_model(cfg: PipelineConfig, y_train: np.ndarray, params: dict = None) -> XGBClassifier:
    p = params if params is not None else base_params(cfg)
    return XGBClassifier(
        **p,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=cfg.model.seed,
        eval_metric=cfg.model.eval_metric,
        enable_categorical=False,
    )


# ============================================================
# Nature-inspired hyperparameter tuning (PSO / GWO) — notebook cell 11
# ============================================================

def tune_hyperparameters(train_df: pd.DataFrame, feature_cols: list, cfg: PipelineConfig) -> dict:
    """Maximize subject-disjoint K-fold CV AUC on the TRAIN set (no eval leakage).
    Returns the winning (PSO vs GWO) parameter dict merged onto BASE_PARAMS.
    Deterministic: seeded with cfg.model.seed.
    """
    HP = [("max_depth", 3, 10, "int"), ("learning_rate", 0.01, 0.30, "float"),
          ("n_estimators", 50, 300, "int"), ("subsample", 0.5, 1.0, "float"),
          ("colsample_bytree", 0.5, 1.0, "float"), ("min_child_weight", 1, 10, "int"),
          ("gamma", 0.0, 5.0, "float"), ("reg_lambda", 0.0, 5.0, "float")]
    LB = np.array([h[1] for h in HP], float)
    UB = np.array([h[2] for h in HP], float)
    seed = cfg.model.seed
    per_subject = cfg.data.normalization == "per_subject"

    def decode(v):
        return {n: (int(round(float(np.clip(x, lo, hi)))) if t == "int" else float(np.clip(x, lo, hi)))
                for x, (n, lo, hi, t) in zip(v, HP)}

    def clip(x):
        return np.minimum(np.maximum(x, LB), UB)

    Xt = train_df[feature_cols].values
    yt = train_df["stress_binary"].values
    gt = train_df["subject"].values
    nsplits = max(2, min(4, len(np.unique(gt))))

    def objective(vec):
        p = decode(vec)
        skf = StratifiedGroupKFold(n_splits=nsplits, shuffle=True, random_state=seed)
        aucs = []
        for tr, te in skf.split(Xt, yt, gt):
            if per_subject:
                Xtr, Xte = Xt[tr].copy(), Xt[te].copy()
                for idx, Xs in ((tr, Xtr), (te, Xte)):
                    for gid in np.unique(gt[idx]):
                        mm = gt[idx] == gid
                        mu = Xs[mm].mean(0); sd = Xs[mm].std(0); sd[sd == 0] = 1.0
                        Xs[mm] = (Xs[mm] - mu) / sd
                Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)
            else:
                sc = StandardScaler()
                Xtr = np.nan_to_num(sc.fit_transform(Xt[tr]))
                Xte = np.nan_to_num(sc.transform(Xt[te]))
            ytr, yte = yt[tr], yt[te]
            if len(set(yte)) < 2 or len(set(ytr)) < 2:
                continue
            spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            m = XGBClassifier(**p, scale_pos_weight=spw, random_state=seed,
                              eval_metric="logloss", n_jobs=1)
            m.fit(Xtr, ytr)
            aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        return 1.0 - (np.mean(aucs) if aucs else 0.0)

    def pso(obj, pop, iters, w=0.7, c1=1.5, c2=1.5):
        rng = np.random.RandomState(seed)
        dim = len(LB); span = UB - LB
        X = rng.uniform(LB, UB, (pop, dim)); V = rng.uniform(-span, span, (pop, dim)) * 0.1
        pb = X.copy(); pbf = np.array([obj(x) for x in X])
        gi = pbf.argmin(); gb = pb[gi].copy(); gbf = pbf[gi]
        for it in range(iters):
            r1, r2 = rng.rand(pop, dim), rng.rand(pop, dim)
            V = w * V + c1 * r1 * (pb - X) + c2 * r2 * (gb - X)
            X = clip(X + V)
            f = np.array([obj(x) for x in X])
            im = f < pbf; pb[im], pbf[im] = X[im], f[im]
            if pbf.min() < gbf:
                gi = pbf.argmin(); gb = pb[gi].copy(); gbf = pbf[gi]
            logger.info(f"  PSO {it+1}/{iters}: CV AUC={1-gbf:.4f}")
        return gb, gbf

    def gwo(obj, pop, iters):
        rng = np.random.RandomState(seed)
        dim = len(LB)
        X = rng.uniform(LB, UB, (pop, dim)); f = np.array([obj(x) for x in X])
        o = f.argsort()
        al, be, de = X[o[0]].copy(), X[o[1]].copy(), X[o[2]].copy()
        af, bf, dff = f[o[0]], f[o[1]], f[o[2]]
        for it in range(iters):
            a = 2 - 2 * it / iters
            A1 = 2 * a * rng.rand(pop, dim) - a; C1 = 2 * rng.rand(pop, dim)
            A2 = 2 * a * rng.rand(pop, dim) - a; C2 = 2 * rng.rand(pop, dim)
            A3 = 2 * a * rng.rand(pop, dim) - a; C3 = 2 * rng.rand(pop, dim)
            X1 = al - A1 * np.abs(C1 * al - X)
            X2 = be - A2 * np.abs(C2 * be - X)
            X3 = de - A3 * np.abs(C3 * de - X)
            X = clip((X1 + X2 + X3) / 3); f = np.array([obj(x) for x in X])
            for i in range(pop):
                if f[i] < af:
                    de, dff = be.copy(), bf; be, bf = al.copy(), af; al, af = X[i].copy(), f[i]
                elif f[i] < bf:
                    de, dff = be.copy(), bf; be, bf = X[i].copy(), f[i]
                elif f[i] < dff:
                    de, dff = X[i].copy(), f[i]
            logger.info(f"  GWO {it+1}/{iters}: CV AUC={1-af:.4f}")
        return al, af

    runners = {"PSO": pso, "GWO": gwo}
    results = {}
    for name in cfg.data.tuners:
        nm = name.upper()
        if nm not in runners:
            continue
        logger.info(f"[{nm}] tuning: {cfg.data.tune_pop} agents x {cfg.data.tune_iters} iters")
        vec, fit = runners[nm](objective, cfg.data.tune_pop, cfg.data.tune_iters)
        results[nm] = dict(params=decode(vec), cv_auc=float(1 - fit))
        logger.info(f"[{nm}] best CV AUC={1-fit:.4f}")

    best = max(results, key=lambda k: results[k]["cv_auc"])
    tuned = base_params(cfg)
    tuned.update(results[best]["params"])
    logger.info(f"WINNER: {best} (CV AUC={results[best]['cv_auc']:.4f}) -> {tuned}")
    return tuned


def train_and_evaluate(train_df: pd.DataFrame, eval_df, cfg: PipelineConfig) -> dict:
    feature_cols = [c for c in FEATURE_ORDER if c in train_df.columns]
    y = train_df["stress_binary"].values
    groups = train_df["subject"].values
    per_subject = cfg.data.normalization == "per_subject"
    train_prior = float((y == 1).mean())

    # ---- Hyperparameter tuning (PSO/GWO on subject-disjoint CV AUC) ----
    if cfg.data.run_tuning:
        logger.info("=== Hyperparameter tuning (PSO/GWO) ===")
        params = tune_hyperparameters(train_df, feature_cols, cfg)
    else:
        params = base_params(cfg)
        logger.info(f"Tuning OFF -> base params: {params}")

    # ---- Within-train Leave-One-Subject-Out CV ----
    logger.info(f"=== LOSO-CV ({len(feature_cols)} features, norm={cfg.data.normalization}) ===")
    logo = LeaveOneGroupOut()
    y_pred_all = np.zeros_like(y)
    y_prob_all = np.zeros_like(y, dtype=float)

    for fold_idx, (tr, te) in enumerate(logo.split(train_df, y, groups)):
        if per_subject:
            Xtr = np.nan_to_num(per_subject_z(train_df.iloc[tr], feature_cols)[feature_cols].values)
            Xte = np.nan_to_num(per_subject_z(train_df.iloc[te], feature_cols)[feature_cols].values)
        else:
            sc = StandardScaler()
            Xtr = np.nan_to_num(sc.fit_transform(train_df.iloc[tr][feature_cols].values))
            Xte = np.nan_to_num(sc.transform(train_df.iloc[te][feature_cols].values))
        m = _create_xgb_model(cfg, y[tr], params)
        m.fit(Xtr, y[tr])
        y_pred_all[te] = m.predict(Xte)
        y_prob_all[te] = m.predict_proba(Xte)[:, 1]

    cv_metrics = {
        "cv_accuracy": accuracy_score(y, y_pred_all),
        "cv_f1_weighted": f1_score(y, y_pred_all, average="weighted"),
        "cv_f1_binary": f1_score(y, y_pred_all, average="binary", zero_division=0),
        "cv_precision": precision_score(y, y_pred_all, zero_division=0),
        "cv_recall": recall_score(y, y_pred_all, zero_division=0),
        "cv_auc_roc": roc_auc_score(y, y_prob_all),
    }
    logger.info("LOSO-CV (threshold=0.5):")
    for k, v in cv_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y, y_pred_all)}")

    # ---- Threshold calibration on out-of-fold probs ----
    if not cfg.data.calibrate_threshold:
        best_threshold = 0.5
    elif cfg.data.threshold_objective == "prior":
        # base-rate matching: predict ~train_prior fraction positive (no labels needed)
        best_threshold = float(np.quantile(y_prob_all, 1 - train_prior))
    else:
        best_threshold = pick_threshold(y, y_prob_all, cfg.data.threshold_objective)
    logger.info(f"Calibrated threshold ({cfg.data.threshold_objective}, "
                f"train_prior={train_prior:.2f}) = {best_threshold:.4f}")

    # ---- Held-out cross-dataset eval (the headline number) ----
    heldout = {}
    if eval_df is not None and len(eval_df):
        if per_subject:
            Xtr = np.nan_to_num(per_subject_z(train_df, feature_cols)[feature_cols].values)
            Xte = np.nan_to_num(per_subject_z(eval_df, feature_cols)[feature_cols].values)
        else:
            sc = StandardScaler()
            Xtr = np.nan_to_num(sc.fit_transform(train_df[feature_cols].values))
            Xte = np.nan_to_num(sc.transform(eval_df[feature_cols].values))
        hm = _create_xgb_model(cfg, y, params)
        hm.fit(Xtr, y)
        ye = eval_df["stress_binary"].values
        qe = hm.predict_proba(Xte)[:, 1]
        # For "prior" objective, base-rate match on the TARGET's own scores
        # (adapts to cross-dataset shift, no target labels used).
        if cfg.data.calibrate_threshold and cfg.data.threshold_objective == "prior":
            thr_h = float(np.quantile(qe, 1 - train_prior))
        else:
            thr_h = best_threshold
        heldout = {
            "heldout_auc": roc_auc_score(ye, qe) if len(set(ye)) > 1 else float("nan"),
            "heldout_accuracy": accuracy_score(ye, (qe >= thr_h).astype(int)),
            "heldout_f1": f1_score(ye, (qe >= thr_h).astype(int), zero_division=0),
            "heldout_auc_ci95": bootstrap_ci(roc_auc_score, ye, qe,
                                             groups=eval_df["subject"].values),
        }
        logger.info(f"\n=== HELD-OUT: train {cfg.data.train_on} -> test {cfg.data.eval_on} ===")
        for k, v in heldout.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

    # ---- Final exported model: GLOBAL StandardScaler (drop-in for the watch) ----
    logger.info("\n=== Training final model on all train data (global norm) ===")
    final_scaler = StandardScaler()
    X_scaled = np.nan_to_num(final_scaler.fit_transform(train_df[feature_cols].values))
    final_model = _create_xgb_model(cfg, y, params)
    final_model.fit(X_scaled, y)

    importances = final_model.feature_importances_
    importance_dict = dict(zip(feature_cols, importances.tolist()))

    return {
        "model": final_model,
        "scaler": final_scaler,
        "feature_cols": feature_cols,
        "importances": importance_dict,
        "best_threshold": best_threshold,
        "y_true": y,
        "y_pred": y_pred_all,
        "y_prob": y_prob_all,
        "X_scaled": X_scaled,
        **cv_metrics,
        **heldout,
    }


# ============================================================
# Model Export (JSON for TypeScript)
# ============================================================

def export_model_json(results: dict, cfg: PipelineConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = results["model"]
    scaler = results["scaler"]
    feature_cols = results["feature_cols"]

    booster = model.get_booster()
    trees = [json.loads(t) for t in booster.get_dump(dump_format="json")]
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
        "learningRate": float(model.learning_rate),
        "decisionThreshold": float(results["best_threshold"]),
        "normalization": normalization,
        "trees": trees,
        "metrics": {k: float(v) for k, v in results.items()
                    if (k.startswith("cv_") or k.startswith("heldout_")) and isinstance(v, (int, float))},
        "importances": results.get("importances", {}),
        "stressLevels": {
            "low": {"min": 0, "max": 25, "label": "Low", "color": "#35e27e"},
            "moderate": {"min": 25, "max": 50, "label": "Moderate", "color": "#9B8EC4"},
            "elevated": {"min": 50, "max": 75, "label": "Elevated", "color": "#E8A87C"},
            "high": {"min": 75, "max": 100, "label": "High", "color": "#C4897B"},
        },
    }

    model_path = out_dir / cfg.export.model_json
    with open(model_path, "w") as f:
        json.dump(model_export, f, indent=2)
    logger.info(f"Exported model to {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    metadata = {
        "version": model_export["version"],
        "features": feature_cols,
        "decisionThreshold": model_export["decisionThreshold"],
        "normalization": normalization,
        "metrics": model_export["metrics"],
        "importances": model_export["importances"],
        "stressLevels": model_export["stressLevels"],
    }
    with open(out_dir / cfg.export.metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)
    return model_path


def write_metrics_json(results: dict, cfg: PipelineConfig, metrics_path: str):
    metrics = {
        "model": "stress",
        "eval_set_id": "wesad_v1",
        "heldout_auc": round(float(results.get("heldout_auc", float("nan"))), 4),
        "heldout_auc_ci95": results.get("heldout_auc_ci95"),
        "heldout_dataset": cfg.data.eval_on,
        "train_datasets": [n for n in cfg.data.train_on.split("+") if n],
        "cv_auc_roc_loso": round(float(results["cv_auc_roc"]), 4),
        "n_features": len(results["feature_cols"]),
        "decision_threshold": round(float(results["best_threshold"]), 4),
        "gate": {
            "primary": "heldout_auc",
            "direction": "max",
            "min_delta": 0.0,
            "no_regress": ["cv_auc_roc_loso"],
        },
    }
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Wrote {metrics_path}")


# ============================================================
# params.yaml override + MLflow
# ============================================================

def apply_params_yaml(cfg: PipelineConfig, params_path: str):
    """Override hyperparameters from the root params.yaml `stress:` block."""
    if not params_path or not Path(params_path).exists():
        return
    import yaml
    with open(params_path) as f:
        s = (yaml.safe_load(f) or {}).get("stress", {}) or {}
    cfg.model.n_estimators = int(s.get("n_estimators", cfg.model.n_estimators))
    cfg.model.max_depth = int(s.get("max_depth", cfg.model.max_depth))
    cfg.model.learning_rate = float(s.get("learning_rate", cfg.model.learning_rate))
    cfg.model.subsample = float(s.get("subsample", cfg.model.subsample))
    cfg.model.colsample_bytree = float(s.get("colsample_bytree", cfg.model.colsample_bytree))
    cfg.data.window_sec = int(s.get("window_sec", cfg.data.window_sec))
    cfg.data.label_purity = float(s.get("label_purity", cfg.data.label_purity))
    cfg.data.train_on = str(s.get("train_on", cfg.data.train_on))
    cfg.data.eval_on = str(s.get("final_test", cfg.data.eval_on))
    cfg.data.normalization = str(s.get("normalization", cfg.data.normalization))


def _log_figs_mlflow(figs_dir, results):
    """Log stress figures (+ key metrics) under the 'seren-stress-train' experiment
    so pull_figures.py finds them. Runs in its own MLflow run because the metrics
    run from maybe_mlflow_run() is already closed by this point."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri or os.environ.get("USE_MLFLOW", "true").lower() != "true":
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("seren-stress-train")
        with mlflow.start_run(run_name="stress"):
            for k, v in results.items():
                if (k.startswith("cv_") or k.startswith("heldout_")) and isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
            for p in sorted(Path(figs_dir).glob("*.png")):
                mlflow.log_artifact(str(p), artifact_path="figures")
        print("Logged stress figures to MLflow (seren-stress-train).")
    except Exception as e:
        print(f"MLflow figure logging skipped ({e}).")


def maybe_mlflow_run(results_fn, cfg: PipelineConfig):
    """Run results_fn() inside an MLflow run if a tracking URI is configured."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    if os.environ.get("USE_MLFLOW", "true").lower() != "true":
        return results_fn()
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        with mlflow.start_run(run_name=f"stress-v{cfg.version}"):
            mlflow.log_params({
                "n_estimators": cfg.model.n_estimators, "max_depth": cfg.model.max_depth,
                "learning_rate": cfg.model.learning_rate, "window_sec": cfg.data.window_sec,
                "train_on": cfg.data.train_on, "eval_on": cfg.data.eval_on,
                "normalization": cfg.data.normalization, "n_features": len(FEATURE_ORDER),
            })
            results = results_fn()
            mlflow.log_metrics({k: float(v) for k, v in results.items()
                                if (k.startswith("cv_") or k.startswith("heldout_"))
                                and isinstance(v, (int, float))})
            return results
    except Exception as e:
        logger.warning(f"MLflow disabled ({e}); running without tracking.")
        return results_fn()


# ============================================================
# Entry Point
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Train Seren stress model")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--data", default="ml/data/features/stress")
    ap.add_argument("--out", default="assets/ml/stress")
    ap.add_argument("--metrics", default="ml/stress/metrics.json")
    ap.add_argument("--figures", default="ml/stress/figures")
    args = ap.parse_args()

    cfg = load_config(None)
    apply_params_yaml(cfg, args.params)
    logging.basicConfig(level=getattr(logging, cfg.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    train_df, eval_df = load_datasets(cfg, args.data)

    def _run():
        return train_and_evaluate(train_df, eval_df, cfg)

    results = maybe_mlflow_run(_run, cfg)

    out_dir = Path(args.out)
    export_model_json(results, cfg, out_dir)
    write_metrics_json(results, cfg, args.metrics)

    # ---- Figures (confusion / ROC / importance / SHAP) ----
    figs = Path(args.figures)
    fcols = results["feature_cols"]
    viz.confusion_matrix_fig(confusion_matrix(results["y_true"], results["y_pred"]),
                             ["Not Stressed", "Stressed"], str(figs / "confusion_matrix.png"),
                             title="Stress — Confusion Matrix (LOSO)")
    viz.roc_fig(results["y_true"], results["y_prob"], str(figs / "roc_curve.png"),
                title="Stress — ROC Curve (LOSO)")
    viz.feature_importance_fig(fcols, results["model"].feature_importances_,
                               str(figs / "feature_importance.png"),
                               title="Stress — Feature Importance")
    viz.shap_summary_fig(results["model"], pd.DataFrame(results["X_scaled"], columns=fcols),
                         str(figs / "shap_summary.png"), title="Stress — SHAP Summary")
    _log_figs_mlflow(figs, results)

    logger.info("\n" + "=" * 50)
    logger.info("Training complete!")
    logger.info(f"  LOSO-CV AUC:   {results['cv_auc_roc']:.4f}")
    if "heldout_auc" in results:
        logger.info(f"  Held-out AUC:  {results['heldout_auc']:.4f}  ({cfg.data.eval_on})")
    logger.info(f"  Threshold:     {results['best_threshold']:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
