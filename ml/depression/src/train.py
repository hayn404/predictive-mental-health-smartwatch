"""
Seren — Depression Risk: Training Pipeline
============================================
Ports the Depresjon notebook into the CI contract. Extracts daily actigraphy
features (or loads the cached CSV), runs 5-fold CV for held-out metrics, trains
+ exports the on-device TS-tree model, writes metrics.json + figures.

DVC/CI contract:
  python ml/depression/src/train.py --params params.yaml \
      --data ml/data/features/depression --raw ml/depression/data/data \
      --out assets/ml/depression --metrics ml/depression/metrics.json

Feature CSV (ml/data/features/depression/depression_features.csv) is committed
so CI can train without the raw Depresjon dataset; if it's missing the script
extracts it from --raw and caches it.
"""
import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ci"))
import viz  # noqa: E402
from bootstrap import bootstrap_ci  # noqa: E402

AGE_MAP = {
    "20-24": 22, "25-29": 27, "30-34": 32, "35-39": 37, "40-44": 42,
    "45-49": 47, "50-54": 52, "55-59": 57, "60-64": 62,
}
DROP_COLS = ["subject", "date", "label", "number", "madrs2"]


# ── Feature extraction (mirrors notebook + export_depression_ts.py) ──────────

def extract_features(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    a = df["activity"].values
    f: dict = {}
    f["mean_activity"] = float(np.mean(a)); f["std_activity"] = float(np.std(a))
    f["median_activity"] = float(np.median(a)); f["max_activity"] = float(np.max(a))
    f["min_activity"] = float(np.min(a)); f["activity_range"] = float(np.max(a) - np.min(a))
    f["activity_cv"] = float(np.std(a) / (np.mean(a) + 1e-6))
    for p in (10, 25, 50, 75, 90):
        f[f"p{p}"] = float(np.percentile(a, p))
    f["inactive_minutes"] = int((a < 20).sum())
    f["active_minutes"] = int((a > 200).sum())
    f["very_active_minutes"] = int((a > 500).sum())
    day = df[(df["timestamp"].dt.hour >= 7) & (df["timestamp"].dt.hour < 22)]["activity"]
    night = df[(df["timestamp"].dt.hour < 7) | (df["timestamp"].dt.hour >= 22)]["activity"]
    f["day_mean"] = float(day.mean()) if len(day) else 0.0
    f["night_mean"] = float(night.mean()) if len(night) else 0.0
    f["day_night_ratio"] = f["day_mean"] / (f["night_mean"] + 1.0)
    inactive = (a < 20).astype(int)
    runs, count = [], 0
    for x in inactive:
        if x:
            count += 1
        else:
            if count:
                runs.append(count)
            count = 0
    if count:
        runs.append(count)
    f["longest_rest_period"] = int(max(runs)) if runs else 0
    f["rest_period_count"] = int(len(runs))
    hist, _ = np.histogram(a, bins=20)
    p = hist / hist.sum(); p = p[p > 0]
    f["activity_entropy"] = float(-np.sum(p * np.log2(p)))
    return f


def build_dataset(raw_dir: Path) -> pd.DataFrame:
    scores = pd.read_csv(raw_dir / "scores.csv")
    scores["age_num"] = scores["age"].map(AGE_MAP)
    scores_clean = scores[["number", "gender", "days", "age_num"]]
    rows = []
    for group, label in [("control", 0), ("condition", 1)]:
        gdir = raw_dir / group
        for fname in os.listdir(gdir):
            df = pd.read_csv(gdir / fname)
            subj = fname.replace(".csv", "")
            for date, day_df in df.groupby("date"):
                feat = extract_features(day_df)
                feat.update(subject=subj, date=date, label=label)
                rows.append(feat)
    ds = pd.DataFrame(rows).merge(scores_clean, left_on="subject", right_on="number", how="left")
    return ds


def load_or_build(data_dir: Path, raw_dir: Path) -> pd.DataFrame:
    csv = data_dir / "depression_features.csv"
    if csv.exists():
        print(f"Loading cached features: {csv}")
        return pd.read_csv(csv)
    print(f"Extracting features from raw Depresjon: {raw_dir}")
    ds = build_dataset(raw_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ds.to_csv(csv, index=False)
    print(f"Cached -> {csv}  ({ds.shape[0]} day-rows, {ds['subject'].nunique()} subjects)")
    return ds


# ── TS tree export helpers (from export_depression_ts.py) ────────────────────

def fix_splits(node, idx_to_name):
    if "split" in node and isinstance(node["split"], str) and node["split"].startswith("f"):
        try:
            node["split"] = idx_to_name[int(node["split"][1:])]
        except (ValueError, KeyError):
            pass
    for c in node.get("children", []):
        fix_splits(c, idx_to_name)
    return node


def traverse(node, feat):
    if "leaf" in node:
        return node["leaf"]
    v = feat.get(node["split"])
    if v is None or (isinstance(v, float) and np.isnan(v)):
        nid = node.get("missing", node["no"])
    else:
        nid = node["yes"] if v < node["split_condition"] else node["no"]
    for c in node["children"]:
        if c["nodeid"] == nid:
            return traverse(c, feat)
    return 0.0


def main():
    ap = argparse.ArgumentParser(description="Train Seren depression model")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--data", default="ml/data/features/depression")
    ap.add_argument("--raw", default="ml/depression/data/data")
    ap.add_argument("--out", default="assets/ml/depression")
    ap.add_argument("--metrics", default="ml/depression/metrics.json")
    ap.add_argument("--figures", default="ml/depression/figures")
    args = ap.parse_args()

    threshold = 0.40
    if Path(args.params).exists():
        import yaml
        d = (yaml.safe_load(open(args.params)) or {}).get("depression", {}) or {}
        threshold = float(d.get("decision_threshold", threshold))

    dataset = load_or_build(Path(args.data), Path(args.raw))
    features = [c for c in dataset.columns if c not in DROP_COLS]
    X = dataset[features].fillna(0.0)
    y = dataset["label"].astype(int)
    print(f"{len(X)} day-rows | {len(features)} features | "
          f"control {(y==0).sum()} / condition {(y==1).sum()}")

    # ---- 5-fold CV (held-out metrics, threshold 0.40) ----
    print("Cross-validating (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds, truth, probs, subj = [], [], [], []
    subjects = dataset["subject"].values
    for tr, te in cv.split(X, y):
        spw = (y.iloc[tr] == 0).sum() / max((y.iloc[tr] == 1).sum(), 1)
        clf = xgb.XGBClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.01, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, gamma=0.2, reg_alpha=0.5,
            reg_lambda=2.0, scale_pos_weight=spw, random_state=42, eval_metric="logloss")
        clf.fit(X.iloc[tr], y.iloc[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        probs.extend(p); preds.extend((p >= threshold).astype(int))
        truth.extend(y.iloc[te]); subj.extend(subjects[te])

    truth, preds, probs, subj = map(np.array, (truth, preds, probs, subj))
    metrics = {
        "cv_accuracy": round(float(accuracy_score(truth, preds)), 4),
        "cv_precision": round(float(precision_score(truth, preds)), 4),
        "cv_recall": round(float(recall_score(truth, preds)), 4),
        "cv_f1": round(float(f1_score(truth, preds)), 4),
        "cv_auc_roc": round(float(roc_auc_score(truth, probs)), 4),
        "cv_accuracy_ci95": bootstrap_ci(accuracy_score, truth, preds, groups=subj),
        "cv_auc_roc_ci95": bootstrap_ci(roc_auc_score, truth, probs, groups=subj),
    }
    print("CV metrics:", metrics)

    # ---- Final model on all data (export) ----
    final = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    final.fit(X, y)

    # ---- Figures ----
    figs = Path(args.figures)
    cm = confusion_matrix(truth, preds)
    viz.confusion_matrix_fig(cm, ["Healthy", "Depressed"], str(figs / "confusion_matrix.png"),
                             title="Depression — Confusion Matrix (5-fold CV)")
    viz.roc_fig(np.array(truth), np.array(probs), str(figs / "roc_curve.png"),
                title="Depression — ROC Curve")
    viz.feature_importance_fig(features, final.feature_importances_,
                               str(figs / "feature_importance.png"),
                               title="Depression — Feature Importance")
    viz.shap_summary_fig(final, X, str(figs / "shap_summary.png"),
                         title="Depression — SHAP Summary")

    # ---- Export TS-tree model ----
    export_model(final, features, metrics, threshold, Path(args.out))

    # ---- metrics.json ----
    write_metrics(metrics, args.metrics)

    # ---- MLflow ----
    log_mlflow(metrics, figs, Path(args.out), args.metrics)
    print("\nDepression training complete.")


def export_model(final, features, metrics, threshold, out_dir):
    booster = final.get_booster()
    idx_to_name = {i: n for i, n in enumerate(features)}
    trees = [fix_splits(json.loads(d), idx_to_name) for d in booster.get_dump(dump_format="json")]
    cfg = json.loads(booster.save_config())
    bs_prob = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))
    base_score = math.log(bs_prob / (1.0 - bs_prob))
    importances = {f: round(float(v), 6) for f, v in zip(features, final.feature_importances_)}
    norm_mean = {f: 0.0 for f in features}
    norm_std = {f: 1.0 for f in features}
    risk = {
        "minimal": {"min": 0, "max": 30, "label": "Minimal", "color": "#35e27e"},
        "mild": {"min": 30, "max": 50, "label": "Mild", "color": "#9B8EC4"},
        "moderate": {"min": 50, "max": 70, "label": "Moderate", "color": "#E8A87C"},
        "high": {"min": 70, "max": 100, "label": "High", "color": "#C4897B"},
    }
    model_out = {
        "version": "1.0.0", "modelType": "xgboost_binary_classifier", "task": "depression_risk",
        "applySigmoid": True, "features": features, "numFeatures": len(features),
        "numTrees": len(trees), "baseScore": base_score, "learningRate": 0.05,
        "normalization": {"mean": norm_mean, "std": norm_std},
        "decisionThreshold": threshold, "metrics": metrics, "importances": importances,
        "riskLevels": risk, "trees": trees,
    }
    metadata = {"version": "1.0.0", "features": features,
                "normalization": {"mean": norm_mean, "std": norm_std},
                "decisionThreshold": threshold, "metrics": metrics,
                "importances": importances, "riskLevels": risk}
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(model_out, open(out_dir / "depression_model.json", "w"), indent=2)
    json.dump(metadata, open(out_dir / "depression_model_metadata.json", "w"), indent=2)
    print(f"Exported -> {out_dir} ({len(trees)} trees, baseScore={base_score:.4f})")


def write_metrics(metrics, path):
    out = {
        "model": "depression", "eval_set_id": "depresjon_v1",
        **metrics,
        "gate": {"primary": "cv_auc_roc", "direction": "max", "min_delta": 0.0,
                 "no_regress": ["cv_accuracy", "cv_f1"]},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"Wrote {path}")


def log_mlflow(metrics, figs, out_dir, metrics_path=None):
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("MLflow disabled (no MLFLOW_TRACKING_URI).")
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("seren-depression-train")
        with mlflow.start_run(run_name="depression"):
            mlflow.log_params({"dataset": "Depresjon", "decision_threshold": 0.40})
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            viz.log_figs_to_mlflow(str(figs))
            mp = out_dir / "depression_model.json"
            if mp.exists():
                mlflow.log_artifact(str(mp))
            if metrics_path and Path(metrics_path).exists():
                mlflow.log_artifact(str(metrics_path))   # carries the *_ci95 lists
        print("Logged to MLflow.")
    except Exception as e:
        print(f"MLflow logging skipped ({e}).")


if __name__ == "__main__":
    main()
