"""
Seren — Cognitive Focus: Training Pipeline
============================================
Adapts train_cogwear_focus.py into the CI contract: per-user-normalized XGBoost
on CogWear HRV features, Leave-One-Subject-Out evaluation, TS-tree export,
metrics.json + figures + MLflow.

DVC/CI contract:
  python ml/focus/src/train.py --params params.yaml --data ml/focus/data \
      --device samsung --out assets/ml/focus --metrics ml/focus/metrics.json

Needs ml/focus/data/cogwear_features_<device>.csv
(columns: 14 HRV features + label + subject_id), produced by extract_cogwear_features.py.
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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score, confusion_matrix)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ci"))
import viz  # noqa: E402
from bootstrap import bootstrap_ci  # noqa: E402

FEATURES = ["meanRR", "sdnn", "rmssd", "pnn50", "pnn20", "hrMean", "hrStd", "hrRange", "cvRR",
            "sd1", "sd2", "sd1sd2Ratio", "sampleEntropy", "dfaAlpha1"]
PARAMS = dict(objective="binary:logistic", eval_metric="logloss", n_estimators=400,
              learning_rate=0.05, max_depth=2, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=2.0, reg_alpha=0.5, gamma=0.2, random_state=42, n_jobs=-1)


def load_tuned_params():
    """Merge the PSO-tuned hyperparameters (ml/focus/models/pso_best_params.json,
    LOSO-AUC ~0.839) into PARAMS so CI reproduces the deployed model. No-ops if the
    file is absent (falls back to the hand-set PARAMS above)."""
    p = Path(__file__).resolve().parents[1] / "models" / "pso_best_params.json"
    if p.exists():
        best = json.loads(p.read_text()).get("best_params", {})
        PARAMS.update(best)
        print(f"Loaded PSO-tuned params: {best}")
    else:
        print("No pso_best_params.json found — using default PARAMS.")


def per_user_normalize(df, feats):
    out = df.copy()
    for f in feats:
        g = out.groupby("subject_id")[f]
        out[f] = (out[f] - g.transform("mean")) / g.transform("std").replace(0, 1.0).fillna(1.0)
    return out.fillna(0.0)


def loso_eval(X, y, groups):
    logo = LeaveOneGroupOut()
    yt, yp, ypr, gg = [], [], [], []
    for tr, te in logo.split(X, y, groups):
        pos = max(1, (y[tr] == 1).sum()); neg = max(1, (y[tr] == 0).sum())
        m = xgb.XGBClassifier(**PARAMS, scale_pos_weight=neg / pos)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        yt.extend(y[te]); ypr.extend(p); yp.extend((p >= 0.5).astype(int)); gg.extend(groups[te])
    return map(np.array, (yt, yp, ypr, gg))


def fix_splits(node, idx_to_name):
    if "split" in node and isinstance(node["split"], str) and node["split"].startswith("f"):
        try:
            node["split"] = idx_to_name[int(node["split"][1:])]
        except (ValueError, KeyError):
            pass
    for c in node.get("children", []):
        fix_splits(c, idx_to_name)
    return node


def export_ts(final, raw_df, feats, metrics, device, out_dir):
    booster = final.get_booster()
    idx_to_name = {i: n for i, n in enumerate(feats)}
    trees = [fix_splits(json.loads(d), idx_to_name) for d in booster.get_dump(dump_format="json")]
    cfg = json.loads(booster.save_config())
    # XGBoost 3.x serialises base_score as '[5E-1]' -> strip brackets before float()
    bp = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))
    bp = min(max(bp, 1e-6), 1 - 1e-6)
    base_margin = math.log(bp / (1 - bp))
    mean = {f: float(raw_df[f].mean()) for f in feats}
    std = {f: float(raw_df[f].std() or 1.0) for f in feats}
    model = {
        "version": f"cogwear-{device}-1.0", "modelType": "xgboost", "task": "binary_logistic",
        "features": feats, "numFeatures": len(feats), "numTrees": len(trees),
        "baseScore": base_margin, "applySigmoid": True, "learningRate": PARAMS["learning_rate"],
        "normalization": {"mean": mean, "std": std}, "perUserNorm": True, "decisionThreshold": 0.5,
        "trees": trees,
        "importances": {f: float(v) for f, v in zip(feats, final.feature_importances_)},
        "metrics": metrics,
        "trainingNotes": ("Trained on CogWear (rest vs cognitive effort). Per-user z-norm at "
                          "inference; global mean/std cold-start. Higher = more cognitive effort."),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(model, open(out_dir / "focus_model.json", "w"))
    print(f"Exported -> {out_dir / 'focus_model.json'} ({len(trees)} trees)")


def write_metrics(metrics, path):
    out = {
        "model": "focus", "eval_set_id": "cogwear_loso_v1", **metrics,
        "gate": {"primary": "loso_auc", "direction": "max", "min_delta": 0.0,
                 "no_regress": ["loso_f1", "loso_mcc"]},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"Wrote {path}")


def log_mlflow(metrics, figs, out_dir):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print("MLflow disabled (no MLFLOW_TRACKING_URI).")
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("seren-focus-train")
        with mlflow.start_run(run_name="focus"):
            mlflow.log_params({"dataset": "CogWear", **{k: PARAMS[k] for k in
                              ("n_estimators", "max_depth", "learning_rate")}})
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            viz.log_figs_to_mlflow(str(figs))
            mp = out_dir / "focus_model.json"
            if mp.exists():
                mlflow.log_artifact(str(mp))
        print("Logged to MLflow.")
    except Exception as e:
        print(f"MLflow logging skipped ({e}).")


def main():
    ap = argparse.ArgumentParser(description="Train Seren focus model")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--data", default="ml/focus/data")
    ap.add_argument("--device", default="samsung")
    ap.add_argument("--out", default="assets/ml/focus")
    ap.add_argument("--metrics", default="ml/focus/metrics.json")
    ap.add_argument("--figures", default="ml/focus/figures")
    args = ap.parse_args()

    load_tuned_params()

    csv = Path(args.data) / f"cogwear_features_{args.device}.csv"
    if not csv.exists():
        sys.exit(f"Feature CSV not found: {csv}\nRun extract_cogwear_features.py first "
                 "(or provide the CSV).")

    df = pd.read_csv(csv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feats = [f for f in FEATURES if f in df.columns]
    y = df["label"].to_numpy(int)
    groups = df["subject_id"].to_numpy()
    print(f"CogWear[{args.device}]: {len(df)} windows, {df['subject_id'].nunique()} subjects, "
          f"{len(feats)} features")

    dn = per_user_normalize(df, feats)
    X = dn[feats].to_numpy(float)
    yt, yp, ypr, gg = loso_eval(X, y, groups)
    metrics = {
        "loso_acc": round(float(accuracy_score(yt, yp)), 4),
        "loso_f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
        "loso_mcc": round(float(matthews_corrcoef(yt, yp)), 4),
        "loso_auc": round(float(roc_auc_score(yt, ypr)), 4),
        "loso_auc_ci95": bootstrap_ci(roc_auc_score, yt, ypr, groups=gg),
    }
    print("LOSO metrics:", metrics)

    pos = max(1, (y == 1).sum()); neg = max(1, (y == 0).sum())
    final = xgb.XGBClassifier(**PARAMS, scale_pos_weight=neg / pos)
    final.fit(X, y)

    figs = Path(args.figures)
    viz.confusion_matrix_fig(confusion_matrix(yt, yp), ["Rest", "Focus"],
                             str(figs / "confusion_matrix.png"),
                             title="Focus — Confusion Matrix (LOSO)")
    viz.roc_fig(yt, ypr, str(figs / "roc_curve.png"), title="Focus — ROC Curve (LOSO)")
    viz.feature_importance_fig(feats, final.feature_importances_,
                               str(figs / "feature_importance.png"),
                               title="Focus — Feature Importance")
    viz.shap_summary_fig(final, dn[feats], str(figs / "shap_summary.png"),
                         title="Focus — SHAP Summary")

    export_ts(final, df, feats, metrics, args.device, Path(args.out))
    write_metrics(metrics, args.metrics)
    log_mlflow(metrics, figs, Path(args.out))
    print("\nFocus training complete.")


if __name__ == "__main__":
    main()
