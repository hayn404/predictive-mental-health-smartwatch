"""
Seren — Biological Age: Training Pipeline
===========================================
Adapts train_bioage.py + export_bioage_ts.py into the CI contract. HRV -> age
regression, GLOBAL normalization, trained on Autonomic Aging (subject-level LOSO),
cross-dataset tested on Fantasia (young-vs-old AUC). Exports the TS-tree model,
writes metrics.json + figures + MLflow.

DVC/CI contract:
  python ml/bioage/src/train.py --params params.yaml --data ml/bioage/data \
      --out assets/ml/bioage --metrics ml/bioage/metrics.json

Needs ml/bioage/data/bioage_features_aa.csv (+ optional bioage_features_fantasia.csv),
each: subject_id, age, + 12 HRV features.
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ci"))
import viz  # noqa: E402
from bootstrap import bootstrap_ci  # noqa: E402

FEAT = ["meanRR", "sdnn", "rmssd", "pnn50", "pnn20", "hrMean", "cvRR",
        "sd1", "sd2", "sd1sd2Ratio", "sampleEntropy", "dfaAlpha1"]
PARAMS = dict(objective="reg:squarederror", n_estimators=400, learning_rate=0.03,
              max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
              reg_alpha=0.5, random_state=42, n_jobs=-1)


def gnorm(X, mean=None, std=None):
    if mean is None:
        mean, std = X.mean(0), X.std(0) + 1e-9
    return (X - mean) / std, mean, std


def subj_level(y_true, y_pred, groups):
    df = pd.DataFrame({"g": groups, "yt": y_true, "yp": y_pred})
    a = df.groupby("g").agg(yt=("yt", "first"), yp=("yp", "mean"))
    return a.yt.values, a.yp.values


def loso(df):
    X = df[FEAT].values; y = df["age"].values.astype(float); g = df["subject_id"].values
    logo = LeaveOneGroupOut(); yt, yp, gg = [], [], []
    for tr, te in logo.split(X, y, g):
        Xtr, m, s = gnorm(X[tr]); Xte, _, _ = gnorm(X[te], m, s)
        reg = xgb.XGBRegressor(**PARAMS); reg.fit(Xtr, y[tr])
        yp.extend(reg.predict(Xte)); yt.extend(y[te]); gg.extend(g[te])
    return subj_level(np.array(yt), np.array(yp), np.array(gg))


def fix(node, idx_to_name):
    if "split" in node and isinstance(node["split"], str) and node["split"].startswith("f"):
        try:
            node["split"] = idx_to_name[int(node["split"][1:])]
        except (ValueError, KeyError):
            pass
    for c in node.get("children", []):
        fix(c, idx_to_name)
    return node


def export_ts(reg, mean, std, metrics, out_dir):
    booster = reg.get_booster()
    idx_to_name = {i: n for i, n in enumerate(FEAT)}
    trees = [fix(json.loads(d), idx_to_name) for d in booster.get_dump(dump_format="json")]
    cfg = json.loads(booster.save_config())
    # XGBoost 3.x serialises base_score as '[5E-1]' -> strip brackets before float()
    base = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))
    model = {
        "version": "bioage-aa-1.0", "modelType": "xgboost", "task": "regression",
        "applySigmoid": False, "features": FEAT, "baseScore": base,
        "normalization": {"mean": {f: float(mean[i]) for i, f in enumerate(FEAT)},
                          "std": {f: float(std[i]) for i, f in enumerate(FEAT)}},
        "representation": "subject-aggregate (median of recent HRV windows)",
        "trees": trees, "metrics": metrics,
        "trainingNotes": ("Trained on PhysioNet Autonomic Aging. Predicts physiological age "
                          "from 12 HRV features; GLOBAL normalization; age-gap = predicted - chronological. "
                          "Aggregate recent windows (median) before predicting."),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(model, open(out_dir / "bioage_model.json", "w"))
    print(f"Exported -> {out_dir / 'bioage_model.json'} ({len(trees)} trees, baseScore={base:.2f})")


def write_metrics(metrics, path):
    out = {
        "model": "bioage", "eval_set_id": "autonomic_aging_loso_v1", **metrics,
        "gate": {"primary": "loso_mae_years", "direction": "min", "min_delta": 0.0,
                 "no_regress": ["loso_r"]},
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
        mlflow.set_experiment("seren-bioage-train")
        with mlflow.start_run(run_name="bioage"):
            mlflow.log_params({"dataset": "AutonomicAging", **{k: PARAMS[k] for k in
                              ("n_estimators", "max_depth", "learning_rate")}})
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            viz.log_figs_to_mlflow(str(figs))
            mp = out_dir / "bioage_model.json"
            if mp.exists():
                mlflow.log_artifact(str(mp))
        print("Logged to MLflow.")
    except Exception as e:
        print(f"MLflow logging skipped ({e}).")


def main():
    ap = argparse.ArgumentParser(description="Train Seren bio-age model")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--data", default="ml/bioage/data")
    ap.add_argument("--out", default="assets/ml/bioage")
    ap.add_argument("--metrics", default="ml/bioage/metrics.json")
    ap.add_argument("--figures", default="ml/bioage/figures")
    args = ap.parse_args()

    data = Path(args.data)
    aa_csv = data / "bioage_features_aa.csv"
    fan_csv = data / "bioage_features_fantasia.csv"
    if not aa_csv.exists():
        sys.exit(f"Feature CSV not found: {aa_csv}\nRun extract_bioage.py aa first "
                 "(or provide the CSV).")

    df_aa = pd.read_csv(aa_csv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"Autonomic Aging: {len(df_aa)} windows, {df_aa.subject_id.nunique()} subjects, "
          f"age {df_aa.age.min():.0f}-{df_aa.age.max():.0f}")

    # ---- LOSO subject-level metrics ----
    print("LOSO (subject-level)...")
    yt, yp = loso(df_aa)
    metrics = {
        "loso_mae_years": round(float(mean_absolute_error(yt, yp)), 4),
        "loso_r": round(float(np.corrcoef(yt, yp)[0, 1]), 4),
        "loso_r2": round(float(r2_score(yt, yp)), 4),
        "loso_mae_years_ci95": bootstrap_ci(mean_absolute_error, yt, yp),
    }
    print("LOSO metrics:", metrics)

    # ---- Final model on all AA (global norm) ----
    X = df_aa[FEAT].values; y = df_aa["age"].values.astype(float)
    Xn, mean, std = gnorm(X)
    reg = xgb.XGBRegressor(**PARAMS); reg.fit(Xn, y)

    # ---- Cross-dataset AA -> Fantasia (young vs old AUC) ----
    if fan_csv.exists():
        df_fan = pd.read_csv(fan_csv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xf, _, _ = gnorm(df_fan[FEAT].values, mean, std)
        pf = reg.predict(Xf)
        yt_f, yp_f = subj_level(df_fan["age"].values.astype(float), pf, df_fan["subject_id"].values)
        y_bin = (yt_f > 50).astype(int)
        if len(set(y_bin)) > 1:
            metrics["cross_dataset_young_old_auc"] = round(float(roc_auc_score(y_bin, yp_f)), 4)
        metrics["cross_dataset_mae_years"] = round(float(mean_absolute_error(yt_f, yp_f)), 4)
        print(f"Cross-dataset AA->Fantasia: AUC(young/old)="
              f"{metrics.get('cross_dataset_young_old_auc')}, MAE={metrics['cross_dataset_mae_years']}")

    # ---- Figures ----
    figs = Path(args.figures)
    viz.regression_scatter_fig(yt, yp, str(figs / "predicted_vs_actual.png"), unit="yrs",
                               title="Bio-age — AA LOSO (subject-level)")
    viz.feature_importance_fig(FEAT, reg.feature_importances_,
                               str(figs / "feature_importance.png"),
                               title="Bio-age — Feature Importance")
    viz.shap_summary_fig(reg, pd.DataFrame(Xn, columns=FEAT), str(figs / "shap_summary.png"),
                         title="Bio-age — SHAP Summary")

    export_ts(reg, mean, std, metrics, Path(args.out))
    write_metrics(metrics, args.metrics)
    log_mlflow(metrics, figs, Path(args.out))
    print("\nBio-age training complete.")


if __name__ == "__main__":
    main()
