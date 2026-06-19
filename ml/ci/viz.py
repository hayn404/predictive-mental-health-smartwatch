"""
Shared figure helpers for the model training scripts. Every chart is saved as a
PNG into the model's figures dir so the CI pipeline can both (a) log it to MLflow
as an artifact and (b) hand it to the thesis. Matplotlib uses the headless Agg
backend so it works on CI runners with no display.
"""
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path}")


def confusion_matrix_fig(cm, class_names, path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    thr = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _save(fig, path)


def roc_fig(y_true, y_prob, path, title="ROC Curve"):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")
    _save(fig, path)


def feature_importance_fig(names, importances, path, title="Feature Importance", top=20):
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(order))))
    ax.barh([names[i] for i in order][::-1],
            [importances[i] for i in order][::-1], color="#6C8EBF")
    ax.set_xlabel("Importance (gain)"); ax.set_title(title)
    _save(fig, path)


def shap_summary_fig(model, X, path, title="SHAP Summary"):
    """Best-effort SHAP beeswarm; skips cleanly if shap isn't installed."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        plt.figure()
        shap.summary_plot(sv, X, show=False)
        fig = plt.gcf()
        fig.suptitle(title)
        _save(fig, path)
    except Exception as e:
        print(f"  SHAP skipped ({e})")


def regression_scatter_fig(y_true, y_pred, path, unit="yrs", title="Predicted vs Actual"):
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, color="#6C8EBF", edgecolor="white", s=40)
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    ax.set_xlabel(f"Actual ({unit})"); ax.set_ylabel(f"Predicted ({unit})")
    ax.set_title(f"{title}  (MAE={mae:.2f} {unit}, R2={r2:.2f})")
    _save(fig, path)


def log_figs_to_mlflow(fig_dir):
    """Log every PNG in fig_dir to the active MLflow run (no-op if MLflow off)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return
    try:
        import mlflow
        for f in sorted(os.listdir(fig_dir)):
            if f.endswith(".png"):
                mlflow.log_artifact(os.path.join(fig_dir, f), artifact_path="figures")
    except Exception as e:
        print(f"  MLflow figure logging skipped ({e})")
