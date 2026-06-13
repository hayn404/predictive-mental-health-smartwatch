"""
Seren — Sleep Stage Classification: Evaluation
================================================
Loads best checkpoint and computes detailed metrics:
  - Per-class precision, recall, F1
  - Confusion matrix
  - Cohen's Kappa
  - Overall accuracy
"""

import numpy as np
import torch
import json
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)

import config as cfg
from model import create_model
from dataset import build_datasets


def evaluate(checkpoint_path: str = None):
    """Run full evaluation on validation set."""
    if checkpoint_path is None:
        checkpoint_path = cfg.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    print("Loading validation data...")
    _, val_loader, _ = build_datasets()

    # Load model
    print("Loading model checkpoint...")
    model = create_model(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Collect predictions
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for raw, aux, labels in val_loader:
            raw = raw.to(device)
            aux = aux.to(device)

            logits = model(raw, aux)
            probs = torch.softmax(logits, dim=1)

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)

    print("\n" + "=" * 60)
    print("SLEEP STAGE CLASSIFICATION — EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    print(f"Cohen's Kappa:     {kappa:.4f}")

    # Per-class report
    print("\n── Per-Class Metrics ──")
    report = classification_report(
        all_labels, all_preds,
        target_names=cfg.STAGE_NAMES,
        digits=3,
        zero_division=0,
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("── Confusion Matrix ──")
    print(f"{'':>8}", end="")
    for name in cfg.STAGE_NAMES:
        print(f"{name:>6}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{cfg.STAGE_NAMES[i]:>8}", end="")
        for val in row:
            print(f"{val:>6}", end="")
        print()

    # Save results
    results = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "kappa": float(kappa),
        "per_class": json.loads(
            classification_report(all_labels, all_preds,
                                  target_names=cfg.STAGE_NAMES,
                                  output_dict=True, zero_division=0).__repr__()
            if False else "{}"
        ),
        "confusion_matrix": cm.tolist(),
        "checkpoint": str(checkpoint_path),
    }

    # Proper per-class dict
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=cfg.STAGE_NAMES,
        output_dict=True,
        zero_division=0,
    )
    results["per_class"] = {
        name: {
            "precision": report_dict[name]["precision"],
            "recall": report_dict[name]["recall"],
            "f1": report_dict[name]["f1-score"],
            "support": report_dict[name]["support"],
        }
        for name in cfg.STAGE_NAMES
    }

    output_path = cfg.OUTPUT_DIR / "evaluation_results.json"
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate sleep stage model")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint)
