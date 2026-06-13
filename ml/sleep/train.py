"""
Seren — Sleep Stage Classification: Training Pipeline
=======================================================
Trains the CNN-BiLSTM model on DREAMT dataset with:
  - Weighted cross-entropy for class imbalance
  - CosineAnnealingLR scheduler
  - Early stopping on validation macro F1
  - Checkpoint saving
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from pathlib import Path

import config as cfg
from model import create_model
from dataset import build_datasets


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for raw, aux, labels in loader:
        raw = raw.to(device)
        aux = aux.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(raw, aux)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for raw, aux, labels in loader:
        raw = raw.to(device)
        aux = aux.to(device)
        labels = labels.to(device)

        logits = model(raw, aux)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    return avg_loss, acc, f1, kappa, np.array(all_preds), np.array(all_labels)


def train(data_dir: str = None):
    """Main training entry point."""
    if data_dir:
        cfg.DATA_DIR = Path(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create output dirs
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n── Loading DREAMT dataset ──")
    train_loader, val_loader, class_weights = build_datasets()

    # Create model
    print("\n── Creating model ──")
    model = create_model(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)

    # Training loop
    print(f"\n── Training for {cfg.NUM_EPOCHS} epochs ──\n")
    best_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [],
               "val_acc": [], "val_kappa": [], "lr": []}

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_kappa, _, _ = validate(
            model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        history["val_kappa"].append(val_kappa)
        history["lr"].append(lr)

        print(f"Epoch {epoch:3d}/{cfg.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} F1: {train_f1:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} "
              f"Kappa: {val_kappa:.3f} | LR: {lr:.6f} | {elapsed:.1f}s")

        # Checkpoint best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "val_kappa": val_kappa,
            }, cfg.CHECKPOINT_DIR / "best_model.pt")
            print(f"  ✓ New best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg.EARLY_STOP_PATIENCE} epochs)")
                break

    # Save training history
    with open(cfg.OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Checkpoint: {cfg.CHECKPOINT_DIR / 'best_model.pt'}")
    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train sleep stage classifier")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to DREAMT dataset directory")
    args = parser.parse_args()
    train(data_dir=args.data_dir)
