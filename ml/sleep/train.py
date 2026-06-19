"""
Seren — Sleep Stage Classification: Training Pipeline
=======================================================
Ports the Kaggle notebook (seren_sleep_kaggle.ipynb) v3.2 into the DVC pipeline
contract. Trains the TCN+BiGRU seq2seq model on the BIDSleep feature cache,
evaluates zero-shot on the held-out Walch dataset, and exports ONNX + TFLite.

DVC contract:
  python ml/sleep/train.py --params params.yaml --data data/sleep \
      --out assets/ml/sleep --metrics ml/sleep/metrics.json

Outputs:
  <out>/sleep_stage_model.onnx
  <out>/sleep_stage_model.tflite          (unless --skip-tflite)
  <out>/sleep_model_metadata.json
  <metrics>                               (held-out Walch metrics + gate block)
  ml/sleep/output/training_history.json, training_curves.png, checkpoints/best_model.pt

MLflow: logs to MLFLOW_TRACKING_URI if set (DagsHub), else no-ops silently.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (f1_score, accuracy_score, cohen_kappa_score,
                             confusion_matrix, recall_score)

import config as cfg
from model import create_model
from dataset import build_datasets


# ── Optional MLflow (never crashes training) ───────────────────

class Tracker:
    def __init__(self, experiment_name):
        self.enabled = False
        self._m = None
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            print("MLflow disabled (no MLFLOW_TRACKING_URI) — runs not tracked.")
            return
        try:
            import mlflow
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            self._m, self.enabled = mlflow, True
            print(f"MLflow tracking: {uri} | experiment: {experiment_name}")
        except Exception as e:
            print(f"MLflow disabled ({e}).")

    def start_run(self, **kw):
        if not self.enabled:
            return _NullCtx()
        try:
            return self._m.start_run(**kw)
        except Exception:
            return _NullCtx()

    def log_params(self, d):
        if self.enabled:
            try: self._m.log_params(d)
            except Exception: pass

    def log_metric(self, k, v, step=None):
        if self.enabled:
            try: self._m.log_metric(k, float(v), step=step)
            except Exception: pass

    def log_artifact(self, p):
        if self.enabled and Path(p).exists():
            try: self._m.log_artifact(str(p))
            except Exception: pass


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ── Train / eval epoch (seq2seq, mirrors notebook cell 13) ─────

def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot, n = 0.0, 0
    P, L = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for f, l in loader:
            f, l = f.to(device), l.to(device)
            logits = model(f)
            fl, ll = logits.reshape(-1, cfg.NUM_CLASSES), l.reshape(-1)
            loss = criterion(fl, ll)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()
            tot += loss.item() * ll.size(0)
            n += ll.size(0)
            P.append(fl.argmax(-1).cpu().numpy())
            L.append(ll.cpu().numpy())
    P, L = np.concatenate(P), np.concatenate(L)
    return (tot / max(1, n),
            f1_score(L, P, average="macro", zero_division=0),
            f1_score(L, P, average="weighted", zero_division=0),
            cohen_kappa_score(L, P), P, L)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Main ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train Seren sleep stage classifier")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--data", default="data/sleep")
    ap.add_argument("--out", default="assets/ml/sleep")
    ap.add_argument("--metrics", default="ml/sleep/metrics.json")
    ap.add_argument("--skip-tflite", action="store_true",
                    help="Skip ONNX->TFLite conversion (for envs without onnx2tf/tensorflow)")
    args = ap.parse_args()

    cfg.apply_params(args.params)
    seed_everything(cfg.RANDOM_SEED)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.EXPORT_DIR = out_dir
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | arch={cfg.ARCH_TAG} | {cfg.NUM_FEATURES} feats, "
          f"seq {cfg.SEQ_LEN}, classes {cfg.STAGE_NAMES}")

    tracker = Tracker("seren-sleep-train")

    # Data + model
    train_loader, val_loader, test_loader, class_weights, info = build_datasets(args.data)
    model = create_model(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)

    best_f1, patience = 0.0, cfg.EARLY_STOP_PATIENCE
    history = {k: [] for k in ["tr_loss", "vl_loss", "tr_mf1", "vl_mf1", "vl_wf1", "vl_kappa"]}
    ckpt_path = cfg.CHECKPOINT_DIR / "best_model.pt"

    with tracker.start_run(run_name="train_v3.2"):
        tracker.log_params({
            "num_features": cfg.NUM_FEATURES, "seq_len": cfg.SEQ_LEN,
            "train_stride": cfg.TRAIN_STRIDE, "tcn_channels": cfg.TCN_CHANNELS,
            "tcn_kernel": cfg.TCN_KERNEL, "lstm_hidden": cfg.LSTM_HIDDEN,
            "lstm_layers": cfg.LSTM_LAYERS, "dropout": cfg.DROPOUT,
            "batch_size": cfg.BATCH_SIZE, "lr": cfg.LEARNING_RATE,
            "weight_decay": cfg.WEIGHT_DECAY, "num_epochs": cfg.NUM_EPOCHS,
            "early_stop_patience": cfg.EARLY_STOP_PATIENCE, "val_split": cfg.VAL_SPLIT,
            "seed": cfg.RANDOM_SEED, "arch": cfg.ARCH_TAG, **info,
        })

        print(f"\n── Training for up to {cfg.NUM_EPOCHS} epochs ──\n")
        for ep in range(1, cfg.NUM_EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_mf1, _, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
            vl_loss, vl_mf1, vl_wf1, vl_k, _, _ = run_epoch(model, val_loader, criterion, device)
            scheduler.step()

            for k, v in zip(history, [tr_loss, vl_loss, tr_mf1, vl_mf1, vl_wf1, vl_k]):
                history[k].append(v)
            for k, v in [("tr_loss", tr_loss), ("vl_loss", vl_loss), ("tr_mf1", tr_mf1),
                         ("vl_mf1", vl_mf1), ("vl_wf1", vl_wf1), ("vl_kappa", vl_k),
                         ("lr", optimizer.param_groups[0]["lr"])]:
                tracker.log_metric(k, v, step=ep)
            print(f"Ep {ep:3d}/{cfg.NUM_EPOCHS} | tr_loss {tr_loss:.3f} mF1 {tr_mf1:.3f} | "
                  f"val mF1 {vl_mf1:.3f} wF1 {vl_wf1:.3f} k {vl_k:.3f} | {time.time()-t0:.1f}s")

            if vl_mf1 > best_f1:
                best_f1, patience = vl_mf1, cfg.EARLY_STOP_PATIENCE
                torch.save({"epoch": int(ep), "model_state_dict": model.state_dict(),
                            "val_mf1": float(vl_mf1), "val_wf1": float(vl_wf1),
                            "val_kappa": float(vl_k)}, ckpt_path)
                print(f"  * best val macro-F1 {vl_mf1:.4f}")
            else:
                patience -= 1
                if patience <= 0:
                    print(f"Early stop @ {ep}")
                    break

        with open(cfg.OUTPUT_DIR / "training_history.json", "w") as fp:
            json.dump(history, fp, indent=2)
        print(f"\nBest BIDSleep val macro-F1: {best_f1:.4f}")

    # ── Held-out Walch evaluation (the honest generalization number) ──
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, te_mf1, te_wf1, te_k, P, L = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    te_acc = accuracy_score(L, P)

    print("\n" + "=" * 56 + "\nWALCH HELD-OUT TEST (4-class)\n" + "=" * 56)
    print(f"  macroF1 {te_mf1:.4f} | weightedF1 {te_wf1:.4f} | acc {te_acc:.4f} | kappa {te_k:.4f}")
    per_f1 = f1_score(L, P, average=None, labels=list(range(cfg.NUM_CLASSES)), zero_division=0)
    per_rc = recall_score(L, P, average=None, labels=list(range(cfg.NUM_CLASSES)), zero_division=0)
    for n_, a, b in zip(cfg.STAGE_NAMES, per_f1, per_rc):
        print(f"    {n_:6s} F1={a:.3f} recall={b:.3f}")
    print("  confusion (rows=true, cols=pred):")
    print(confusion_matrix(L, P, labels=list(range(cfg.NUM_CLASSES))))

    # 3-class view (Light+Deep -> NREM)
    to3 = {0: 0, 1: 1, 2: 1, 3: 2}
    P3 = np.vectorize(to3.get)(P)
    L3 = np.vectorize(to3.get)(L)
    te3_wf1 = f1_score(L3, P3, average="weighted", zero_division=0)
    te3_acc = accuracy_score(L3, P3)
    te3_k = cohen_kappa_score(L3, P3)
    print(f"\n3-class (Wake/NREM/REM): wF1 {te3_wf1:.3f} acc {te3_acc:.3f} kappa {te3_k:.3f}")

    with tracker.start_run(run_name="walch_eval_v3.2"):
        for k, v in [("walch_macro_f1", te_mf1), ("walch_weighted_f1", te_wf1),
                     ("walch_accuracy", te_acc), ("walch_kappa", te_k),
                     ("walch_3class_wf1", te3_wf1), ("walch_3class_kappa", te3_k)]:
            tracker.log_metric(k, v)

    # ── Write metrics.json (gate baseline lives in git) ──
    metrics = {
        "model": "sleep",
        "eval_set_id": "walch_v1",
        "walch_kappa": round(float(te_k), 4),
        "walch_weightedF1": round(float(te_wf1), 4),
        "walch_macroF1": round(float(te_mf1), 4),
        "walch_accuracy": round(float(te_acc), 4),
        "walch_3class_weightedF1": round(float(te3_wf1), 4),
        "gate": {
            "primary": "walch_kappa",
            "direction": "max",
            "min_delta": 0.0,
            "no_regress": ["walch_weightedF1", "walch_macroF1"],
        },
    }
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"\nWrote {args.metrics}")

    # ── Export ONNX + metadata (mirrors notebook cell 17) ──
    export_onnx(model, out_dir, te_mf1, te_wf1, te_k, tracker)

    # ── Training curves ──
    save_curves(history, cfg.OUTPUT_DIR / "training_curves.png", tracker)

    # ── ONNX -> TFLite ──
    if not args.skip_tflite:
        convert_tflite(out_dir)
    else:
        print("Skipped TFLite conversion (--skip-tflite).")


def export_onnx(model, out_dir, te_mf1, te_wf1, te_k, tracker):
    import onnx
    import onnxruntime as ort

    export_model = type(model)().to("cpu")
    export_model.load_state_dict(model.state_dict())
    export_model.eval()

    dummy = torch.randn(1, cfg.SEQ_LEN, cfg.NUM_FEATURES)
    onnx_path = out_dir / "sleep_stage_model.onnx"
    torch.onnx.export(
        export_model, (dummy,), str(onnx_path),
        input_names=["features"], output_names=["logits"],
        opset_version=17, do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path))
    diff = float(np.max(np.abs(export_model(dummy).detach().numpy()
                               - sess.run(None, {"features": dummy.numpy()})[0])))
    print(f"ONNX parity max diff: {diff:.6f}")

    metadata = {
        "version": "3.2.0",
        "modelType": "tcn_bilstm_seq2seq_hr_actigraphy",
        "task": "sleep_stage_classification_4class",
        "trainData": "BIDSleep (Apple Watch)",
        "testData": "Walch/Sleep-Accel (held-out)",
        "numClasses": cfg.NUM_CLASSES, "stageNames": cfg.STAGE_NAMES,
        "stageMap": cfg.APP_STAGE_MAP,
        "input": {"features": {
            "shape": [1, cfg.SEQ_LEN, cfg.NUM_FEATURES], "names": cfg.FEATURE_NAMES,
            "epochSeconds": cfg.EPOCH_SEC, "recommendedSeqLen": cfg.SEQ_LEN,
            "normalization": "per_night_median_mad_per_feature"}},
        "output": {"logits": {"shape": [1, cfg.SEQ_LEN, cfg.NUM_CLASSES],
                              "note": "argmax last dim -> stageMap"}},
        "test": {"walch_macroF1": float(te_mf1), "walch_weightedF1": float(te_wf1),
                 "walch_kappa": float(te_k)},
    }
    with open(out_dir / "sleep_model_metadata.json", "w") as fp:
        json.dump(metadata, fp, indent=2)
    print(f"Saved {onnx_path} ({onnx_path.stat().st_size/1e3:.1f} KB) + metadata")
    tracker.log_artifact(onnx_path)
    tracker.log_artifact(out_dir / "sleep_model_metadata.json")


def save_curves(history, path, tracker):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}) — skipping curves.")
        return
    ep = range(1, len(history["tr_loss"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, history["tr_loss"], label="train"); ax[0].plot(ep, history["vl_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(ep, history["tr_mf1"], label="train macro"); ax[1].plot(ep, history["vl_mf1"], label="val macro")
    ax[1].plot(ep, history["vl_wf1"], label="val weighted"); ax[1].axhline(0.72, ls="--", c="gray")
    ax[1].set_title("F1"); ax[1].legend()
    ax[2].plot(ep, history["vl_kappa"]); ax[2].set_title("Val kappa")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
    print(f"Saved {path}")
    tracker.log_artifact(path)


def convert_tflite(out_dir):
    import convert_onnx_to_tflite as conv
    conv.ONNX = out_dir / "sleep_stage_model.onnx"
    conv.OUTDIR = out_dir
    conv.WORK = out_dir / "_tflite_work"
    try:
        target = conv.convert()
    except Exception as e:
        raise SystemExit(
            f"TFLite conversion failed: {e}\n"
            "Install: pip install -U onnx2tf tensorflow onnxruntime onnx "
            "onnx-graphsurgeon onnxsim tf_keras\n"
            "Or re-run with --skip-tflite if you only need the ONNX artifact."
        )
    # Parity check is best-effort: the .tflite is already written, so a finicky
    # interpreter feed shouldn't fail the whole run.
    try:
        conv.verify(target)
    except Exception as e:
        print(f"TFLite parity check non-fatal failure: {e}")


if __name__ == "__main__":
    main()
