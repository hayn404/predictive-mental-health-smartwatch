# Seren Focus Model — Model Card

## What it predicts
On-device **cognitive engagement / effort** from wearable PPG-derived HRV. Output is a
0–100 score = probability the user's physiology shows active cognitive effort
(high = locked into a task; low = at rest / disengaged).

> **Scope / honesty disclosure:** this is a *wellness* signal, **not** a clinical or
> diagnostic measure. It was trained on a **laboratory cognitive-load task** (Stroop),
> **not** real-world studying, and the deployed (Samsung) model was validated on **20
> subjects**. Generalization to real exam/study focus is reasonable but **not proven**.

## Why not the previous (exam-grade) model
The prior focus model tried to predict exam grade (PhysioNet exam-stress set, 10 subjects,
one grade per session). Under leave-one-subject-out CV it **failed to beat the mean
baseline** (LOSO-MAE 12.8 vs 11.99) and output a near-constant ~76 for everyone — the
exam-grade task has no usable cross-subject signal at N=10. We report that negative result
honestly and moved to a task-labeled cognitive-load dataset.

## Data
- **CogWear** (Grzeszczyk et al., 2023, PhysioNet) — rest (baseline) vs Stroop (cognitive_load).
- Deployed model uses the **Samsung Galaxy Watch4 PPG** (~23 Hz) — the exact signal the app
  receives on-device. Empatica E4 (64 Hz) trained as a research-grade reference.

## Method
- Raw PPG → RR intervals → 14 HRV features (time-domain + non-linear), reusing the shared
  extractor `ml/stress/src/features.py`.
- **Enhanced RR extraction** for low-rate consumer PPG: bandpass to the HR band, then
  cubic-upsample the filtered Galaxy Watch4 PPG (~23 Hz) to 128 Hz before peak detection so
  RR timing isn't quantized to the coarse native sample period. (LOSO AUC 0.74→0.79.)
- **120 s windows** (vs 60 s) for stabler HRV estimates. (LOSO AUC 0.79→0.82.)
- **Per-user (per-subject) z-normalization** — the headline accuracy lever: each feature is
  z-scored against the subject's own distribution. On-device this uses the user's recent
  windows, with the model's global (population) scaler as a cold-start fallback.
- **Temporal smoothing at deployment**: focus is a sustained state, so the app averages the
  model's scores over ~2 min of cycles — the runtime analogue of per-session aggregation.
- XGBoost (`binary:logistic`, depth 2, 400 trees, lr 0.05, class-balanced).
- Exported directly to the TS tree format (`assets/ml/focus/focus_model.json`) consumed by
  `services/ai/focusModel.ts` (same engine as the stress model).

## Results — Leave-One-Subject-Out CV (honest, cross-subject)

| Model | Accuracy | F1 | MCC | AUC |
|-------|----------|----|-----|-----|
| Majority baseline | 0.70 | — | 0.00 | — |
| **Samsung per-window (deployed)** | **0.78** | **0.84** | **0.48** | **0.82** |
| **Samsung sustained-state (≈2 min agg.)** | **0.85** | **0.86** | **0.71** | **0.95** |
| Empatica (research-grade reference) | 0.77 | 0.81 | 0.52 | 0.82 |

Two honest readings: **per-window** (single 120 s window, the conservative number) and
**sustained-state** (the model's scores pooled over a couple minutes, which is how the app
actually reports focus). Enhanced RR + longer windows brought the consumer Galaxy Watch4 up
to the research-grade Empatica's per-window AUC. **Per-user normalization is the lever**
(MCC 0.36→0.48 vs raw). Top features: `dfaAlpha1`, `hrMean`, `sd1sd2Ratio`, `meanRR`.

## Deployment fidelity caveat
Training features come from raw watch PPG; on-device the app currently derives HRV from
Health Connect HR (BPM) samples, which are coarser than raw PPG. Per-user normalization and
temporal smoothing bridge this gap, but live accuracy will trail the dataset figures — the
same train-on-raw / deploy-on-BPM trade-off as the stress and anxiety models.

## Reproduce
```
python3 ml/focus/src/extract_cogwear_features.py        # PPG -> HRV feature CSVs (enhanced RR, 120s)
python3 ml/focus/src/train_cogwear_focus.py samsung     # train + LOSO + export TS model
python3 ml/focus/src/experiment_enhance.py              # ablations: RR method / window / aggregation
```
