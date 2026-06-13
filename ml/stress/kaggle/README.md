# Seren — Stress model: Kaggle notebook (single source of truth)

`seren_stress_kaggle.ipynb` is the **only** place the stress model is produced.
Everything — data loading, preprocessing, feature extraction, per-subject
normalization, PSO/GWO hyperparameter tuning, threshold calibration, training,
evaluation, and model export — runs **inside the notebook on Kaggle**. There is
no local training pipeline; the notebook is fully self-contained.

**Output → app:** the notebook writes `stress_model.json` + `model_metadata.json`
to `/kaggle/working/`. Download them and drop them into the mobile app
(`assets/ml/` and `ml/models/`).

## Run on Kaggle
1. New Notebook → **Add Data** → add **WESAD** (folders `S2/S2.pkl … S17/S17.pkl`).
2. (Recommended) Add **Stress-Predict / SIPD** (`italha-d/Stress-Predict-Dataset`)
   for cross-dataset training/eval.
3. Upload `seren_stress_kaggle.ipynb`, set the **CONFIG** cell, **Run All**.
4. Download `stress_model.json` + `model_metadata.json` from the **Output** tab.

## Datasets (all Empatica E4 → watch-only: BVP/ACC/TEMP read, EDA/ECG ignored)
| Dataset | Access | Notes |
|---|---|---|
| WESAD | open (Kaggle) | required base |
| Stress-Predict / SIPD | open (GitHub) | recommended; richer, transfers better |
| **PhysioStress** (PhysioNet wearable-device-dataset) | **open, no agreement** | recommended 1st fusion add (verified: 34 subj, 648 windows, 33% stress); set `USE_PHYSIOSTRESS`, point `PHYSIOSTRESS_ROOT` at `.../Wearable_Dataset/STRESS` |
| ForDigitStress | access-by-request | `USE_FORDIGIT` |
| VerBIO | public (TAMU) | `USE_VERBIO` |

`TRAIN_ON` accepts any `+`-combo, e.g. `"WESAD+SIPD+PhysioStress"`; `EVAL_ON` is a
held-out dataset for the honest cross-dataset number.

## Key CONFIG knobs
- `TRAIN_ON` / `EVAL_ON` — training fusion set and held-out test (default `SIPD` → `WESAD`).
- `USE_MORPHOLOGY` — 8 raw-PPG pulse-shape features (helps within-dataset).
- `NORMALIZATION` — `"per_subject"` (best for generalization) | `"global"` (drop-in).
- `RUN_TUNING` / `TUNERS=["pso","gwo"]` — nature-inspired tuning, no test leakage.
- `CALIBRATE_THRESHOLD` / `THRESHOLD_OBJECTIVE="prior"` — base-rate-matched decision threshold.
- `MLFLOW_TRACKING_URI` — DagsHub by default (needs Kaggle internet + `DAGSHUB_TOKEN` secret).
- `FINAL_EVAL` / `FINAL_TEST` — leakage-free final number for the thesis: picks the
  config + tunes using ONLY the dev datasets (leave-one-dev-dataset-out), then evaluates
  `FINAL_TEST` (default WESAD) **exactly once**. Removes model-selection-on-test bias.
  Needs >=3 datasets loaded. Logs `final_clean_auc` to MLflow.

## Label-file note (fusion datasets)
- **PhysioStress** labels are FINALIZED & verified against the real download:
  protocol-based (stressor task vs baseline) using the exact tag→stage mapping
  from the dataset's own `Wearable_Dataset.ipynb` (f07 + f14 excluded).
- **ForDigitStress / VerBIO** labels are still best-effort (continuous-annotation
  threshold); the `ANN_GLOB` pattern may need a tweak once you obtain them.

## ⚠️ Before you SHIP an improved model
The exported file is only drop-in if it matches the on-device feature pipeline:
- `USE_MORPHOLOGY=True` → the TS inference (`services/ai/stressModel.ts`) must also
  compute the 8 `ppg*` features from raw PPG.
- `NORMALIZATION="per_subject"` → the watch must z-score against a per-user baseline.
- The exported `decisionThreshold` should be used instead of a hard 0.5.

For a strictly drop-in model with no TS changes: `USE_MORPHOLOGY=False`,
`NORMALIZATION="global"`, `CALIBRATE_THRESHOLD=False`.

## Files
- `seren_stress_kaggle.ipynb` — the notebook (the deliverable).
- `build_notebook.py` — regenerates the .ipynb (edit cells here, re-run to rebuild).
- `_validate_local.py` — dev-only smoke test (runs the notebook's cells against a
  local WESAD copy + a synthetic dataset to catch bugs before you run on Kaggle).
