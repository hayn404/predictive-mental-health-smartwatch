# Seren — Mental Fatigue / Cognitive Depletion Model

Detects cognitive depletion (the flip side of focus: engaged → drained) from wrist HRV.
Same architecture as the focus model: 14 HRV features + per-subject z-normalization +
XGBoost, validated with LOSO (cross-subject) and an external dataset (cross-dataset).

## Datasets (free, within-subject fatigue contrast)

| Role | Dataset | Subjects | Signals | Label | Link |
|------|---------|----------|---------|-------|------|
| TRAIN | MEFAR | 23 | Empatica E4 wrist BVP/EDA/HR/temp/accel (+1ch EEG) | Chalder Fatigue Scale, morning vs evening (cutoff 12) | https://data.mendeley.com/datasets/z3g26tphnv/5 |
| EXTERNAL TEST | SAE | 346 | ECG + EDA + RESP | driving fatigue (KSS/questionnaire) | https://doi.org/10.5281/zenodo.7214953 |
| (alt test) | DD-Database | 10 | ECG | driving drowsiness (KSS) | https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c |

MEFAR uses the Empatica E4 (same sensor family as CogWear), so its BVP → HRV extraction
reuses the focus enhanced-RR pipeline (`ml/focus/src/extract_cogwear_features.py`).

## Pipeline
1. Adapters (to be written after download) → standardized `fatigue_features_<dataset>.csv`
   with `subject_id, label, [14 HRV features]` (label: 0 = fresh, 1 = fatigued).
   - MEFAR: E4 BVP (64 Hz) → enhanced RR → HRV; label from Chalder (morning=fresh / evening=fatigued, or score cutoff).
   - SAE: ECG → R-peaks (Pan–Tompkins) → HRV; label from KSS/fatigue rating (binarized).
2. `train_fatigue.py` → per-subject z-norm + XGBoost + LOSO + external (MEFAR→SAE).

## Honest checks (lessons from the focus/anxiety work)
- Confirm labels are WITHIN-SUBJECT (fresh vs fatigued over time), not one score per person.
- Construct nuance: MEFAR = occupational mental fatigue; SAE = driving fatigue. The external
  test is therefore a genuine cross-construct generalization check (report honestly).
- AUC is the headline metric; per-user calibration handles thresholds at deployment.

## Status
Scaffolding ready. Awaiting MEFAR + SAE download → build adapters → run.
