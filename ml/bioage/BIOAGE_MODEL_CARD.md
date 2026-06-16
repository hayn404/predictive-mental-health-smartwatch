# Seren Biological-Age ("Heart Age") Model — Model Card

## What it predicts
**Physiological age (years) from wearable HRV**, and the **age-gap** = predicted − chronological.
A positive gap means the body looks "older than its years" — a state accelerated by chronic
stress and poor recovery. The age-gap is the headline metric that ties the app's stress,
sleep, and recovery signals into one number.

> Wellness/screening estimate — not a clinical or diagnostic measure.

## Data
| Role | Dataset | Subjects | Signal | Label |
|------|---------|----------|--------|-------|
| TRAIN | PhysioNet **Autonomic Aging** | 1,095 | ECG lead II (→ RR → HRV) | age group (15 bins → midpoint) |
| TEST | PhysioNet **Fantasia** | 40 | ECG + beat annotations | exact age (21–34, 68–85) |

## Method
- ECG → R-peaks (Pan–Tompkins, ECG downsampled to 250 Hz) → RR → **14 HRV features**
  (reuses `ml/stress/src/features.py`). Same features the watch computes from PPG.
- **Subject-aggregate**: median of a subject's 120 s windows (age is a stable trait).
- **GLOBAL normalization** (NOT per-user) — age is between-subject, so per-user z-scoring
  would erase the signal.
- **XGBoost regressor** (depth 3, 400 trees, lr 0.03). Exported to the TS tree format
  (`assets/ml/bioage/bioage_model.json`) consumed on-device by `services/ai/bioAgeModel.ts`.
- On-device: aggregate the user's recent windows (median) → global-normalize → predict →
  age-gap vs the user's chronological age.

## Results
| Validation | MAE (yrs) | r | R² |
|------------|-----------|---|-----|
| **Autonomic Aging — 10-fold subject-level** | **7.58** | **0.68** | 0.46 |
| Fantasia — within-dataset LOSO | 10.95 | 0.80 | 0.64 |
| **Cross-dataset AA → Fantasia (young vs old)** | — | — | **AUC 0.94** |
| Cross-dataset AA → Fantasia (absolute age) | 15 | 0.77 | — |

- **Within-population: literature-grade** (published HRV-age ≈ 5–7 yrs MAE).
- **Cross-dataset discrimination generalizes strongly** (young-vs-old AUC 0.94) across a
  different device and population.
- **Absolute cross-dataset MAE is inflated** (~15 yrs): the model under-predicts the elderly
  on Fantasia (young predicted 29 vs true 28 = exact; old predicted ~50 vs true 76). The
  *ranking* transfers; the *absolute calibration* for old ages does not, due to device +
  age-group-label coarseness + Fantasia's bimodal ages. Deployment surfaces the *relative*
  age-gap and recalibrates per device.

## Mental-health link
Chronic stress measurably accelerates biological aging (epigenetic age acceleration,
telomere shortening ~3–6 yrs, cortisol). So the age-gap is framed as: *"your stress / sleep /
recovery are shaping your physiological age — manage them to lower it."* This unifies the
other Seren vitals into one motivating headline.

## Deployment (in-app)
- `services/ai/bioAgeModel.ts` — on-device inference (median aggregation + global norm + tree
  traversal → age + age-gap).
- `hooks/useWellness.tsx` — loads the model, computes bio-age each cycle from the rolling
  feature history, exposes `bioAge`, `chronologicalAge`, `setChronologicalAge`.
- `app/(tabs)/index.tsx` — "Physiological Age" card (predicted age + age-gap badge + message).

## Limitations
- **Chronological age must be provided** (no onboarding field yet; demo default = 25). The
  age-gap needs it; predicted age alone does not.
- **ECG-trained / PPG-deployed gap** — like the other vitals; HRV features bridge it.
- **Age labels are group midpoints** (≈5-yr bins) → quantization floor on MAE.
- **Cross-device absolute calibration** under-predicts the elderly; surface the relative gap.
- Wellness screening, not diagnosis.

## Reproduce
```
python3 ml/bioage/src/extract_bioage.py aa          # ECG -> RR -> HRV features (AA)
python3 ml/bioage/src/extract_bioage.py fantasia    # (Fantasia)
python3 ml/bioage/src/finalize_bioage.py            # 10-fold CV + train final + export native model
python3 ml/bioage/src/export_bioage_ts.py           # convert to TS tree format for the app
```
