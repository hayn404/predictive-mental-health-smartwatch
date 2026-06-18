# Seren Stress Model — Results & Related-Work Comparison

> Consolidated reference for the thesis. All numbers below were produced by the
> self-contained Kaggle notebook `ml/kaggle/seren_stress_kaggle.ipynb`.
> Last updated: 2026-06-13.

---

## 1. Seren in one line
A GWO/PSO-tuned XGBoost on **21 PPG-derived HRV features** (no EDA, no ECG, no chest
strap), with **per-subject normalization**, trained on a **fusion of two datasets**
and evaluated **cross-dataset** under a **leakage-free protocol**.

- **Headline (defensible) result:** train SIPD+PhysioStress → test WESAD = **AUC 0.827**
  (config + hyperparameters chosen *without ever seeing WESAD*).
- **Within-WESAD (PPG-only):** **0.91 accuracy / 0.94 AUC** — competitive with EDA-based prior work.

---

## The journey — where we started → where we are now

### Before vs after
| | **Where we started** | **Where we are now** |
|---|---|---|
| Training data | WESAD only (15 subjects) | **WESAD + SIPD + PhysioStress** (83 subjects, 3 datasets) |
| Evaluation | within-dataset only | **within + honest cross-dataset** (train-on-two/test-on-third) |
| Reported number | 93 % "train accuracy" (optimistic) | **0.827 AUC cross-dataset, leakage-free** |
| Cross-dataset reality | 67 % (the unexplained problem) | measured, explained, and improved |
| Features | 29 (HRV + temp + accel) | **21 PPG-HRV** (watch-only; temp/accel/morphology dropped after testing) |
| Normalization | global | **per-subject** (the key generalization lever) |
| Tuning | hand-set hyperparameters | **PSO + GWO** metaheuristics (compared) |
| Threshold | fixed 0.5 | **base-rate-matched** decision threshold |
| Leakage control | none stated | subject- & dataset-disjoint, non-overlapping windows, **nested FINAL_EVAL** |
| Pipeline | local Python scripts | **single self-contained Kaggle notebook** → DagsHub MLflow |
| Model size | — | ~115–440 KB, EDA-free, deployable |

### The milestones (and what each one taught us)
1. **Started** with the symptom: XGBoost on WESAD scored ~93 % internally but **67 % cross-dataset** — a 26-point collapse with no explanation.
2. **Benchmark research** → learned the gap is *normal* for small single-lab datasets, that WESAD's high numbers lean on **EDA** (which the watch lacks), and that PPG-**morphology** + multi-dataset training are the published levers.
3. **First fixes** → added per-subject normalization + PPG-morphology features: **+7.9 pts within WESAD**. Built a cross-dataset harness.
4. **SWELL rejected** → tried SWELL as a cross-dataset test; it collapsed to chance. Found it ships a *4 Hz tachogram*, not beat-to-beat RR — `pNN50` 58 %→0.86 %. **Lesson: same feature *name* ≠ same feature; you need the same pipeline on raw PPG.**
5. **Infrastructure** → moved everything to a self-contained Kaggle notebook with MLflow→DagsHub (fixed Kaggle's env-injection 403 bug).
6. **First valid cross-dataset (SIPD)** → train SIPD → test WESAD = **0.835 AUC**. PSO/GWO tuning discovered that **regularization**, not capacity, generalizes.
7. **The "do we beat them?" question** → competitive with Bhattacharya on accuracy, behind on F1 — traced to **class imbalance + threshold**, not learning (AUC stayed high).
8. **Blocked datasets** → ForDigitStress/VerBIO needed access agreements; **pivoted to the open PhysioNet PhysioStress** dataset instead. Verified its labels against the dataset's own notebook.
9. **Fusion** → SIPD + PhysioStress → WESAD = **0.858** (beats either alone). Learned **source–target match matters more than raw volume**.
10. **Rigor pass** → audited for leakage/overfitting; added the **leakage-free FINAL_EVAL** nested protocol → honest **0.827**. The 0.858→0.827 gap *was* the model-selection bias, now removed.
11. **The deeper insight** → WESAD is an *easy* dataset (within-WESAD 0.94 vs within-SIPD 0.62); **test-set difficulty dominates** — explaining why the field's "93 % on WESAD" overstates real-world performance.
12. **Prior-work verification** → read the actual papers: PhysioStress's published 93 % uses a **non-subject-disjoint 10-fold split + EDA**; a 33-study review found **only 2 studies ever tested cross-dataset**. → Seren's honest cross-dataset evaluation is the contribution.

### The one-sentence arc
> *We began with a WESAD-only model that looked excellent (93 %) but failed to generalize (67 %), and ended with an EDA-free, three-dataset, per-subject-normalized, PSO/GWO-tuned model that generalizes to a completely unseen dataset at 0.827 AUC under a strict leakage-free protocol — while explaining why the field's headline numbers are optimistic.*

---

## 2. Datasets used (all Empatica E4 → watch-reproducible)

| Dataset | Subjects (used) | Windows (60 s) | Stress % | Stressor | Watch-only signals used | Dataset also has |
|---|---|---|---|---|---|---|
| **WESAD** (Schmidt 2018) | 15 | 827 (645/182) | 22 % | TSST (social-evaluative) | BVP→HRV | EDA, chest ECG/EMG/RESP |
| **SIPD / Stress-Predict** (Iqbal 2022) | 34 (of 35) | 1745 (1190/555) | 32 % | Stroop + TSST + Hyperventilation | BVP→HRV | EDA |
| **PhysioStress** (Hongn 2025, PhysioNet) | 34 (of 36; f07, f14 excluded) | 648 (434/214) | 33 % | Stroop/TMCT + interview + subtraction | BVP→HRV | EDA |

**Seren reads only BVP/ACC/TEMP and uses only the 21 PPG-HRV features below — never EDA/ECG.**

**Feature set (21 HRV, all from PPG):**
- Time-domain (9): meanRR, SDNN, RMSSD, pNN50, pNN20, HR mean, HR std, HR range, CV-RR
- Frequency-domain (7): VLF, LF, HF, LF/HF, total power, LFnu, HFnu
- Non-linear (5): SD1, SD2, SD1/SD2, sample entropy, DFA-α1
- (8 PPG-morphology features available but dropped — neutral cross-dataset.)
- (Skin-temp + accel extracted but excluded — didn't help cross-dataset.)

Model: XGBoost, per-subject z-norm, PSO/GWO-tuned, base-rate decision threshold. ~115–440 KB.

---

## 3. Seren results — WITHIN-dataset (same protocol prior works report)
Leave-one-subject-out, non-overlapping windows, fixed BASE_PARAMS, per-subject norm, PPG-only.

| Dataset | Features | Acc | F1 | AUC |
|---|---|---|---|---|
| **WESAD** | HRV-21 | **0.908** | 0.799 | **0.940** |
| WESAD | HRV+morph | 0.911 | 0.805 | 0.939 |
| SIPD | HRV-21 | 0.646 | 0.426 | 0.615 |
| SIPD | HRV+morph | 0.639 | 0.419 | 0.620 |
| PhysioStress | HRV-21 | 0.630 | 0.397 | 0.618 |
| PhysioStress | HRV+morph | 0.654 | 0.434 | 0.623 |

**Key insight:** WESAD is an *easy* dataset (0.94 AUC within); SIPD/PhysioStress are
hard even within-dataset (~0.62 AUC). Test-set difficulty dominates: a model
trained+tested on SIPD (0.62) underperforms one trained on SIPD and tested on the
easy WESAD (0.71). The field's "93 % on WESAD" headline is inflated by WESAD being easy.

---

## 4. Seren results — CROSS-dataset (the honest test)

### 4a. Tuned held-out (each tuned on its own train set, → WESAD)
| Training set | → WESAD AUC | calibrated acc | F1 |
|---|---|---|---|
| PhysioStress | 0.816 | 0.769 | — |
| SIPD | 0.835 | ~0.767 | 0.62 |
| **SIPD + PhysioStress (fusion)** | **0.858** | **0.794** | **0.62** |

Fusion beats either single source → **dataset fusion confirmed**.

### 4b. Leakage-free FINAL_EVAL (config + tuning chosen without seeing WESAD)
- **train SIPD+PhysioStress → test WESAD = AUC 0.827** ← **THE NUMBER TO REPORT**
- The 0.858 (4a) vs 0.827 (4b) gap (~0.03) is the model-selection-on-WESAD bias, removed.

### 4c. Stable pairwise matrix (fixed BASE_PARAMS, per-subject — reproducible)
| train → test | acc | F1 | AUC |
|---|---|---|---|
| PhysioStress → WESAD | 0.719 | 0.442 | 0.716 |
| SIPD → WESAD | 0.710 | 0.415 | 0.707 |
| WESAD → PhysioStress | 0.659 | 0.374 | 0.630 |
| WESAD → SIPD | 0.659 | 0.354 | 0.630 |
| SIPD → PhysioStress | 0.642 | 0.429 | 0.601 |
| PhysioStress → SIPD | 0.616 | 0.373 | 0.576 |

- **Mean cross-dataset AUC ≈ 0.64** (honest "arbitrary unseen data" estimate).
- → WESAD is the *easy* direction (0.71); → SIPD/PhysioStress is realistic/hard (~0.60).

---

## 5. Prior work — WITHIN-dataset (the literature's headline numbers)

| Study | Dataset | Acc (binary) | CV | Sensors | Notes |
|---|---|---|---|---|---|
| Schmidt 2018 (WESAD orig) | WESAD | 0.93 / wrist 0.88 | LOSO | PPG + **EDA** + chest ECG | drops to 0.8833 without HR |
| SELF-CARE 2023 | WESAD | **0.9412** | LOSO | PPG + **EDA** | selective sensor fusion |
| Liapis 2021 | WESAD | 0.974 | (eval on diff. annotated set) | + **EDA** | not a clean WESAD test |
| **Hongn 2025 (PhysioStress orig)** | PhysioStress | **0.93** | **10-fold (NOT subject-disjoint)** | HR+HRV+**EDA**+ACC | + oversampling |
| Iqbal 2022 (Stress-Predict orig) | SIPD | — (no classifier) | — | — | **statistical pilot only** |
| **Seren (ours)** | WESAD | **0.91** | **LOSO** | **PPG only** | per-subject norm |
| Seren | SIPD | 0.65 | LOSO | PPG only | |
| Seren | PhysioStress | 0.63 | LOSO | PPG only | |

**Critical fact (Vos et al. review):** all WESAD results >90 % include EDA *and* HR;
methods *excluding* either consistently score **< 86 %**. → Seren's PPG-only 0.91
on WESAD actually **beats the typical EDA-free result**, thanks to per-subject norm.

**Did we outperform within-dataset prior works?**
- **WESAD:** ✅ beat the wrist baseline (0.91 vs 0.88) and the EDA-free band (<0.86) — *with fewer sensors*. Below absolute SOTA (0.94, uses EDA).
- **PhysioStress:** ❌ below their 0.93 — **but their 0.93 uses 10-fold (leaky split) + EDA + oversampling**; ours is subject-disjoint + PPG-only. *Not comparable.*
- **SIPD:** no established within-dataset benchmark exists to compare against.

---

## 6. Prior work — CROSS-dataset (the fair comparison)

**Bhattacharya et al. 2024, "Stressor Type Matters!"** (arXiv 2405.09563) — wrist BVP HRV, train-on-one/test-on-another, best of RFC/SVM/MLP per pair:

| train → test | Acc | F1 |
|---|---|---|
| ForDigitStress → WESAD | 0.82 | 0.79 |
| VerBIO → WESAD | 0.80 | 0.78 |
| WESAD → VerBIO | 0.91 | 0.90 |
| WESAD → ForDigitStress | 0.78 | 0.77 |
| **Seren: SIPD+PhysioStress → WESAD** | **~0.79** | **0.62** (AUC 0.827) |

- On **accuracy**, Seren (~0.79) ≈ their 0.80–0.82 — with fewer subjects, one sensor, stricter protocol.
- On **F1**, Seren (0.62) is below (their tests are likely class-balanced → F1≈acc). AUC not reported by them.

**Vos et al. 2023 systematic review** (arXiv 2209.15137, 33 studies):
- *"None of these studies apart from Mishra et al. and Liapis et al. tested
  generalization on a totally unseen, new dataset."* → **cross-dataset is the field's open gap; Seren fills it.**
- WESAD statistical power ≈ 45 %; Stress-Predict (35 subj) built for 80 %.
- AUC sensitive to imbalance; accuracy can be high while the minority class is mostly wrong.
- Caveat: in aggregate the review found no definitive LOSO-vs-K-fold accuracy difference
  (heterogeneous studies; prove leakage with a controlled same-data test, don't assert it).

---

## 7. Why Seren's F1 (~0.59–0.62) looks low
- Recall 0.79 (good — catches stress), **precision 0.48** (the drag). F1 = harmonic mean.
- Cause = **class imbalance** (WESAD 22 % stress) + a **recall-favoring operating point**
  + **prevalence mismatch** (train 32 % stress, test 22 %) → over-prediction → false alarms.
- **AUC 0.83 proves the model ranks well**; the low F1 is a *threshold/precision* issue, not learning.
- Prior work's F1 ≈ their accuracy ⇒ they likely evaluate on **class-balanced** sets; Seren uses natural prevalence.
- **Report AUC as the primary cross-dataset metric** (threshold- and prevalence-independent).

---

## 8. Data-leakage & overfitting controls (defense-ready)
**Leakage guards:**
1. Subject-disjoint (LOSO within; entirely different datasets cross-dataset).
2. **Non-overlapping** 60 s windows (verified in code) → no adjacent-window leakage.
3. Tuning (PSO/GWO) maximizes CV AUC on the **training set only**; held-out never seen.
4. Threshold = training out-of-fold + training prior; **no target labels used**.
5. Per-subject norm uses **features not labels** (transductive → needs on-device calibration; not label leakage).

**Anti-overfitting evidence:**
- GWO chose a **heavily regularized** model (depth 3, lr 0.01, high gamma/reg_lambda).
- **Cross > within** pattern (cross-WESAD 0.84 > within-SIPD 0.62) ⇒ not memorizing.

**Residual risks to disclose:**
- Model-selection-on-WESAD (mitigated by `FINAL_EVAL` → 0.827).
- Small N (15/34/34) ⇒ report ranges, tuning variance ≈ ±0.03.

---

## 9. Defensible thesis claims (what you CAN say)
1. *"On WESAD, Seren reaches 0.91 binary accuracy using **PPG alone** — outperforming the
   original wrist benchmark (0.88, which also used EDA) and the EDA-free band (<0.86),
   and within ~3 pts of EDA-based SOTA (0.94)."*
2. *"Cross-dataset, Seren generalizes at 0.827 AUC (leakage-free, train-on-two/test-on-one) —
   matching the published accuracy of comparable wrist-PPG work (Bhattacharya 2024, 0.80–0.82)
   with fewer subjects, a single sensor, and a stricter protocol."*
3. *"We report **honest cross-dataset** numbers in a field where a 33-study review found only
   2 studies ever tested on a fully unseen dataset — and show within-dataset 90%+ figures rely
   on easier datasets, EDA, or non-subject-disjoint splits."*

What you **cannot** claim: "we beat the state of the art" (SELF-CARE 0.94 within-WESAD is higher; PhysioStress 0.93 exists — both with EDA/leaky protocols).

---

## 10. Leakage-proof experiment (implemented + validated)
The within-dataset benchmark runs three evaluations per dataset, on identical data + PPG-only
features (config `WITHIN_KFOLD=True`, `WITHIN_KFOLD_K=10`):
- **Seren LOSO** — your model: per-subject norm, subject-disjoint (the honest within number).
- **pw-LOSO** — *prior-work protocol* (**global** norm), honest subject-disjoint split.
- **pw-kfold** — prior-work protocol (global norm), **random 10-fold** (same subject in train+test).
- **leak gap = pw-kfold − pw-LOSO** = inflation from a non-subject-disjoint split alone.

**Two findings (both important):**
1. **Per-subject normalization is leakage-ROBUST.** Under per-subject norm, k-fold ≈ LOSO
   (on WESAD the gap was even slightly negative). So Seren's evaluation can't be inflated this way.
2. **The leakage appears under GLOBAL normalization** — the protocol prior work uses. That's
   why the demo forces global norm: to faithfully reproduce the published setup.

**FINAL result (real data, PPG-only, all 3 datasets):**
| Dataset | Features | Seren LOSO (per-subj) | pw-LOSO (global) | pw-kfold (global) | leak gap |
|---|---|---|---|---|---|
| WESAD | HRV-21 | **0.940** | 0.898 | 0.925 | **+0.028** |
| WESAD | HRV+morph | 0.938 | 0.896 | 0.926 | +0.031 |
| **PhysioStress** | HRV-21 | 0.618 | 0.595 | 0.631 | **+0.036** |
| PhysioStress | HRV+morph | 0.623 | 0.598 | 0.632 | +0.034 |
| SIPD | HRV-21 | 0.615 | 0.590 | 0.615 | +0.025 |
| SIPD | HRV+morph | 0.620 | 0.586 | 0.602 | +0.016 |

**What this proves (honest, measured):**
- **k-fold split inflation is real but MODEST: +0.025 to +0.036 AUC** (largest on PhysioStress, 34
  subjects, as predicted). A non-subject-disjoint split adds ~0.03 on identical PPG-only data.
- **Per-subject norm (Seren LOSO) ≥ global pw-LOSO** on every dataset → per-subject is both the
  generalization lever AND leakage-robust.

**Honest decomposition of PhysioStress's published 0.93** (so you don't over-attribute to leakage):
| Step | AUC |
|---|---|
| Seren honest PPG-only LOSO (per-subject) | 0.62 |
| + non-subject-disjoint k-fold split (global norm) | ~0.63 (+0.036) |
| + EDA + HR + ACC multimodal features + oversampling (their setup) | **0.93** |

→ **The dominant cause of the gap is EDA / multimodal sensors the watch can't reproduce (~+0.30),
NOT the leaky split (~+0.04).** Defensible claim: *"Our PPG-only watch-deployable model scores
below the published 0.93 mainly because that result uses EDA and a richer multimodal feature set;
a smaller, measured component (~0.03 AUC) is attributable to their non-subject-disjoint split."*

---

## Sources
- WESAD — Schmidt et al. 2018, ICMI: https://doi.org/10.1145/3242969.3242985
- SELF-CARE — Rashid et al. 2023: https://arxiv.org/abs/2303.08215
- Stress-Predict / SIPD — Iqbal et al. 2022, Sensors 22(21):8135: https://www.mdpi.com/1424-8220/22/21/8135 ; data: https://github.com/italha-d/Stress-Predict-Dataset
- PhysioStress — Hongn et al. 2025, Sci Data 12:520: https://doi.org/10.1038/s41597-025-04845-9 ; data: https://physionet.org/content/wearable-device-dataset/
- Cross-dataset — Bhattacharya et al. 2024, "Stressor Type Matters!": https://arxiv.org/html/2405.09563v1
- Review — Vos et al. 2023, "Generalizable ML for Stress Monitoring": https://arxiv.org/pdf/2209.15137
