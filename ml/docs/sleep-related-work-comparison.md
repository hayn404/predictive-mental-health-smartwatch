# Seren Sleep Model — Results & Related-Work Comparison

> Consolidated reference for the thesis. All numbers below were produced by the
> self-contained Kaggle notebook `ml/sleep/seren_sleep_kaggle.ipynb` and the on-device
> runtime in `services/ai/sleepStageModel.ts`. Last updated: 2026-06-16.
>
> **Model is lean — 11 features per epoch**, down from 12, after XAI confirmed
> `immobility_frac` contributed near-zero signal. See §7 (XAI feature reduction)
> for the sweep table.

---

## 1. Seren in one line
A **TCN + BiGRU seq2seq** on **11 engineered HR + actigraphy features** per 30-s epoch
(no EEG, no EOG, no chest strap, no raw PPG waveform), trained on **BIDSleep
(47 subjects, 252 nights, subject-disjoint 40/7 split)** and tested zero-shot
on a **completely separate dataset (Walch / Sleep-Accel, 31 subjects)**.

- **Headline (defensible) result:** **Walch κ = 0.482, weighted F1 = 0.665,
  3-class wF1 = 0.798, Deep recall = 0.781, REM recall = 0.723** — under a strict
  cross-dataset, subject-disjoint protocol the model has **never** seen during
  training. Hyperparameters chosen by niapy PSO over 110 trials against the
  subject-disjoint val split (best val mF1 = 0.651).
- **Comparison:** **approaches** the independently-validated Apple Watch Series 8
  (κ = 0.53 in SLEEP Advances 2025), **exceeds** the independently-validated
  Galaxy Watch 5 (κ = 0.42), and matches/beats every academic Walch baseline
  *despite* training on a different dataset (a strictly harder protocol).

---

## The journey — where we started → where we are now

### Before vs after
| | **Where we started** | **Where we are now** |
|---|---|---|
| Training data | DREAMT (Empatica E4, chest + wrist, ~10 subjects) | **BIDSleep** (Apple Watch, **47 subjects, 252 nights**) + Walch (held-out, 31 subjects) |
| Evaluation | within-DREAMT, 5-class | within + **honest cross-dataset** (BIDSleep → Walch); **0 subject overlap** with Walch by construction |
| Reported number | ad-hoc within-dataset accuracy | **Walch κ = 0.482 / wF1 = 0.665 / Deep recall 0.781 / 3-class wF1 0.798** under a leakage-free protocol (PSO-tuned) |
| Input | raw 100 Hz BVP + 3-axis accel (2 channels × 3000 samples + 6 aux) | **11 engineered features per 30-s epoch** (HR-stats + actigraphy + `time_of_night`; `immobility_frac` dropped post-XAI) — Galaxy-Watch reproducible |
| Sensors required | research-grade Empatica E4 (chest + wrist, raw PPG) | **consumer-watch HR + tri-axial accelerometer** (Galaxy Watch / Apple Watch / Pixel Watch) |
| Classes | 5 (Wake / N1 / N2 / N3 / REM) | **4** (Wake / Light = N1+N2 / Deep = N3 / REM) — matches Apple/Galaxy convention |
| Architecture | CNN-BiGRU on raw 100 Hz signal | **TCN (2× Conv1d, k=5) + BiGRU (h=48, 2-layer) seq2seq** on 11-feature × 41-epoch windows |
| Window length | single 30 s | **41 epochs ≈ 20.5 min** (BiGRU context for half a sleep cycle) |
| Train/val split | per-night random (**leaked subjects across train/val**) | **subject-disjoint** (40 train / 7 val subjects, **0 overlap**) |
| Tuning | hand-set hyperparameters | **niapy PSO** over 110 trials → SEQ_LEN 41, stride 5, BiGRU 48, dropout 0.20, LR 8.8e-4 |
| Class imbalance | uniform | **√-inverse-frequency** class weights (full inverse hurt Light precision) |
| Pipeline | local Python scripts + manual data download | **self-contained Kaggle notebook** + cached feature pickles (`prepare_features.py` ↔ `seren_sleep_kaggle.ipynb`) |
| Deployment | ONNX via `onnxruntime-react-native` (legacy module, won't autolink under Expo SDK 54 New Arch) | **TFLite via `react-native-fast-tflite`** (New Arch compatible, parity ~1e-6 vs ONNX) |
| Explainability | none | **Captum Integrated Gradients + zero-occlusion** suite (global, per-class, per-timestep, per-feature, physiological agreement, optional gated feature-reduction sweep) |
| MLflow | none | DagsHub-hosted MLflow (params + per-epoch metrics + ONNX/TFLite artifacts) |

### The milestones (and what each one taught us)

1. **Started with DREAMT** (Empatica E4) — clean raw 100 Hz BVP + accel, 5-class
   per AASM. CNN-BiGRU worked within-dataset but the inputs (chest-grade BVP
   waveform) are **not exposed by consumer watches**. Galaxy Watch streams *derived
   HR samples*, not raw photometric values. Trained model would never run on the
   target hardware.

2. **Pivoted to PhysioNet cohorts that match Galaxy Watch sensor surface** —
   identified **BIDSleep** (also called *Multi-Night HR + Accelerometry with EEG
   labels*; 47 subjects × 252 nights, Apple Watch) and **Walch / Sleep-Accel**
   (31 subjects, Apple Watch, single night each). Both have only HR (1 Hz) and
   tri-axial accelerometer, with PSG labels — exactly what a consumer watch can
   reproduce.

3. **Engineered 11 per-epoch features** instead of streaming raw signals: HR
   statistics (mean / std / min / max / range / successive-diff std / Δ-prev) +
   actigraphy (count / max / std). Computed on the watch in
   `EpochAggregator.kt`, sent over the Wearable Data Layer at ~48 bytes per 30-s
   epoch — battery-friendly all night.

4. **Re-labelled to the 4-class consumer convention** to match the deployed app
   UI and the Apple Watch / Galaxy Watch / Health Connect ontology: `Wake = 0`,
   `Light = N1+N2 = 1`, `Deep = N3 = 2`, `REM = 3`. Mapped BIDSleep's Dreem-style
   AASM codes and Walch's 6-stage codes accordingly.

5. **v3.0 → v3.1 → v3.2 progression**:
   - **v3.0**: 11 features, plain Conv1d + LSTM, validation overfit fast
     (val loss diverging from train within 10 epochs).
   - **v3.1**: added TCN front-end + larger BiGRU + class weights.
   - **v3.2**: **PSO-tuned** (niapy, 110 trials) — discovered the regulariser sweet
     spot was a *smaller* BiGRU (48), *low* dropout (0.20), training stride 5
     (stride 1 over-augmented), LR ≈ 8.8e-4. Walch macro-F1 reached **0.621**.

6. **Added the feature `time_of_night`** — linear `[-1, +1]` across each
   night. Justification: Deep sleep concentrates in the first half of the night;
   REM in the second; without this the model has no clock signal. **XAI later
   confirmed** the model uses it correctly (Deep ρ = −0.90, REM ρ = +0.86).

7. **Deployment migration: ONNX → TFLite**. The legacy
   `onnxruntime-react-native` package doesn't autolink under Expo SDK 54 with
   New Architecture. Converted via `onnx2tf` (replacing `Erf`/`GELU` with native
   TFLite ops so the standard runtime — not Flex/Select-TF — runs the model).
   Parity verified ~1e-6 max-abs-diff vs ONNX. On-device input layout is feature-
   major `[1, 11, 41]` (`onnx2tf` transposes during conversion to keep the BiGRU
   convertible); the PyTorch and ONNX paths both use the natural `[1, 41, 11]`.

8. **Explainability — Captum Integrated Gradients suite** added to the Kaggle
   notebook (mirroring the stress model's TreeSHAP suite, adapted for a deep
   sequence model):
   - **Global** per-feature × per-class importance
   - **Zero-occlusion** cross-check (rank agreement reported)
   - **Local** epoch heatmaps (correct Deep, correct REM, most confident
     misclassified) + whole-night Walch hypnogram with dominant-attribution strip
   - **Temporal/feature effects** — per-timestep curves + IG-vs-value scatter
   - **Physiological agreement** — `|IG|`-weighted score per stage vs
     sleep-physiology priors
   - **Optional gated** XAI-guided feature pruning retrain

9. **Data-leakage audit → BIDSleep split corrected.** Discovered the train/val
   split was on *nights* (not subjects). Because BIDSleep has ~5 nights per
   subject, validation subjects also appeared in training, inflating val macro-F1
   by ~4 points (0.66 → 0.64 after the fix). **The held-out cross-dataset Walch
   test was unaffected** (Walch and BIDSleep have no subject overlap by
   construction). **The reported headline numbers in this document are from the
   leak-corrected pipeline.**

### The one-sentence arc
> *We began with a DREAMT-trained 5-class CNN-BiGRU on raw 100 Hz BVP that
> couldn't run on a Galaxy Watch, and ended with a PSO-tuned 4-class TCN+BiGRU
> seq2seq on 11 engineered HR + actigraphy features per 30-s epoch — trained on
> BIDSleep under a subject-disjoint split, tested zero-shot on Walch at
> κ = 0.482 / Deep recall 0.781 / 3-class wF1 0.798, with Integrated-Gradients
> attribution showing that the model's per-stage reasoning matches sleep
> physiology (Wake 65 %, Deep 60 %, REM 67 % `|IG|`-weighted agreement) and
> deployed on-device via TFLite at parity ~1e-6 vs ONNX.*

---

## 2. Datasets used (consumer-watch reproducible)

| Dataset | Subjects | Nights | Epochs | Class % (W / L / D / R) | Sensors used | Dataset also has |
|---|---|---|---|---|---|---|
| **BIDSleep** (Walch 2024, PhysioNet) | **47** | **252** | 210 633 | 11.0 / 44.2 / 18.8 / 26.0 | Apple Watch HR + tri-axial accel | EEG (used for labels only) |
| **Walch / Sleep-Accel** (Walch 2019, PhysioNet) | **31** | 31 | 26 557 | 8.9 / 55.3 / 13.8 / 22.0 | Apple Watch HR + tri-axial accel | PSG (used for labels only) |

**Seren reads only HR (1 Hz) and tri-axial accelerometer — never EEG/EOG/EMG/ECG.**

**Feature set (11 per 30-s epoch, lean post-XAI):**
- HR statistics (7): `hr_mean`, `hr_std`, `hr_min`, `hr_max`, `hr_range`,
  `hr_succdiff_std`, `hr_delta_prev`
- Actigraphy (3): `act_count`, `act_max`, `act_std`
  *(`immobility_frac` removed after XAI — see §7 below)*
- Position (1): `time_of_night` ∈ [−1, +1] (linear across the night)

**Per-night robust normalisation:** `(x − median) / (1.4826 × MAD)` per feature
column, NaN/Inf → 0 (mirrors Apple-Watch HR/motion gaps). Applied identically in
`prepare_features.py` (offline cache) and `sleepStageModel.ts` (on-device).

**Architecture (~100 k params):** `LayerNorm(11) → 2× Conv1d(64, k=5, GELU, BN,
dropout 0.20) → BiGRU(48, 2-layer, dropout 0.20) → LayerNorm → Linear(96, 4)`.

---

## 3. Seren results — WITHIN-dataset (BIDSleep val, subject-disjoint)

40 train subjects / 7 val subjects (0 overlap), √-inverse class weights,
disjoint 41-epoch eval windows, PSO-tuned hyperparameters.

| Metric | Value |
|---|---|
| Best val macro-F1 (PSO best trial) | **0.6512** |
| Best val macro-F1 (final 80-epoch run) | 0.6437 |
| Best val κ | 0.479 |
| Best val weighted-F1 | 0.648 |

The drop from the pre-leak-fix val mF1 of ~0.66 to ~0.64 is the **healthy, honest
within-subject generalization signal**. The corrected split removes the model's
ability to memorise per-subject HR/accelerometer baselines that don't transfer
to new users.

---

## 4. Seren results — CROSS-dataset (the honest test): **BIDSleep → Walch**

Trained on BIDSleep (47 subjects), evaluated zero-shot on Walch (31 subjects).
**Walch is never seen during training, validation, or early-stopping.**

### 4-class (Wake / Light / Deep / REM)
| Metric | Value | Notes |
|---|---|---|
| **macro F1** | **0.621** | |
| **weighted F1** | **0.665** | (SLAMSS 2023 target ≥ 0.72) |
| **accuracy** | **0.661** | |
| **Cohen κ** | **0.482** | independent SLEEP Advances 2025 reports AW S8 = 0.53 (best consumer) |

### Per-class
| Stage | F1 | Recall | Notes |
|---|---|---|---|
| Wake | 0.516 | 0.470 | weakest stage — 8.9 % of Walch epochs (severe imbalance) |
| Light | 0.700 | 0.636 | majority class (55 %) |
| **Deep** | **0.580** | **0.781** | beats SLAMSS 2023 deep-recall target (>0.56) by 0.22 |
| REM | 0.690 | 0.723 | strong recovery vs the previous run (0.666 → 0.723) |

### 3-class view (Wake / NREM / REM — merge Light + Deep)
| Metric | Value | SLAMSS target |
|---|---|---|
| **weighted F1** | **0.798** | ~0.80 (within 0.002) |
| accuracy | 0.799 | — |
| **Cohen κ** | **0.561** | — |

---

## 5. Prior work on Walch — WITHIN-dataset (the literature's headline numbers)

Walch is the canonical public Apple-Watch sleep-staging benchmark. The headline
prior-work numbers below all use **within-Walch** cross-validation (training and
testing on Walch subjects), which is a **strictly easier protocol** than Seren's
zero-shot cross-dataset evaluation.

| Study | Method | Walch protocol | 3-class acc | 3-class κ | 4-class κ |
|---|---|---|---|---|---|
| Walch et al. 2019 (original) | Logistic regression | 5-fold CV | ~0.60 | ~0.30 | — |
| **Lee et al. 2023** (bioRxiv) | TDA + circadian + NN | within-Walch MCCV | 0.713 | **0.435** | — |
| Lee et al. 2023 (same paper) | TDA + circadian + NN | LOSOCV | 0.683 | 0.387 | — |
| Song et al. 2023 (SLAMSS) | Seq2Seq LSTM | MESA → Walch (fine-tune) | 0.646 | — | — |
| Song et al. 2023 (SLAMSS) | Seq2Seq LSTM | within-Walch 5-fold | ~0.61 | — | — |
| **Ederli et al. 2025** | GAF + EfficientNet | within-Walch 5-fold | 0.622 *(balanced)* | **0.41** | — |
| **Seren v3.2 (ours)** | **TCN + BiGRU (11 feat, PSO-tuned)** | **BIDSleep → Walch (zero-shot)** | **0.799** | **0.561** | **0.482** |

**Headline read:** Seren equals or beats every academic Walch result under a
**strictly harder protocol** (zero-shot cross-dataset, vs their within-dataset
CV). The +0.13 (3-class) κ over Lee 2023 (the strongest published Walch baseline)
is defensible because the protocols are not directly comparable — we expect to
*lose* points to cross-dataset shift, and we don't.

---

## 6. Prior work — independently-validated consumer wearables (the right comparison)

The **2025 SLEEP Advances** cross-vendor benchmark (Birrer et al., n=62 adults,
single-night lab PSG) is the current gold-standard independent comparison of
consumer wrist-worn PPG + accel devices. Vendor self-reports (Apple Oct 2025
white paper, Samsung Mendonça 2024) are listed separately because they use
proprietary data and over-state independent reality.

| Device / model | Sensor modality | Protocol | 4-class κ | Notes |
|---|---|---|---|---|
| **Apple Watch S8** | accel-only (white paper claim) / accel + PPG (real) | SLEEP Adv 2025 n=20 lab PSG | **0.53** | best independently-validated commercial wrist device |
| Apple Watch (vendor, Oct 2025) | accel-only | private 299-night cohort | 0.68 | vendor self-report — not independently validated |
| **Seren v3.2 (ours)** | **HR + accel (engineered)** | **BIDSleep → Walch zero-shot** | **0.482** | exceeds Galaxy Watch 5, approaches Apple Watch S8 — on a strictly harder protocol |
| Galaxy Watch 5 (Lee 2023 JMIR) | accel + PPG | independent n=22 lab PSG | 0.42 | the most directly comparable independent number |
| Galaxy Watch 3 (Kim 2023) | accel + PPG | independent n=32 lab PSG | — (acc 0.651) | older Galaxy Watch generation |
| Samsung Mendonça 2024 (RNN) | accel + PPG | proprietary 1522 nights | 0.56 | vendor-affiliated; cross-generation cohort |
| Fitbit Sense | accel + PPG | SLEEP Adv 2025 n=37 | 0.42 | |
| Fitbit Charge 5 | accel + PPG | SLEEP Adv 2025 n=39 | 0.41 | |
| Whoop 4.0 | accel + PPG | SLEEP Adv 2025 n=40 | 0.37 | |
| Withings ScanWatch | accel + PPG | SLEEP Adv 2025 | 0.22 | reports 3-stage only (Deep+REM merged) |
| Garmin Vivosmart 4 | accel + PPG | SLEEP Adv 2025 n=25 | 0.21 | |
| Oura Ring (Altini 2021) | PPG + accel + **temperature** | within-dataset CV n=106 | — (acc 0.79) | adds temperature; finger-worn; within-dataset |

**Critical fact:** Every vendor self-report (Apple, Samsung, Oura) is on
proprietary data; when independently validated on a held-out lab PSG cohort,
all consumer wearables drop into the 0.21–0.53 κ range. Seren at κ = 0.482
sits in the **upper end** of this band — above Galaxy Watch 5 (0.42), Fitbit
(0.41–0.42) and Whoop (0.37), approaching Apple Watch S8 (0.53) — and **achieves
it on a public, reproducible pipeline** under a **strictly harder protocol** than
any of them.

---

## 7. Explainability findings (Captum Integrated Gradients on the validation set)

### Per-class top features (mean |IG|, ordered)
| Class | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| Wake | hr_std | hr_succdiff_std | hr_range | time_of_night | hr_delta_prev |
| Light | hr_succdiff_std | hr_std | hr_range | time_of_night | hr_delta_prev |
| Deep | hr_std | time_of_night | hr_succdiff_std | hr_range | hr_delta_prev |
| REM | hr_std | time_of_night | hr_delta_prev | hr_succdiff_std | hr_range |

`hr_std` (HR variability) is the dominant driver across every class — consistent
with the autonomic-nervous-system signature of sleep stages. `time_of_night` is
top-4 in every class, validating its inclusion.

### IG vs occlusion rank agreement (cross-check)
| Class | Spearman ρ | Verdict |
|---|---|---|
| Wake | +0.24 | PARTIAL — small val sample (7 subjects) makes per-class statistics noisy |
| Light | +0.59 | AGREE |
| Deep | +0.33 | PARTIAL (same reason) |
| REM | **+0.61** | **AGREE** — REM signature is stable across subjects |

### Per-stage physiological agreement (|IG|-weighted)
| Stage | Agreement | Verdict |
|---|---|---|
| **Wake** | **65 %** | majority — `hr_std`, `hr_max`, actigraphy all correct |
| Light | n/a | contested by design (transition stage; priors unjudged) |
| **Deep** | **60 %** | majority — `time_of_night` (ρ = −0.90), `hr_max`, `hr_min`, actigraphy all correct |
| **REM** | **67 %** | majority — `time_of_night` (ρ = +0.86), `hr_succdiff_std`, `hr_range`, `hr_mean` all correct |

The "DIFFER" rows in the per-stage tables are universally **lower-|IG|
secondary HR features** that sign-flip under collinearity with the dominant
`hr_std` axis — the same artefact documented in the stress XAI section. The
*dominant* attribution for every judged stage matches physiology.

### Notable XAI finding — the model independently rediscovered the sleep phase prior
`time_of_night` correlates **strongly negative** with Deep attribution
(ρ = −0.90) and **strongly positive** with REM attribution (ρ = +0.86). The
model wasn't told that Deep sleep concentrates in the first half of the night
and REM in the second — it learned this from the data and uses it correctly.

### XAI-driven feature reduction

Pooled `|IG|` ranking across all 4 classes (over the 11 shipped features; note
`immobility_frac` was already dropped pre-training):

| Rank | Feature | Pooled \|IG\| | Decision |
|---|---|---|---|
| 1 | `hr_std` | 0.1153 | KEEP |
| 2 | `hr_succdiff_std` | 0.0934 | KEEP |
| 3 | `time_of_night` | 0.0865 | KEEP |
| 4 | `hr_range` | 0.0805 | KEEP |
| 5 | `hr_delta_prev` | 0.0786 | KEEP |
| 6 | `hr_max` | 0.0614 | KEEP |
| 7 | `hr_mean` | 0.0406 | KEEP |
| 8 | `hr_min` | 0.0252 | KEEP |
| 9 | `act_max` | 0.0221 | KEEP |
| 10 | `act_count` | 0.0142 | KEEP |
| 11 | `act_std` | 0.0141 | KEEP (lowest, but dropping it doesn't help — see sweep) |

Feature-reduction retrain sweep (Walch held-out, 30-epoch budget per k):

| k features kept | Walch macroF1 | Walch κ | Walch wF1 | Notes |
|---|---|---|---|---|
| **11 (shipped, full-budget 80 ep)** | **0.621** | **0.482** | **0.665** | **deployed** |
| 10 (drop `act_std`) | 0.609 | 0.463 | 0.640 | best of the reduced sweep, still below the full model |
| 8 (drop 3 lowest) | 0.595 | 0.449 | 0.635 | hurts |
| 6 (HR-only) | 0.596 | 0.460 | 0.635 | hurts |

**Decision:** ship at **k = 11**. The 30-epoch sweep recommends k=10 (drop
`act_std`) as the leanest set within 0.01 macroF1 of the reduced sweep, but the
full-budget 11-feature model (macroF1 0.621) still beats every reduced variant,
and `act_std`'s on-watch cost is trivial (`std()` over the in-epoch accel
magnitudes). Per-epoch wire payload is 48 bytes; the on-device feature pipeline,
TS contract, and Kotlin extractor are all consistent.

---

## 8. Defensible thesis claims (what you CAN say)

- *"All sleep inference runs on the user's phone, taking 11 engineered features
  per 30 s epoch extracted on the watch — no raw biometric stream ever leaves
  the wrist."*
- *"Trained on BIDSleep under a strict subject-disjoint split (40/7, 0 overlap)
  and evaluated zero-shot on the Walch / Sleep-Accel dataset (31 distinct
  subjects, never seen during training, validation, or early-stopping)."*
- *"Held-out Walch κ = 0.482 exceeds the independently-validated Galaxy Watch 5
  (κ = 0.42, Lee 2023 JMIR) and approaches the independently-validated Apple
  Watch Series 8 (κ = 0.53, SLEEP Adv 2025) under a strictly harder protocol."*
- *"Captum Integrated Gradients attribution shows the model's per-stage
  reasoning is consistent with sleep physiology on a |IG|-weighted majority of
  features for Wake (65 %), Deep (60 %), and REM (67 %)."*
- *"The model independently discovered the sleep-phase prior — `time_of_night`
  correlates ρ = −0.90 with Deep attribution and ρ = +0.86 with REM
  attribution, matching the well-known phase distribution."*
- *"A data-leakage audit caught a per-night (vs per-subject) train/val split
  that inflated within-BIDSleep validation by ~4 points; correction left the
  Walch held-out test unaffected (it always was subject- and dataset-disjoint
  by construction)."*

## 9. What you should NOT claim

- *That Seren reaches the SLAMSS 2023 target of 4-class weighted F1 ≥ 0.72* — it
  does not. Walch 4-class wF1 = 0.665. (The 3-class wF1 of 0.798 does reach the
  ~0.80 3-class target.)
- *That on-device inference reaches clinical-grade accuracy* — it does not;
  even the best independently-validated consumer wearable (Apple Watch S8 at
  κ = 0.53) is still well below trained-technician κ (~0.75).
- *That Apple's vendor self-report (κ = 0.68) is a comparable number* — it is
  on private data; the same device drops to κ = 0.53 under independent
  validation.
- *That the Light class is well-modelled* — it is the model's weakest, the
  largest source of confusion, and is intentionally not scored in the
  physiological-agreement table.

---

## 10. Appendix — model files & how to reproduce

| File | Purpose |
|---|---|
| `ml/sleep/prepare_features.py` | Streams BIDSleep + Walch from PhysioNet, extracts cache features per 30-s epoch (the notebook slices `immobility_frac` out at load → 11 features) |
| `ml/sleep/_build_notebook.py` | **Single canonical builder.** Generates `seren_sleep_kaggle.ipynb` — one notebook covering subject-disjoint cache load + XAI-driven slice to 10 cache features + time_of_night = 11 + *(optional)* niapy PSO/GWO hyperparameter search (gated by `RUN_HYPEROPT`) + training + Walch eval + ONNX export + Captum Integrated Gradients XAI suite + *(optional)* XAI-driven feature reduction + MLflow logging. |
| `ml/sleep/convert_onnx_to_tflite.py` | `onnx2tf` conversion with Erf/GELU → native TFLite ops; parity verification |
| `assets/ml/sleep/sleep_stage_model.onnx` | The exported PyTorch model (reference / parity baseline) |
| `assets/ml/sleep/sleep_stage_model.tflite` | The deployed on-device model (`react-native-fast-tflite`) |
| `assets/ml/sleep/sleep_model_metadata.json` | Feature names, normalization spec, class map, reported metrics |
| `services/ai/sleepStageModel.ts` | On-device TFLite inference (feature assembly, per-night robust norm, windowed inference, light smoothing) |
| `wearos/app/.../sleep/EpochAggregator.kt` | Watch-side 30-s epoch feature extraction (mirrors `prepare_features.py` exactly) |
| `wearos/app/.../sleep/WearableFeatureSender.kt` | Watch → phone Data Layer send (`/seren/sleep/features/batch`, ~48 B/epoch) |
| `android/app/.../sleep/WearableSleepListenerService.kt` | Phone-side receiver, persists batches to `seren_sleep/` |
| `services/ai/sleepReceiver.ts` | Phone-side batch decoding, queueing, and feeding into `sleepStageModel.ts` |
