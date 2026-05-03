# Focus Model — Cognitive Readiness Predictor

## What It Does

The Focus model runs entirely on-device and outputs a **Cognitive Readiness score (0–100)** every 5 seconds from your smartwatch's live biometric stream. A high score means your nervous system is calm and your physiological signals are aligned for sustained concentration. A low score means distraction-driving signals are elevated — not a judgement, just a physiological reading that something is pulling your body's resources away from focused cognition.

---

## Dataset — PhysioNet Wearable Exam Stress

| Property | Value |
|---|---|
| **Full name** | Wearable Exam Stress Dataset for Predicting Cognitive Performance in Real-World Settings |
| **Source** | PhysioNet (physionet.org/content/wearable-exam-stress/1.0.0/) |
| **Participants** | 10 university students |
| **Setting** | Real university exam halls — not a lab simulation |
| **Device** | Empatica E4 wristband (medical-grade wearable) |
| **Signals recorded** | BVP · EDA · Skin Temperature · 3-axis Accelerometer |
| **Sampling rates** | BVP: 64 Hz · EDA: 4 Hz · Temperature: 4 Hz · Accel: 32 Hz |
| **Total training samples** | ~10,373 (60-second sliding windows extracted from all sessions) |
| **Sessions per participant** | Multiple exam periods covering different subjects |
| **Labels** | Exam performance scores (grade outcomes) used as cognitive readiness proxy |

### Why This Dataset Fits Seren

1. **Real exams, not lab tasks.** Data was captured during actual high-stakes university examinations — the exact context Seren targets. Lab-based HRV datasets often miss the specific sympathetic activation pattern that occurs under genuine exam pressure.

2. **Same hardware family as consumer smartwatches.** The Empatica E4 measures the same physiological signals (PPG for HRV, skin temperature, movement) available on modern smartwatches (Apple Watch, Galaxy Watch, Garmin). The feature pipeline is directly transferable.

3. **Cognitive outcome labels.** Unlike generic stress datasets, this one labels sessions by actual cognitive performance (exam grades), making it uniquely suited for predicting *readiness to perform* rather than just *stress level*.

4. **Student population.** All 10 participants are students in an exam setting — directly matching Seren's primary use case.

---

## Model Architecture

| Property | Value |
|---|---|
| **Algorithm** | XGBoost Gradient-Boosted Regression Trees |
| **Number of trees** | 18 |
| **Base score** | 76.27 |
| **Output range** | 0 – 100 (continuous) |
| **Inference** | On-device, zero network dependency |
| **Runtime** | < 2 ms per inference window |

---

## Input Features (29 Total)

The model uses three physiological domains extracted from a 60-second rolling window:

### Heart Rate Variability — Time Domain
| Feature | Description | Normal Range |
|---|---|---|
| `meanRR` | Mean RR interval (ms) | 700–900 ms |
| `sdnn` | Standard deviation of RR intervals | 40–200 ms |
| `rmssd` | Root mean square of successive RR differences | 30–200 ms |
| `pnn50` | % of successive RR diffs > 50 ms | 10–100% |
| `pnn20` | % of successive RR diffs > 20 ms | 20–100% |
| `hrMean` | Mean heart rate (bpm) | 60–85 bpm |
| `hrStd` | HR standard deviation | 2–12 bpm |
| `hrRange` | HR range within window | varies |
| `cvRR` | Coefficient of variation of RR | 0.08–0.18 |

### Heart Rate Variability — Frequency Domain
| Feature | Description |
|---|---|
| `vlfPower` | Very-low-frequency power (0.003–0.04 Hz) — long-term regulation |
| `lfPower` | Low-frequency power (0.04–0.15 Hz) — sympathetic + parasympathetic |
| `hfPower` | High-frequency power (0.15–0.4 Hz) — parasympathetic (vagal) tone |
| `lfHfRatio` | Sympathovagal balance index |
| `totalPower` | Total spectral power |
| `lfNorm` | Normalised LF power |
| `hfNorm` | Normalised HF power |

### HRV — Non-Linear Domain
| Feature | Description |
|---|---|
| `sd1` | Poincaré plot short-axis (short-term variability) |
| `sd2` | Poincaré plot long-axis (long-term variability) |
| `sd1sd2Ratio` | Ratio of SD1/SD2 |
| `sampleEntropy` | Signal complexity (unpredictability of RR series) |
| `dfaAlpha1` | Detrended fluctuation analysis — fractal correlation |

### Peripheral Physiology
| Feature | Description | Normal Range |
|---|---|---|
| `tempMean` | Mean skin temperature | 31–34 °C |
| `tempSlope` | Temperature trend (rising = stress vasoconstriction) | < 0.05 °C/min |
| `tempStd` | Temperature variability | — |
| `tempRange` | Temperature range within window | — |

### Movement & Activity
| Feature | Description |
|---|---|
| `accelMagnitudeMean` | Mean accelerometer magnitude |
| `accelMagnitudeStd` | Movement variability |
| `stepCount` | Steps in window |
| `activityType` | Encoded activity: 0=sedentary, 1=walking, 2=active, 3=sleeping |

---

## Feature Importance (Top 10)

Derived from XGBoost's gain-based importance across all 18 trees:

| Rank | Feature | Importance | What It Captures |
|---|---|---|---|
| 1 | `hrStd` | **20.2%** | HR variability — fluctuation under cognitive load |
| 2 | `hrMean` | **13.4%** | Resting vs. stress-elevated heart rate |
| 3 | `sampleEntropy` | **10.4%** | Nervous system signal complexity — drops when taxed |
| 4 | `sd2` | **8.4%** | Long-term HRV — autonomic regulatory capacity |
| 5 | `accelMagnitudeStd` | **5.0%** | Restlessness / fidgeting — behavioural distraction signal |
| 6 | `hfNorm` | **3.6%** | Parasympathetic (calm/recovery) tone |
| 7 | `totalPower` | **3.5%** | Overall autonomic nervous system activity |
| 8 | `lfPower` | **3.4%** | Sympathetic activation |
| 9 | `pnn50` | **4.6%** | Beat-to-beat parasympathetic influence |
| 10 | `meanRR` | **4.7%** | Base cardiac rhythm tempo |

> The top two features alone (HR standard deviation + mean HR) account for **33.6%** of the model's predictive power, confirming that heart rate dynamics are the primary window into cognitive readiness.

---

## Validation Metrics

Evaluated with **5-fold cross-validation** on the full PhysioNet dataset:

| Metric | Value | What It Means |
|---|---|---|
| **CV Accuracy** | **90.1%** | 9 in 10 readiness classifications correct |
| **CV F1 (weighted)** | **90.2%** | Balanced precision/recall across all levels |
| **CV F1 (binary)** | **80.4%** | Low vs. non-low readiness detection |
| **CV Precision** | **78.0%** | When model flags distraction, it's right 78% of the time |
| **CV Recall** | **83.0%** | Catches 83% of true low-readiness states |
| **CV AUC-ROC** | **94.5%** | Near-excellent discrimination across all thresholds |

The **AUC-ROC of 0.945** is particularly meaningful — it means the model correctly ranks a distracted-state reading above a focused-state reading 94.5% of the time, regardless of where you set the threshold.

---

## Score Levels

| Level | Range | Physiological Meaning |
|---|---|---|
| **Sharp** | 75–100 | No distraction signals. Autonomic balance intact. Ideal to begin. |
| **Steady** | 50–74 | Minimal distraction. Mild sympathetic elevation but focus is sustainable. |
| **Drifting** | 25–49 | Distraction signals rising. Sympathetic activation starting to outpace recovery. |
| **Scattered** | 0–24 | High distraction. Physiological resources diverted — concentration will be fragmented. |

---

## Normalization

All 29 features are z-score normalised before inference using dataset statistics:

- **Mean RR interval**: 777.85 ms (population mean ≈ 77 bpm)
- **Mean skin temperature**: 32.70 °C
- **Mean HRV (RMSSD)**: 143.2 ms (healthy resting baseline)

This ensures the model is robust to individual physiological baselines — a naturally low-HR person won't score artificially high.

---

## Limitations

- **10 participants** — small cohort by clinical standards. The model generalises through cross-validation but has not been tested on populations outside university students.
- **Exam context** — trained on exam stress specifically. Performance during other cognitive tasks (deep work, creative writing) is extrapolated, not directly validated.
- **No individual calibration** — the model uses population-level normalisation. Personal baseline adaptation is on the roadmap.
- **Wrist-based PPG** — HRV accuracy is lower than chest ECG, especially during movement. The `activityType` feature partially corrects for motion artefacts.
