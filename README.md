# Seren — Real-Time Multi-Model Machine Learning for Mental-Health Monitoring

**Seren** turns the physiological signals a smartwatch already captures into actionable
mental-health insights, using a suite of **on-device machine-learning models**. A Wear OS
companion streams heart rate, HRV, motion, ambient light and location to the phone, where
six models run locally — stress, sleep staging, cognitive focus, biological "heart" age,
depression screening, and voice-emotion — plus PHQ-9 / GAD-7 self-report tools.

> Graduation project • Field: **Machine Learning & AI**. A wellness/research system — **not a
> medical device**.

---

## System at a glance

```
  Wear OS watch (Kotlin/Compose)                Phone (React Native / Expo)
  ──────────────────────────────                ───────────────────────────
  Health Services: HR, HRV, accel       ── Wearable Data Layer ──▶  receivers persist batches
  Sensors: ambient light, GPS                                       │
  EpochAggregator → feature batches                                 ▼
  (sleep 48 B/epoch; env light+GPS)                          on-device models
                                                              ├─ stress   (XGBoost / HRV)
                                                              ├─ sleep    (TCN+BiGRU / TFLite)
                                                              ├─ focus    (XGBoost / HRV)
                                                              ├─ bio-age  (XGBoost / HRV)
                                                              ├─ depression (XGBoost / actigraphy)
                                                              └─ voice    (acoustic → arousal/valence)
                                                              + PHQ-9 / GAD-7 screeners
                                              All inference offline; data stays on the phone.
```

## The models

| Model | Task | Data (train → test) | Headline | Runtime |
|---|---|---|---|---|
| **Stress** | binary stress | SIPD + PhysioStress → WESAD (cross-dataset) | AUC **0.86** | XGBoost (TS tree) |
| **Sleep** | 4-class staging (Wake/Light/Deep/REM) | BIDSleep → Walch (zero-shot) | κ **0.48** · wF1 0.665 · Deep recall 0.78 · 3-class wF1 0.80 | TCN+BiGRU (TFLite) |
| **Focus** | cognitive engagement | CogWear (LOSO) → MAUS (external) | AUC **0.82** (0.95 sustained) | XGBoost (TS tree) |
| **Bio-age** | physiological age + age-gap | Autonomic Aging → Fantasia | MAE **7.6 yr** · young-vs-old AUC 0.94 | XGBoost (TS tree) |
| **Depression** | screening | Depresjon (actigraphy) | acc **0.92** · ROC-AUC 0.97 | XGBoost (TS tree) |
| **Voice** | acoustic emotion | RAVDESS-style check-ins | pitch/energy/rate → arousal + valence | on-phone DSP |

Engineering themes: **subject-independent + cross-dataset validation**, **per-user
normalization** (bridges research-grade vs consumer sensors), and **explainability**
(SHAP for the trees, Captum Integrated Gradients for the sleep net). The neural sleep model
runs via `react-native-fast-tflite`; the gradient-boosted models run through a custom
TypeScript tree-traversal engine. See `ml/docs/` for per-model cards and results.

## Repository layout

```
app/                 Expo Router screens (onboarding, tabs, auth, screening PHQ-9/GAD-7)
hooks/useWellness    Orchestrator: loads models, ingests watch data, exposes app state
services/ai/         On-device inference, feature engineering, receivers, SQLite, LLM/voice
assets/ml/<model>/   Deployed model artifacts (tflite + JSON trees + metadata)
wearos/              Wear OS companion app (Kotlin, Compose, Health Services, Data Layer)
android/             Native Android project (Expo prebuild; gitignored)
ml/<model>/          Training pipelines, notebooks, model cards (sleep, stress, focus, bioage, …)
ml/ci/, dvc.yaml     CI model-gate + DVC pipeline (see docs/MLOPS.md)
docs/                MLOPS, submission, install/testing notes
```

## Getting started (development)

```bash
npm install --legacy-peer-deps        # legacy-peer-deps is required for this project
npx expo start --clear                # first run (clears Metro cache); then `npx expo start`
```
Create a `.env` with `EXPO_PUBLIC_SUPABASE_URL`, `EXPO_PUBLIC_SUPABASE_ANON_KEY`, and any
`EXPO_PUBLIC_*` LLM keys (see `.env.example`). The on-device ML (TFLite) needs a **dev/release
build**, not Expo Go — Expo Go falls back to mock data for the sleep model.

## Build for a real device

Both apps are signed with the **same key** so the watch↔phone Data Layer pairs.

```bash
# Phone (arm64 — the -P flag avoids a Windows 260-char path failure)
cd android && ./gradlew assembleRelease -PreactNativeArchitectures=arm64-v8a
#  → android/app/build/outputs/apk/release/app-release.apk

# Watch
cd wearos && ./gradlew assembleRelease
#  → wearos/app/build/outputs/apk/release/app-release.apk
```

**Install + pair:** sideload the phone APK (allow "install unknown apps"); sideload the watch
APK over `adb` (enable Wireless debugging → `adb pair` / `adb connect` / `adb install`). The
phone + watch must be paired in **Wear OS** (or **Galaxy Wearable**). Grant all permissions on
first launch, and un-pause Supabase (free tier auto-pauses).

## MLOps / CI-CD

Training runs on a **self-hosted GitHub Actions runner**, data + model artifacts are versioned
with **DVC on DagsHub** (no Kaggle), metrics log to **DagsHub MLflow**, and a **champion/
challenger gate** blocks any PR whose model regresses on a pinned held-out set. On merge,
**EAS Update** ships the new model over-the-air. Sleep and stress are wired; see `docs/MLOPS.md`
and `dvc.yaml`.

## Datasets

WESAD · Stress-Predict (SIPD) · PhysioStress · BIDSleep · Walch/Sleep-Accel · CogWear · MAUS ·
PhysioNet Autonomic Aging · Fantasia · Depresjon · RAVDESS. (Citations in `ml/docs/`.)

## Tech stack

| | |
|---|---|
| Mobile | React Native 0.81 · Expo SDK 54 (New Architecture) · TypeScript · Expo Router v6 |
| On-device ML | `react-native-fast-tflite` (sleep net) · custom TS tree engine (XGBoost models) |
| Storage / auth | expo-sqlite (local) · Supabase |
| Wearable | Wear OS · Kotlin · Jetpack Compose · Health Services · Wearable Data Layer |
| ML training | XGBoost · PyTorch · TensorFlow/Keras · scikit-learn · onnx2tf · SHAP · Captum · PSO |
| Voice / LLM | Whisper (STT) · OpenRouter (LLM) · acoustic feature extraction |
| MLOps | DVC · DagsHub (DVC remote + MLflow) · GitHub Actions · EAS |

## App pages

Onboarding **Welcome → Profile (age) → Device Sync → Privacy & Permissions**, then the tabs:
**Home** (live stress/sleep/anxiety/sunlight/location + hypnogram), **Voice Check-in**,
**Insights** (trends), **Recommendations** (breathing + AI tips), **Settings** (data control,
watch status, PHQ-9 / GAD-7 screeners).

## Honest caveats

- **Wellness/screening, not diagnosis.** Clinical-grade sleep/affect labelling needs PSG/clinician.
- On-device HRV is derived from Health Connect HR (BPM), coarser than the raw PPG used in
  training — live accuracy trails the dataset numbers (per-user normalization mitigates this).
- Continuous GPS + sensor capture on the watch uses more battery than stock.

## Contributing

`feature/*` / `fix/*` branches → PR → squash-merge. Run `npx tsc --noEmit` and `npx jest`
before opening a PR (185 tests). Model PRs must pass the CI gate.

## License

Private. For collaboration inquiries, contact the authors.
