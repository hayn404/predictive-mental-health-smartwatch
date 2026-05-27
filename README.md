# Seren — Predictive Mental Health Monitoring

> AI-powered mental well-being companion that continuously monitors stress, sleep, and HRV via smartwatch biometrics and voice check-ins — all processed privately on your device.

**Team #14 · Zewail City of Science, Technology and Innovation**
School of Computational Sciences and Artificial Intelligence · Spring 2025–2026
Supervisor: Prof. Khaled Mostafa El Sayed

---

## What is Seren?

Seren is a React Native application paired with a Samsung Galaxy Watch that passively monitors mental health indicators throughout the day. It combines physiological signals (heart rate variability, sleep stages, skin temperature, SpO₂) with optional voice mood check-ins to generate personalized, evidence-based well-being recommendations — entirely on-device, with no cloud dependency by default.

The core insight driving the project: stress and mood patterns are detectable from wrist-worn biometrics days before a person consciously notices them. Seren makes that signal visible and actionable.

---

## Key Features

| Feature | Description |
|---|---|
| **Real-time stress scoring** | XGBoost model (200 trees, 29 HRV features) runs inference in <5 ms on-device |
| **Sleep quality analysis** | AASM-aligned clinical scoring: efficiency, fragmentation, circadian stability |
| **Voice check-ins** | Whisper STT transcription + LLM emotional analysis; VADER lexicon fallback when offline |
| **Personal baseline** | 14-day rolling mean/std; z-score anomaly detection for HR, HRV, temperature |
| **Recommendations engine** | 12 evidence-based interventions with clinical citations; de-duplicated, priority-ranked |
| **Privacy-first architecture** | All data stays on device; optional encrypted cloud backup only if user enables it |
| **Depression risk module** | Internal-only risk score (C09) adjusts recommendations — never surfaced to the user |

---

## Architecture Overview

```
Samsung Galaxy Watch (Wear OS)
         │  Samsung Health syncs via BLE
         ▼
Google Health Connect (Android)
         │  react-native-health-connect reads every 5 min
         ▼
┌─────────────────────────────────────────────┐
│              SEREN PHONE APP                │
│                                             │
│  Data Ingestion → SQLite (expo-sqlite)      │
│       │                                     │
│  Feature Engineering (29 HRV features)      │
│       │                                     │
│  Personal Baseline (14-day rolling)         │
│       │                                     │
│  XGBoost Inference (pure TypeScript)        │
│       │                                     │
│  Sleep Analysis + Depression Risk (C09)     │
│       │                                     │
│  Recommendations Engine → UI                │
└─────────────────────────────────────────────┘
```

Full system architecture: [Eraser.io diagram](https://app.eraser.io/workspace/zpWl6f2em2QvhOHgq9oL?origin=share&elements=3PUK5tpXhYBNuh0ldqtqOw)

Full data flow diagram: [Eraser.io diagram](https://app.eraser.io/workspace/yBsvcfKaM2yqlkDhRXPm?origin=share&elements=iQqV08de3zafDIqV-BgFeQ)

UI mockups: [Figma](https://www.figma.com/design/1HdiLHIqX9jHMNDkim14Q2/Seren---Mental-Health-Monitoring?node-id=0-1&t=KY1mjsC0NKa8EVzx-1)

---

## Technology Stack

| Layer | Technology |
|---|---|
| App Framework | React Native + Expo SDK 54 |
| Language | TypeScript 5.9 (strict) |
| Navigation | Expo Router v6 (file-based) |
| ML Training | Python + XGBoost + scikit-learn (LOSO-CV on WESAD + SWELL-KW) |
| ML Inference | Pure TypeScript tree traversal — no native dependencies |
| STT / NLP | Groq Whisper large-v3-turbo + Llama 3.1 8B; VADER offline fallback |
| Database | expo-sqlite (on-device, encrypted) |
| Health Data | react-native-health-connect (Health Connect API) |
| Testing | Jest + ts-jest (TypeScript), pytest (Python ML pipeline) |
| Build | EAS (Expo Application Services) |

---

## Component Map (16 components)

| ID | Component | Layer | Status |
|---|---|---|---|
| C01 | Sensor Data Collection | Data Acquisition | ✅ Done |
| C02 | Data Sync & Preprocessing | Mobile Processing | ✅ Done |
| C03 | Local Data Storage | On-Device Storage | ✅ Done |
| C04 | User Profile & Privacy Management | On-Device Security | ✅ Done |
| C05 | Physiological Feature Extraction | On-Device ML | ✅ Done |
| C06 | Behavioral Feature Extraction | On-Device ML | ✅ Done |
| C07 | Mental State Prediction Engine | On-Device AI | ✅ Done |
| C08 | Sleep & Circadian Analysis Engine | On-Device AI | ✅ Done |
| C09 | Depression Risk Analyzer (internal-only) | On-Device AI | ✅ Done |
| C10 | Recommendations Engine | User Feedback | ✅ Done |
| C11 | Voice Check-in & STT Processor | User Interaction | ✅ Done |
| C12 | Conversational Companion | User Interaction | ✅ Done |
| C13 | Notification & Scheduling Manager | User Interaction | ✅ Done |
| C14 | Subscription & Access Control | Security / Access | 🔄 Partial |
| C15 | Cloud Sync & Backup (Optional) | Cloud (Opt-In) | 🔄 Partial |
| C16 | Model Training & Update Pipeline | Cloud / ML Dev | ✅ Done |

---

## ML Model: Stress Prediction

- **Algorithm:** XGBoost — gradient boosted decision trees
- **Training data:** WESAD dataset (15 subjects, Trier Social Stress Test) + SWELL-KW (office knowledge-work conditions) + synthetic augmentation
- **Validation:** Leave-One-Subject-Out Cross-Validation (LOSO-CV) — zero data leakage between subjects
- **CV Accuracy:** >70% on held-out subjects
- **Input:** 29 biometric features per 5-minute window
- **Output:** Stress score 0–100 → Low / Moderate / Elevated / High
- **Inference:** Pure TypeScript tree traversal, <5 ms, no native modules

**Top 5 feature importances:**
1. RMSSD — primary parasympathetic HRV indicator
2. LF/HF Ratio — sympathovagal balance; rises under mental stress
3. Heart Rate Mean — sympathetic nervous system proxy
4. SDNN — overall HRV; declines under sustained stress
5. SD1 (Poincaré) — short-term beat-to-beat variability

---

## Test Coverage

164 tests total — all passing.

| Module | Line Coverage |
|---|---|
| featureEngineering.ts | 99.5% |
| voiceAnalysis.ts | 96.8% |
| sleepAnalysis.ts | 96.0% |
| baseline.ts | 95.0% |
| recommendations.ts | 94.8% |
| stressModel.ts | 75.5% |
| **Overall** | **74% statements · 60% branches · 82% functions** |

```bash
# Run all TypeScript tests
npm test

# Run with coverage report
npm run test:coverage

# Run Python ML pipeline tests
cd ml && pytest tests/ -v
```

---

## Getting Started

### Requirements

- Node.js 20+
- Android phone with Google Health Connect installed
- Samsung Galaxy Watch 4 or later (Wear OS 3+)
- For real data: EAS dev build (see below)

### Run in Expo Go (mock data mode)

```bash
npm install --legacy-peer-deps
npx expo start --clear
```

Scan the QR code with Expo Go. The app runs fully with realistic simulated biometric data.

### Build for real device (Health Connect enabled)

```bash
npm install --legacy-peer-deps
npx eas build --platform android --profile development
```

Install the resulting `.apk` on your Android device. Grant Health Connect permissions for Heart Rate, HRV, Sleep, Steps, and Temperature.

### Run ML training pipeline

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Place WESAD dataset in ml/data/wesad/ and SWELL-KW in ml/data/swell/
python src/train.py
```

---

## Repository Structure

```
seren-app/
├── app/                    # 10 screens (Expo Router)
│   ├── (tabs)/
│   │   ├── index.tsx       # Home dashboard
│   │   ├── insights.tsx    # Trends & HRV charts
│   │   ├── checkin.tsx     # Voice check-in
│   │   ├── recommendations.tsx
│   │   └── settings.tsx    # Privacy controls
│   └── onboarding.tsx
├── services/ai/            # 17 AI modules (5,250 lines)
│   ├── featureEngineering.ts   # 29 HRV features
│   ├── stressModel.ts          # XGBoost inference
│   ├── sleepAnalysis.ts        # Sleep scoring
│   ├── voiceAnalysis.ts        # VADER sentiment
│   ├── recommendations.ts      # Intervention engine
│   ├── depressionRisk.ts       # C09 internal risk module
│   ├── llmService.ts           # Groq / LLM integration
│   ├── whisperService.ts       # Whisper STT
│   ├── baseline.ts             # Personal baseline
│   ├── healthConnect.ts        # Health Connect bridge
│   └── db.ts                   # SQLite layer
├── ml/                     # Python training pipeline
│   ├── src/
│   │   ├── train.py            # XGBoost LOSO-CV training
│   │   ├── features.py         # 29 HRV features (Python)
│   │   ├── preprocessing.py    # WESAD + SWELL-KW loading
│   │   └── synthetic_data.py   # Augmentation generator
│   └── tests/
├── assets/ml/
│   ├── stress_model.json       # 200-tree XGBoost model
│   └── model_metadata.json     # Normalization parameters
└── __tests__/              # 8 Jest suites (2,138 lines)
```

---

## Privacy & Security

- **On-device by default:** All biometric data, predictions, and voice transcripts are stored in device-local SQLite only. Nothing leaves the device unless the user explicitly enables cloud backup.
- **PII stripping:** Voice transcripts are stripped of names, emails, phone numbers, and SSNs before any external API call.
- **Encryption:** Database encrypted via SQLCipher; profile data via EncryptedSharedPreferences + Android Keystore.
- **No ads, no tracking, no data selling.**
- **Depression risk score (C09):** Computed internally only. Never shown to the user. Used solely to calibrate recommendation intensity.

---

## Roadmap

| Milestone | Description |
|---|---|
| **v1.0 — Current** | Full phone app with mock + Health Connect data, trained XGBoost model, 164 tests passing |
| **v1.1** | EAS production build, Play Store submission |
| **v1.2** | Wear OS companion app — Tiles UI, complications, haptic stress alerts |
| **v1.3** | Model retraining on real user cohort data; AUC target >0.80 |
| **v2.0** | iOS HealthKit adapter; subscription tier via Google Play Billing |

---

## Team

| Name | ID | Program | Primary Contribution |
|---|---|---|---|
| Haneen Alaa | 202201463 | DSAI | AI pipeline architecture, HRV feature engineering, Python training pipeline |
| Kareem Mohamed | 202200402 | DSAI | XGBoost inference engine, model export/import, build configuration |
| Mariam Zakary | 202202092 | DSAI | Sleep analysis, baseline & anomaly detection, data pipeline evaluation |
| Youssef Mahmoud | 202202048 | DSAI | Voice check-in pipeline, recommendation engine, integration tests |

---

## License

Private project — Zewail City of Science, Technology and Innovation, 2025–2026.
For collaboration inquiries contact the team via GitHub.
