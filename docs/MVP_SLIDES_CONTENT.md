# Seren — Two-Slide MVP Summary

---

## SLIDE 1: MVP Overview

### What Seren Does
Seren is a **privacy-first mental wellness companion** that uses smartwatch biometric data (heart rate, HRV, sleep) and on-device machine learning to **predict stress levels, detect anxiety patterns, and deliver personalized interventions** — all without sending health data to the cloud.

### Key Implemented Features
- **On-device XGBoost stress model** — 90.1% accuracy, trained on WESAD dataset, runs in pure TypeScript
- **29-feature HRV engineering** — time-domain, frequency-domain, and non-linear analysis per 5-min window
- **Sleep stage analysis** — clinical quality scoring with deep/REM/light stage breakdown
- **Voice check-in** — sentiment analysis + LLM-powered emotional understanding
- **AI recommendation engine** — 20+ evidence-based interventions triggered by biometric thresholds
- **Location diversity & sunlight tracking** — detects monotonous routines linked to depression
- **Clinical screening** — PHQ-9 (depression) and GAD-7 (anxiety) validated assessments
- **Wear OS companion app** — 8 screens with stress gauge, breathing exercises, and haptic feedback
- **User authentication** — Supabase email/password + Google OAuth

### System Architecture
```
┌─────────────────────┐     ┌──────────────────────┐
│  Samsung Galaxy      │     │  Wear OS App         │
│  Watch 7 Sensors     │     │  (Kotlin/Compose)    │
│  HR, HRV, Sleep,     │     │  Stress gauge, HR,   │
│  Steps, SpO2, Temp   │     │  Breathing, Insights │
└─────────┬───────────┘     └──────────────────────┘
          │ Health Connect API
┌─────────▼──────────────────────────────────────────┐
│              Seren Mobile App                      │
│         (React Native + Expo SDK 54)               │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Feature  │  │ XGBoost  │  │ Recommendation   │ │
│  │ Engine   │→ │ Stress   │→ │ Engine (20+      │ │
│  │ (29 feat)│  │ Model    │  │ interventions)   │ │
│  └──────────┘  └──────────┘  └──────────────────┘ │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Sleep    │  │ Voice    │  │ Location +       │ │
│  │ Analysis │  │ Check-in │  │ Sunlight Track   │ │
│  └──────────┘  └──────────┘  └──────────────────┘ │
│  ┌──────────────────────────────────────────────┐  │
│  │ SQLite (on-device) │ Supabase (auth only)    │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

---

## SLIDE 2: Learning & Contributions

### Key Technical Challenges
1. **On-device ML inference** — Porting XGBoost from Python to pure TypeScript with no native dependencies; required implementing tree traversal + sigmoid activation from scratch
2. **Real-time biometric pipeline** — Processing 29 HRV features per 5-minute window with FFT-based frequency analysis running on a mobile device at 5-second intervals
3. **Health Connect integration** — Bridging Samsung Watch sensor data through Health Connect API with graceful fallback to mock data for development
4. **Windows path length limits** — Android CMake builds exceeded 260-character limit; required project restructuring
5. **Privacy-first architecture** — All ML inference, sleep scoring, and voice analysis run locally; only auth tokens leave the device

### What the Team Learned
- Building production ML pipelines that run entirely on-device (no cloud inference)
- Designing modular AI services with graceful degradation (real sensors → mock fallback)
- Integrating native Kotlin Wear OS apps alongside a React Native mobile app
- Clinical foundations for mental health interventions (evidence-based citations, PHQ-9/GAD-7 validity)
- End-to-end feature engineering: raw sensor data → engineered features → model prediction → user-facing recommendation

### Team Responsibilities

| Member | Responsibility |
|--------|---------------|
| Haneen Alaa | AI/ML Pipeline — XGBoost stress model, feature engineering, baseline computation, voice analysis |
| Kareem Mohamed | Mobile Frontend — React Native UI, dashboard, insights charts, onboarding flow |
| Youssef Mahmoud | Backend & Integration — Supabase auth, Health Connect bridge, SQLite persistence, data export |
| Mariem Zakary | Wear OS App & Sensors — Kotlin/Compose watch app, Health Services API, sunlight/location tracking |

---