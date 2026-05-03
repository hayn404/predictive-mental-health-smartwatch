# Seren — What's New Before Push

> Summary of all changes made on top of the last GitHub commit (`c103647`).

---

## Focus Model — New Feature (Shipped)

The largest addition in this build. A full on-device cognitive readiness predictor, AI-generated insight, and a dedicated screen — all new.

### New Files
| File | What It Does |
|---|---|
| `app/(tabs)/focus.tsx` | New Focus tab screen with cognitive readiness card, live signals bar, and Groq insight |
| `services/ai/focusModel.ts` | On-device XGBoost inference engine — 29 features → 0-100 readiness score |
| `services/ai/focusRecommendations.ts` | Groq (llama-3.1-8b-instant) generates one science-based, calming tip from live biometrics |
| `components/ui/FocusGauge.tsx` | Animated SVG ring gauge with score and level colour |
| `assets/ml/focus_model.json` | Trained XGBoost model (18 trees, PhysioNet Exam Stress dataset) |
| `ml/docs/focus_model.md` | Full model documentation (dataset, features, metrics, importances) |

### What Changed in Existing Files
| File | Change |
|---|---|
| `app/(tabs)/_layout.tsx` | Added Focus tab with `psychology` icon |
| `services/ai/types.ts` | Added `FocusLevel`, `ElevatedFeature`, `FocusPrediction` types |
| `hooks/useWellness.tsx` | Loads focus model on startup; runs inference every 5 s; triggers Groq only when score changes ≥ 7 points |

### How It Works
1. Smartwatch streams 29 biometric features (HRV, heart rate, skin temperature, accelerometer) in a 60-second window
2. XGBoost model runs on-device in < 2 ms and outputs a score 0–100
3. Score is mapped to four levels: **Sharp** (75–100) · **Steady** (50–74) · **Drifting** (25–49) · **Scattered** (0–24)
4. If the score shifts by ≥ 7 points, Groq generates a single gentle, science-grounded observation (e.g. *"Your heart rate is slightly elevated — two slow exhales can bring it back toward baseline"*)
5. The previous Groq tip persists on screen until a new one is generated — no blank states

### UI Design Decisions
- Label is **COGNITIVE READINESS** (not "distraction" — framing is indirect to avoid exam anxiety)
- AI tip sits directly under the gauge with a `✨` icon — no separate card/box
- **LIVE SIGNALS** features bar pinned at the bottom: Heart Rate · HRV · Temp · Balance
- Signal colours: amber if the reading is outside normal baseline, green if within range
- Gauge size is computed from screen dimensions via `onLayout` so it fills available space on every device

---

## Anxiety Model — Under R&D

> **Status: Research & Development — not yet surfaced in the app UI**

The anxiety model has been built and integrated into the inference pipeline but is intentionally not shown to users in this release. Anxiety scoring is a clinically sensitive signal and the model requires more validation before being displayed.

### What Exists
| File | Status |
|---|---|
| `services/ai/anxietyModel.ts` | Built — separate XGBoost model, independent of stress |
| `components/ui/AnxietyGauge.tsx` | Built — not rendered anywhere in the UI |
| `assets/ml/anxiety_model.json` | Trained model file present |
| `ml/models/anxiety_model.json` | Same model, ml directory copy |
| `ml/models/ANXIETY_MODEL_TRAINING_LOG.md` | Training log |

### What Changed
- `predictAnxiety()` was previously a quick approximation inside `stressModel.ts` (stress score + HRV deviation + sleep penalty + LF/HF ratio). That function has been **removed** from `stressModel.ts` and replaced by the dedicated `anxietyModel.ts` which runs its own XGBoost trees
- The anxiety prediction is computed in the `useWellness` inference loop but the result is **not exposed in any screen** until the model clears further validation

### Roadmap for R&D Completion
- [ ] Personal baseline calibration (7-day adaptation period)
- [ ] GAD-7 ground truth correlation study
- [ ] Clinical sensitivity review before showing anxiety scores to users
- [ ] UI design that presents anxiety gently (not alarming)

---

## Stress Gauge — Visual Fix

| File | Change |
|---|---|
| `services/mockData.ts` | `getStressLabel()` colours changed from purple/orange/red to a **unified green palette** across all four levels: `#35e27e` (low) → `#2DBD6A` (moderate) → `#1F9952` (elevated) → `#157A3E` (high) |

The previous palette (purple moderate, orange elevated, red high) was visually inconsistent with the rest of the app's green health-indicator language. The new palette keeps the semantic meaning (darker = more elevated) while staying on-brand and less alarming.

---

## Other Changes

### Background Services
- `services/background/` — new background task infrastructure added (details in directory)

### Components
- `components/ui/PanicInterventionBanner.tsx` — new component for panic/crisis intervention prompts (not yet wired to a trigger)

### Auth & Notifications
- `hooks/useAuth.ts` — session handling improvements
- `services/ai/notifications.ts` / `utils/notifications.ts` — notification scheduling updates
- `services/ai/speechRecognition.ts` — minor additions

### Type System
- `services/ai/types.ts` — added Focus types (see above); minor additions to existing interfaces

### Tests
- `__tests__/stressModel.test.ts` — updated to match stress model refactor
- `__tests__/integration.test.ts` — updated integration assertions
- `__tests__/proof_different_readings.test.ts` — new test proving stress and focus models produce distinct readings from the same biometric input

### Dependencies
- `package.json` / `package-lock.json` — 2 new packages added

---

## File Count Summary

| Category | New Files | Modified Files |
|---|---|---|
| Focus feature | 6 | 3 |
| Anxiety model (R&D) | 5 | 1 |
| Stress gauge fix | 0 | 1 |
| UI components | 1 | 1 |
| Services & hooks | 1 | 7 |
| Tests | 1 | 2 |
| Docs | 2 | 0 |
| Other | 3 | 4 |
| **Total** | **19** | **19** |
