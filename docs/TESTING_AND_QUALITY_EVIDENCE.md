# Seren - Testing and Quality Evidence

**Project:** Predictive Mental Health Smartwatch Companion  
**Date verified:** May 3, 2026  
**Stack:** React Native / Expo SDK 54, TypeScript, Jest, Python ML pipeline, Wear OS/Kotlin, XGBoost JSON inference  
**Team:** Kareem, Mariam, Haneen, Youssef  

---

## 4. Testing and Quality Evidence

### Project Progress Snapshot

This testing plan uses `PROGRESS_REPORT.md` as the implementation source of truth.

| Status | Count | Components |
|---|---:|---|
| Fully done | 6 | C03 Local Storage, C05 Physiological Features, C06 Behavioral Features, C07 Mental State Prediction, C08 Sleep/Circadian, C10 Recommendations |
| Partial / mock-only | 7 | C01 Sensors, C02 Sync/Preprocessing, C04 User/Profile Privacy, C11 Voice/STT, C12 Conversational Companion, C13 Notifications, C16 Training/Update Pipeline |
| Not started / stub only | 3 | C09 Depression Risk Analyzer, C14 Subscription/Access Control, C15 Cloud Sync/Backup |

**Progress percentage:** about **59% production-ready** using a simple component-weighted estimate: fully done = 100%, partial = 50%, not started = 0%. Calculation: `(6 full + 7 partial * 0.5) / 16 = 59.4%`. This matches the progress report's broader estimate of **55-60% toward a production-ready MVP**.

**Testing priority based on progress:** preserve evidence for fully done components, add missing automated/device evidence for partial components, and clearly mark not-started components as gaps rather than testing them as if they exist.

### 4.1 Test Coverage

Current automated evidence comes from the Jest suite in `__tests__/`. On May 3, 2026, `npm run test:coverage -- --runInBand` passed with **9 test suites** and **159 tests**.

The Python ML test files exist in `ml/tests/`, but they were **not verified in this environment today** because `python3 -m pytest tests/ -v` failed with `No module named pytest`. Install `ml/requirements.txt` in a Python environment and re-run before final submission.

| Component / Module | Test Type | Coverage % | Notes |
|---|---:|---:|---|
| Overall TypeScript AI services | Unit / integration | 75.58% lines | 159 Jest tests passed. Main covered area is AI/model logic, not UI/device behavior. |
| `featureEngineering.ts` | Unit | 99.47% lines | Strong coverage for RR conversion, artifact filtering, HRV, frequency/nonlinear features, temperature, activity, and 29-feature vector output. |
| `voiceAnalysis.ts` | Unit | 96.80% lines | Covers sentiment, negation, intensifiers, emotion scoring, biometric/text mismatch, insights, and trends. |
| `sleepAnalysis.ts` | Unit | 96.00% lines | Covers duration, efficiency, WASO, stage percentages, recovery, consistency, and trends. |
| `baseline.ts` | Unit | 95.00% lines | Covers minimum samples, baseline calculation, recompute rules, and anomaly flags. |
| `anxietyModel.ts` | R&D / exploratory | 90.19% lines | Anxiety is still treated as R&D and is not assigned as a production testing responsibility. Keep existing tests as research evidence only. |
| `recommendations.ts` | Unit | 76.38% lines | Covers stress, sleep, check-in triggers, dedupe, priority, freshness, and outcomes. Anxiety paths are documented as R&D, not final product evidence. |
| `stressModel.ts` | Unit / model | 75.67% lines | Covers model loading, score range, confidence, contributors, fallback path. Needs more branch coverage. |
| `llmService.ts` | Unit / integration | 7.14% lines | Largest gap. Needs mocked tests for Groq failures, PII stripping, prompt safety, response parsing, timeout, and local encryption helpers. |
| Model JSON files | Validation | Covered by model validation tests | Checks file integrity, required fields, tree count, feature order, normalization, metadata, and split references. |
| Full AI pipeline | Integration | Covered by 5 tests | Tests relaxed/stressed states, poor sleep, distressed check-in, and baseline anomaly path. |
| ML Python pipeline | Unit / training | Files exist, not verified today | `ml/tests/` includes synthetic data, feature, and training tests, but pytest is missing locally. |
| UI screens and visual rendering | Planned | Not measured | Focus, dashboard, auth, onboarding, activity, and screening screens need React Native Testing Library plus screenshots/device evidence. |
| Auth and session lifecycle | Planned | Not measured | Login/signup UI exists, but expired session, logout, storage failure, PIN/biometric, and route protection need tests. |
| Health Connect / real watch data | Manual planned | Not measured | Mock/dev path exists. Real Galaxy Watch + Health Connect testing requires Android dev build and paired device evidence. |
| Wear OS app | Manual planned | Not measured | Kotlin files exist, but build/run evidence on an emulator or watch must be collected. |

**Verified Jest coverage summary:**

| Metric | Result |
|---|---:|
| Test suites | 9 passed / 9 total |
| Tests | 159 passed / 159 total |
| Statements | 73.75% |
| Branches | 58.40% |
| Functions | 80.89% |
| Lines | 75.58% |

### 4.2 Testing Depth

#### Edge Cases Already Tested

| Area | Edge Cases Tested |
|---|---|
| Feature engineering | Fewer than required samples, impossible BPM values, RR outliers beyond 30% from median, constant intervals, low-frequency input, missing temperature/activity data, direct RMSSD from Health Connect, and complete 29-feature vector. |
| Stress model | Model not loaded, JSON loading, score clamping to 0-100, confidence bounds, low versus high RMSSD, contributor shape, and fallback prediction. |
| Anxiety R&D checks | Existing anxiety tests may be kept as research evidence, but they are not part of production testing ownership. |
| Sleep analysis | No HRV data, no sleep baseline, awake stage exclusion, poor versus good sleep, fragmentation, bedtime consistency, and empty trend data. |
| Voice analysis | Positive, neutral, negative, and strongly negative text; negation; intensifiers; anxiety/sadness/fatigue/gratitude keywords; no-keyword fallback; max 3 insights. |
| Recommendations | Low stress should not trigger stress intervention, poor sleep path, distressed check-in path, max 3 recommendations, priority sorting, dedupe, and recent item filtering. |
| Baseline | Insufficient samples, fresh versus stale baseline, normal readings, low HRV anomaly, high HR anomaly, temperature anomaly, and severity labels. |

#### Failure Scenarios Already Tested

| Failure Scenario | Current Validation |
|---|---|
| Invalid or missing model state | `isModelLoaded`, `loadModel`, string loading, and fallback paths are covered. |
| Corrupt or incomplete biometric windows | Feature functions return defaults instead of crashing when sample counts are too low. |
| No baseline available | Baseline returns `null` when not enough samples exist; sleep recovery defaults to neutral. |
| Recent duplicate recommendations | Recently shown interventions are filtered out. |
| Poor sleep input | Poor sleep session triggers sleep-focused recommendation path. |
| Distressed check-in with stressed biometrics | Integration test validates targeted recommendation generation. |
| Model schema mismatch risk | Model validation checks model structure, features, normalization, and tree split references. |

#### Example Inputs and Outputs

| Scenario | Example Input | Expected Output |
|---|---|---|
| Relaxed biometric state | HR near 60 bpm, high RMSSD/HRV, stable temperature, low activity | Low stress score, no elevated stress path, few or no stress recommendations. |
| Stressed biometric state | HR near 95 bpm, low RMSSD, higher LF/HF ratio, unstable HR | Higher stress score and stress-related recommendations such as breathing or grounding. |
| Poor sleep | Short session, fragmented stages, low sleep efficiency | Lower sleep quality score and sleep hygiene recommendation. |
| Positive check-in | "I feel good and calm today" with calm biometrics | Positive sentiment, low concern, supportive response. |
| Distressed check-in | "I am overwhelmed and stressed" with high biometric stress | Distressed/concerned sentiment, stress-aligned insight, targeted recommendation. |
| Normal baseline reading | HR/HRV near computed baseline | No anomaly flags. |
| Low HRV anomaly | HRV more than 2 standard deviations below baseline | Anomaly flag with severity and recommendation context. |

### 4.3 Bugs and Issues

| Bug ID | Description | Severity | Status | Fix / Action |
|---|---|---|---|---|
| B01 | Whisper showed `[not configured]` despite API key being present. | High | Fixed | Updated SDK 54 file-system import path to `expo-file-system/legacy`. |
| B02 | `FileSystemUploadType.MULTIPART` not found. | High | Fixed | Same SDK 54 migration fix as B01. |
| B03 | Groq rejected non-standard `audio/m4a` MIME type. | Medium | Fixed | Changed upload MIME type to `audio/mp4`. |
| B04 | Error message did not separate missing configuration from API failure. | Low | Fixed | Added configuration check before transcription attempt. |
| B05 | Negation only detects immediate preceding negator in local sentiment. | Low | Known limitation | Documented as a local sentiment limitation; improve with wider negation window later. |
| B06 | `llmService.ts` has very low test coverage. | High | Open | Add tests for PII stripping, Groq fallback, timeout, response parsing, prompt safety, and encryption helpers. |
| B07 | Auth/session behavior is not fully tested. | High | Open | Add tests for login validation, logout, expired session, storage failure, PIN/biometric auth, and route guards. |
| B08 | Real Health Connect and Galaxy Watch path is not tested. | High | Open | Requires Android dev build, real paired device or emulator, permission screenshots, and logs. |
| B09 | UI visual regression is not automated. | Medium | Open | Add React Native Testing Library tests and screenshot evidence for small/large screens. |
| B10 | Background inference is not validated on device. | High | Open | Test background-fetch/task behavior after app backgrounding and device lock. |
| B11 | Python ML pytest suite could not run in the current environment. | Medium | Open | Install pytest from `ml/requirements.txt`, run `python3 -m pytest tests/ -v`, and attach logs. |
| B12 | PHQ-9/GAD-7 screens are not proven to route scores into AI risk logic. | High | Open | Add integration tests from screening score to recommendation/risk modules. |

### 4.4 Fix Validation

| Fix | Before Behavior | After Behavior | Validation Method |
|---|---|---|---|
| Whisper SDK 54 file upload fix | STT returned `[not configured]` or crashed on missing upload API. | File upload path uses SDK 54-compatible legacy file-system API. | Re-test voice check-in with configured and missing API key paths. |
| Audio MIME type fix | Groq could reject M4A upload due to non-standard `audio/m4a`. | Upload uses `audio/mp4`, matching the M4A container. | Manual API path validation and reduced upload errors. |
| Configuration error handling | Missing key and API failure looked similar. | Service checks configuration before transcription. | Manual negative test with missing key and configured-key paths. |
| Recommendation deduplication | Same intervention could be repeatedly suggested. | Recent recommendations are filtered and deduped by highest priority. | `recommendations.test.ts` covers dedupe and freshness. |
| Model schema validation | Broken model JSON could fail at runtime. | Model JSON tests verify structure, feature order, normalization, and tree split references. | `modelValidation.test.ts` passed in Jest run. |
| Feature artifact filtering | Impossible HR/RR values could contaminate downstream features. | Bounds and median outlier removal clean the RR interval set. | `featureEngineering.test.ts` artifact filtering tests passed. |

**Re-test commands:**

```bash
npm run test:coverage -- --runInBand
npx jest __tests__/featureEngineering.test.ts
npx jest __tests__/stressModel.test.ts
npx jest __tests__/recommendations.test.ts
npx jest __tests__/integration.test.ts
cd ml && python3 -m pytest tests/ -v
```

### 4.5 What Is Not Tested Yet

These gaps should be stated honestly in the final report.

| Area Not Tested Yet | Why It Matters | Planned Evidence |
|---|---|---|
| Real Galaxy Watch / Health Connect ingestion | Core product depends on real sensor data, not only mock data. | Android dev build, paired Samsung/Galaxy Watch session, screenshots/logs of HR, HRV, temperature, steps, and sleep records. |
| Wear OS app run proof | The watch app exists in code but needs runtime evidence. | Build/run on Wear OS emulator or watch, capture dashboard/breathing/sensor screens. |
| End-to-end UI automation | Screens are user-facing but not represented in Jest coverage. | React Native Testing Library tests for tabs, auth, onboarding, screening, settings, and focus screen. |
| Auth security and session lifecycle | Mental-health and biometric data require protected access. | Tests for PIN/biometric auth, expired token, logout, session timeout, storage failure, and route protection. |
| Real notification behavior | Notification scheduling changes by permission/device/background state. | Device tests for permission denied, quiet hours, high-stress alert, daily reminder, and rate limits. |
| Background inference | Continuous monitoring cannot rely only on foreground execution. | Background-fetch/task tests showing inference behavior after backgrounding and device lock. |
| LLM service edge cases | External API failures can affect user-facing guidance. | Mock fetch tests for 401/429/500, timeout, empty response, long response, PII stripping, and fallback. |
| Security audit | Biometric and mental-health data is sensitive. | No-secrets scan, encrypted storage validation, PII stripping tests, data deletion tests, and export privacy checks. |
| Real-world model validation | Synthetic/WESAD evidence is not the same as deployment evidence. | User study comparing predictions with self-reports and PHQ-9/GAD-7 check-ins. |
| PHQ-9/GAD-7 routing into AI pipeline | Screening screens must influence risk/recommendation logic. | Integration tests proving PHQ-9/GAD-7 scores are stored and routed to AI/risk modules. |
| Python ML tests in this machine | Python tests could not run because pytest is missing. | Install dependencies and attach `python3 -m pytest tests/ -v` logs. |

---

## Equal Team Division

All four members are assigned an equal **25% contribution**. The testing work is redistributed so each person has:

- one core ML/model area,
- one AI-service or pipeline area,
- one SWD/mobile/device area,
- one documentation/evidence responsibility.

| Member | Core ML / Model Testing | AI Service / Pipeline Testing | SWD / App Testing | Evidence Required | Contribution |
|---|---|---|---|---|---:|
| Kareem | Focus model + focus feature/gauge behavior. | Model JSON validation and focus recommendation trigger path. | Focus/dashboard gauge render checks and model-output display screenshots. | Focus/model validation logs, focus screenshots, current ML gap notes. | 25% |
| Mariam | Feature-engineering quality checks + screening score UI, excluding anxiety production claims. | Recommendation safety, focus-tip fallback, and Groq failure mocks. | Auth, onboarding, PHQ-9/GAD-7 screen validation, and recommendation UI. | Feature/recommendation logs, LLM fallback tests, auth/screening screenshots. | 25% |
| Haneen | Stress model + sleep/baseline edge checks. | Stress-to-recommendation path and voice check-in sentiment. | Health Connect, notifications, background behavior, and Wear OS runtime proof. | Stress/sleep/voice logs, Android/device logs, notification and Wear OS screenshots. | 25% |
| Youssef | Full integration pipeline and storage-backed data flow. | `useWellness`, LLM privacy/PII tests, PHQ-9/GAD-7 routing gap proof. | SQLite/storage, data export, auth/session service checks. | Integration/coverage logs, storage/export logs, auth/session proof, routing gap notes. | 25% |

### Member Deliverables

| Member | Must Submit |
|---|---|
| Kareem | Focus model proof, model validation output, focus/dashboard UI evidence, WESAD/Python test gap note. |
| Mariam | Feature-engineering/screening UI proof, recommendation/focus-tip safety output, Groq fallback tests, auth/onboarding/screening UI evidence. |
| Haneen | Stress model proof, sleep/baseline/voice analysis proof, Health Connect/device evidence, notification/background/Wear OS evidence. |
| Youssef | Full integration proof, `useWellness` evidence, storage/export/auth checks, PHQ-9/GAD-7 routing gap proof. |

### Final Submission Checklist

| Evidence Item | Owner | Current Status |
|---|---|---|
| Full Jest coverage output | Youssef | Done today: 159 tests passed, 75.58% line coverage. Re-run before final hand-in. |
| Focus model and model-validation evidence | Kareem | Needed; anxiety remains R&D only. |
| Feature and recommendation safety evidence | Mariam | Needed. |
| Stress, sleep, baseline, and voice evidence | Haneen | Partially done in Jest; re-run and attach logs. |
| Integration pipeline evidence | Youssef | Partially done in Jest; re-run and attach logs. |
| LLM/recommendation failure evidence | Mariam | Needed, especially `llmService.ts`. |
| UI screenshots and automated screen tests | Kareem, Mariam | Needed. |
| Android dev build and Health Connect logs | Haneen | Needed. |
| Notification, background, and Wear OS proof | Haneen | Needed. |
| Auth, SQLite, and screening route tests | Youssef | Needed. |
| GitHub commits mapped to each member | All | Needed before final marking. |
