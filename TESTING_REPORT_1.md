# Implementation & Testing Report 1

**Project Title:** Seren — Predictive Mental Health Monitoring via Smartwatch Biometrics
**Program:** Data Science & Artificial Intelligence (DSAI)
**Date:** March 7, 2026
**Team Members:** [Add team member names here]
**Supervisor:** [Add supervisor name here]

---

## 1. Project Status & Engineering Transparency (15 marks)

### 1.1 Current State of the Project

Seren is a privacy-first mental wellness companion that monitors stress, anxiety, and sleep quality using smartwatch biometric data. The application processes heart rate, HRV (heart rate variability), skin temperature, and accelerometer data through an on-device AI pipeline to deliver real-time stress predictions and evidence-based wellness recommendations.

**Current development stage:** Core application and AI inference pipeline are implemented and testable. The system runs on mock data with the architecture fully prepared for real Health Connect integration via a dev build.

### 1.2 Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Framework | React Native + Expo | SDK 54, RN 0.81.5 |
| Language | TypeScript (strict mode) | 5.9.2 |
| Navigation | Expo Router (file-based) | v6 |
| ML Training | Python (XGBoost, scikit-learn) | Python 3.x |
| ML Inference | Pure TypeScript XGBoost tree traversal | Custom |
| NLP | Groq Whisper (STT) + Llama 3.1 8B (analysis) | API |
| Local NLP | VADER-style sentiment lexicon | Custom |
| Database | expo-sqlite (on-device) | SDK 54 |
| Testing | Jest + ts-jest (TS), pytest (Python) | Jest 30, pytest |
| State | React hooks + Context | React 19.1 |

### 1.3 Repository Structure

```
seren-app/
+-- app/                       # 10 screens (Expo Router file-based routing)
|   +-- (tabs)/                # Main tab screens
|   |   +-- index.tsx          # Home dashboard (stress gauge, sleep, anxiety)
|   |   +-- insights.tsx       # Trends & bio-correlation charts
|   |   +-- checkin.tsx        # Voice check-in with waveform
|   |   +-- recommendations.tsx # AI-generated wellness suggestions
|   |   +-- settings.tsx       # Privacy controls, data export/delete
|   +-- onboarding.tsx         # 3-slide onboarding + watch setup
+-- services/ai/              # 17 AI service modules (5,250 lines)
|   +-- featureEngineering.ts  # 29 HRV feature extraction (472 lines)
|   +-- stressModel.ts        # XGBoost inference engine (413 lines)
|   +-- sleepAnalysis.ts      # Clinical sleep quality scoring (395 lines)
|   +-- voiceAnalysis.ts      # VADER sentiment + emotion detection (343 lines)
|   +-- recommendations.ts    # Evidence-based intervention engine (483 lines)
|   +-- llmService.ts         # LLM integration (Groq/OpenAI) (610 lines)
|   +-- whisperService.ts     # Speech-to-text via Whisper API (151 lines)
|   +-- baseline.ts           # Personal baseline & anomaly detection (260 lines)
|   +-- healthConnect.ts      # Samsung Health Connect interface
|   +-- db.ts                 # SQLite persistence layer
|   +-- types.ts              # 407-line type system (29 interfaces)
|   +-- ...                   # notifications, encryption, data export
+-- ml/                       # Python ML training pipeline
|   +-- src/                   # 7 modules (1,636 lines)
|   |   +-- train.py           # XGBoost LOSO-CV training
|   |   +-- features.py        # 29 HRV feature extraction (Python)
|   |   +-- preprocessing.py   # WESAD + SWELL-KW data loading
|   |   +-- synthetic_data.py  # Physiologically realistic synthetic data
|   +-- tests/                 # 3 test files (pytest)
+-- components/ui/            # 5 reusable UI components
+-- assets/ml/                # Trained model artifacts
|   +-- stress_model.json     # 200-tree XGBoost model
|   +-- model_metadata.json   # Normalization parameters
+-- __tests__/                # 8 Jest test suites (2,138 lines)
+-- hooks/                    # React hooks (useHealthData, useWellness, useCheckin)
+-- constants/                # Design tokens (theme.ts)
```

### 1.4 Architecture Overview

```
Samsung Galaxy Watch (Health Connect API)
        |
        v
  [Health Connect Service] -- Raw HR, HRV, Temp, Sleep, Steps
        |
        v
  [Feature Engineering] -- 29 biometric features per 5-min window
        |
        v
  [XGBoost Stress Model] -- Stress score 0-100
        |                      |
        v                      v
  [Anxiety Predictor]    [Personal Baseline]
        |                      |
        v                      v
  [Recommendation Engine] <-- [Sleep Analysis]
        |                      |
        v                      v
  [UI Dashboard]          [Anomaly Detection]
        ^
        |
  [Voice Check-in] -- Whisper STT --> LLM Analysis --> Sentiment + Emotions
```

### 1.5 Gaps and Honest Limitations

| Area | Status | Limitation |
|---|---|---|
| Real smartwatch data | Not yet integrated | Requires Expo dev build for Health Connect native module; currently running on mock data |
| LLM dependency | External API (Groq) | Requires network; VADER fallback works offline but with less depth |
| Model training data | Synthetic + WESAD | Real user data collection not started; model accuracy on diverse populations unvalidated |
| User study | Not conducted | No real participants tested yet |
| Security audit | Not performed | Encryption layer exists (XOR-based) but not formally audited |
| iOS support | Partial | Health Connect is Android-only; iOS would need HealthKit adapter |

---

## 2. Implementation Progress (30 marks)

### 2.1 Implemented Components Summary

| # | Component | Status | Description |
|---|---|---|---|
| C01 | Onboarding Flow | Done | 3 slides + watch connection setup screen |
| C02 | Home Dashboard | Done | StressGauge, sleep/anxiety cards, HRV insight, mic FAB |
| C03 | Stress Gauge | Done | Animated SVG circular gauge with color-coded levels |
| C04 | Trend Charts | Done | SVG line charts with area fill, grid lines, 7-day view |
| C05 | Sleep Heatmap | Done | Monthly grid of colored circles for sleep quality |
| C06 | Voice Check-in | Done | Real audio recording (expo-av) + Whisper STT + LLM analysis |
| C07 | Recommendations | Done | Rule-based engine with clinical citations, 12 interventions |
| C08 | Settings/Privacy | Done | Data export (JSON), full delete, device toggles |
| C09 | HRV Feature Engine | Done | 29 features: time-domain, frequency-domain, non-linear, temp, activity |
| C10 | XGBoost Inference | Done | Pure TS tree traversal, no native dependencies |
| C11 | Sleep Analysis | Done | Weighted quality scoring with clinical thresholds |
| C12 | Sentiment Analysis | Done | VADER-style lexicon + 8-emotion detection |
| C13 | Personal Baseline | Done | 14-day rolling baseline + z-score anomaly detection |
| C14 | Health Connect Bridge | Partial | Interface built, mock service active; real requires dev build |
| C15 | SQLite Persistence | Done | Full schema for biometrics, features, sleep, check-ins |
| C16 | LLM Integration | Done | Groq Llama 3.1 8B for deep emotional analysis with PII stripping |

### 2.2 Key Implementation Details

#### 2.2.1 Feature Engineering Pipeline (featureEngineering.ts — 472 lines)

The feature engineering module computes 29 biometric features from raw Health Connect data, matching the Python training pipeline:

- **Time-domain HRV (9 features):** meanRR, SDNN, RMSSD, pNN50, pNN20, hrMean, hrStd, hrRange, cvRR
- **Frequency-domain HRV (7 features):** VLF/LF/HF power via simplified Welch's method (DFT with Hann window, resampled to 4 Hz), LF/HF ratio, normalized LF/HF
- **Non-linear HRV (5 features):** Poincare SD1/SD2, sample entropy (m=2, r=0.2*SD), DFA alpha1 (box sizes 4-16)
- **Temperature (4 features):** mean, slope (linear regression), std, range
- **Activity (4 features):** estimated accelerometer magnitude from step rate, activity type classification

#### 2.2.2 Stress Model Inference (stressModel.ts — 413 lines)

Pure TypeScript XGBoost tree traversal:
1. Feature normalization using z-score (mean/std from training)
2. Binary decision tree traversal for each of 200 trees
3. Leaf value summation with sigmoid activation
4. Stress score scaling to 0-100 with level mapping (low/moderate/elevated/high)
5. Feature importance-based contributor identification
6. Rule-based fallback when model is not loaded

#### 2.2.3 Voice Check-in Pipeline

1. **Audio Recording:** expo-av with 16kHz mono M4A, real-time metering for waveform visualization
2. **Transcription:** Groq Whisper (whisper-large-v3-turbo) via multipart file upload
3. **Analysis:** LLM (Llama 3.1 8B) for deep emotional understanding with biometric context, or local VADER fallback
4. **Privacy:** PII stripping before API calls, transcript encryption for storage

### 2.3 Codebase Metrics

| Metric | Value |
|---|---|
| Total AI service code | 5,250 lines (17 TypeScript modules) |
| ML training pipeline | 1,636 lines (7 Python modules) |
| Test code | 2,138 lines (8 Jest suites) + 262 lines (3 pytest files) |
| UI code | ~3,500 lines (10 screens + 5 components) |
| Type definitions | 407 lines (29 interfaces, 6 enums) |
| Trained model | 200 trees, 29 features, exported as JSON |

---

## 3. Program-Specific Technical Depth — DSAI (20 marks)

### 3.1 Data Pipeline Architecture

```
Raw Sensor Data (Health Connect)
    |
    v
[RR Interval Extraction] -- BPM to RR (ms): RR = 60000/BPM
    |                        Artifact filter: 300ms < RR < 2000ms
    |                        Outlier removal: >30% from median
    v
[5-Minute Windowing] -- Sliding windows, no overlap
    |
    v
[29-Feature Extraction] -- Time-domain, Freq-domain, Non-linear, Temp, Activity
    |
    v
[Z-Score Normalization] -- Using training set mean/std per feature
    |
    v
[XGBoost Inference] -- 200 trees, binary classification
    |
    v
[Score Scaling] -- Sigmoid -> 0-100 stress score
    |
    v
[Contextualization] -- Personal baseline comparison, anxiety derivation
```

**Data sources used for training:**
- **WESAD dataset:** 15 subjects, Empatica E4 wrist sensor, labeled stress/baseline/amusement/meditation from the Trier Social Stress Test (TSST)
- **SWELL-KW dataset:** Office workers, HRV features during knowledge work under stress
- **Synthetic data generator:** Physiologically realistic data for augmentation and testing (configurable stress ratio, subject count)

### 3.2 Baseline Model Performance

The XGBoost model was trained using Leave-One-Subject-Out Cross-Validation (LOSO-CV) to ensure generalizability:

| Metric | Value |
|---|---|
| Model | XGBoost (gradient boosted trees) |
| Estimators | 200 trees |
| Max depth | 6 |
| Learning rate | 0.1 |
| Cross-validation | LOSO-CV (leave-one-subject-out) |
| Features | 29 biometric features |
| CV Accuracy | >70% (on synthetic data; WESAD pending) |
| Binary task | Stressed (1) vs. Baseline (0) |

**Top feature importances (from model):**
1. RMSSD (root mean square of successive RR differences) — parasympathetic indicator
2. LF/HF Ratio — sympathovagal balance
3. Heart Rate Mean — sympathetic activation
4. SDNN — overall HRV
5. SD1 (Poincare) — short-term variability

### 3.3 Evaluation Approach

**Offline evaluation (completed):**
- LOSO-CV ensures no data leakage between subjects
- Metrics: accuracy, AUC-ROC, F1-weighted
- Synthetic data validation: stressed samples have lower RMSSD, higher HR, higher LF/HF (verified in test_synthetic_data.py)

**Online evaluation (planned for MVP):**
- Pre/post intervention comparison: stress score before and after recommendation completion
- Effectiveness scoring: weighted combination of stress reduction (0-0.3) and HRV improvement (0-0.2)
- Baseline deviation tracking: z-score anomaly flags for HR, HRV, temperature

### 3.4 Early Model Integration

The model is fully integrated into the TypeScript application:

1. **Model loading:** JSON model file loaded at app startup from `assets/ml/stress_model.json`
2. **Feature parity:** TypeScript feature extraction produces the same 29 features in the same order as the Python training pipeline (verified via `FEATURE_NAMES` constant shared between both)
3. **Normalization alignment:** Mean/std from training scaler stored in model JSON and applied identically during inference
4. **End-to-end pipeline:** Raw HR samples → feature extraction → normalization → tree traversal → stress score → anxiety derivation → recommendations (tested in integration tests)

---

## 4. Testing Discipline (20 marks)

### 4.1 Testing Strategy

We employ a multi-layer testing approach:

| Layer | Framework | Scope | Count |
|---|---|---|---|
| Unit Tests (TypeScript) | Jest + ts-jest | AI service modules | 139 tests |
| Integration Tests (TypeScript) | Jest | End-to-end AI pipeline | 5 tests |
| Model Validation Tests | Jest | Model file structure & schema | 14 tests (cross-validated as 9 in suite) |
| ML Pipeline Tests (Python) | pytest | Training, synthetic data, export | 11 tests |
| **Total** | | | **153 TS + 11 Python = 164 tests** |

### 4.2 Test Coverage

```
-----------------------|---------|----------|---------|---------
File                   | % Stmts | % Branch | % Funcs | % Lines
-----------------------|---------|----------|---------|---------
All files              |   74.07 |    59.93 |   82.14 |   76.3
 baseline.ts           |   90.62 |    72.00 |   87.87 |  95.00
 featureEngineering.ts |   95.73 |    83.50 |  100.00 |  99.47
 llmService.ts         |    5.88 |     0.00 |    0.00 |   6.74
 recommendations.ts    |   95.00 |    84.21 |  100.00 |  94.82
 sleepAnalysis.ts      |   93.05 |    81.52 |  100.00 |  96.00
 stressModel.ts        |   66.37 |    51.35 |   87.50 |  75.53
 types.ts              |  100.00 |   100.00 |  100.00 | 100.00
 voiceAnalysis.ts      |   97.16 |    92.53 |   94.44 |  96.80
-----------------------|---------|----------|---------|---------
```

**Note:** `llmService.ts` has low coverage because its primary function (`analyzeCheckinWithLLM`) requires a live API call to Groq. The VADER fallback path (`voiceAnalysis.ts`) is tested at 97% coverage. The LLM service's PII stripping and encryption utilities are unit-testable and will be added in the next iteration.

### 4.3 Unit Tests Conducted

#### 4.3.1 Feature Engineering Tests (22 tests)

| Test | What It Validates |
|---|---|
| RR interval extraction — empty/single sample | Edge case handling |
| BPM to RR conversion (60 BPM = 1000ms) | Mathematical correctness |
| Physiological artifact filtering (BPM 20, 250) | Input validation |
| Outlier removal (>30% from median) | Signal quality |
| Time-domain: mean, SDNN, RMSSD, pNN50, pNN20 | Statistical correctness |
| Time-domain: constant intervals yield SDNN=0 | Degenerate case |
| Time-domain: cvRR = SDNN/meanRR | Formula verification |
| Frequency-domain: non-negative power values | Physical validity |
| Frequency-domain: lfNorm + hfNorm = 100 | Conservation law |
| Non-linear: SD1 higher for variable vs steady RR | Poincare validity |
| Non-linear: sample entropy non-negative | Mathematical bound |
| Temperature: slope direction (warming/cooling) | Trend detection |
| Activity: sedentary/walking/active/sleeping classification | Threshold logic |
| Full 29-feature vector completeness | Schema compliance |
| Direct RMSSD substitution when divergent | Health Connect integration |

#### 4.3.2 Stress Model Tests (16 tests)

| Test | What It Validates |
|---|---|
| Model loading from object and JSON string | Serialization |
| Stress score bounded 0-100 | Output range |
| Stress level categories (low/moderate/elevated/high) | Mapping correctness |
| Confidence bounded 0-1 | Output range |
| Low RMSSD yields higher stress than high RMSSD | Clinical validity |
| Top contributors limited to 3 with required fields | Output structure |
| Anxiety index bounded 0-100 | Output range |
| Poor sleep increases anxiety | Clinical correlation |
| High LF/HF ratio increases anxiety | Sympathetic dominance |
| HRV below baseline increases anxiety | Personalized scoring |
| Baseline deviation clamped [-1, 1] | Output bounds |

#### 4.3.3 Sleep Analysis Tests (18 tests)

| Test | What It Validates |
|---|---|
| Total in-bed time from session boundaries | Duration calculation |
| Total sleep time excluding awake stages | Stage filtering |
| Sleep onset latency computation | Time to first sleep stage |
| WASO (Wake After Sleep Onset) | Mid-sleep awakenings |
| Stage percentages sum to 1.0 | Proportion conservation |
| Sleep efficiency = sleep/in-bed | Formula correctness |
| Quality score bounded 0-100 | Output range |
| Good sleep (8hr, balanced stages) = high score | Clinical validity |
| Poor sleep (4hr, fragmented) = low score | Clinical validity |
| HR/HRV stats during sleep | Biometric integration |
| Recovery score uses HRV baseline comparison | Personalization |
| Consistency score uses bedtime baseline | Schedule tracking |
| Sleep trend: improving/declining/stable detection | Longitudinal analysis |

#### 4.3.4 Voice Analysis Tests (24 tests)

| Test | What It Validates |
|---|---|
| Positive text ("great and happy") = positive sentiment | Lexicon accuracy |
| Negative text ("terrible and stressed") = concerned/distressed | Lexicon accuracy |
| Strongly negative = distressed (score < -0.5) | Severity differentiation |
| Sentiment score bounded [-1, 1] | Output range |
| Negation flips valence ("not good" < "good") | Negator handling |
| Intensifiers amplify sentiment ("very good" > "good") | Modifier handling |
| "extremely" stronger than "very" | Intensifier ranking |
| Joy/anxiety/sadness/fatigue/gratitude keyword detection | 8-emotion classification |
| Default to mild calm when no keywords match | Fallback behavior |
| Emotion scores bounded [0, 1] | Output range |
| Verbal + biometric stress alignment detection | Cross-modal insight |
| Verbal concern + calm biometrics = mismatch insight | Discrepancy detection |
| HRV < 25 triggers breathing suggestion | Context-aware insight |
| Max 3 insights per check-in | Output limiting |
| Complete CheckinAnalysis output structure | Schema compliance |
| Empathy response varies by sentiment | Response personalization |
| Check-in trend: improving/declining/stable | Longitudinal tracking |

#### 4.3.5 Recommendation Engine Tests (18 tests)

| Test | What It Validates |
|---|---|
| High stress (>=70) triggers box breathing + grounding | Clinical threshold |
| Moderate stress (50-69) triggers coherent breathing | Graduated response |
| Low stress (<50) = no stress interventions | False-positive avoidance |
| Severe anxiety triggers grounding + 4-7-8 breathing | Anxiety pathway |
| Sustained anxiety suggests social connection | Escalation logic |
| Poor sleep quality triggers wind-down routine | Sleep pathway |
| Low sleep efficiency triggers schedule alignment | Sleep optimization |
| Distressed check-in triggers expressive writing | Voice-triggered |
| Fatigued check-in triggers stretching | Emotion-specific |
| Max 3 recommendations | Output limiting |
| Sorted by priority (descending) | Ranking correctness |
| Deduplication keeps highest priority | No repeats |
| Recently shown recommendations filtered out | Freshness |
| Required fields present (title, citation, etc.) | Schema compliance |
| Outcome recording marks completed | Status tracking |
| Effectiveness score bounded [0, 1] | Output range |
| Stress reduction increases effectiveness | Metric validity |

#### 4.3.6 Baseline & Anomaly Detection Tests (14 tests)

| Test | What It Validates |
|---|---|
| Returns null with <50 samples | Minimum data requirement |
| Valid baseline with 100+ samples | Computation success |
| Realistic HR/HRV baseline ranges | Physiological bounds |
| Sleep baseline computation from sessions | Cross-data integration |
| Default sleep baselines when no sessions | Graceful degradation |
| Recomputation trigger after 24 hours | Freshness enforcement |
| No recomputation when baseline is fresh | Efficiency |
| No anomalies for normal features | Specificity |
| Low HRV anomaly detection (z < -2) | Sensitivity |
| High severity for z < -3 | Severity grading |
| Elevated HR anomaly detection | Multi-metric coverage |
| Temperature drop anomaly detection | Vasoconstriction signal |
| Anomaly flags have required properties | Schema compliance |

### 4.4 Integration Tests (5 tests)

| Test | Pipeline |
|---|---|
| Relaxed state: low HR + high HRV -> features -> stress -> anxiety -> recommendations | Full pipeline, low-stress path |
| Stressed state: high HR + low HRV -> elevated stress + targeted recommendations | Full pipeline, high-stress path |
| Poor sleep session -> sleep analysis -> sleep recommendations | Sleep pathway |
| Distressed check-in + high biometrics -> voice analysis -> recommendations | Voice pathway |
| Normal baseline -> anomalous features -> anomaly flags | Baseline + anomaly detection |

### 4.5 Model Validation Tests (9 tests)

| Test | What It Validates |
|---|---|
| Model JSON file exists and parses | File integrity |
| Required top-level fields present | Schema compliance |
| Correct number of trees (200) | Training consistency |
| Valid tree node structure (split/leaf) | Model format |
| Normalization parameters for all features | Completeness |
| Base score within reasonable range | Model validity |
| Learning rate between 0 and 1 | Hyperparameter validity |
| Feature schema alignment (TS matches model) | Cross-platform consistency |
| Split features reference valid indices | Tree integrity |

### 4.6 Python ML Pipeline Tests (11 tests)

| Test | What It Validates |
|---|---|
| Synthetic data shape (subjects x windows) | Generator correctness |
| All 29 features present in synthetic data | Feature completeness |
| Label distribution matches target ratio | Class balance |
| Physiological ranges (HR 40-180, RMSSD >5, temp 28-38) | Data realism |
| Stressed vs baseline statistical separation | Discriminability |
| Reproducibility with fixed seed | Determinism |
| Training produces metrics (accuracy, AUC, F1) | Pipeline completion |
| CV accuracy > 0.5 (better than random) | Model validity |
| Fold metrics per subject | LOSO-CV correctness |
| JSON model export with trees and normalization | Export format |
| Metadata export with mean/std | Normalization export |

### 4.7 Bug Documentation

| # | Bug | Severity | Status | Resolution |
|---|---|---|---|---|
| B01 | Whisper shows "[not configured]" despite having API key | High | Fixed | Changed `expo-file-system` import to `expo-file-system/legacy` (SDK 54 migration) |
| B02 | `FileSystemUploadType.MULTIPART` not found | High | Fixed | Same root cause as B01 — `uploadAsync` moved to legacy module in SDK 54 |
| B03 | Non-standard MIME type `audio/m4a` rejected by Groq | Medium | Fixed | Changed to standard `audio/mp4` MIME type |
| B04 | Error message doesn't distinguish "not configured" from "API call failed" | Low | Fixed | Added `isWhisperConfigured()` check before transcription attempt |
| B05 | Negation only detects immediately preceding negator | Low | Known | VADER-style limitation — "I do not feel good" doesn't flip "good" because "not" is 2 tokens before "good". Documented as expected behavior. |

### 4.8 Responsible Handling

- **No hardcoded credentials:** API keys are configured at runtime via `initializeAIServices()` and never committed to version control
- **PII stripping:** The LLM service strips names, emails, phone numbers, and SSNs from transcripts before sending to external APIs
- **On-device storage:** All biometric data stays on-device in SQLite; no cloud sync
- **Data deletion:** Full data wipe available via `deleteAllData()` in settings
- **Model transparency:** Stress contributors show which biometric features drove the prediction

---

## 5. Work Plan Toward MVP (10 marks)

### 5.1 Remaining Work

| Task | Priority | Estimated Complexity | Dependencies |
|---|---|---|---|
| Expo dev build for Health Connect | Critical | Medium | EAS Build setup, Android device |
| Real smartwatch data collection | Critical | High | Dev build, Galaxy Watch pairing |
| Model retraining on real data | High | Medium | Real data collection |
| LLM service unit tests (PII, encryption) | Medium | Low | None |
| User onboarding study (5-10 participants) | Medium | Medium | Working dev build |
| iOS HealthKit adapter | Low | High | Apple Developer account |
| Security audit of encryption layer | Medium | Medium | None |

### 5.2 Sprint Plan

**Sprint 1 (Weeks 1-2): Real Device Integration**
- Set up EAS Build for Android dev build
- Test Health Connect integration with Galaxy Watch
- Validate feature extraction on real biometric data
- Compare real vs mock data distributions

**Sprint 2 (Weeks 3-4): Model Validation & Improvement**
- Collect 1-2 weeks of real biometric data
- Retrain XGBoost model with real data included
- Evaluate model on real-world stress episodes
- Add LLM service tests and increase overall coverage to >85%

**Sprint 3 (Weeks 5-6): User Study & Polish**
- Conduct usability study with 5-10 participants
- Collect subjective stress ratings for ground truth
- Compare model predictions against self-reports
- UI/UX refinements based on feedback

### 5.3 Risk Mitigation

| Risk | Mitigation |
|---|---|
| Health Connect permissions rejected | Fallback to manual entry + Expo sensors |
| Model accuracy insufficient on real data | Rule-based fallback always available; retrain with expanded dataset |
| API rate limits (Groq free tier) | VADER fallback works fully offline; cache recent transcriptions |
| Galaxy Watch battery drain | Configurable polling interval (default 5 min); background task optimization |

---

## 6. Individual Contribution Transparency (5 marks)

[To be filled by each team member]

| Team Member | Primary Contributions | % of Work |
|---|---|---|
| [Name 1] | [e.g., AI pipeline, feature engineering, model training] | [%] |
| [Name 2] | [e.g., UI/UX design, screen implementation, components] | [%] |
| [Name 3] | [e.g., Testing, documentation, Health Connect integration] | [%] |

**Collaboration tools used:** Git (version control), [Add: Jira/Trello/Notion for task tracking], [Add: communication tools]

---

## Appendix A: How to Run Tests

### TypeScript Tests (Jest)
```bash
# Run all tests
npm test

# Run with coverage report
npm run test:coverage

# Run in watch mode (re-runs on file changes)
npm run test:watch

# Run a specific test suite
npx jest __tests__/featureEngineering.test.ts
```

### Python ML Tests (pytest)
```bash
cd ml
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pytest tests/ -v
```

## Appendix B: Test Results Summary

```
Test Suites: 8 passed, 8 total
Tests:       153 passed, 153 total
Snapshots:   0 total
Time:        ~25 seconds

Coverage:    74% statements | 60% branches | 82% functions | 76% lines
```

Core modules (excluding external API-dependent llmService.ts):
- Feature Engineering: **99.5% line coverage**
- Voice Analysis: **96.8% line coverage**
- Sleep Analysis: **96.0% line coverage**
- Baseline: **95.0% line coverage**
- Recommendations: **94.8% line coverage**
- Stress Model: **75.5% line coverage**
