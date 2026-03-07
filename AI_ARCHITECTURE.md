# Seren AI Architecture

## System Overview

```
Samsung Galaxy Watch (Wear OS)
    | Samsung Health syncs to
    v
Google Health Connect (on Android phone)
    | react-native-health-connect reads
    v
+--------------------------------------------------+
|              SEREN REACT NATIVE APP               |
|                                                   |
|  [1] DATA INGESTION SERVICE                       |
|      healthConnect.ts                             |
|      - Reads HR, HRV, Sleep, Steps, SpO2, Temp   |
|      - Polls every 5 minutes                      |
|      - Writes raw samples to SQLite               |
|                                                   |
|  [2] LOCAL DATABASE (expo-sqlite)                 |
|      db.ts                                        |
|      - biometric_samples (HR, temp, motion)       |
|      - hrv_samples (RMSSD, SDNN, LF/HF)         |
|      - sleep_sessions + sleep_stages              |
|      - checkin_results                            |
|      - user_baselines                             |
|      - recommendations_log                        |
|                                                   |
|  [3] FEATURE ENGINEERING SERVICE                  |
|      featureEngineering.ts                        |
|      - Computes 29 HRV features per 5-min window  |
|      - Time-domain: RMSSD, SDNN, pNN50, etc.     |
|      - Frequency-domain: LF, HF, LF/HF ratio     |
|      - Normalizes against personal baselines      |
|                                                   |
|  [4] PERSONAL BASELINE SERVICE                    |
|      baseline.ts                                  |
|      - Rolling 14-day statistics per user          |
|      - Resting HR mean/std                        |
|      - HRV RMSSD mean/std                         |
|      - Sleep schedule, avg quality                |
|      - Recomputed daily                           |
|                                                   |
|  [5] STRESS MODEL SERVICE                         |
|      stressModel.ts                               |
|      - Loads exported XGBoost JSON tree            |
|      - Takes 29 features -> stress score 0-100    |
|      - Pure TypeScript tree traversal             |
|      - No native modules needed                   |
|                                                   |
|  [6] SLEEP ANALYSIS SERVICE                       |
|      sleepAnalysis.ts                             |
|      - Consumes Health Connect sleep stages        |
|      - Computes quality score (weighted formula)   |
|      - Tracks consistency and trends              |
|                                                   |
|  [7] VOICE CHECK-IN SERVICE                       |
|      voiceAnalysis.ts                             |
|      - STT: expo-speech-recognition (Phase 1)     |
|      - Sentiment: VADER lexicon (Phase 1)         |
|      - Upgrade: whisper.rn + DistilBERT (Phase 2) |
|                                                   |
|  [8] RECOMMENDATION ENGINE                        |
|      recommendations.ts                           |
|      - Rule-based on clinical thresholds           |
|      - Inputs: stress, sleep, anxiety, time        |
|      - Outputs: prioritized interventions          |
|      - Tracks effectiveness (pre/post biometrics)  |
|                                                   |
+--------------------------------------------------+

## Data Flow Per 5-Minute Cycle

Health Connect -> Raw samples -> SQLite
                                  |
                    Feature Engineering (29 features)
                                  |
                    Personal Baseline comparison
                                  |
                    Stress Model inference
                                  |
                    Recommendation check
                                  |
                    UI update via React hooks
```

## Health Connect Data Types Used

| Health Connect Record | Samsung Watch Sensor | Our Use |
|----------------------|---------------------|---------|
| HeartRateRecord | PPG (optical) | HR BPM samples |
| HeartRateVariabilityRmssdRecord | PPG derived | RMSSD values |
| SleepSessionRecord | Accel + HR | Sleep stages/duration |
| StepsRecord | Accelerometer | Activity level |
| OxygenSaturationRecord | SpO2 sensor | Blood oxygen |
| SkinTemperatureRecord | Temp sensor | Skin temperature |
| RespiratoryRateRecord | PPG derived | Breathing rate |
| ExerciseSessionRecord | Multi-sensor | Activity context |

## ML Training Pipeline (Python)

```
ml/
  data/
    wesad/               # WESAD dataset (download separately)
    swell/               # SWELL-KW HRV dataset
  notebooks/
    01_explore_wesad.ipynb
    02_feature_engineering.ipynb
    03_train_stress_model.ipynb
  src/
    preprocessing.py     # Load & clean WESAD pickle files
    features.py          # Extract 29 HRV features per window
    train.py             # Train XGBoost, evaluate, export
    export.py            # Export to JSON for TypeScript
  models/
    stress_model.json    # Exported XGBoost decision trees
    model_metadata.json  # Feature names, normalization params
  requirements.txt
```

## Model: Stress Prediction

- **Algorithm**: XGBoost (gradient boosted trees)
- **Input**: 29 features per 5-min window
- **Output**: stress_score 0-100 (regression) or class (low/moderate/elevated/high)
- **Training data**: WESAD (wrist-only) + SWELL-KW
- **Export**: JSON decision tree dump -> TypeScript tree traversal
- **Size**: ~50-200 KB
- **Inference**: <1ms on any phone
