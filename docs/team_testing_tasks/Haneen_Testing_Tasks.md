# Haneen - Balanced Testing Tasks and Quality Evidence

**Project:** Seren Predictive Mental Health Smartwatch Companion  
**Member contribution:** 25%  
**Balanced split:** Stress model + AI signal analysis + SWD/device + evidence reporting  
**Date:** May 3, 2026  

## Progress Context

From `PROGRESS_REPORT.md`, the project is about **59% production-ready**. Haneen now owns the production-facing **stress model** testing. Anxiety remains **R&D only** and is not part of final testing ownership.

| Component | Status | Haneen's Testing Meaning |
|---|---|---|
| C07 Mental State Prediction | Fully done | Own stress-model behavior and fallback testing. |
| C08 Sleep/Circadian Engine | Fully done | Add sleep/baseline edge evidence as supporting work. |
| C01 Sensor Collection | Partial / mock-only | Real Health Connect/watch ingestion still needs proof. |
| C13 Notifications | Partial | Notifications exist, but rate limiting/background scheduling need evidence or gap notes. |

## Assigned Workload

| Work Type | Assigned Scope | Why This Is Equal |
|---|---|---|
| Core ML/model | Stress model score, level, confidence, fallback, and contributors. | One production model area. |
| AI signal analysis | Stress-to-recommendation path plus voice/sleep support checks. | Owns model-to-user outcome evidence. |
| SWD/device | Health Connect, notifications, background behavior, Wear OS run proof. | Device/platform slice. |
| Evidence | Stress/sleep/voice logs and Android/device screenshots. | Same evidence load as everyone else. |

## Already Done / Existing Evidence

| Item | Evidence |
|---|---|
| Stress model implementation | C07 is fully done; TypeScript XGBoost inference and fallback exist. |
| Stress tests | `stressModel.test.ts` exists and passed in the latest full Jest run. |
| Sleep and baseline tests | `sleepAnalysis.test.ts` and `baseline.test.ts` exist and passed in the latest full Jest run. |
| Notification helpers | Local notification code exists, but device validation is still needed. |

## Testing Tasks

| ID | Type | Task | Expected Result |
|---|---|---|---|
| H-ML-01 | Unit | Re-run `npx jest __tests__/stressModel.test.ts`. | Stress score, levels, confidence, fallback, and contributors pass. |
| H-ML-02 | Boundary | Add/confirm stress tests for all-zero input, HR=200, HRV=0, and missing features. | No crash, no NaN/Infinity, output stays 0-100. |
| H-AI-01 | Pipeline | Verify high stress input creates stress-focused recommendation. | Breathing/grounding recommendation appears from high stress input. |
| H-AI-02 | Support | Re-run `npx jest __tests__/sleepAnalysis.test.ts` and `baseline.test.ts`. | Sleep/baseline edge tests pass as supporting evidence. |
| H-SWD-01 | Device | Build Android dev app and test Health Connect permission path. | Permission screenshot/log captured. |
| H-SWD-02 | Device | Test real or no-record Health Connect read behavior. | Either records are logged or safe empty state is documented. |
| H-SWD-03 | Device | Test notification permission allowed/denied and high-stress alert. | Correct behavior or limitation is documented. |
| H-SWD-04 | Device | Build/run Wear OS app or document current build/runtime blocker. | Screenshot/log attached, or blocker stated clearly. |
| H-EV-01 | Evidence | Document background behavior after app backgrounding. | Current limitation is clear: production WorkManager/background monitoring remains open. |

## Not Done / Must Not Be Claimed Complete

| Gap | Evidence To Provide |
|---|---|
| Anxiety is R&D only. | Do not claim anxiety as production evidence. |
| Real Galaxy Watch stress validation is not verified. | Attach Health Connect/watch logs only if tested. |
| Background inference is not production-complete. | Document current foreground/background behavior. |
| Notification rate limiting/dynamic scheduling is not complete. | Test current notification helpers and mark missing behavior. |

## Commands

```bash
npx jest __tests__/stressModel.test.ts
npx jest __tests__/sleepAnalysis.test.ts
npx jest __tests__/baseline.test.ts
npm run android
cd wearos && ./gradlew assembleDebug
```

