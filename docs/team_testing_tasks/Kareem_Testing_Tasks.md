# Kareem - Balanced Testing Tasks and Quality Evidence

**Project:** Seren Predictive Mental Health Smartwatch Companion  
**Member contribution:** 25%  
**Balanced split:** Focus model + AI pipeline + SWD UI + evidence reporting  
**Date:** May 3, 2026  

## Progress Context

From `PROGRESS_REPORT.md`, the project is about **59% production-ready**. Anxiety remains **R&D only**, so Kareem owns the production-facing **focus model** testing instead.

| Component | Status | Kareem's Testing Meaning |
|---|---|---|
| Focus model | Implemented / needs focused evidence | Own focus score, level mapping, and focus recommendation behavior. |
| C05 Physiological Feature Extraction | Fully done | Use feature tests as support evidence for focus inputs. |
| C16 Training/Update Pipeline | Partial | Document Python/WESAD/test-environment gaps. |
| UI screens | Mostly built | Prove focus/dashboard model output renders correctly. |

## Assigned Workload

| Work Type | Assigned Scope | Why This Is Equal |
|---|---|---|
| Core ML/model | Focus model score bounds, level mapping, and calm/drifting/sharp examples. | One production model area. |
| AI pipeline | Model JSON validation + focus recommendation trigger path. | Connects model output to user-facing app behavior. |
| SWD/app | Focus/dashboard score display and gauge screenshots. | Comparable UI work to the other members. |
| Evidence | Focus logs, model validation, screenshots, Python/WESAD gap note. | Same reporting load as everyone else. |

## Already Done / Existing Evidence

| Item | Evidence |
|---|---|
| Feature engineering implementation | C05 is fully done; latest Jest coverage showed 99.47% line coverage. |
| Focus model file exists | `services/ai/focusModel.ts` is implemented and should now receive direct test evidence. |
| Model validation suite | `modelValidation.test.ts` exists and passed in the latest full Jest run. |
| Focus UI exists | `app/(tabs)/focus.tsx` and `components/ui/FocusGauge.tsx` exist and need screenshot/render evidence. |

## Testing Tasks

| ID | Type | Task | Expected Result |
|---|---|---|---|
| K-ML-01 | Model | Add/verify focus model tests for score bounds. | Focus score always stays 0-100. |
| K-ML-02 | Model | Add/verify focus level mapping tests. | Scattered, drifting, steady, and sharp map correctly. |
| K-ML-03 | Boundary | Test calm, steady, drifting, and distracted focus examples. | Output changes logically or limitation is documented. |
| K-ML-04 | Boundary | Test all-zero, extreme HR, low HRV, and missing feature values. | No crash, no NaN/Infinity, safe output. |
| K-AI-01 | Validation | Re-run `npx jest __tests__/modelValidation.test.ts`. | Model schema, feature order, normalization, and tree references pass. |
| K-AI-02 | Pipeline | Verify focus score changes drive focus recommendation/tip behavior where implemented. | Focus advice appears or current gap is documented. |
| K-SWD-01 | UI | Screenshot focus/dashboard score display on small and large screens. | Score, `/100`, and live signals are readable with no overlap. |
| K-SWD-02 | UI boundary | Check FocusGauge states for 0, 50, and 100. | Arc and text render correctly in every state. |
| K-EV-01 | Evidence | Run or document Python ML tests. | If pytest is still missing, record the exact blocker honestly. |

## Not Done / Must Not Be Claimed Complete

| Gap | Evidence To Provide |
|---|---|
| Anxiety is R&D only. | Do not claim anxiety as production testing evidence. |
| Real-world focus calibration is not validated. | Document current focus output as not validated on real Samsung/Galaxy Watch data. |
| Python pytest was blocked locally by missing pytest. | Install dependencies and run, or document the blocker. |
| UI visual evidence is not automated yet. | Add screenshots or UI test output before final submission. |

## Commands

```bash
npx jest __tests__/modelValidation.test.ts
npx jest __tests__/featureEngineering.test.ts
npm run test:coverage -- --runInBand
cd ml && python3 -m pytest tests/ -v
```

