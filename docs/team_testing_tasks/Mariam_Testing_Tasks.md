# Mariam - Balanced Testing Tasks and Quality Evidence

**Project:** Seren Predictive Mental Health Smartwatch Companion  
**Member contribution:** 25%  
**Balanced split:** Feature quality + AI service + SWD UI + evidence reporting  
**Date:** May 3, 2026  

## Progress Context

From `PROGRESS_REPORT.md`, the project is about **59% production-ready**. Anxiety is still **R&D only**, so Mariam's scope excludes anxiety production testing.

| Component | Status | Mariam's Testing Meaning |
|---|---|---|
| C05 Physiological Feature Extraction | Fully done | Own feature-quality evidence that supports all production models. |
| C10 Recommendations | Fully done | Preserve recommendation test evidence and add safety/fallback checks. |
| C04 User/Profile Privacy | Partial | Auth UI exists, but real PIN/biometric/session behavior is not complete. |
| C11 Voice/STT | Partial | Screening screens exist, but PHQ-9/GAD-7 routing is missing. |

## Assigned Workload

| Work Type | Assigned Scope | Why This Is Equal |
|---|---|---|
| Core quality/model support | Feature-engineering edge cases and screening score UI checks. | Supports ML without assigning R&D anxiety. |
| AI service | Recommendations, focus-tip safety, Groq failure fallback. | Owns user-facing AI text quality. |
| SWD/app | Auth, onboarding, PHQ-9, GAD-7, and recommendation UI checks. | Comparable frontend slice. |
| Evidence | Feature/recommendation logs, LLM fallback logs, UI screenshots. | Same evidence load as everyone else. |

## Already Done / Existing Evidence

| Item | Evidence |
|---|---|
| Recommendation engine implementation | C10 is fully done. |
| Recommendation tests | `recommendations.test.ts` exists and passed in the latest full Jest run. |
| Feature tests | `featureEngineering.test.ts` exists and passed in the latest full Jest run. |
| Auth/onboarding/screening UI | Screens exist, but automated tests and screenshots still need to be attached. |

## Testing Tasks

| ID | Type | Task | Expected Result |
|---|---|---|---|
| M-ML-01 | Feature | Re-run `npx jest __tests__/featureEngineering.test.ts`. | Feature extraction edge cases pass. |
| M-ML-02 | Boundary | Document feature behavior for missing sensor data, RR outliers, and low sample count. | Defaults are safe and documented. |
| M-AI-01 | Unit | Re-run `npx jest __tests__/recommendations.test.ts`. | Recommendation triggers, priority, dedupe, and outcome tests pass. |
| M-AI-02 | Failure | Add/verify Groq fallback tests for 401, 429, 500, empty response, and timeout. | Rule-based/fallback text appears with no crash. |
| M-AI-03 | Safety | Verify focus/recommendation tips avoid banned or harmful wording. | Tip is short, safe, and user-facing. |
| M-SWD-01 | UI | Test login/signup empty fields and validation states. | Validation appears and no invalid API call is made. |
| M-SWD-02 | UI | Test onboarding navigation and screenshots. | Welcome, privacy, and sync screens render cleanly. |
| M-SWD-03 | UI | Test PHQ-9 and GAD-7 completion screens. | Screen score works; AI-routing gap is documented separately. |
| M-EV-01 | Evidence | Produce a screenshot pack for auth, onboarding, screening, and recommendation screens. | Small and large screen evidence is ready. |

## Not Done / Must Not Be Claimed Complete

| Gap | Evidence To Provide |
|---|---|
| Anxiety is R&D only. | Do not claim anxiety model testing as final production evidence. |
| PHQ-9/GAD-7 routing into AI is not implemented. | Show screen scoring only; document missing routing as a gap. |
| On-device LLM and offline STT are not implemented. | Test Groq/cloud fallback and state the privacy limitation. |
| Real auth system is partial. | Do not claim PIN/biometric/session timeout is complete. |

## Commands

```bash
npx jest __tests__/featureEngineering.test.ts
npx jest __tests__/recommendations.test.ts
npm run test:coverage -- --runInBand
npx tsc --noEmit
```

