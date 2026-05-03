# Youssef - Balanced Testing Tasks and Quality Evidence

**Project:** Seren Predictive Mental Health Smartwatch Companion  
**Member contribution:** 25%  
**Balanced split:** Integration pipeline + AI privacy + SWD services + evidence reporting  
**Date:** May 3, 2026  

## Progress Context

From `PROGRESS_REPORT.md`, the project is about **59% production-ready**. Focus model ownership moved to Kareem, and stress model ownership moved to Haneen. Youssef owns integration and service-layer proof.

| Component | Status | Youssef's Testing Meaning |
|---|---|---|
| C03 Local Storage | Fully done | Own storage/export testing evidence. |
| C10 Recommendations | Fully done | Own full pipeline regression evidence. |
| C04 User/Profile Privacy | Partial | Auth/session gaps must be tested or documented honestly. |
| C09 Depression Risk Analyzer | Not implemented | PHQ-9 routing cannot be marked complete. |

## Assigned Workload

| Work Type | Assigned Scope | Why This Is Equal |
|---|---|---|
| Core integration | Full biometric -> features -> model -> recommendations pipeline. | Cross-module equivalent to model ownership. |
| AI service/privacy | `useWellness`, LLM privacy/PII tests, PHQ-9/GAD-7 routing gap proof. | Owns orchestration and safety evidence. |
| SWD/services | SQLite/storage, data export, auth/session service checks. | Service-layer SWD slice. |
| Evidence | Full coverage log, integration logs, storage/export proof, routing gap notes. | Same evidence load as everyone else. |

## Already Done / Existing Evidence

| Item | Evidence |
|---|---|
| Integration tests | `integration.test.ts` exists and passed in the latest full Jest run. |
| Local storage implementation | C03 is fully done. |
| Foreground wellness loop | Progress report notes inference currently runs with foreground `setInterval`. |
| Full Jest run | Latest run passed 159 tests with 75.58% line coverage. |

## Testing Tasks

| ID | Type | Task | Expected Result |
|---|---|---|---|
| Y-AI-01 | Integration | Re-run `npx jest __tests__/integration.test.ts`. | Relaxed, stressed, poor sleep, voice, and anomaly paths pass. |
| Y-AI-02 | Regression | Re-run `npm run test:coverage -- --runInBand`. | Full suite passes and coverage summary is attached. |
| Y-AI-03 | Privacy | Add/verify LLM PII stripping and fallback tests. | Sensitive data is removed and fallback works safely. |
| Y-AI-04 | Gap proof | Inspect PHQ-9/GAD-7 routing into AI modules. | Document as not implemented unless code has changed. |
| Y-SWD-01 | Hook | Test `useWellness` model startup and 5-second inference loop with fake timers. | Foreground inference behavior is proven. |
| Y-SWD-02 | Hook | Test Groq refresh threshold behavior. | Tip refresh triggers only when score changes by at least 7 points. |
| Y-SWD-03 | Storage | Test SQLite write/read/delete/retention behavior. | Data persists and deletes correctly. |
| Y-SWD-04 | Privacy | Test data export/deletion and auth/session current behavior. | Export is structured; auth gaps are documented honestly. |

## Not Done / Must Not Be Claimed Complete

| Gap | Evidence To Provide |
|---|---|
| Anxiety is R&D only. | Do not claim anxiety as production evidence. |
| Depression risk analyzer C09 is not implemented. | State PHQ-9 routing to C09 is missing. |
| Auth/session lifecycle is partial. | Test current behavior, but do not claim PIN/biometric/session timeout. |
| Background monitoring is not production-complete. | Only foreground `useWellness` loop can be proven unless implementation changes. |

## Commands

```bash
npx jest __tests__/integration.test.ts
npm run test:coverage -- --runInBand
npx tsc --noEmit
```

