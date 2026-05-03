# Seren — Testing Plan
## Sprint QA Allocation — 4 Team Members

> App: Predictive Mental Health Smartwatch Companion  
> Stack: React Native / Expo SDK 54 · TypeScript · XGBoost (on-device) · Groq API (llama-3.1-8b-instant)

---

## Member 1 — ML Model Validation (On-Device Inference)

**Scope:** Stress model, Focus model, Anxiety model (R&D), Feature engineering pipeline

### What to Test

| Test ID | Component | Type | What to Verify |
|---|---|---|---|
| M1.1 | `focusModel.ts` | Unit | Given a valid 29-feature input, `predictFocus()` returns a score in [0, 100] |
| M1.2 | `focusModel.ts` | Unit | Score decreases with elevated HR + low HRV (high distraction scenario) |
| M1.3 | `focusModel.ts` | Unit | Score increases with low HR + high HRV (calm scenario) |
| M1.4 | `focusModel.ts` | Unit | `getFocusLevel()` maps 0–24 → `scattered`, 25–49 → `drifting`, 50–74 → `steady`, 75–100 → `sharp` |
| M1.5 | `stressModel.ts` | Unit | `predictStress()` returns 0–100 for any valid input |
| M1.6 | `stressModel.ts` | Unit | High HRV input → lower stress score than low HRV input |
| M1.7 | `anxietyModel.ts` | Unit | `predictAnxiety()` returns a numeric score without throwing |
| M1.8 | `featureEngineering.ts` | Unit | Z-score normalisation of features stays within ±4σ for typical physiological ranges |
| M1.9 | `proof_different_readings.test.ts` | Integration | Same biometric input yields different stress vs. focus scores |
| M1.10 | All models | Boundary | All-zero feature vector → no crash, score stays in [0, 100] |
| M1.11 | All models | Boundary | Extreme values (HR = 200, HRV = 0) → graceful output, not NaN/Infinity |
| M1.12 | `focusModel.ts` | Performance | Inference completes in < 10 ms (model spec: < 2 ms) |

### Edge Cases to Cover
- NaN or undefined in feature array → should default to population mean, not crash
- Missing accelerometer data (sedentary = `activityType: 0`) → model handles correctly
- `pnn50` > 100 or negative RR intervals (corrupt sensor data)

### Example Inputs / Outputs

**High distraction (should score ≈ 20–35):**
```ts
{ hrMean: 95, hrStd: 14, rmssd: 18, pnn50: 4, lfHfRatio: 3.8, sampleEntropy: 0.4, ... }
// Expected: score ≤ 40, level = 'scattered' or 'drifting'
```

**Peak readiness (should score ≈ 80–95):**
```ts
{ hrMean: 62, hrStd: 3, rmssd: 72, pnn50: 32, lfHfRatio: 1.1, sampleEntropy: 1.8, ... }
// Expected: score ≥ 70, level = 'sharp' or 'steady'
```

### Fix Validation
- Run `npx jest __tests__/stressModel.test.ts __tests__/proof_different_readings.test.ts` and confirm 0 failures
- If a model fix changes the score range, re-run `modelValidation.test.ts` to catch regressions

---

## Member 2 — AI Services & Groq Integration

**Scope:** `focusRecommendations.ts`, `recommendations.ts`, `llmService.ts`, `aiConfig.ts`, `notifications.ts`

### What to Test

| Test ID | Component | Type | What to Verify |
|---|---|---|---|
| A2.1 | `focusRecommendations.ts` | Unit | `getFocusTips()` returns exactly 1 tip (not 3) |
| A2.2 | `focusRecommendations.ts` | Unit | Tip is non-empty string, under 100 characters |
| A2.3 | `focusRecommendations.ts` | Unit | Tip does NOT contain banned words: "distracted", "unfocused", "scattered" |
| A2.4 | `focusRecommendations.ts` | Unit | `ruleBasedTips()` returns a tip for every possible elevated feature label |
| A2.5 | `focusRecommendations.ts` | Unit | `ruleBasedTips([])` (no elevated features) → returns `DEFAULT_TIP`, not empty |
| A2.6 | `focusRecommendations.ts` | Fallback | When no API key configured → falls back to rule-based, no crash |
| A2.7 | `focusRecommendations.ts` | Fallback | When Groq returns HTTP 429/500 → rule-based fallback triggered |
| A2.8 | `focusRecommendations.ts` | Fallback | When Groq response is < 8 chars → falls back to rule-based |
| A2.9 | `focusRecommendations.ts` | Timeout | 10-second abort controller fires for slow responses |
| A2.10 | `recommendations.ts` | Unit | Returns 5 recommendations from `getRecommendations()` |
| A2.11 | `llmService.ts` | Unit | `getLLMConfig()` returns null when no key is set |
| A2.12 | `notifications.ts` | Unit | Scheduled notifications don't fire during quiet hours (e.g., midnight) |

### Edge Cases to Cover
- Groq returns a multi-line response → only first line used (strip bullet/list formatting)
- API key is whitespace-only string → treated as missing, fallback triggered
- Network offline → no unhandled promise rejection, graceful degradation

### Example Inputs / Outputs

**Groq fallback on 500:**
```ts
// Mock fetch to return { status: 500 }
const tips = await getFocusTips(45, 'drifting', [{ feature: 'hrMean', label: 'Heart Rate', value: 88, direction: 'high' }], true);
// Expected: tips.length === 1, tips[0] includes 'heart rate' (rule-based)
```

**No elevated features:**
```ts
const tips = await getFocusTips(82, 'sharp', [], true);
// Expected: tips[0] === DEFAULT_TIP (settled body message)
```

### Fix Validation
- Run `npx jest __tests__/recommendations.test.ts` — 0 failures
- Manually test with a real Groq API key: confirm single-line tip is returned with no banned words

---

## Member 3 — UI Components & Screen Rendering

**Scope:** `FocusGauge.tsx`, `StressGauge.tsx`, `AnxietyGauge.tsx`, `focus.tsx`, `index.tsx`, `PanicInterventionBanner.tsx`

### What to Test

| Test ID | Component | Type | What to Verify |
|---|---|---|---|
| U3.1 | `FocusGauge` | Snapshot | Renders without crashing at size=180, value=50, level='steady' |
| U3.2 | `FocusGauge` | Visual | Center shows score number + `/100`, NO level label text (Sharp/Drifting etc.) |
| U3.3 | `FocusGauge` | Visual | Ring color matches `getFocusColor()` output for each level |
| U3.4 | `FocusGauge` | Boundary | value=0 → ring is near-empty arc, no crash |
| U3.5 | `FocusGauge` | Boundary | value=100 → ring is full arc, no crash |
| U3.6 | `FocusGauge` | Boundary | size=100 (small screen) → renders without overflow |
| U3.7 | `focus.tsx` | Integration | All 4 LIVE SIGNALS items render (Heart Rate, HRV, Temp, Balance) |
| U3.8 | `focus.tsx` | Integration | Signal values show `—` when elevatedFeatures is empty (no watch connected) |
| U3.9 | `focus.tsx` | Integration | ActivityIndicator shown when `focus.groqLoading === true` |
| U3.10 | `focus.tsx` | Integration | `displayTip` falls back to hardcoded string when `groqTips` is empty AND `lastTip` is null |
| U3.11 | `focus.tsx` | Integration | Tip updates when `groqTips[0]` changes → `lastTip` also persists |
| U3.12 | `StressGauge` | Visual | All 4 stress levels render in green palette (#35e27e, #2DBD6A, #1F9952, #157A3E) |
| U3.13 | `StressGauge` | Visual | No purple/orange/red colors appear in stress gauge |
| U3.14 | `PanicInterventionBanner` | Unit | Component renders without props; does not crash when not triggered |
| U3.15 | `index.tsx` | Regression | Main screen matches last GitHub commit state (no unauthorised changes) |

### Edge Cases to Cover
- `gaugeZoneH === 0` before layout measured → FocusGauge renders with default size=180, not zero-size
- Screen height < 700px (small phone) → card doesn't push features bar off-screen
- `focus.focusScore` is NaN → `Math.round(NaN)` shows as `NaN` — validate input sanitisation upstream

### Example Inputs / Outputs

**FocusGauge visual test:**
```tsx
render(<FocusGauge value={30} level="drifting" size={200} />);
// Expect: NO text containing "Drifting" or "DRIFTING" in output
// Expect: score text shows "30"
// Expect: ring color matches getFocusColor('drifting')
```

### Fix Validation
- Run `npx tsc --noEmit` — 0 TypeScript errors
- Device test on both small (iPhone SE) and large (iPhone Pro Max) screen sizes
- Screenshot each gauge state (sharp/steady/drifting/scattered) and confirm colors

---

## Member 4 — Auth, Hooks, & Integration Pipeline

**Scope:** `useAuth.ts`, `useWellness.tsx`, `db.ts`, `healthConnect.ts`, `integration.test.ts`, `app/auth/`, `app/screening/`

### What to Test

| Test ID | Component | Type | What to Verify |
|---|---|---|---|
| I4.1 | `useWellness.tsx` | Integration | Focus model loads on startup without blocking the main thread |
| I4.2 | `useWellness.tsx` | Integration | Inference runs every 5 seconds (mock timers, advance by 5s) |
| I4.3 | `useWellness.tsx` | Integration | Groq tip is triggered ONLY when score changes ≥ 7 points |
| I4.4 | `useWellness.tsx` | Integration | Groq tip is NOT retriggered if score changes < 7 points |
| I4.5 | `useWellness.tsx` | State | `focus.groqLoading` toggles true → false after Groq call resolves |
| I4.6 | `useAuth.ts` | Unit | Session persists after app reload (AsyncStorage mock) |
| I4.7 | `useAuth.ts` | Unit | `signOut()` clears session and redirects to login |
| I4.8 | `useAuth.ts` | Unit | Expired session token → user redirected to login, not silent failure |
| I4.9 | `db.ts` | Unit | Health snapshot is written and read back correctly |
| I4.10 | `healthConnect.ts` | Unit | Returns mock data when watch is not connected (no crash) |
| I4.11 | `integration.test.ts` | Integration | Full pipeline: raw biometrics → feature extraction → model inference → score |
| I4.12 | `app/auth/signup.tsx` | E2E | Empty email/password → validation error shown, no API call made |
| I4.13 | `app/auth/signup.tsx` | E2E | Valid credentials → navigates to main app |
| I4.14 | `app/screening/gad7.tsx` | E2E | All 7 questions answered → score computed and stored |
| I4.15 | `app/screening/phq9.tsx` | E2E | All 9 questions answered → PHQ-9 score computed and stored |
| I4.16 | `useWellness.tsx` | Error | Model fails to load (corrupt JSON) → app doesn't crash, shows error state |

### Edge Cases to Cover
- Watch disconnects mid-session → inference loop continues with last known data, no crash
- Groq call fires while previous call is still in-flight → no double-call, `groqLoading` stays true
- User denies health permissions → `healthConnect.ts` returns empty values, not throws
- PHQ-9 / GAD-7 score of 0 (no symptoms) → stored correctly, no divide-by-zero

### Example Inputs / Outputs

**Groq 7-point threshold:**
```ts
// Score starts at 60
// advance timer → score becomes 64 (delta = 4) → Groq NOT called
// advance timer → score becomes 67 (delta = 7) → Groq IS called
// Expected: callGroq mock called exactly once
```

**Auth session expiry:**
```ts
// Mock AsyncStorage to return expired JWT
// Call useAuth() initialisation
// Expected: navigation.replace('/(auth)/login') called
```

### Fix Validation
- Run `npx jest __tests__/integration.test.ts` — 0 failures
- Run full test suite: `npx jest --passWithNoTests` — confirm no regressions introduced

---

## Shared Quality Gates (All Members)

Before marking any test area complete:

- [ ] `npx tsc --noEmit` → 0 TypeScript errors
- [ ] All tests in relevant `__tests__/` files pass
- [ ] No `console.error` outputs in test runner that aren't mocked
- [ ] No hardcoded API keys or secrets in any file
- [ ] Features marked as "R&D" (anxiety model, `AnxietyGauge`) are NOT rendered in any screen

---

## Bug Tracking

| Bug ID | Description | Severity | Owner | Status |
|---|---|---|---|---|
| B1 | Login crash on null session token | High | Member 4 | Fixed — null check added in `useAuth.ts` |
| B2 | FocusGauge shows "Drifting" label in circle | Medium | Member 3 | Fixed — removed `LEVEL_LABELS` pill |
| B3 | Groq returned 3 bullets instead of 1 tip | Medium | Member 2 | Fixed — `max_tokens: 60`, single-line prompt |
| B4 | LIVE SIGNALS bar overlapping card on small screens | High | Member 3 | Fixed — static `cardHeight` from screen dimensions |
| B5 | Stress gauge using purple/orange/red colors | Low | Member 3 | Fixed — unified green palette in `mockData.ts` |

---

## Test Run Commands

```bash
# Run all tests
npx jest --passWithNoTests

# Run by area
npx jest __tests__/stressModel.test.ts           # Member 1 — stress model
npx jest __tests__/proof_different_readings.test.ts  # Member 1 — model separation
npx jest __tests__/recommendations.test.ts       # Member 2 — AI tips
npx jest __tests__/integration.test.ts           # Member 4 — pipeline
npx jest __tests__/featureEngineering.test.ts    # Member 1 — features

# TypeScript check (all members)
npx tsc --noEmit
```
