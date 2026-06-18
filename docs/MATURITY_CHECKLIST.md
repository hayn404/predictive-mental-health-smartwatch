# Seren — Emotional Understanding Maturity Checklist

This tracks *how well* the voice/check-in pipeline understands the user — not
whether a feature exists (see `MVP_FEATURES_CHECKLIST.md` for that), but
whether it's actually working well in practice. It's the dev-facing view of
`services/ai/emotionalMaturity.ts`'s `MaturitySnapshot`; the same data drives
the user-facing "How well Seren is reading you" card
(`components/ui/EmotionalMaturityCard.tsx`), just with warmer copy there.

Levels: **0** Not yet measured · **1** Heuristic · **2** Calibrated · **3** Consistent · **4** Validated

| # | Dimension | Signal source | Level (now) | Notes |
|---|-----------|---------------|-------------|-------|
| 1 | Voice capture quality | `meta.asrProvider` per check-in (Nemotron vs. Whisper) | 1–2 | `nemotronAsrService.ts` + `server/nemotron-relay/` exist and `voiceAssistant.ts`'s `transcribeSpeech()` (Nemotron-first, Whisper-fallback) is wired into both `useCheckin()` and `useVoiceAssistant()`. Still capped low because nobody's deployed the relay with real NVIDIA credentials yet — until then every check-in falls back to Whisper, which is honest (correct fallback behavior), just not yet "mostly Nemotron". |
| 2 | Understanding depth (LLM vs. local fallback) | `meta.llmProvider` per check-in | 1–2 | Approximated from "is an LLM provider configured", not "did this specific call succeed" — `llmService.ts`'s internal try/catch fallback isn't currently reported back to the caller. |
| 3 | Emotional vocabulary breadth | `emotionScores` across `checkinHistory` | grows with use | No code gap — purely a function of how many check-ins exist and how varied they are. |
| 4 | Body-mind cross-referencing | text-scan of `keyInsights` for alignment phrases | capped at 1 | `llmService.ts` already computes `bodyMindAlignment` ("aligned"/"mixed"/"contradictory") in its parsed response but never persists it as its own field — it's baked into `keyInsights` strings instead. Persisting it directly would unlock levels 2–4. |
| 5 | Conversational responsiveness (follow-ups) | `suggestedFollowUp` presence rate | 0 until v2 migration | **Pre-existing bug, now fixed**: `db.ts`'s `saveCheckin`/`getRecentCheckins` never persisted `themes`, `emotionalIntensity`, `empathyResponse`, or `suggestedFollowUp` — they were computed every check-in and then silently dropped on the next app load. Fixed via the `SCHEMA_VERSION = 2` migration; verify on a real device that follow-ups survive a restart. |
| 6 | Voice duet latency (record → speak) | `meta.latencyMs` | 0 until exercised | `voiceAssistant.ts` + `useVoiceAssistant()` (hooks/useHealthData.ts) ship the full record→transcribe→analyze→speak loop and record latency via `recordVoiceTurnMeta` once a turn finishes — but no UI screen calls `useVoiceAssistant()` yet (only `useCheckin()`, which doesn't include a "speak back" step). Wire a screen to it to start collecting real numbers. |
| 7 | Trend stability (enough data to trust patterns) | count + days covered in `checkinHistory` | grows with use | No code gap — `computeCheckinTrend()` already exists; this just tracks whether there's enough history behind it. |

## How to read this

Dimensions 3 and 7 improve automatically just from people using the app —
nothing to build. Dimension 1 needs someone to actually deploy
`server/nemotron-relay/` with real NVIDIA credentials (see its README) —
the code path is ready and tested (proto loading, gRPC client construction,
and the relay's HTTP contract are all verified; the live NVCF round-trip
isn't, since that requires real credentials). Dimension 6 needs a UI screen
wired to `useVoiceAssistant()` — it's not yet surfaced anywhere, unlike
`useCheckin()` which `app/(tabs)/checkin.tsx` already uses. Dimension 4
needs a small, well-scoped schema addition (persist `bodyMindAlignment`
directly instead of inferring it from insight text). Dimension 5 was a real
bug, not a missing feature — it's the kind of thing this checklist exists
to catch: the LLM was doing good work that the app was throwing away on
every restart.

## Updating this doc

When a dimension's underlying mechanism changes (e.g. `nemotronAsrService.ts`
ships, or `bodyMindAlignment` gets persisted), update the "Level (now)" column
and prune the note once it's no longer a gap. `emotionalMaturity.ts`'s
`evidence`/`nextStep` strings are generated live from real data and are the
fastest way to sanity-check this table is still accurate — they're visible in
dev builds via `console.log(computeEmotionalMaturity(checkinHistory, ctx))`.
