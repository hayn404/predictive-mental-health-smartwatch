/**
 * Seren AI — Emotional Maturity Tracker
 * ========================================
 * "How well is Seren reading me?" — a structured answer instead of a vibe.
 *
 * Same shape as services/observability/modelMonitor.ts (pure, in-memory,
 * no backend): reads checkinHistory + which providers are currently
 * configured, and returns a snapshot of how mature the understanding
 * pipeline actually is right now — not a feature checklist, a *quality*
 * checklist.
 *
 * Two consumers read the same snapshot:
 *  - docs/MATURITY_CHECKLIST.md (dev-facing — blunt, technical)
 *  - components/ui/EmotionalMaturityCard.tsx (user-facing — warm, framed
 *    as "how well Seren is getting to know you", via useWellness())
 *
 * Several dimensions need CheckinAnalysis.meta (asrProvider/llmProvider/
 * latencyMs — see types.ts) which only exists on check-ins saved after the
 * v2 DB migration (db.ts). Older rows are handled gracefully — they just
 * can't move certain dimensions past level 1.
 */

import { CheckinAnalysis, EmotionScores } from './types';

// ============================================================
// Types
// ============================================================

export type MaturityLevel = 0 | 1 | 2 | 3 | 4;

export interface MaturityDimension {
  id: string;
  label: string;
  level: MaturityLevel;
  /** What data actually produced this level — for the dev checklist's "Evidence" column */
  evidence: string;
  /** What would move this to the next level — for the dev checklist's "Path to next level" column */
  nextStep: string;
}

export interface MaturitySnapshot {
  /** 0-100, average of dimension levels normalized */
  score: number;
  /** Friendly, non-clinical label for the user-facing card */
  overallTier: string;
  dimensions: MaturityDimension[];
  /** Check-ins this snapshot was computed over */
  sampleSize: number;
}

export interface MaturityContext {
  llmConfigured: boolean;
  nemotronAvailable: boolean;
  whisperConfigured: boolean;
}

const LEVEL_LABELS: Record<MaturityLevel, string> = {
  0: 'Not yet measured',
  1: 'Heuristic',
  2: 'Calibrated',
  3: 'Consistent',
  4: 'Validated',
};

const TIER_LABELS = [
  'Just getting to know you',   // avg level 0-0.99
  'Learning your patterns',     // 1-1.99
  'Reading you well',           // 2-2.99
  'Deeply attuned',             // 3-3.99
  'Fully attuned',              // 4
];

const EMOTION_KEYS: (keyof EmotionScores)[] = [
  'joy', 'sadness', 'anxiety', 'anger', 'calm', 'fear', 'gratitude', 'fatigue',
];

// Phrases voiceAnalysis.ts / llmService.ts already use when body and words
// agree or disagree — used as a (heuristic, level-1-ceiling) proxy for
// body-mind cross-referencing until bodyMindAlignment is persisted directly.
const ALIGNMENT_PHRASES = [
  'both your words and body',
  'your body may be carrying',
  'your body might be telling',
  'body metrics look steady',
];

// ============================================================
// Public API
// ============================================================

export function computeEmotionalMaturity(
  checkins: CheckinAnalysis[],
  ctx: MaturityContext,
): MaturitySnapshot {
  const dimensions: MaturityDimension[] = [
    voiceCaptureQuality(checkins, ctx),
    understandingDepth(checkins, ctx),
    emotionalVocabularyBreadth(checkins),
    bodyMindCrossReferencing(checkins),
    conversationalResponsiveness(checkins),
    duetLatency(checkins),
    trendStability(checkins),
  ];

  const avgLevel = dimensions.reduce((sum, d) => sum + d.level, 0) / dimensions.length;
  const score = Math.round((avgLevel / 4) * 100);
  const tierIndex = Math.min(TIER_LABELS.length - 1, Math.floor(avgLevel));

  return {
    score,
    overallTier: TIER_LABELS[tierIndex],
    dimensions,
    sampleSize: checkins.length,
  };
}

// ============================================================
// Dimensions
// ============================================================

function voiceCaptureQuality(checkins: CheckinAnalysis[], ctx: MaturityContext): MaturityDimension {
  const withMeta = checkins.filter(c => c.meta?.inputMode === 'voice' && c.meta?.asrProvider);
  const id = 'voice_capture_quality';
  const label = 'Voice capture quality';

  if (withMeta.length === 0) {
    const level: MaturityLevel = ctx.nemotronAvailable || ctx.whisperConfigured ? 1 : 0;
    return {
      id, label, level,
      evidence: ctx.whisperConfigured || ctx.nemotronAvailable
        ? 'An ASR provider is configured, but no check-in has recorded which one transcribed it yet (meta.asrProvider missing — likely pre-v2 data).'
        : 'No ASR provider configured — voice check-ins fall back to manual text entry.',
      nextStep: 'Route transcription through nemotronAsrService/voiceAssistant.ts so each check-in records meta.asrProvider.',
    };
  }

  const nemotronCount = withMeta.filter(c => c.meta?.asrProvider === 'nemotron').length;
  const nemotronRate = nemotronCount / withMeta.length;
  let level: MaturityLevel = 1;
  if (nemotronRate >= 0.75) level = 4;
  else if (nemotronRate >= 0.4) level = 3;
  else if (withMeta.length >= 3) level = 2;

  return {
    id, label, level,
    evidence: `${nemotronCount}/${withMeta.length} recent voice check-ins used Nemotron ASR (rest used Whisper fallback).`,
    nextStep: level < 4
      ? 'Confirm the Nemotron relay health check is passing consistently (server/nemotron-relay/ GET /health) so fewer check-ins fall back to Whisper.'
      : 'At ceiling — Nemotron is the primary ASR path.',
  };
}

function understandingDepth(checkins: CheckinAnalysis[], ctx: MaturityContext): MaturityDimension {
  const id = 'understanding_depth';
  const label = 'Understanding depth (LLM vs. local fallback)';

  if (checkins.length === 0) {
    return { id, label, level: 0, evidence: 'No check-ins yet.', nextStep: 'Complete a check-in to start measuring.' };
  }

  const withMeta = checkins.filter(c => c.meta?.llmProvider);
  if (withMeta.length === 0) {
    const level: MaturityLevel = ctx.llmConfigured ? 1 : 0;
    return {
      id, label, level,
      evidence: ctx.llmConfigured
        ? 'An LLM provider is configured, but check-ins don\'t record which path analyzed them yet (meta.llmProvider missing — likely pre-v2 data).'
        : 'No LLM configured — every check-in uses the local VADER-style lexicon (voiceAnalysis.ts fallback).',
      nextStep: 'Have performCheckin() in useWellness.tsx stamp meta.llmProvider on each saved check-in.',
    };
  }

  const llmCount = withMeta.filter(c => c.meta?.llmProvider !== 'local').length;
  const llmRate = llmCount / withMeta.length;
  let level: MaturityLevel = 1;
  if (llmRate >= 0.9) level = 4;
  else if (llmRate >= 0.6) level = 3;
  else if (llmRate > 0) level = 2;

  return {
    id, label, level,
    evidence: `${llmCount}/${withMeta.length} recent check-ins were analyzed by the configured LLM; the rest used the local lexicon fallback.`,
    nextStep: level < 4
      ? 'Check LLM provider uptime/rate limits (llmService.ts) — frequent fallback usually means API failures, not missing config.'
      : 'At ceiling — LLM analysis is the primary path.',
  };
}

function emotionalVocabularyBreadth(checkins: CheckinAnalysis[]): MaturityDimension {
  const id = 'emotional_vocabulary_breadth';
  const label = 'Emotional vocabulary breadth';

  if (checkins.length === 0) {
    return { id, label, level: 0, evidence: 'No check-ins yet.', nextStep: 'Complete a check-in to start measuring.' };
  }

  const seen = new Set<string>();
  for (const c of checkins) {
    for (const key of EMOTION_KEYS) {
      if ((c.emotionScores?.[key] ?? 0) > 0.3) seen.add(key);
    }
  }
  const breadth = seen.size / EMOTION_KEYS.length;
  let level: MaturityLevel = 1;
  if (breadth >= 0.75) level = 4;
  else if (breadth >= 0.5) level = 3;
  else if (breadth >= 0.25) level = 2;

  return {
    id, label, level,
    evidence: `${seen.size}/${EMOTION_KEYS.length} distinct emotions detected with meaningful confidence across ${checkins.length} check-in(s).`,
    nextStep: level < 4
      ? 'More check-ins across varied moods (not just stressful ones) will naturally widen this — no code change needed.'
      : 'At ceiling — a broad emotional range is being captured.',
  };
}

function bodyMindCrossReferencing(checkins: CheckinAnalysis[]): MaturityDimension {
  const id = 'body_mind_cross_referencing';
  const label = 'Body-mind cross-referencing';

  if (checkins.length === 0) {
    return { id, label, level: 0, evidence: 'No check-ins yet.', nextStep: 'Complete a check-in to start measuring.' };
  }

  const withAlignmentInsight = checkins.filter(c =>
    c.keyInsights?.some(insight =>
      ALIGNMENT_PHRASES.some(phrase => insight.toLowerCase().includes(phrase)),
    ),
  ).length;

  // Heuristic, text-scan based — capped at level 1 until bodyMindAlignment
  // (already computed in llmService.ts's parsed response) is persisted as
  // its own field instead of being inferred back out of keyInsights strings.
  const level: MaturityLevel = withAlignmentInsight > 0 ? 1 : 0;

  return {
    id, label, level,
    evidence: `${withAlignmentInsight}/${checkins.length} check-ins surfaced a verbal/biometric alignment insight (detected via text match — heuristic).`,
    nextStep: 'Persist llmService.ts\'s parsed.bodyMindAlignment ("aligned"|"mixed"|"contradictory") as its own CheckinAnalysis/DB field instead of inferring it from insight text — would unlock levels 2-4 here.',
  };
}

function conversationalResponsiveness(checkins: CheckinAnalysis[]): MaturityDimension {
  const id = 'conversational_responsiveness';
  const label = 'Conversational responsiveness (follow-up questions)';

  if (checkins.length === 0) {
    return { id, label, level: 0, evidence: 'No check-ins yet.', nextStep: 'Complete a check-in to start measuring.' };
  }

  const withFollowUp = checkins.filter(c => !!c.suggestedFollowUp).length;
  const rate = withFollowUp / checkins.length;
  let level: MaturityLevel = 1;
  if (rate >= 0.6) level = 4;
  else if (rate >= 0.3) level = 3;
  else if (rate > 0) level = 2;

  return {
    id, label, level,
    evidence: `${withFollowUp}/${checkins.length} check-ins included a gentle follow-up question.`,
    nextStep: rate === 0
      ? 'If this is persistently 0 right after reinstalling the app, confirm the v2 DB migration ran (db.ts) — suggestedFollowUp wasn\'t persisted before it.'
      : 'Naturally improves as the LLM has more context (recentMood, stress trend) to decide when a follow-up is warranted.',
  };
}

function duetLatency(checkins: CheckinAnalysis[]): MaturityDimension {
  const id = 'duet_latency';
  const label = 'Voice duet latency (record → speak)';

  const withLatency = checkins.filter(c => typeof c.meta?.latencyMs === 'number');
  if (withLatency.length === 0) {
    return {
      id, label, level: 0,
      evidence: 'No latency data yet — only voiceAssistant.ts\'s full duet flow records this (text check-ins and the simple useCheckin() flow don\'t).',
      nextStep: 'Use the voice duet (useVoiceAssistant) at least once to start measuring end-to-end latency.',
    };
  }

  const avgMs = withLatency.reduce((sum, c) => sum + (c.meta!.latencyMs ?? 0), 0) / withLatency.length;
  let level: MaturityLevel = 1;
  if (avgMs <= 2500) level = 4;
  else if (avgMs <= 4500) level = 3;
  else if (avgMs <= 7000) level = 2;

  return {
    id, label, level,
    evidence: `Average end-to-end latency over ${withLatency.length} duet turn(s): ${Math.round(avgMs)}ms.`,
    nextStep: level < 4
      ? 'Biggest wins are usually ASR/LLM network round-trips — check whether Nemotron or Whisper is the bottleneck, and consider streaming ASR partials instead of waiting for the full transcript.'
      : 'At ceiling — latency feels conversational.',
  };
}

function trendStability(checkins: CheckinAnalysis[]): MaturityDimension {
  const id = 'trend_stability';
  const label = 'Trend stability (enough data to trust patterns)';

  const days = checkins.length > 0
    ? Math.ceil((Date.now() - checkins[checkins.length - 1].timestamp) / (24 * 60 * 60 * 1000))
    : 0;

  let level: MaturityLevel = 0;
  if (checkins.length >= 14 && days >= 14) level = 4;
  else if (checkins.length >= 7 && days >= 7) level = 3;
  else if (checkins.length >= 3) level = 2;
  else if (checkins.length >= 1) level = 1;

  return {
    id, label, level,
    evidence: `${checkins.length} check-in(s) spanning ${days} day(s).`,
    nextStep: level < 4
      ? 'No code change — this grows automatically as the user checks in regularly over time.'
      : 'At ceiling — trends (computeCheckinTrend) are backed by a solid history.',
  };
}
