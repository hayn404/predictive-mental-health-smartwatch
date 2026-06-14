/**
 * Runtime model-health / drift monitoring (gaps S6 / L8).
 *
 * Tracks a rolling window of recent on-device predictions and surfaces simple,
 * dependency-free health signals — so "how do you detect model degradation / bad outputs?"
 * has a concrete answer without any backend. Pure + in-memory; the UI (or Sentry, when a
 * DSN is set) can read getModelHealth().
 */

interface Sample {
  ts: number;
  score: number;       // 0-100
  confidence: number;  // 0-1
}

const MAX_SAMPLES = 200;
const MIN_FOR_HEALTH = 20;
const LOW_CONFIDENCE = 0.5;

const recent: Sample[] = [];

export interface ModelHealth {
  samples: number;
  meanConfidence: number;
  lowConfidenceRate: number; // fraction below LOW_CONFIDENCE
  degenerate: boolean;       // outputs stuck (≈zero variance) -> sensor/feature issue
  drift: boolean;            // a health signal worth surfacing
  reasons: string[];
}

/** Record one stress prediction. Cheap; call on every inference. */
export function recordStressPrediction(p: { stressScore: number; confidence: number; timestamp?: number }): void {
  recent.push({ ts: p.timestamp ?? 0, score: p.stressScore, confidence: p.confidence });
  if (recent.length > MAX_SAMPLES) recent.shift();
}

/** Compute current model-health signals over the rolling window. */
export function getModelHealth(): ModelHealth {
  const n = recent.length;
  if (n < MIN_FOR_HEALTH) {
    return { samples: n, meanConfidence: 1, lowConfidenceRate: 0, degenerate: false, drift: false, reasons: [] };
  }

  const meanConfidence = recent.reduce((s, x) => s + x.confidence, 0) / n;
  const lowConfidenceRate = recent.filter((x) => x.confidence < LOW_CONFIDENCE).length / n;

  const meanScore = recent.reduce((s, x) => s + x.score, 0) / n;
  const scoreVariance = recent.reduce((s, x) => s + (x.score - meanScore) ** 2, 0) / n;
  const degenerate = scoreVariance < 1; // ~no movement across 20+ windows

  const reasons: string[] = [];
  if (lowConfidenceRate > 0.5) reasons.push(`low confidence on ${Math.round(lowConfidenceRate * 100)}% of recent windows`);
  if (degenerate) reasons.push('stress score is not varying (possible sensor/feature issue)');

  return {
    samples: n,
    meanConfidence,
    lowConfidenceRate,
    degenerate,
    drift: reasons.length > 0,
    reasons,
  };
}

/** Test/reset hook. */
export function _resetModelMonitor(): void {
  recent.length = 0;
}
