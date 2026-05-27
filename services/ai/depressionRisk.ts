/**
 * services/ai/depressionRisk.ts
 *
 * C09 — Depression Risk Analyzer (Internal-Only)
 *
 * IMPORTANT: This module is INTERNAL ONLY.
 * The risk score it produces is NEVER shown to the user in any form.
 * It is used solely to adjust the intensity and type of recommendations
 * produced by the Recommendations Engine (C10).
 *
 * Clinical rationale:
 * Depression and sustained stress share overlapping physiological signatures
 * (chronically depressed HRV, disrupted sleep architecture, circadian
 * instability) but require different recommendation strategies. A user
 * showing acute stress gets breathing exercises; a user showing prolonged
 * low HRV + poor sleep consistency + social withdrawal indicators gets
 * gentler, socially oriented interventions and a more conservative tone.
 *
 * This module does NOT:
 *  - Diagnose depression
 *  - Display any risk score, label, or indicator to the user
 *  - Log risk scores to any analytics or cloud service
 *  - Use risk scores to alter the app's visual appearance
 *
 * This module DOES:
 *  - Compute an internal composite risk index (0–100) from existing features
 *  - Produce non-diagnostic risk flags consumed only by recommendations.ts
 *  - Apply strict access control: only recommendations.ts may call this module
 *
 * Access control enforcement: the module checks its call stack context via
 * a caller token. Any component other than the Recommendations Engine
 * receives only a null result.
 */

// ─── Types ────────────────────────────────────────────────────────────────────

/**
 * Input features — all sourced from C05 (physiological) and C08 (sleep).
 * No additional data collection required.
 */
export interface DepressionRiskInput {
  // From C05 / C07
  hrv_rmssd_14day_mean: number;        // Personal 14-day mean RMSSD
  hrv_rmssd_14day_std: number;         // Std of RMSSD over 14 days
  current_hrv_rmssd: number;           // Today's RMSSD
  stress_score_7day_mean: number;      // Average stress score over 7 days
  hr_resting_14day_mean: number;       // 14-day resting HR mean

  // From C08
  sleep_quality_7day_mean: number;     // Average sleep quality score (0–100)
  sleep_consistency_score: number;     // Bedtime consistency score (0–100)
  deep_sleep_percentage_7day: number;  // % of deep sleep averaged over 7 days
  sleep_onset_latency_7day_mean: number; // Minutes to fall asleep, 7-day avg

  // Behavioral (from C06)
  activity_variance_7day: number;      // How much daily steps vary (0–1 normalized)
  circadian_stability: number;         // Regularity of circadian pattern (0–1)

  // Voice check-in derived (optional — from C11, may be undefined)
  checkin_sentiment_7day_mean?: number; // -1 to +1; undefined if no check-ins yet
  checkin_count_7days?: number;         // How many check-ins in past 7 days
}

/**
 * Output — risk flags for C10 (Recommendations Engine) only.
 * Intentionally avoids any clinical terminology that could be
 * interpreted as a diagnosis by a developer reading the code
 * and accidentally surfacing it.
 */
export interface DepressionRiskFlags {
  /**
   * Composite internal index 0–100.
   * MUST NOT be displayed, logged externally, or used outside C10.
   */
  _internalRiskIndex: number;

  /** Recommendation tone should be gentler and more supportive */
  useGentleTone: boolean;

  /** Prioritise social connection and routine-building recommendations */
  prioritiseSocialAndRoutine: boolean;

  /** Flag to add a "consider speaking to someone you trust" recommendation */
  suggestSocialSupport: boolean;

  /** Suppress high-intensity exercise recommendations */
  suppressHighIntensityInterventions: boolean;

  /** Data quality flag — insufficient history to compute meaningful risk */
  insufficientHistory: boolean;
}

// ─── Caller access control ────────────────────────────────────────────────────

/**
 * Only the Recommendations Engine should call computeDepressionRisk.
 * Pass RECOMMENDATIONS_CALLER_TOKEN to authenticate.
 * Any other caller receives a null-safe default result.
 */
export const RECOMMENDATIONS_CALLER_TOKEN = 'C10_RECOMMENDATIONS_ENGINE_INTERNAL';

const DEFAULT_FLAGS: DepressionRiskFlags = {
  _internalRiskIndex: 0,
  useGentleTone: false,
  prioritiseSocialAndRoutine: false,
  suggestSocialSupport: false,
  suppressHighIntensityInterventions: false,
  insufficientHistory: true,
};

// ─── Feature scoring helpers ──────────────────────────────────────────────────

/**
 * Score chronically low HRV: RMSSD consistently below personal baseline
 * is the strongest physiological marker in this feature set.
 * Returns 0–30 (highest weight component).
 */
function scoreHRVChronicDepression(input: DepressionRiskInput): number {
  if (input.hrv_rmssd_14day_std < 1) return 0; // No baseline variance yet

  const zScore = (input.current_hrv_rmssd - input.hrv_rmssd_14day_mean) / input.hrv_rmssd_14day_std;

  if (zScore >= -0.5) return 0;          // Normal range
  if (zScore >= -1.0) return 5;          // Mildly below baseline
  if (zScore >= -1.5) return 12;         // Noticeably below baseline
  if (zScore >= -2.0) return 20;         // Significantly below
  return 30;                             // Severely depressed HRV
}

/**
 * Score sustained high stress: 7-day mean stress above 55 indicates
 * chronic, not acute, stress loading.
 * Returns 0–20.
 */
function scoreSustainedStress(input: DepressionRiskInput): number {
  const mean = input.stress_score_7day_mean;
  if (mean < 40) return 0;
  if (mean < 55) return 5;
  if (mean < 65) return 10;
  if (mean < 75) return 15;
  return 20;
}

/**
 * Score sleep disruption: poor quality + long sleep latency + reduced
 * deep sleep are consistently associated with mood disorders.
 * Returns 0–25.
 */
function scoreSleepDisruption(input: DepressionRiskInput): number {
  let score = 0;

  // Sleep quality
  if (input.sleep_quality_7day_mean < 40) score += 10;
  else if (input.sleep_quality_7day_mean < 55) score += 5;

  // Sleep consistency (irregular bedtimes)
  if (input.sleep_consistency_score < 40) score += 7;
  else if (input.sleep_consistency_score < 60) score += 3;

  // Reduced deep sleep (normal is 15–25% of total)
  if (input.deep_sleep_percentage_7day < 8) score += 5;
  else if (input.deep_sleep_percentage_7day < 13) score += 2;

  // Long sleep onset latency (>30 min is clinically significant)
  if (input.sleep_onset_latency_7day_mean > 30) score += 3;

  return Math.min(score, 25);
}

/**
 * Score circadian and activity disruption.
 * Returns 0–15.
 */
function scoreCircadianDisruption(input: DepressionRiskInput): number {
  let score = 0;

  if (input.circadian_stability < 0.4) score += 8;
  else if (input.circadian_stability < 0.6) score += 4;

  // High activity variance (boom-bust pattern) can signal mood instability
  if (input.activity_variance_7day > 0.7) score += 4;
  else if (input.activity_variance_7day > 0.5) score += 2;

  return Math.min(score, 15);
}

/**
 * Score voice check-in sentiment trend.
 * Returns 0–10.
 */
function scoreSentimentTrend(input: DepressionRiskInput): number {
  if (
    input.checkin_sentiment_7day_mean === undefined ||
    input.checkin_count_7days === undefined ||
    input.checkin_count_7days < 3  // Need at least 3 check-ins for trend
  ) {
    return 0;
  }

  const sentiment = input.checkin_sentiment_7day_mean;
  if (sentiment > -0.2) return 0;
  if (sentiment > -0.4) return 3;
  if (sentiment > -0.6) return 6;
  return 10;
}

// ─── Minimum data requirements ────────────────────────────────────────────────

function hasMinimumHistory(input: DepressionRiskInput): boolean {
  return (
    input.hrv_rmssd_14day_mean > 0 &&
    input.hrv_rmssd_14day_std > 0 &&
    input.sleep_quality_7day_mean > 0 &&
    input.stress_score_7day_mean > 0
  );
}

// ─── Main export ──────────────────────────────────────────────────────────────

/**
 * Compute internal depression risk flags from physiological and
 * behavioural features. Only callable by the Recommendations Engine.
 *
 * @param input   Feature vector from C05, C06, C08
 * @param callerToken  Must equal RECOMMENDATIONS_CALLER_TOKEN
 * @returns DepressionRiskFlags, or default (all false) if caller is unauthorized
 */
export function computeDepressionRisk(
  input: DepressionRiskInput,
  callerToken: string
): DepressionRiskFlags {
  // Access control: only C10 may receive meaningful output
  if (callerToken !== RECOMMENDATIONS_CALLER_TOKEN) {
    console.warn('[C09] Unauthorized access attempt to DepressionRiskAnalyzer — returning defaults');
    return { ...DEFAULT_FLAGS };
  }

  // Minimum data check
  if (!hasMinimumHistory(input)) {
    return { ...DEFAULT_FLAGS, insufficientHistory: true };
  }

  // Compute component scores
  const hrvScore = scoreHRVChronicDepression(input);
  const stressScore = scoreSustainedStress(input);
  const sleepScore = scoreSleepDisruption(input);
  const circadianScore = scoreCircadianDisruption(input);
  const sentimentScore = scoreSentimentTrend(input);

  const riskIndex = Math.min(
    100,
    hrvScore + stressScore + sleepScore + circadianScore + sentimentScore
  );

  // Derive recommendation flags from risk thresholds
  // These thresholds are conservative by design
  const flags: DepressionRiskFlags = {
    _internalRiskIndex: riskIndex,
    insufficientHistory: false,

    // Gentle tone from moderate risk onward
    useGentleTone: riskIndex >= 35,

    // Prioritise social + routine from high risk
    prioritiseSocialAndRoutine: riskIndex >= 50,

    // Suggest social support at significant risk
    // (kept intentionally vague — no clinical language)
    suggestSocialSupport: riskIndex >= 60,

    // Suppress high-intensity interventions when HRV is
    // already depleted (additional stress load is counterproductive)
    suppressHighIntensityInterventions: hrvScore >= 20 || riskIndex >= 55,
  };

  return flags;
}
