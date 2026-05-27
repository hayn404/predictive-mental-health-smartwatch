/**
 * __tests__/depressionRisk.test.ts
 *
 * Unit tests for C09 — Depression Risk Analyzer
 *
 * Run with: npx jest __tests__/depressionRisk.test.ts
 */

import {
  computeDepressionRisk,
  RECOMMENDATIONS_CALLER_TOKEN,
  type DepressionRiskInput,
} from '../services/ai/depressionRisk';

// ─── Fixture helpers ──────────────────────────────────────────────────────────

/** Healthy baseline — all metrics in normal range */
function healthyInput(overrides: Partial<DepressionRiskInput> = {}): DepressionRiskInput {
  return {
    hrv_rmssd_14day_mean: 42,
    hrv_rmssd_14day_std: 8,
    current_hrv_rmssd: 40,
    stress_score_7day_mean: 35,
    hr_resting_14day_mean: 68,
    sleep_quality_7day_mean: 72,
    sleep_consistency_score: 70,
    deep_sleep_percentage_7day: 18,
    sleep_onset_latency_7day_mean: 12,
    activity_variance_7day: 0.25,
    circadian_stability: 0.75,
    checkin_sentiment_7day_mean: 0.15,
    checkin_count_7days: 5,
    ...overrides,
  };
}

/** High-risk profile — multiple deteriorated metrics */
function highRiskInput(overrides: Partial<DepressionRiskInput> = {}): DepressionRiskInput {
  return {
    hrv_rmssd_14day_mean: 42,
    hrv_rmssd_14day_std: 8,
    current_hrv_rmssd: 18,         // z ≈ -3.0, severely low HRV
    stress_score_7day_mean: 72,    // sustained high stress
    hr_resting_14day_mean: 78,
    sleep_quality_7day_mean: 32,   // poor sleep quality
    sleep_consistency_score: 30,   // irregular bedtimes
    deep_sleep_percentage_7day: 6, // very low deep sleep
    sleep_onset_latency_7day_mean: 40, // long to fall asleep
    activity_variance_7day: 0.75,
    circadian_stability: 0.3,
    checkin_sentiment_7day_mean: -0.55,
    checkin_count_7days: 4,
    ...overrides,
  };
}

// ─── Access control ───────────────────────────────────────────────────────────

describe('C09 access control', () => {
  test('returns default flags when no caller token provided', () => {
    const result = computeDepressionRisk(highRiskInput(), '');
    expect(result._internalRiskIndex).toBe(0);
    expect(result.useGentleTone).toBe(false);
    expect(result.insufficientHistory).toBe(true);
  });

  test('returns default flags when wrong caller token provided', () => {
    const result = computeDepressionRisk(highRiskInput(), 'C07_STRESS_ENGINE');
    expect(result._internalRiskIndex).toBe(0);
    expect(result.useGentleTone).toBe(false);
  });

  test('returns meaningful flags when correct caller token provided', () => {
    const result = computeDepressionRisk(highRiskInput(), RECOMMENDATIONS_CALLER_TOKEN);
    expect(result._internalRiskIndex).toBeGreaterThan(0);
    expect(result.insufficientHistory).toBe(false);
  });
});

// ─── Insufficient data ────────────────────────────────────────────────────────

describe('C09 insufficient data handling', () => {
  test('returns insufficientHistory=true when no baseline exists', () => {
    const result = computeDepressionRisk(
      healthyInput({ hrv_rmssd_14day_mean: 0, hrv_rmssd_14day_std: 0 }),
      RECOMMENDATIONS_CALLER_TOKEN
    );
    expect(result.insufficientHistory).toBe(true);
    expect(result._internalRiskIndex).toBe(0);
  });

  test('returns insufficientHistory=true when sleep data missing', () => {
    const result = computeDepressionRisk(
      healthyInput({ sleep_quality_7day_mean: 0 }),
      RECOMMENDATIONS_CALLER_TOKEN
    );
    expect(result.insufficientHistory).toBe(true);
  });

  test('returns insufficientHistory=true when stress data missing', () => {
    const result = computeDepressionRisk(
      healthyInput({ stress_score_7day_mean: 0 }),
      RECOMMENDATIONS_CALLER_TOKEN
    );
    expect(result.insufficientHistory).toBe(true);
  });
});

// ─── Healthy profile — all flags off ─────────────────────────────────────────

describe('C09 healthy profile', () => {
  test('risk index is low for healthy metrics', () => {
    const result = computeDepressionRisk(healthyInput(), RECOMMENDATIONS_CALLER_TOKEN);
    expect(result._internalRiskIndex).toBeLessThan(35);
  });

  test('all recommendation flags are false for healthy profile', () => {
    const result = computeDepressionRisk(healthyInput(), RECOMMENDATIONS_CALLER_TOKEN);
    expect(result.useGentleTone).toBe(false);
    expect(result.prioritiseSocialAndRoutine).toBe(false);
    expect(result.suggestSocialSupport).toBe(false);
    expect(result.suppressHighIntensityInterventions).toBe(false);
    expect(result.insufficientHistory).toBe(false);
  });
});

// ─── HRV scoring ─────────────────────────────────────────────────────────────

describe('C09 HRV component scoring', () => {
  test('normal HRV (within 0.5 SD) contributes 0 to risk', () => {
    const normal = computeDepressionRisk(
      healthyInput({ current_hrv_rmssd: 40 }), // z = -0.25
      RECOMMENDATIONS_CALLER_TOKEN
    );
    const highNormal = computeDepressionRisk(
      healthyInput({ current_hrv_rmssd: 45 }), // z = +0.375
      RECOMMENDATIONS_CALLER_TOKEN
    );
    expect(normal._internalRiskIndex).toBeLessThanOrEqual(highNormal._internalRiskIndex + 1);
  });

  test('severely low HRV (z < -2) triggers suppressHighIntensityInterventions', () => {
    const result = computeDepressionRisk(
      healthyInput({ current_hrv_rmssd: 16 }), // z = (16-42)/8 = -3.25
      RECOMMENDATIONS_CALLER_TOKEN
    );
    expect(result.suppressHighIntensityInterventions).toBe(true);
  });

  test('HRV risk increases as RMSSD drops further below baseline', () => {
    const mild = computeDepressionRisk(
      healthyInput({ current_hrv_rmssd: 34 }), // z ≈ -1.0
      RECOMMENDATIONS_CALLER_TOKEN
    );
    const severe = computeDepressionRisk(
      healthyInput({ current_hrv_rmssd: 20 }), // z ≈ -2.75
      RECOMMENDATIONS_CALLER_TOKEN
    );
    expect(severe._internalRiskIndex).toBeGreaterThan(mild._internalRiskIndex);
  });
});

// ─── Sleep scoring ────────────────────────────────────────────────────────────

describe('C09 sleep disruption scoring', () => {
  test('poor sleep quality increases risk index', () => {
    const good = computeDepressionRisk(healthyInput({ sleep_quality_7day_mean: 72 }), RECOMMENDATIONS_CALLER_TOKEN);
    const poor = computeDepressionRisk(healthyInput({ sleep_quality_7day_mean: 32 }), RECOMMENDATIONS_CALLER_TOKEN);
    expect(poor._internalRiskIndex).toBeGreaterThan(good._internalRiskIndex);
  });

  test('irregular bedtimes (low consistency) increase risk', () => {
    const consistent = computeDepressionRisk(healthyInput({ sleep_consistency_score: 80 }), RECOMMENDATIONS_CALLER_TOKEN);
    const irregular = computeDepressionRisk(healthyInput({ sleep_consistency_score: 25 }), RECOMMENDATIONS_CALLER_TOKEN);
    expect(irregular._internalRiskIndex).toBeGreaterThan(consistent._internalRiskIndex);
  });

  test('long sleep onset latency adds to risk', () => {
    const fast = computeDepressionRisk(healthyInput({ sleep_onset_latency_7day_mean: 8 }), RECOMMENDATIONS_CALLER_TOKEN);
    const slow = computeDepressionRisk(healthyInput({ sleep_onset_latency_7day_mean: 40 }), RECOMMENDATIONS_CALLER_TOKEN);
    expect(slow._internalRiskIndex).toBeGreaterThan(fast._internalRiskIndex);
  });
});

// ─── High-risk profile — all flags on ────────────────────────────────────────

describe('C09 high-risk profile', () => {
  test('risk index is high for deteriorated metrics', () => {
    const result = computeDepressionRisk(highRiskInput(), RECOMMENDATIONS_CALLER_TOKEN);
    expect(result._internalRiskIndex).toBeGreaterThan(55);
  });

  test('useGentleTone is true at high risk', () => {
    const result = computeDepressionRisk(highRiskInput(), RECOMMENDATIONS_CALLER_TOKEN);
    expect(result.useGentleTone).toBe(true);
  });

  test('prioritiseSocialAndRoutine is true at high risk', () => {
    const result = computeDepressionRisk(highRiskInput(), RECOMMENDATIONS_CALLER_TOKEN);
    expect(result.prioritiseSocialAndRoutine).toBe(true);
  });

  test('suggestSocialSupport is true at high risk', () => {
    const result = computeDepressionRisk(highRiskInput(), RECOMMENDATIONS_CALLER_TOKEN);
    expect(result.suggestSocialSupport).toBe(true);
  });

  test('suppressHighIntensityInterventions is true at high risk', () => {
    const result = computeDepressionRisk(highRiskInput(), RECOMMENDATIONS_CALLER_TOKEN);
    expect(result.suppressHighIntensityInterventions).toBe(true);
  });
});

// ─── Risk index bounds ────────────────────────────────────────────────────────

describe('C09 output bounds', () => {
  test('risk index is always 0–100', () => {
    const low = computeDepressionRisk(healthyInput(), RECOMMENDATIONS_CALLER_TOKEN);
    const high = computeDepressionRisk(highRiskInput(), RECOMMENDATIONS_CALLER_TOKEN);
    expect(low._internalRiskIndex).toBeGreaterThanOrEqual(0);
    expect(high._internalRiskIndex).toBeLessThanOrEqual(100);
  });

  test('all flag values are boolean', () => {
    const result = computeDepressionRisk(healthyInput(), RECOMMENDATIONS_CALLER_TOKEN);
    expect(typeof result.useGentleTone).toBe('boolean');
    expect(typeof result.prioritiseSocialAndRoutine).toBe('boolean');
    expect(typeof result.suggestSocialSupport).toBe('boolean');
    expect(typeof result.suppressHighIntensityInterventions).toBe('boolean');
    expect(typeof result.insufficientHistory).toBe('boolean');
  });
});

// ─── Sentiment scoring ────────────────────────────────────────────────────────

describe('C09 sentiment scoring', () => {
  test('insufficient check-ins (<3) contribute 0 sentiment score', () => {
    const withCheckins = computeDepressionRisk(
      healthyInput({ checkin_sentiment_7day_mean: -0.7, checkin_count_7days: 5 }),
      RECOMMENDATIONS_CALLER_TOKEN
    );
    const withoutCheckins = computeDepressionRisk(
      healthyInput({ checkin_sentiment_7day_mean: -0.7, checkin_count_7days: 2 }),
      RECOMMENDATIONS_CALLER_TOKEN
    );
    expect(withCheckins._internalRiskIndex).toBeGreaterThan(withoutCheckins._internalRiskIndex);
  });

  test('strongly negative sentiment increases risk', () => {
    const positive = computeDepressionRisk(
      healthyInput({ checkin_sentiment_7day_mean: 0.3, checkin_count_7days: 5 }),
      RECOMMENDATIONS_CALLER_TOKEN
    );
    const negative = computeDepressionRisk(
      healthyInput({ checkin_sentiment_7day_mean: -0.65, checkin_count_7days: 5 }),
      RECOMMENDATIONS_CALLER_TOKEN
    );
    expect(negative._internalRiskIndex).toBeGreaterThan(positive._internalRiskIndex);
  });
});
