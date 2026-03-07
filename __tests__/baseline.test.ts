/**
 * Unit Tests — Personal Baseline & Anomaly Detection
 * =====================================================
 * Tests baseline computation, recomputation triggers,
 * and anomaly detection against personal baselines.
 */

import {
  computeBaseline,
  shouldRecomputeBaseline,
  detectAnomalies,
} from '@/services/ai/baseline';
import type { BiometricFeatureVector, SleepAnalysis, PersonalBaseline } from '@/services/ai/types';

// ============================================================
// Test Helpers
// ============================================================

function makeFeatureWindow(overrides: Partial<BiometricFeatureVector> = {}, ts?: number): BiometricFeatureVector {
  return {
    timestamp: ts ?? Date.now(),
    windowSeconds: 300,
    meanRR: 857, sdnn: 50, rmssd: 45, pnn50: 25, pnn20: 60,
    hrMean: 70, hrStd: 3, hrRange: 10, cvRR: 0.058,
    vlfPower: 500, lfPower: 400, hfPower: 600, lfHfRatio: 0.67,
    totalPower: 1500, lfNorm: 40, hfNorm: 60,
    sd1: 32, sd2: 45, sd1sd2Ratio: 0.71, sampleEntropy: 1.5, dfaAlpha1: 1.0,
    tempMean: 33.5, tempSlope: 0, tempStd: 0.1, tempRange: 0.3,
    accelMagnitudeMean: 1.0, accelMagnitudeStd: 0.02, stepCount: 100, activityType: 'sedentary',
    ...overrides,
  };
}

function makeSleepSession(overrides: Partial<SleepAnalysis> = {}): SleepAnalysis {
  return {
    date: '2026-03-07',
    sessionStart: new Date('2026-03-06T23:00:00').getTime(),
    sessionEnd: new Date('2026-03-07T07:00:00').getTime(),
    totalInBedMin: 480, totalSleepMin: 450, onsetLatencyMin: 10, wasoMin: 10,
    lightSleepMin: 200, deepSleepMin: 100, remSleepMin: 110, awakeMin: 20,
    deepSleepPct: 0.22, remSleepPct: 0.24, lightSleepPct: 0.44,
    sleepEfficiency: 0.94, fragmentationIndex: 1, awakeningCount: 1,
    avgHrSleep: 55, minHrSleep: 48, avgHrvSleep: 55, maxHrvSleep: 70,
    qualityScore: 82, recoveryScore: 75, consistencyScore: 85,
    ...overrides,
  };
}

function generateWindows(count: number): BiometricFeatureVector[] {
  return Array.from({ length: count }, (_, i) =>
    makeFeatureWindow({
      hrMean: 65 + Math.random() * 15,
      rmssd: 35 + Math.random() * 20,
      sdnn: 40 + Math.random() * 20,
      tempMean: 33 + Math.random(),
    }, Date.now() - (count - i) * 300000)
  );
}

// ============================================================
// computeBaseline
// ============================================================

describe('computeBaseline', () => {
  test('returns null with insufficient samples (<50)', () => {
    const windows = generateWindows(30);
    const result = computeBaseline('user1', windows, []);
    expect(result).toBeNull();
  });

  test('returns a valid baseline with sufficient data', () => {
    const windows = generateWindows(100);
    const sleepSessions = [makeSleepSession(), makeSleepSession()];
    const result = computeBaseline('user1', windows, sleepSessions);

    expect(result).not.toBeNull();
    expect(result!.userId).toBe('user1');
    expect(result!.sampleCount).toBe(100);
    expect(result!.periodDays).toBe(14);
  });

  test('computes realistic HR baselines', () => {
    const windows = generateWindows(100);
    const result = computeBaseline('user1', windows, []);

    expect(result!.restingHrMean).toBeGreaterThan(40);
    expect(result!.restingHrMean).toBeLessThan(120);
  });

  test('computes HRV baselines', () => {
    const windows = generateWindows(100);
    const result = computeBaseline('user1', windows, []);

    expect(result!.rmssdMean).toBeGreaterThan(0);
    expect(result!.sdnnMean).toBeGreaterThan(0);
    expect(result!.rmssdStd).toBeGreaterThanOrEqual(0);
  });

  test('computes sleep baselines from sessions', () => {
    const windows = generateWindows(100);
    const sessions = [
      makeSleepSession({ totalSleepMin: 420, sleepEfficiency: 0.88, qualityScore: 75 }),
      makeSleepSession({ totalSleepMin: 450, sleepEfficiency: 0.92, qualityScore: 82 }),
    ];
    const result = computeBaseline('user1', windows, sessions);

    expect(result!.avgSleepDurationMin).toBeCloseTo(435, 0);
    expect(result!.avgSleepEfficiency).toBeCloseTo(0.90, 1);
  });

  test('defaults sleep baselines when no sessions provided', () => {
    const windows = generateWindows(100);
    const result = computeBaseline('user1', windows, []);

    expect(result!.typicalBedtimeHour).toBe(23);
    expect(result!.typicalWakeHour).toBe(7);
    expect(result!.avgSleepDurationMin).toBe(0);
  });

  test('computes average daily steps', () => {
    const windows = generateWindows(100);
    const result = computeBaseline('user1', windows, []);
    expect(result!.avgDailySteps).toBeGreaterThanOrEqual(0);
  });
});

// ============================================================
// shouldRecomputeBaseline
// ============================================================

describe('shouldRecomputeBaseline', () => {
  test('returns true when baseline is null', () => {
    expect(shouldRecomputeBaseline(null)).toBe(true);
  });

  test('returns true when baseline is >24h old', () => {
    const old: PersonalBaseline = {
      userId: 'user1', computedAt: Date.now() - 25 * 60 * 60 * 1000,
      periodDays: 14, sampleCount: 100,
      restingHrMean: 65, restingHrStd: 5, sleepingHrMean: 55,
      rmssdMean: 45, rmssdStd: 10, sdnnMean: 50, sdnnStd: 12, nocturnalRmssdMean: 55,
      typicalBedtimeHour: 23, typicalWakeHour: 7, avgSleepDurationMin: 450,
      avgSleepEfficiency: 0.88, avgSleepQuality: 75, avgDeepSleepPct: 0.2,
      avgRemSleepPct: 0.22, avgDailySteps: 8000, restingTempMean: 33.5, restingTempStd: 0.3,
    };
    expect(shouldRecomputeBaseline(old)).toBe(true);
  });

  test('returns false when baseline is fresh (<24h)', () => {
    const fresh: PersonalBaseline = {
      userId: 'user1', computedAt: Date.now() - 1 * 60 * 60 * 1000,
      periodDays: 14, sampleCount: 100,
      restingHrMean: 65, restingHrStd: 5, sleepingHrMean: 55,
      rmssdMean: 45, rmssdStd: 10, sdnnMean: 50, sdnnStd: 12, nocturnalRmssdMean: 55,
      typicalBedtimeHour: 23, typicalWakeHour: 7, avgSleepDurationMin: 450,
      avgSleepEfficiency: 0.88, avgSleepQuality: 75, avgDeepSleepPct: 0.2,
      avgRemSleepPct: 0.22, avgDailySteps: 8000, restingTempMean: 33.5, restingTempStd: 0.3,
    };
    expect(shouldRecomputeBaseline(fresh)).toBe(false);
  });
});

// ============================================================
// detectAnomalies
// ============================================================

describe('detectAnomalies', () => {
  const baseline: PersonalBaseline = {
    userId: 'user1', computedAt: Date.now(), periodDays: 14, sampleCount: 200,
    restingHrMean: 65, restingHrStd: 5, sleepingHrMean: 55,
    rmssdMean: 45, rmssdStd: 10, sdnnMean: 50, sdnnStd: 12, nocturnalRmssdMean: 55,
    typicalBedtimeHour: 23, typicalWakeHour: 7, avgSleepDurationMin: 450,
    avgSleepEfficiency: 0.88, avgSleepQuality: 75, avgDeepSleepPct: 0.2,
    avgRemSleepPct: 0.22, avgDailySteps: 8000, restingTempMean: 33.5, restingTempStd: 0.3,
  };

  test('no anomalies for normal features', () => {
    const features = makeFeatureWindow({ rmssd: 45, hrMean: 70, tempMean: 33.5 });
    const flags = detectAnomalies(features, baseline);
    expect(flags.length).toBe(0);
  });

  test('detects low HRV anomaly (>2 std below baseline)', () => {
    // RMSSD 10 is (10 - 45) / 10 = -3.5 std devs below
    const features = makeFeatureWindow({ rmssd: 10 });
    const flags = detectAnomalies(features, baseline);
    const hrvFlag = flags.find(f => f.type === 'hrv_low');
    expect(hrvFlag).toBeDefined();
    expect(hrvFlag!.severity).toBe('high'); // zScore < -3
  });

  test('detects elevated HR anomaly (>2 std above baseline)', () => {
    // HR 80 is (80 - 65) / 5 = 3 std devs above
    const features = makeFeatureWindow({ hrMean: 80 });
    const flags = detectAnomalies(features, baseline);
    const hrFlag = flags.find(f => f.type === 'hr_elevated');
    expect(hrFlag).toBeDefined();
  });

  test('detects temperature drop anomaly', () => {
    // Temp 32.5 is (32.5 - 33.5) / 0.3 = -3.3 std devs below
    const features = makeFeatureWindow({ tempMean: 32.5 });
    const flags = detectAnomalies(features, baseline);
    const tempFlag = flags.find(f => f.type === 'temp_drop');
    expect(tempFlag).toBeDefined();
  });

  test('anomaly flags have required properties', () => {
    const features = makeFeatureWindow({ rmssd: 10 });
    const flags = detectAnomalies(features, baseline);
    for (const flag of flags) {
      expect(flag).toHaveProperty('type');
      expect(flag).toHaveProperty('severity');
      expect(flag).toHaveProperty('message');
      expect(flag).toHaveProperty('zScore');
      expect(['moderate', 'high']).toContain(flag.severity);
    }
  });
});
