/**
 * Unit Tests — Sleep Analysis
 * ============================
 * Tests sleep quality scoring, stage computation, recovery scoring,
 * consistency scoring, and sleep trend analysis.
 */

import { analyzeSleepSession, computeSleepTrend } from '@/services/ai/sleepAnalysis';
import type {
  RawSleepSession,
  RawSleepStage,
  SleepAnalysis,
  PersonalBaseline,
  RawHeartRateSample,
  RawHRVSample,
} from '@/services/ai/types';

// ============================================================
// Test Helpers
// ============================================================

function makeSession(overrides: Partial<RawSleepSession> = {}): RawSleepSession {
  const start = new Date('2026-03-06T23:00:00').getTime();
  const end = new Date('2026-03-07T07:00:00').getTime(); // 8 hours

  return {
    startTime: start,
    endTime: end,
    stages: [
      { startTime: start, endTime: start + 15 * 60000, stage: 'awake' },         // 15 min onset
      { startTime: start + 15 * 60000, endTime: start + 120 * 60000, stage: 'light' },  // 105 min light
      { startTime: start + 120 * 60000, endTime: start + 180 * 60000, stage: 'deep' },  // 60 min deep
      { startTime: start + 180 * 60000, endTime: start + 240 * 60000, stage: 'rem' },   // 60 min REM
      { startTime: start + 240 * 60000, endTime: start + 250 * 60000, stage: 'awake' }, // 10 min awake
      { startTime: start + 250 * 60000, endTime: start + 360 * 60000, stage: 'light' }, // 110 min light
      { startTime: start + 360 * 60000, endTime: start + 420 * 60000, stage: 'deep' },  // 60 min deep
      { startTime: start + 420 * 60000, endTime: start + 480 * 60000, stage: 'rem' },   // 60 min REM
    ],
    source: 'test',
    ...overrides,
  };
}

function makeBaseline(overrides: Partial<PersonalBaseline> = {}): PersonalBaseline {
  return {
    userId: 'test', computedAt: Date.now(), periodDays: 14, sampleCount: 200,
    restingHrMean: 65, restingHrStd: 5, sleepingHrMean: 55,
    rmssdMean: 45, rmssdStd: 10, sdnnMean: 50, sdnnStd: 12, nocturnalRmssdMean: 55,
    typicalBedtimeHour: 23, typicalWakeHour: 7, avgSleepDurationMin: 450,
    avgSleepEfficiency: 0.88, avgSleepQuality: 75, avgDeepSleepPct: 0.2,
    avgRemSleepPct: 0.22, avgDailySteps: 8000, restingTempMean: 33.5, restingTempStd: 0.3,
    ...overrides,
  };
}

function makeHRSamples(avgBpm: number, count: number): RawHeartRateSample[] {
  return Array.from({ length: count }, (_, i) => ({
    timestamp: Date.now() + i * 60000,
    bpm: avgBpm + Math.round(Math.random() * 4 - 2),
    source: 'test',
  }));
}

function makeHRVSamples(avgRmssd: number, count: number): RawHRVSample[] {
  return Array.from({ length: count }, (_, i) => ({
    timestamp: Date.now() + i * 60000,
    rmssd: avgRmssd + Math.round(Math.random() * 5 - 2.5),
  }));
}

// ============================================================
// analyzeSleepSession
// ============================================================

describe('analyzeSleepSession', () => {
  test('computes total in-bed time from session start/end', () => {
    const result = analyzeSleepSession(makeSession());
    expect(result.totalInBedMin).toBe(480); // 8 hours
  });

  test('computes total sleep time (excluding awake stages)', () => {
    const result = analyzeSleepSession(makeSession());
    // Total sleep = 105 + 60 + 60 + 110 + 60 + 60 = 455 min
    expect(result.totalSleepMin).toBe(455);
  });

  test('computes sleep onset latency', () => {
    const result = analyzeSleepSession(makeSession());
    expect(result.onsetLatencyMin).toBe(15); // 15 min of initial awake
  });

  test('computes WASO (wake after sleep onset)', () => {
    const result = analyzeSleepSession(makeSession());
    expect(result.wasoMin).toBe(10); // 10 min awake in middle
  });

  test('computes stage percentages that sum to ~1.0', () => {
    const result = analyzeSleepSession(makeSession());
    const total = result.deepSleepPct + result.remSleepPct + result.lightSleepPct;
    expect(total).toBeCloseTo(1.0, 1);
  });

  test('sleep efficiency = totalSleep / totalInBed', () => {
    const result = analyzeSleepSession(makeSession());
    const expected = 455 / 480;
    expect(result.sleepEfficiency).toBeCloseTo(expected, 2);
  });

  test('quality score is between 0 and 100', () => {
    const result = analyzeSleepSession(makeSession());
    expect(result.qualityScore).toBeGreaterThanOrEqual(0);
    expect(result.qualityScore).toBeLessThanOrEqual(100);
  });

  test('good sleep yields high quality score', () => {
    // 8 hours, good stage distribution, low fragmentation
    const result = analyzeSleepSession(makeSession());
    expect(result.qualityScore).toBeGreaterThan(60);
  });

  test('poor sleep yields lower quality score', () => {
    const start = new Date('2026-03-07T01:00:00').getTime();
    const end = new Date('2026-03-07T05:00:00').getTime(); // Only 4 hours
    const poorSession = makeSession({
      startTime: start,
      endTime: end,
      stages: [
        { startTime: start, endTime: start + 30 * 60000, stage: 'awake' },      // Long onset
        { startTime: start + 30 * 60000, endTime: start + 120 * 60000, stage: 'light' },
        { startTime: start + 120 * 60000, endTime: start + 140 * 60000, stage: 'deep' },
        { startTime: start + 140 * 60000, endTime: start + 150 * 60000, stage: 'awake' },
        { startTime: start + 150 * 60000, endTime: start + 200 * 60000, stage: 'light' },
        { startTime: start + 200 * 60000, endTime: start + 220 * 60000, stage: 'rem' },
        { startTime: start + 220 * 60000, endTime: start + 240 * 60000, stage: 'awake' },
      ],
    });
    const result = analyzeSleepSession(poorSession);
    expect(result.qualityScore).toBeLessThan(60);
  });

  test('counts awakenings correctly', () => {
    const result = analyzeSleepSession(makeSession());
    // One awakening in the middle of the night
    expect(result.awakeningCount).toBe(1);
  });

  test('computes HR stats when provided', () => {
    const hr = makeHRSamples(55, 30);
    const result = analyzeSleepSession(makeSession(), hr);
    expect(result.avgHrSleep).toBeGreaterThan(0);
    expect(result.minHrSleep).toBeGreaterThan(0);
    expect(result.minHrSleep).toBeLessThanOrEqual(result.avgHrSleep);
  });

  test('computes HRV stats when provided', () => {
    const hrv = makeHRVSamples(50, 20);
    const result = analyzeSleepSession(makeSession(), undefined, hrv);
    expect(result.avgHrvSleep).toBeGreaterThan(0);
    expect(result.maxHrvSleep).toBeGreaterThanOrEqual(result.avgHrvSleep);
  });

  test('recovery score defaults to 50 without HRV data', () => {
    const result = analyzeSleepSession(makeSession());
    expect(result.recoveryScore).toBe(50);
  });

  test('recovery score uses baseline when available', () => {
    const hrv = makeHRVSamples(60, 20);
    const baseline = makeBaseline({ nocturnalRmssdMean: 50 });
    const result = analyzeSleepSession(makeSession(), undefined, hrv, baseline);
    // HRV 60 vs baseline 50 => ratio 1.2 => score = 1.2 * 70 = 84
    expect(result.recoveryScore).toBeGreaterThan(50);
  });

  test('consistency score uses baseline bedtime', () => {
    const baseline = makeBaseline({ typicalBedtimeHour: 23, avgSleepDurationMin: 480 });
    const result = analyzeSleepSession(makeSession(), undefined, undefined, baseline);
    // Session starts at 23:00 (matches baseline), 480 min (matches)
    expect(result.consistencyScore).toBeGreaterThan(60);
  });

  test('date format is YYYY-MM-DD', () => {
    const result = analyzeSleepSession(makeSession());
    expect(result.date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });
});

// ============================================================
// computeSleepTrend
// ============================================================

describe('computeSleepTrend', () => {
  function makeSleepAnalysis(quality: number, durationMin: number): SleepAnalysis {
    return {
      date: '2026-03-07', sessionStart: Date.now(), sessionEnd: Date.now(),
      totalInBedMin: durationMin + 30, totalSleepMin: durationMin,
      onsetLatencyMin: 10, wasoMin: 5,
      lightSleepMin: durationMin * 0.5, deepSleepMin: durationMin * 0.2,
      remSleepMin: durationMin * 0.25, awakeMin: 15,
      deepSleepPct: 0.2, remSleepPct: 0.25, lightSleepPct: 0.55,
      sleepEfficiency: 0.85, fragmentationIndex: 1.5, awakeningCount: 2,
      avgHrSleep: 55, minHrSleep: 48, avgHrvSleep: 50, maxHrvSleep: 70,
      qualityScore: quality, recoveryScore: 70, consistencyScore: 80,
    };
  }

  test('returns stable trend with no data', () => {
    const result = computeSleepTrend([]);
    expect(result.trend).toBe('stable');
    expect(result.daysAnalyzed).toBe(0);
  });

  test('computes average quality and duration', () => {
    const sessions = [
      makeSleepAnalysis(70, 420),
      makeSleepAnalysis(80, 450),
      makeSleepAnalysis(90, 480),
    ];
    const result = computeSleepTrend(sessions);
    expect(result.avgQuality).toBeCloseTo(80, 0);
    expect(result.avgDuration).toBeCloseTo(450, 0);
  });

  test('detects improving trend', () => {
    const sessions = [
      makeSleepAnalysis(50, 360),
      makeSleepAnalysis(55, 380),
      makeSleepAnalysis(75, 420),
      makeSleepAnalysis(85, 450),
    ];
    const result = computeSleepTrend(sessions);
    expect(result.trend).toBe('improving');
  });

  test('detects declining trend', () => {
    const sessions = [
      makeSleepAnalysis(85, 450),
      makeSleepAnalysis(80, 440),
      makeSleepAnalysis(60, 380),
      makeSleepAnalysis(50, 360),
    ];
    const result = computeSleepTrend(sessions);
    expect(result.trend).toBe('declining');
  });

  test('detects stable trend for similar scores', () => {
    const sessions = [
      makeSleepAnalysis(75, 420),
      makeSleepAnalysis(73, 430),
      makeSleepAnalysis(76, 425),
      makeSleepAnalysis(74, 420),
    ];
    const result = computeSleepTrend(sessions);
    expect(result.trend).toBe('stable');
  });
});
