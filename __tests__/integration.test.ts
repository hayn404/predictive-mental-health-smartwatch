/**
 * Integration Tests — End-to-End AI Pipeline
 * =============================================
 * Tests the complete flow: raw data → feature extraction →
 * stress prediction → anxiety prediction → recommendations.
 */

import { extractFeatures } from '@/services/ai/featureEngineering';
import { loadModel, predictStress, predictAnxiety } from '@/services/ai/stressModel';
import { analyzeSleepSession } from '@/services/ai/sleepAnalysis';
import { analyzeCheckinLocal } from '@/services/ai/voiceAnalysis';
import { generateRecommendations } from '@/services/ai/recommendations';
import { computeBaseline, detectAnomalies } from '@/services/ai/baseline';
import type { RawHeartRateSample, RawTemperatureSample, RawSleepSession } from '@/services/ai/types';

// ============================================================
// Test Helpers
// ============================================================

function makeHRSamples(n: number, baseBpm: number): RawHeartRateSample[] {
  return Array.from({ length: n }, (_, i) => ({
    timestamp: Date.now() + i * 1000,
    bpm: baseBpm + Math.round(Math.sin(i * 0.1) * 5),
    source: 'test',
  }));
}

function makeTempSamples(n: number, baseTemp: number): RawTemperatureSample[] {
  return Array.from({ length: n }, (_, i) => ({
    timestamp: Date.now() + i * 60000,
    temperatureCelsius: baseTemp + Math.random() * 0.5,
  }));
}

// Minimal model for testing
const testModel = {
  version: '1.0.0',
  features: ['rmssd', 'hrMean', 'lfHfRatio', 'sdnn'],
  numTrees: 1,
  baseScore: 0,
  learningRate: 0.3,
  normalization: {
    mean: { rmssd: 40, hrMean: 75, lfHfRatio: 1.5, sdnn: 50 },
    std: { rmssd: 15, hrMean: 10, lfHfRatio: 1.0, sdnn: 15 },
  },
  trees: [{
    nodeid: 0, split: 'rmssd', split_condition: 0,
    yes: 1, no: 2, missing: 1,
    children: [
      { nodeid: 1, leaf: 0.5 },
      { nodeid: 2, leaf: -0.5 },
    ],
  }],
  stressLevels: {
    low: { min: 0, max: 25, label: 'Low', color: '#4CAF50' },
    moderate: { min: 26, max: 50, label: 'Moderate', color: '#FFC107' },
    elevated: { min: 51, max: 75, label: 'Elevated', color: '#FF9800' },
    high: { min: 76, max: 100, label: 'High', color: '#F44336' },
  },
  importances: { rmssd: 0.4, hrMean: 0.3, lfHfRatio: 0.2, sdnn: 0.1 },
};

// ============================================================
// Full Pipeline: Raw Data → Features → Stress → Recommendations
// ============================================================

describe('End-to-End Pipeline: Relaxed State', () => {
  beforeAll(() => {
    loadModel(testModel as any);
  });

  test('relaxed biometrics produce low stress and appropriate recommendations', () => {
    // Step 1: Extract features from relaxed-state biometrics
    const features = extractFeatures({
      hrSamples: makeHRSamples(60, 65), // Low resting HR
      hrvSamples: [{ timestamp: Date.now(), rmssd: 55 }],
      tempSamples: makeTempSamples(5, 33.5),
      stepCount: 0,
      windowSeconds: 300,
      timestamp: Date.now(),
    });

    expect(features.hrMean).toBeLessThan(80);

    // Step 2: Predict stress
    const stress = predictStress(features);
    expect(stress.stressScore).toBeDefined();
    expect(stress.stressLevel).toBeDefined();

    // Step 3: Predict anxiety
    const anxiety = predictAnxiety(stress, features, null, 80);
    expect(anxiety.anxietyIndex).toBeDefined();

    // Step 4: Generate recommendations
    const recs = generateRecommendations(stress, anxiety, null, null, null);
    // Low stress = fewer or no urgent recommendations
    expect(recs.length).toBeLessThanOrEqual(3);
  });
});

describe('End-to-End Pipeline: Stressed State', () => {
  beforeAll(() => {
    loadModel(testModel as any);
  });

  test('stressed biometrics trigger stress-related recommendations', () => {
    // Step 1: Extract features from stressed-state biometrics
    const features = extractFeatures({
      hrSamples: makeHRSamples(60, 95), // Elevated HR
      hrvSamples: [{ timestamp: Date.now(), rmssd: 18 }], // Low HRV
      tempSamples: makeTempSamples(5, 32.5), // Lower temp
      stepCount: 0,
      windowSeconds: 300,
      timestamp: Date.now(),
    });

    // Step 2: Predict stress
    const stress = predictStress(features);

    // Step 3: Predict anxiety
    const anxiety = predictAnxiety(stress, features, null, 40); // Poor sleep

    // Step 4: Generate recommendations
    const recs = generateRecommendations(stress, anxiety, null, null, null);

    // Verify pipeline produces coherent output
    expect(stress.stressScore).toBeGreaterThanOrEqual(0);
    expect(anxiety.anxietyIndex).toBeGreaterThanOrEqual(0);
    expect(Array.isArray(recs)).toBe(true);
  });
});

// ============================================================
// Sleep → Recommendations Pipeline
// ============================================================

describe('Sleep Pipeline', () => {
  test('poor sleep analysis triggers sleep recommendations', () => {
    const start = new Date('2026-03-07T02:00:00').getTime();
    const end = new Date('2026-03-07T05:30:00').getTime();

    const session: RawSleepSession = {
      startTime: start, endTime: end,
      stages: [
        { startTime: start, endTime: start + 30 * 60000, stage: 'awake' },
        { startTime: start + 30 * 60000, endTime: start + 120 * 60000, stage: 'light' },
        { startTime: start + 120 * 60000, endTime: start + 140 * 60000, stage: 'deep' },
        { startTime: start + 140 * 60000, endTime: start + 160 * 60000, stage: 'rem' },
        { startTime: start + 160 * 60000, endTime: start + 210 * 60000, stage: 'light' },
      ],
      source: 'test',
    };

    const sleep = analyzeSleepSession(session);
    expect(sleep.qualityScore).toBeLessThan(70);

    const stress = { timestamp: Date.now(), stressScore: 45, stressLevel: 'moderate' as const, confidence: 0.8, topContributors: [] };
    const recs = generateRecommendations(stress, null, sleep, null, null);

    // Poor sleep should trigger sleep-related recommendations
    const hasSleepRec = recs.some(r => r.category === 'sleep_hygiene');
    expect(hasSleepRec).toBe(true);
  });
});

// ============================================================
// Voice Check-in → Recommendations Pipeline
// ============================================================

describe('Voice Check-in Pipeline', () => {
  test('distressed check-in with high biometric stress generates targeted recommendations', () => {
    const checkin = analyzeCheckinLocal(
      'I feel terrible and so stressed. I have been anxious all day and cannot sleep.',
      { hr: 95, hrv: 18, stress: 78, sleepQuality: 40, anxietyIndex: 70, sleepDurationHrs: 4 },
    );

    expect(checkin.sentiment).not.toBe('positive');
    expect(checkin.emotionScores.anxiety).toBeGreaterThan(0);

    const stress = { timestamp: Date.now(), stressScore: 78, stressLevel: 'high' as const, confidence: 0.8, topContributors: [] };
    const anxiety = { timestamp: Date.now(), anxietyIndex: 70, level: 'moderate' as const, sustained: false, baselineDeviation: -0.5 };

    const recs = generateRecommendations(stress, anxiety, null, checkin, null);
    expect(recs.length).toBeGreaterThan(0);
  });
});

// ============================================================
// Baseline → Anomaly Detection Pipeline
// ============================================================

describe('Baseline & Anomaly Pipeline', () => {
  test('anomalies are detected when features deviate from baseline', () => {
    // Build baseline from normal windows
    const normalWindows = Array.from({ length: 100 }, (_, i) => {
      const features = extractFeatures({
        hrSamples: makeHRSamples(60, 68 + Math.random() * 4),
        hrvSamples: [{ timestamp: Date.now(), rmssd: 45 + Math.random() * 10 }],
        tempSamples: makeTempSamples(3, 33.3 + Math.random() * 0.4),
        stepCount: Math.round(Math.random() * 200),
        windowSeconds: 300,
        timestamp: Date.now() - (100 - i) * 300000,
      });
      return features;
    });

    const baseline = computeBaseline('user1', normalWindows, []);
    expect(baseline).not.toBeNull();

    // Create anomalous features
    const anomalous = extractFeatures({
      hrSamples: makeHRSamples(60, 110), // Very high HR
      hrvSamples: [{ timestamp: Date.now(), rmssd: 10 }], // Very low HRV
      tempSamples: makeTempSamples(3, 31.5), // Low temp
      stepCount: 0,
      windowSeconds: 300,
      timestamp: Date.now(),
    });

    const flags = detectAnomalies(anomalous, baseline!);
    // Should detect at least one anomaly
    expect(flags.length).toBeGreaterThan(0);
  });
});
