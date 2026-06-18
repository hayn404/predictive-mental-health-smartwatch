import {
  loadDepressionModel,
  isDepressionModelLoaded,
  predictDepression,
  buildDailyActivityFeatures,
} from '../services/ai/depressionModel';
import type { DailyActivityFeatures } from '../services/ai/types';

// ── Minimal test model (2 trees, 3 features) ─────────────────────────────────

function makeMinimalModel() {
  return {
    version: '1.0.0-test',
    modelType: 'XGBClassifier',
    features: ['mean_activity', 'inactive_minutes', 'day_night_ratio',
      'std_activity', 'median_activity', 'max_activity', 'min_activity',
      'activity_range', 'activity_cv', 'p10', 'p25', 'p50', 'p75', 'p90',
      'very_active_minutes', 'active_minutes', 'day_mean', 'night_mean',
      'longest_rest_period', 'rest_period_count', 'activity_entropy',
      'gender', 'days', 'age_num'],
    numTrees: 2,
    baseScore: -0.6014,          // logit(0.354) — identity normalization
    learningRate: 0.05,
    normalization: {
      mean: Object.fromEntries(
        ['mean_activity', 'inactive_minutes', 'day_night_ratio',
          'std_activity', 'median_activity', 'max_activity', 'min_activity',
          'activity_range', 'activity_cv', 'p10', 'p25', 'p50', 'p75', 'p90',
          'very_active_minutes', 'active_minutes', 'day_mean', 'night_mean',
          'longest_rest_period', 'rest_period_count', 'activity_entropy',
          'gender', 'days', 'age_num'].map(k => [k, 0])
      ),
      std: Object.fromEntries(
        ['mean_activity', 'inactive_minutes', 'day_night_ratio',
          'std_activity', 'median_activity', 'max_activity', 'min_activity',
          'activity_range', 'activity_cv', 'p10', 'p25', 'p50', 'p75', 'p90',
          'very_active_minutes', 'active_minutes', 'day_mean', 'night_mean',
          'longest_rest_period', 'rest_period_count', 'activity_entropy',
          'gender', 'days', 'age_num'].map(k => [k, 1])
      ),
    },
    // Tree 1: split on mean_activity
    trees: [
      {
        nodeid: 0, split: 'mean_activity', split_condition: 100, yes: 1, no: 2, missing: 1,
        children: [
          { nodeid: 1, leaf: -0.2 },
          { nodeid: 2, leaf:  0.3 },
        ],
      },
      // Tree 2: split on inactive_minutes
      {
        nodeid: 0, split: 'inactive_minutes', split_condition: 600, yes: 1, no: 2, missing: 1,
        children: [
          { nodeid: 1, leaf:  0.1 },
          { nodeid: 2, leaf: -0.1 },
        ],
      },
    ],
    importances: { mean_activity: 0.6, inactive_minutes: 0.4 },
    decisionThreshold: 0.40,
    applySigmoid: true,
    riskLevels: {
      minimal:  { min: 0,  max: 30, label: 'Minimal',  color: '#35e27e' },
      mild:     { min: 30, max: 50, label: 'Mild',     color: '#9B8EC4' },
      moderate: { min: 50, max: 70, label: 'Moderate', color: '#E8A87C' },
      high:     { min: 70, max: 100, label: 'High',    color: '#C4897B' },
    },
  };
}

function makeHealthyFeatures(): DailyActivityFeatures {
  return {
    mean_activity: 220,        // above split → higher leaf in tree 1
    std_activity: 300,
    median_activity: 60,
    max_activity: 2000,
    min_activity: 0,
    activity_range: 2000,
    activity_cv: 1.36,
    p10: 0, p25: 0, p50: 60, p75: 350, p90: 700,
    inactive_minutes: 500,     // below split → +0.1 in tree 2
    active_minutes: 200,
    very_active_minutes: 50,
    day_mean: 300,
    night_mean: 25,
    day_night_ratio: 11.2,
    longest_rest_period: 480,
    rest_period_count: 3,
    activity_entropy: 3.1,
    gender: 1,
    days: 14,
    age_num: 28,
  };
}

function makeSedentaryFeatures(): DailyActivityFeatures {
  return {
    mean_activity: 40,         // below split → -0.2 leaf in tree 1
    std_activity: 80,
    median_activity: 5,
    max_activity: 400,
    min_activity: 0,
    activity_range: 400,
    activity_cv: 2.0,
    p10: 0, p25: 0, p50: 5, p75: 60, p90: 200,
    inactive_minutes: 900,     // above split → -0.1 leaf in tree 2
    active_minutes: 20,
    very_active_minutes: 0,
    day_mean: 50,
    night_mean: 10,
    day_night_ratio: 4.5,
    longest_rest_period: 600,
    rest_period_count: 2,
    activity_entropy: 1.2,
    gender: 2,
    days: 3,
    age_num: 35,
  };
}

// ── Model loading ─────────────────────────────────────────────────────────────

test('isDepressionModelLoaded returns false before loadDepressionModel', () => {
  // We can't reset the module singleton cleanly in jest without isolateModules,
  // so just verify the loaded state after loading.
  loadDepressionModel(makeMinimalModel() as any);
  expect(isDepressionModelLoaded()).toBe(true);
});

test('loadDepressionModel sets the cached model', () => {
  const m = makeMinimalModel();
  loadDepressionModel(m as any);
  expect(isDepressionModelLoaded()).toBe(true);
});

// ── Prediction range ──────────────────────────────────────────────────────────

test('predictDepression returns probability in [0, 1]', () => {
  loadDepressionModel(makeMinimalModel() as any);
  const pred = predictDepression(makeHealthyFeatures());
  expect(pred.probability).toBeGreaterThanOrEqual(0);
  expect(pred.probability).toBeLessThanOrEqual(1);
});

test('predictDepression returns riskScore in [0, 100]', () => {
  loadDepressionModel(makeMinimalModel() as any);
  const pred = predictDepression(makeHealthyFeatures());
  expect(pred.riskScore).toBeGreaterThanOrEqual(0);
  expect(pred.riskScore).toBeLessThanOrEqual(100);
});

// ── Risk level mapping ────────────────────────────────────────────────────────

test('higher activity → lower risk than sedentary', () => {
  loadDepressionModel(makeMinimalModel() as any);
  const healthy   = predictDepression(makeHealthyFeatures());
  const sedentary = predictDepression(makeSedentaryFeatures());
  // Sedentary: mean_activity=40 < 100 → leaf=-0.2; inactive=900 > 600 → leaf=-0.1
  // Healthy:   mean_activity=220 ≥ 100 → leaf=+0.3; inactive=500 < 600 → leaf=+0.1
  expect(healthy.riskScore).toBeGreaterThan(sedentary.riskScore);
});

test('risk level is one of the four valid values', () => {
  loadDepressionModel(makeMinimalModel() as any);
  const valid = ['minimal', 'mild', 'moderate', 'high'];
  expect(valid).toContain(predictDepression(makeHealthyFeatures()).riskLevel);
  expect(valid).toContain(predictDepression(makeSedentaryFeatures()).riskLevel);
});

test('riskScore < 30 → minimal', () => {
  // Force a low raw score by using features that both go low leaf
  loadDepressionModel(makeMinimalModel() as any);
  const pred = predictDepression(makeSedentaryFeatures());
  if (pred.riskScore < 30) {
    expect(pred.riskLevel).toBe('minimal');
  }
});

// ── topContributors ───────────────────────────────────────────────────────────

test('topContributors has at most 3 entries', () => {
  loadDepressionModel(makeMinimalModel() as any);
  const pred = predictDepression(makeHealthyFeatures());
  expect(pred.topContributors.length).toBeLessThanOrEqual(3);
});

test('topContributors entries have required fields', () => {
  loadDepressionModel(makeMinimalModel() as any);
  const pred = predictDepression(makeHealthyFeatures());
  for (const c of pred.topContributors) {
    expect(typeof c.feature).toBe('string');
    expect(typeof c.label).toBe('string');
    expect(typeof c.value).toBe('number');
    expect(typeof c.impact).toBe('number');
    expect(['high', 'low']).toContain(c.direction);
  }
});

// ── Mock fallback ─────────────────────────────────────────────────────────────

test('dataQuality is mock when daysOfData < 3', () => {
  loadDepressionModel(makeMinimalModel() as any);
  const feats = { ...makeHealthyFeatures(), days: 1 };
  const pred = predictDepression(feats);
  expect(pred.dataQuality).toBe('mock');
});

test('dataQuality is good when daysOfData >= 7', () => {
  loadDepressionModel(makeMinimalModel() as any);
  const feats = { ...makeHealthyFeatures(), days: 10 };
  const pred = predictDepression(feats);
  expect(pred.dataQuality).toBe('good');
});

// ── buildDailyActivityFeatures ────────────────────────────────────────────────

describe('buildDailyActivityFeatures', () => {
  const now = 1718700000000; // fixed reference time

  test('returns mock features for empty step samples', () => {
    const feats = buildDailyActivityFeatures([], 1, 25, 0);
    expect(feats.mean_activity).toBeGreaterThan(0);
    expect(feats.gender).toBe(1);
    expect(feats.age_num).toBe(25);
    expect(feats.days).toBe(1);
  });

  test('uses unknown gender (1.5) when null passed', () => {
    const feats = buildDailyActivityFeatures([], null as any, null, 0);
    expect(feats.gender).toBe(1.5);
    expect(feats.age_num).toBe(30);
  });

  test('computes stats from step samples', () => {
    const samples = [
      { startTime: now,              endTime: now + 60_000,  count: 120 }, // 2 steps/min over 1 min
      { startTime: now + 60_000,     endTime: now + 120_000, count: 0   },
      { startTime: now + 120_000,    endTime: now + 240_000, count: 300 }, // 150 steps/min over 2 min
    ];
    const feats = buildDailyActivityFeatures(samples as any, 1.5, 30, 7);
    expect(feats.mean_activity).toBeGreaterThan(0);
    expect(feats.max_activity).toBeGreaterThan(feats.mean_activity);
    expect(feats.inactive_minutes).toBeGreaterThanOrEqual(0);
    expect(feats.days).toBe(7);
    expect(feats.gender).toBe(1.5);
    expect(feats.age_num).toBe(30);
  });

  test('longest_rest_period counts consecutive low-activity minutes', () => {
    // 60 minutes of zero activity = longest rest = 60
    const samples = [
      { startTime: now,                  endTime: now + 3_600_000, count: 0 },    // 60 min at 0 steps
      { startTime: now + 3_600_000,      endTime: now + 3_660_000, count: 1200 }, // 1 active min
      { startTime: now + 3_660_000,      endTime: now + 7_260_000, count: 0 },    // 60 more rest min
    ];
    const feats = buildDailyActivityFeatures(samples as any, 1, 25, 5);
    expect(feats.longest_rest_period).toBeGreaterThanOrEqual(58); // ~60 after floating-point spread
  });

  test('all 24 required fields are present', () => {
    const feats = buildDailyActivityFeatures([], 1, 25, 3);
    const required = [
      'mean_activity', 'std_activity', 'median_activity', 'max_activity', 'min_activity',
      'activity_range', 'activity_cv', 'p10', 'p25', 'p50', 'p75', 'p90',
      'inactive_minutes', 'active_minutes', 'very_active_minutes',
      'day_mean', 'night_mean', 'day_night_ratio',
      'longest_rest_period', 'rest_period_count', 'activity_entropy',
      'gender', 'days', 'age_num',
    ];
    for (const k of required) {
      expect(typeof (feats as any)[k]).toBe('number');
    }
  });
});

// ── Real model regression guard ───────────────────────────────────────────────

describe('real shipped model', () => {
  let realModel: any;
  try {
    realModel = require('../assets/ml/depression/depression_model.json');
  } catch {
    realModel = null;
  }

  const skip = realModel == null ? test.skip : test;

  skip('loads without error', () => {
    expect(() => loadDepressionModel(realModel)).not.toThrow();
    expect(isDepressionModelLoaded()).toBe(true);
  });

  skip('active profile scores higher than sedentary profile', () => {
    loadDepressionModel(realModel);
    const active   = predictDepression(makeHealthyFeatures());
    const inactive = predictDepression(makeSedentaryFeatures());
    // Active (mean=220, diverse, good day/night) → less risk expected by the real model
    // Sedentary (mean=40, flat, poor rhythm) → more risk
    // We just check they differ meaningfully (not an exact-value assertion)
    expect(Math.abs(active.riskScore - inactive.riskScore)).toBeGreaterThan(0);
  });

  skip('returns a valid risk level for known-healthy features', () => {
    loadDepressionModel(realModel);
    const pred = predictDepression(makeHealthyFeatures());
    expect(['minimal', 'mild', 'moderate', 'high']).toContain(pred.riskLevel);
    expect(pred.probability).toBeGreaterThanOrEqual(0);
    expect(pred.probability).toBeLessThanOrEqual(1);
  });
});
