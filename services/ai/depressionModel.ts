/**
 * Seren — Depression Risk Model (On-Device Inference)
 * =====================================================
 * Pure-TS XGBoost tree traversal, same engine as stress/focus/bioAge models.
 *
 * Input:  DailyActivityFeatures — 24 daily actigraphy aggregates computed from
 *         a rolling 24-hour window of Health Connect step data.
 * Output: DepressionPrediction with probability 0–1, risk level, top contributors.
 *
 * Key differences from HRV models:
 *  - Features are DAILY aggregates (not per-5s window).
 *  - No feature normalization: XGBoost was trained on raw values (Depresjon dataset).
 *    The JSON stores identity normalization (mean=0, std=1) as a no-op.
 *  - Cadence: computed once per day on app open, not in the live inference cycle.
 *
 * Training: Depresjon actigraphy dataset (55 subjects: 23 depressed, 32 control).
 * Held-out CV metrics: Accuracy 0.926, F1 0.898, ROC-AUC 0.967, threshold 0.40.
 *
 * ⚠ NON-CLINICAL: this is a wellness screening signal, not a diagnosis.
 *   Cross-reference with the in-app PHQ-9 for a fuller picture.
 */

import {
  DailyActivityFeatures,
  DepressionPrediction,
  DepressionRiskLevel,
  DepressionContributor,
  RawStepsSample,
} from './types';

// ── XGBoost model types (same shape as stress/focus/bioAge) ──────────────────

interface XGBoostNode {
  nodeid: number;
  split?: string;
  split_condition?: number;
  yes?: number;
  no?: number;
  missing?: number;
  leaf?: number;
  children?: XGBoostNode[];
}

interface DepressionModel {
  version: string;
  features: string[];
  baseScore: number;             // log-odds (logit of empirical prior)
  normalization: { mean: Record<string, number>; std: Record<string, number> };
  trees: XGBoostNode[];
  decisionThreshold: number;    // 0.40 from CV tuning
  importances: Record<string, number>;
  riskLevels: Record<string, { min: number; max: number; label: string; color: string }>;
}

let cached: DepressionModel | null = null;

export function loadDepressionModel(json: DepressionModel): void {
  cached = json;
}

export function isDepressionModelLoaded(): boolean {
  return cached !== null && Array.isArray(cached.features);
}

// ── Tree traversal ────────────────────────────────────────────────────────────

function traverse(node: XGBoostNode, features: Record<string, number>): number {
  if (node.leaf !== undefined) return node.leaf;
  const v = features[node.split!];
  if (v === undefined || v === null || isNaN(v)) {
    const id = node.missing ?? node.no!;
    const child = node.children?.find(c => c.nodeid === id);
    return child ? traverse(child, features) : 0;
  }
  const id = v < node.split_condition! ? node.yes! : node.no!;
  const child = node.children?.find(c => c.nodeid === id);
  return child ? traverse(child, features) : 0;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ── Feature normalization (identity for this model) ──────────────────────────

function toFeatureRecord(features: DailyActivityFeatures, model: DepressionModel): Record<string, number> {
  const rec: Record<string, number> = {};
  model.features.forEach((name, i) => {
    const raw = (features as any)[name] ?? 0;
    const mean = model.normalization.mean[name] ?? 0;
    const std  = model.normalization.std[name]  ?? 1;
    const z = std > 0 ? (raw - mean) / std : 0;
    rec[name]    = z;
    rec[`f${i}`] = z;          // positional alias for f0..fN split keys
  });
  return rec;
}

// ── Risk level ────────────────────────────────────────────────────────────────

function getRiskLevel(score: number): DepressionRiskLevel {
  if (score < 30) return 'minimal';
  if (score < 50) return 'mild';
  if (score < 70) return 'moderate';
  return 'high';
}

// ── Human-readable labels for top contributors ────────────────────────────────

const FEATURE_LABELS: Record<string, { high: string; low: string }> = {
  mean_activity:        { high: 'Higher overall activity',        low: 'Low overall activity' },
  std_activity:         { high: 'Variable activity levels',       low: 'Uniform (flat) activity' },
  median_activity:      { high: 'Higher typical activity',        low: 'Low typical activity' },
  max_activity:         { high: 'High peak activity',             low: 'Low peak activity' },
  activity_cv:          { high: 'Highly variable routine',        low: 'Very regular routine' },
  inactive_minutes:     { high: 'Many sedentary minutes',         low: 'Few sedentary minutes' },
  active_minutes:       { high: 'More active minutes',            low: 'Fewer active minutes' },
  very_active_minutes:  { high: 'High-intensity activity present','low': 'Little intense activity' },
  day_mean:             { high: 'Active daytime',                 low: 'Low daytime activity' },
  night_mean:           { high: 'Restless nights',                low: 'Still nights' },
  day_night_ratio:      { high: 'Clear day/night pattern',        low: 'Disrupted day/night rhythm' },
  longest_rest_period:  { high: 'Long rest stretches',            low: 'Fragmented rest' },
  rest_period_count:    { high: 'Many rest intervals',            low: 'Few rest breaks' },
  activity_entropy:     { high: 'Diverse activity pattern',       low: 'Monotonous activity pattern' },
  p10:                  { high: 'Active at low percentiles',      low: 'Very inactive at baseline' },
  p90:                  { high: 'High peak bursts',               low: 'Low activity peaks' },
  age_num:              { high: 'Older age group',                low: 'Younger age group' },
  gender:               { high: 'Female pattern',                 low: 'Male pattern' },
  days:                 { high: 'More days tracked',              low: 'Limited tracking history' },
};

function identifyContributors(
  features: DailyActivityFeatures,
  model: DepressionModel,
): DepressionContributor[] {
  const imp = model.importances ?? {};
  const mean = model.normalization?.mean ?? {};
  const total = model.features.reduce((s, f) => s + (imp[f] ?? 0), 0) || 1;

  const ranked = [...model.features].sort((a, b) => (imp[b] ?? 0) - (imp[a] ?? 0));
  const out: DepressionContributor[] = [];

  for (const f of ranked) {
    const val = (features as any)[f];
    const labels = FEATURE_LABELS[f];
    if (typeof val !== 'number' || !labels || (imp[f] ?? 0) <= 0) continue;
    const ref = mean[f] ?? val;
    const direction: 'high' | 'low' = val >= ref ? 'high' : 'low';
    out.push({
      feature: f,
      label: val >= ref ? labels.high : labels.low,
      value: val,
      impact: (imp[f] ?? 0) / total,
      direction,
    });
    if (out.length >= 3) break;
  }
  return out;
}

// ── Public inference API ──────────────────────────────────────────────────────

/**
 * Predict depression risk from a DailyActivityFeatures vector.
 * Returns a full DepressionPrediction including risk level and top contributors.
 */
export function predictDepression(features: DailyActivityFeatures): DepressionPrediction {
  const now = Date.now();

  if (!isDepressionModelLoaded()) {
    return mockDepressionPrediction(features, now);
  }

  const model = cached!;
  const rec = toFeatureRecord(features, model);

  let rawScore = model.baseScore;
  for (const tree of model.trees) {
    rawScore += traverse(tree, rec);
  }

  const probability = sigmoid(rawScore);
  const riskScore   = Math.round(Math.min(100, Math.max(0, probability * 100)));
  const riskLevel   = getRiskLevel(riskScore);
  const threshold   = model.decisionThreshold ?? 0.40;
  const topContributors = identifyContributors(features, model);

  // Data quality: good if we have a full day (≥ 720 minutes), partial otherwise
  const totalMinutes = features.inactive_minutes + features.active_minutes +
    (1440 - features.inactive_minutes - features.active_minutes);
  const dataQuality = features.days >= 7 ? 'good' : features.days >= 3 ? 'partial' : 'mock';

  return {
    timestamp:       now,
    probability,
    riskScore,
    riskLevel,
    topContributors,
    dataQuality,
    daysOfData:      features.days,
  };
}

// ── Daily feature builder from Health Connect step data ───────────────────────

/**
 * Build DailyActivityFeatures from 24h of Health Connect step samples.
 *
 * Health Connect returns steps in arbitrary intervals (not per-minute).
 * We distribute each interval's count evenly over its duration in minutes,
 * matching the per-minute actigraphy representation of the Depresjon dataset.
 *
 * @param stepSamples  24h of RawStepsSample from service.readSteps()
 * @param gender       1=male, 2=female, 1.5=unknown
 * @param chronoAge    chronological age in years (from onboarding)
 * @param daysAvailable rolling number of days with data (from history length)
 */
export function buildDailyActivityFeatures(
  stepSamples: RawStepsSample[],
  gender: number,
  chronoAge: number | null,
  daysAvailable: number,
): DailyActivityFeatures {
  if (stepSamples.length === 0) {
    return mockActivityFeatures(gender, chronoAge, daysAvailable);
  }

  // Build a per-minute activity array (1440 slots for 24h, or however many minutes covered)
  const windowStart = stepSamples[0].startTime;
  const windowEnd   = stepSamples[stepSamples.length - 1].endTime;
  const totalMinutes = Math.max(1, Math.ceil((windowEnd - windowStart) / 60_000));
  const activity = new Array<number>(totalMinutes).fill(0);

  for (const s of stepSamples) {
    const startMin = Math.floor((s.startTime - windowStart) / 60_000);
    const endMin   = Math.ceil((s.endTime - windowStart) / 60_000);
    const durationMin = Math.max(1, endMin - startMin);
    const countPerMin = s.count / durationMin;
    for (let m = startMin; m < Math.min(endMin, totalMinutes); m++) {
      activity[m] += countPerMin;
    }
  }

  // ── Basic statistics ──────────────────────────────────────────────────────
  const n = activity.length;
  const mean_activity = activity.reduce((s, v) => s + v, 0) / n;
  const variance = activity.reduce((s, v) => s + (v - mean_activity) ** 2, 0) / n;
  const std_activity = Math.sqrt(variance);
  const sorted = [...activity].sort((a, b) => a - b);
  const median_activity = n % 2 === 0
    ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    : sorted[Math.floor(n / 2)];
  const max_activity = sorted[n - 1];
  const min_activity = sorted[0];
  const activity_range = max_activity - min_activity;
  const activity_cv = std_activity / (mean_activity + 1e-6);

  // ── Percentiles ────────────────────────────────────────────────────────────
  const pct = (p: number) => {
    const idx = (p / 100) * (n - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  };

  // ── Activity levels ────────────────────────────────────────────────────────
  const inactive_minutes    = activity.filter(v => v < 20).length;
  const active_minutes      = activity.filter(v => v > 200).length;
  const very_active_minutes = activity.filter(v => v > 500).length;

  // ── Circadian rhythm ───────────────────────────────────────────────────────
  // Assign timestamps to each minute slot
  const dayActivity: number[]   = [];
  const nightActivity: number[] = [];
  for (let m = 0; m < n; m++) {
    const abs = new Date(windowStart + m * 60_000);
    const h = abs.getHours();
    if (h >= 7 && h < 22) dayActivity.push(activity[m]);
    else nightActivity.push(activity[m]);
  }
  const day_mean   = dayActivity.length   ? dayActivity.reduce((s, v) => s + v, 0) / dayActivity.length   : 0;
  const night_mean = nightActivity.length ? nightActivity.reduce((s, v) => s + v, 0) / nightActivity.length : 0;
  const day_night_ratio = day_mean / (night_mean + 1);

  // ── Sleep proxy ────────────────────────────────────────────────────────────
  const runs: number[] = [];
  let count = 0;
  for (const v of activity) {
    if (v < 20) {
      count++;
    } else {
      if (count > 0) { runs.push(count); count = 0; }
    }
  }
  if (count > 0) runs.push(count);
  const longest_rest_period = runs.length ? Math.max(...runs) : 0;
  const rest_period_count   = runs.length;

  // ── Entropy ────────────────────────────────────────────────────────────────
  const BINS = 20;
  const binSize = (max_activity - min_activity + 1e-9) / BINS;
  const hist = new Array<number>(BINS).fill(0);
  for (const v of activity) {
    const b = Math.min(BINS - 1, Math.floor((v - min_activity) / binSize));
    hist[b]++;
  }
  const total = hist.reduce((s, v) => s + v, 0) || 1;
  const activity_entropy = hist
    .map(h => h / total)
    .filter(p => p > 0)
    .reduce((s, p) => s - p * Math.log2(p), 0);

  return {
    mean_activity, std_activity, median_activity, max_activity, min_activity,
    activity_range, activity_cv,
    p10: pct(10), p25: pct(25), p50: pct(50), p75: pct(75), p90: pct(90),
    inactive_minutes, active_minutes, very_active_minutes,
    day_mean, night_mean, day_night_ratio,
    longest_rest_period, rest_period_count, activity_entropy,
    gender: gender ?? 1.5,
    days: Math.max(1, daysAvailable),
    age_num: chronoAge ?? 30,
  };
}

// ── Mock fallbacks ────────────────────────────────────────────────────────────

function mockActivityFeatures(
  gender: number,
  chronoAge: number | null,
  daysAvailable: number,
): DailyActivityFeatures {
  return {
    mean_activity: 180, std_activity: 260, median_activity: 45,
    max_activity: 1800, min_activity: 0, activity_range: 1800, activity_cv: 1.44,
    p10: 0, p25: 0, p50: 45, p75: 280, p90: 620,
    inactive_minutes: 720, active_minutes: 180, very_active_minutes: 40,
    day_mean: 260, night_mean: 30, day_night_ratio: 8.4,
    longest_rest_period: 480, rest_period_count: 3,
    activity_entropy: 2.8,
    gender: gender ?? 1.5,
    days: Math.max(1, daysAvailable),
    age_num: chronoAge ?? 30,
  };
}

function mockDepressionPrediction(
  features: DailyActivityFeatures,
  now: number,
): DepressionPrediction {
  return {
    timestamp:       now,
    probability:     0.12,
    riskScore:       12,
    riskLevel:       'minimal',
    topContributors: [],
    dataQuality:     'mock',
    daysOfData:      features.days,
  };
}
