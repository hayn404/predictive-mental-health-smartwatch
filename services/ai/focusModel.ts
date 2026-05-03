/**
 * Seren — Focus Mood Model (On-Device Inference)
 * ================================================
 * XGBoost regressor predicting cognitive performance readiness (0-100)
 * from the same 29 wearable features as the anxiety model.
 *
 * Trained on the PhysioNet Wearable Exam Stress dataset —
 * real students wearing Empatica E4 during actual university exams.
 * Higher score = sharper focus. Lower score = scattered, needs intervention.
 */

import { BiometricFeatureVector, FocusPrediction, FocusLevel, ElevatedFeature } from './types';

interface XGBoostNode {
  nodeid: number;
  depth?: number;
  split?: string;
  split_condition?: number;
  yes?: number;
  no?: number;
  missing?: number;
  leaf?: number;
  children?: XGBoostNode[];
}

interface FocusModel {
  version: string;
  features: string[];
  numTrees: number;
  baseScore: number;
  learningRate: number;
  normalization?: { mean: Record<string, number>; std: Record<string, number> };
  trees: XGBoostNode[];
  importances: Record<string, number>;
}

let cachedFocusModel: FocusModel | null = null;

export function loadFocusModel(modelJson: FocusModel): void {
  cachedFocusModel = modelJson;
}

export function isFocusModelLoaded(): boolean {
  return cachedFocusModel !== null;
}

// ── Tree Traversal ────────────────────────────────────────────

function traverseTree(node: XGBoostNode, features: Record<string, number>): number {
  if (node.leaf !== undefined) return node.leaf;
  const val = features[node.split!];
  if (val === undefined || val === null || isNaN(val)) {
    const targetId = node.missing ?? node.yes!;
    const child = node.children?.find(c => c.nodeid === targetId);
    return child ? traverseTree(child, features) : 0;
  }
  const targetId = val < node.split_condition! ? node.yes! : node.no!;
  const child = node.children?.find(c => c.nodeid === targetId);
  return child ? traverseTree(child, features) : 0;
}

function xgboostPredict(model: FocusModel, features: Record<string, number>): number {
  let raw = model.baseScore;
  for (const tree of model.trees) raw += traverseTree(tree, features);
  return Math.max(0, Math.min(100, raw));
}

// ── Feature Extraction ────────────────────────────────────────

const ACT_MAP: Record<string, number> = { sedentary: 0, walking: 1, active: 2, sleeping: 3 };

function toRawFeatures(fv: BiometricFeatureVector, model: FocusModel): Record<string, number> {
  const raw: Record<string, number> = {};
  for (const name of model.features) {
    const val = (fv as any)[name] ?? 0;
    raw[name] = name === 'activityType'
      ? (typeof val === 'string' ? (ACT_MAP[val] ?? 0) : val)
      : val;
  }
  return raw;
}

// ── Feature Elevation Detection ───────────────────────────────
// Normal ranges derived from resting baseline. Features outside
// these ranges during cognitive tasks indicate physiological strain.

const FEATURE_NORMAL_RANGES: Record<string, { low: number; high: number; label: string; badDir: 'high' | 'low' | 'both' }> = {
  hrMean:       { low: 60,  high: 85,  label: 'Heart Rate',         badDir: 'high' },
  hrStd:        { low: 2,   high: 12,  label: 'Heart Rate Variability', badDir: 'high' },
  rmssd:        { low: 30,  high: 200, label: 'HRV (RMSSD)',        badDir: 'low'  },
  sdnn:         { low: 40,  high: 200, label: 'HRV (SDNN)',         badDir: 'low'  },
  lfHfRatio:    { low: 0.5, high: 2.5, label: 'Sympathovagal Balance', badDir: 'high' },
  sampleEntropy:{ low: 1.2, high: 3.0, label: 'Mental Complexity',  badDir: 'low'  },
  tempMean:     { low: 31,  high: 34,  label: 'Skin Temperature',   badDir: 'high' },
  tempSlope:    { low: -0.1,high: 0.05,label: 'Temperature Trend',  badDir: 'high' },
  dfaAlpha1:    { low: 0.5, high: 1.2, label: 'Stress Pattern',     badDir: 'high' },
  pnn50:        { low: 10,  high: 100, label: 'Parasympathetic Tone', badDir: 'low' },
};

function detectElevatedFeatures(
  raw: Record<string, number>,
  importances: Record<string, number>,
): ElevatedFeature[] {
  const elevated: ElevatedFeature[] = [];

  for (const [feat, range] of Object.entries(FEATURE_NORMAL_RANGES)) {
    const val = raw[feat];
    if (val === undefined) continue;
    const imp = importances[feat] ?? 0;

    if (range.badDir === 'high' && val > range.high) {
      elevated.push({ feature: feat, label: range.label, value: val, direction: 'high', impact: imp });
    } else if (range.badDir === 'low' && val < range.low) {
      elevated.push({ feature: feat, label: range.label, value: val, direction: 'low', impact: imp });
    }
  }

  return elevated.sort((a, b) => b.impact - a.impact).slice(0, 4);
}

// ── Focus Level ───────────────────────────────────────────────

function getFocusLevel(score: number): FocusLevel {
  if (score >= 75) return 'sharp';
  if (score >= 50) return 'steady';
  if (score >= 25) return 'drifting';
  return 'scattered';
}

// ── Rule-Based Fallback ───────────────────────────────────────

function ruleBasedFocus(fv: BiometricFeatureVector): number {
  let score = 60;
  if (fv.hrMean > 90)      score -= 20;
  else if (fv.hrMean > 80) score -= 8;
  if (fv.rmssd < 20)       score -= 20;
  else if (fv.rmssd < 35)  score -= 10;
  if (fv.lfHfRatio > 3)    score -= 15;
  if (fv.sampleEntropy < 1.0) score -= 10;
  if (fv.tempMean > 34.5)  score -= 8;
  return Math.max(0, Math.min(100, score));
}

// ── Public API ────────────────────────────────────────────────

export function predictFocus(fv: BiometricFeatureVector): FocusPrediction {
  if (!cachedFocusModel) {
    const score = ruleBasedFocus(fv);
    return {
      timestamp: fv.timestamp,
      focusScore: score,
      focusLevel: getFocusLevel(score),
      elevatedFeatures: [],
      groqTips: [],
      groqLoading: false,
    };
  }

  const raw = toRawFeatures(fv, cachedFocusModel);
  const focusScore = xgboostPredict(cachedFocusModel, raw);
  const elevatedFeatures = detectElevatedFeatures(raw, cachedFocusModel.importances);

  return {
    timestamp: fv.timestamp,
    focusScore,
    focusLevel: getFocusLevel(focusScore),
    elevatedFeatures,
    groqTips: [],
    groqLoading: false,
  };
}

export function getFocusColor(level: FocusLevel): string {
  switch (level) {
    case 'sharp':    return '#35e27e';
    case 'steady':   return '#6B8EFF';
    case 'drifting': return '#F59E0B';
    case 'scattered':return '#EF4444';
  }
}

export function getFocusDescription(level: FocusLevel, _isStudent: boolean): string {
  switch (level) {
    case 'sharp':
      return 'No distraction signals — clear mind, full cognitive readiness.';
    case 'steady':
      return 'Minimal distraction. Steady enough to sustain concentrated effort.';
    case 'drifting':
      return 'Distraction signals rising — your body may pull attention away mid-task.';
    case 'scattered':
      return 'High distraction detected. A brief reset before your task is strongly advised.';
  }
}
