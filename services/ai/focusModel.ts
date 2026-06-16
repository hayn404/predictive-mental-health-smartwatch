/**
 * Seren — Focus / Cognitive-Effort Model (On-Device Inference)
 * ============================================================
 * Pure-TS XGBoost tree traversal, same engine as the stress model. The model is a
 * binary classifier trained on the CogWear dataset (rest vs. Stroop cognitive effort)
 * using the Samsung Galaxy Watch4 PPG — the exact signal this app gets on-device.
 *
 * Output = probability of cognitive effort/engagement (0–100). Higher = the user's
 * physiology shows active cognitive engagement; lower = at rest / disengaged.
 *
 * Headline accuracy lever (same as stress): PER-USER normalization. We z-score the
 * live feature window against the USER's own recent history, falling back to the
 * model's global (population) scaler until enough personal history exists.
 *
 * LOSO-CV (leave-one-subject-out) on CogWear Galaxy Watch4 PPG:
 *   acc 0.73 · F1 0.78 · MCC 0.42 · AUC 0.77  (vs 0.64 majority baseline)
 */

import { BiometricFeatureVector, FocusPrediction, FocusLevel, ElevatedFeature } from './types';
import { computePerUserNorm, type PerUserNorm } from './perUserNorm';

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

interface FocusModel {
  version: string;
  task?: string;
  features: string[];
  baseScore: number;
  applySigmoid?: boolean;
  normalization: { mean: Record<string, number>; std: Record<string, number> };
  trees: XGBoostNode[];
  importances?: Record<string, number>;
  decisionThreshold?: number;
}

let cachedFocusModel: FocusModel | null = null;

export function loadFocusModel(modelJson: FocusModel): void {
  cachedFocusModel = modelJson;
}

export function isFocusModelLoaded(): boolean {
  return cachedFocusModel !== null && Array.isArray(cachedFocusModel.features);
}

// ── Tree Traversal ────────────────────────────────────────────

function traverseTree(node: XGBoostNode, features: Record<string, number>): number {
  if (node.leaf !== undefined) return node.leaf;
  const val = features[node.split!];
  if (val === undefined || val === null || isNaN(val)) {
    const id = node.missing ?? node.yes!;
    const child = node.children?.find(c => c.nodeid === id);
    return child ? traverseTree(child, features) : 0;
  }
  const id = val < node.split_condition! ? node.yes! : node.no!;
  const child = node.children?.find(c => c.nodeid === id);
  return child ? traverseTree(child, features) : 0;
}

function modelPredictProb(model: FocusModel, features: Record<string, number>): number {
  let margin = model.baseScore;
  for (const tree of model.trees) margin += traverseTree(tree, features);
  return model.applySigmoid === false ? margin : 1 / (1 + Math.exp(-margin));
}

// ── Per-user (+ global fallback) normalization — the accuracy lever ────────────

function normalizeFeatures(
  fv: BiometricFeatureVector,
  model: FocusModel,
  perUser: PerUserNorm | null,
): Record<string, number> {
  const out: Record<string, number> = {};
  model.features.forEach((name, i) => {
    const raw = (fv as any)[name] ?? 0;
    const useUser = !!perUser && name in perUser.mean && (perUser.std[name] ?? 0) > 0;
    const mean = useUser ? perUser!.mean[name] : (model.normalization.mean[name] ?? 0);
    const std = useUser ? perUser!.std[name] : (model.normalization.std[name] ?? 1);
    const z = std > 0 ? (raw - mean) / std : 0;
    out[name] = z;
    out[`f${i}`] = z; // alias both name and index so traversal resolves either dump format
  });
  return out;
}

// ── Focus Level Buckets ───────────────────────────────────────

export function getFocusLevel(score: number): FocusLevel {
  if (score >= 75) return 'sharp';
  if (score >= 50) return 'steady';
  if (score >= 25) return 'drifting';
  return 'scattered';
}

// ── Elevated-signal detection (drives the amber "LIVE SIGNALS" tint) ───────────

const FEATURE_NORMAL_RANGES: Record<string, { low: number; high: number; label: string; badDir: 'high' | 'low' }> = {
  hrMean:        { low: 60,  high: 85,  label: 'Heart Rate',             badDir: 'high' },
  hrStd:         { low: 2,   high: 12,  label: 'Heart Rate Variability', badDir: 'high' },
  rmssd:         { low: 30,  high: 200, label: 'HRV (RMSSD)',            badDir: 'low'  },
  sdnn:          { low: 40,  high: 200, label: 'HRV (SDNN)',             badDir: 'low'  },
  sampleEntropy: { low: 1.2, high: 3.0, label: 'Mental Complexity',      badDir: 'low'  },
  pnn50:         { low: 10,  high: 100, label: 'Parasympathetic Tone',   badDir: 'low'  },
};

function detectElevatedFeatures(fv: BiometricFeatureVector): ElevatedFeature[] {
  const out: (ElevatedFeature & { _sev: number })[] = [];
  for (const [feat, r] of Object.entries(FEATURE_NORMAL_RANGES)) {
    const val = (fv as any)[feat];
    if (typeof val !== 'number') continue;
    const span = r.high - r.low || 1;
    if (r.badDir === 'high' && val > r.high) out.push({ feature: feat, label: r.label, value: val, direction: 'high', impact: 0, _sev: (val - r.high) / span });
    else if (r.badDir === 'low' && val < r.low) out.push({ feature: feat, label: r.label, value: val, direction: 'low', impact: 0, _sev: (r.low - val) / span });
  }
  return out.sort((a, b) => b._sev - a._sev).slice(0, 4).map(({ _sev, ...e }) => e);
}

// ── Rule-based fallback (only if the model fails to load) ──────────────────────

function ruleBasedFocus(fv: BiometricFeatureVector): number {
  let s = 60;
  if (fv.hrMean > 90) s -= 20; else if (fv.hrMean > 80) s -= 8;
  if (fv.rmssd < 20) s -= 20; else if (fv.rmssd < 35) s -= 10;
  if (fv.lfHfRatio > 3) s -= 15;
  if (fv.sampleEntropy < 1.0) s -= 10;
  return Math.max(0, Math.min(100, s));
}

// ── Public API ────────────────────────────────────────────────

/**
 * @param fv             live biometric feature window from the watch
 * @param recentWindows  the user's recent windows, for per-user normalization
 *                       (same history the stress model uses)
 */
export function predictFocus(
  fv: BiometricFeatureVector,
  recentWindows?: BiometricFeatureVector[] | null,
): FocusPrediction {
  if (!isModelUsable()) {
    const score = ruleBasedFocus(fv);
    return base(fv, score);
  }
  try {
    const perUser = computePerUserNorm(recentWindows ?? null, cachedFocusModel!.features, 20);
    const normalized = normalizeFeatures(fv, cachedFocusModel!, perUser);
    const prob = modelPredictProb(cachedFocusModel!, normalized);
    const focusScore = Math.max(0, Math.min(100, Math.round(prob * 100)));
    return base(fv, focusScore);
  } catch {
    return base(fv, ruleBasedFocus(fv));
  }
}

function isModelUsable(): boolean {
  return cachedFocusModel !== null && Array.isArray(cachedFocusModel.features) && Array.isArray(cachedFocusModel.trees);
}

function base(fv: BiometricFeatureVector, focusScore: number): FocusPrediction {
  return {
    timestamp: fv.timestamp,
    focusScore,
    focusLevel: getFocusLevel(focusScore),
    elevatedFeatures: detectElevatedFeatures(fv),
    groqTips: [],
    groqLoading: false,
  };
}

export function getFocusColor(level: FocusLevel): string {
  switch (level) {
    case 'sharp':     return '#35e27e';
    case 'steady':    return '#6B8EFF';
    case 'drifting':  return '#F59E0B';
    case 'scattered': return '#EF4444';
  }
}

export function getFocusDescription(level: FocusLevel, _isStudent: boolean): string {
  switch (level) {
    case 'sharp':     return 'Strong cognitive engagement — your body is locked into the task.';
    case 'steady':    return 'Moderate engagement. Steady enough to sustain concentrated effort.';
    case 'drifting':  return 'Engagement is low — your body may be drifting away from the task.';
    case 'scattered': return 'Very low engagement detected. A brief reset before your task may help.';
  }
}
