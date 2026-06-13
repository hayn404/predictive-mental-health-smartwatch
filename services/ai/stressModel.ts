/**
 * Seren AI — Stress Prediction Model (On-Device Inference)
 * ==========================================================
 * Pure TypeScript XGBoost tree traversal for stress prediction.
 *
 * The model is loaded from a JSON file exported by the Python training pipeline.
 * No native ML libraries are needed — this runs anywhere JavaScript runs.
 *
 * How XGBoost inference works:
 * 1. Each tree is a binary decision tree with split conditions
 * 2. For each tree, traverse from root to leaf based on feature values
 * 3. Sum all leaf values across all trees
 * 4. Apply sigmoid to get probability
 * 5. Scale to 0-100 stress score
 */

import {
  BiometricFeatureVector,
  StressPrediction,
  StressLevel,
  StressContributor,
  PersonalBaseline,
  FEATURE_NAMES,
} from './types';

// ============================================================
// XGBoost Model Types (matches JSON export format)
// ============================================================

interface XGBoostNode {
  nodeid: number;
  depth?: number;
  split?: string;          // Feature name for split
  split_condition?: number; // Threshold value
  yes?: number;            // Node ID if condition is true (go left)
  no?: number;             // Node ID if condition is false (go right)
  missing?: number;        // Node ID for missing values
  leaf?: number;           // Leaf value (only for leaf nodes)
  children?: XGBoostNode[];
}

interface XGBoostModel {
  version: string;
  features: string[];
  numTrees: number;
  baseScore: number;
  learningRate: number;
  normalization: {
    mean: Record<string, number>;
    std: Record<string, number>;
  };
  trees: XGBoostNode[];
  stressLevels: Record<string, { min: number; max: number; label: string; color: string }>;
  importances: Record<string, number>;
  decisionThreshold?: number;   // calibrated prob cutoff for "stressed" (base-rate matched)
}

// ============================================================
// Model Loading & Caching
// ============================================================

let cachedModel: XGBoostModel | null = null;

/**
 * Load the stress model from JSON.
 * In the real app, this loads from assets. For now, it can be passed directly.
 */
export function loadModel(modelJson: XGBoostModel): void {
  cachedModel = modelJson;
}

/**
 * Load model from a JSON string (e.g., from expo-file-system or bundled asset).
 */
export function loadModelFromString(jsonString: string): void {
  cachedModel = JSON.parse(jsonString);
}

export function isModelLoaded(): boolean {
  return cachedModel !== null;
}

// ============================================================
// Tree Traversal (Core XGBoost Inference)
// ============================================================

/**
 * Traverse a single decision tree to get its leaf value.
 *
 * XGBoost tree structure:
 *   Internal node: has 'split' (feature name) and 'split_condition' (threshold)
 *     - If feature_value < split_condition -> go to 'yes' child
 *     - Else -> go to 'no' child
 *   Leaf node: has 'leaf' (prediction value)
 */
function traverseTree(node: XGBoostNode, features: Record<string, number>): number {
  // Base case: leaf node
  if (node.leaf !== undefined) {
    return node.leaf;
  }

  // Get feature value
  const featureValue = features[node.split!];
  const threshold = node.split_condition!;

  // Handle missing values
  if (featureValue === undefined || featureValue === null || isNaN(featureValue)) {
    const targetId = node.missing ?? node.yes!;
    const child = node.children?.find(c => c.nodeid === targetId);
    return child ? traverseTree(child, features) : 0;
  }

  // Decision: go left (yes) if value < threshold, else right (no)
  const targetId = featureValue < threshold ? node.yes! : node.no!;
  const child = node.children?.find(c => c.nodeid === targetId);

  return child ? traverseTree(child, features) : 0;
}

/**
 * Run full XGBoost inference: traverse all trees and sum leaf values.
 *
 * For binary classification:
 *   raw_score = base_score + sum(tree_leaf_values)
 *   probability = sigmoid(raw_score)
 */
function xgboostPredict(model: XGBoostModel, features: Record<string, number>): number {
  let rawScore = model.baseScore;

  for (const tree of model.trees) {
    rawScore += traverseTree(tree, features);
  }

  // Sigmoid activation for binary classification
  const probability = 1 / (1 + Math.exp(-rawScore));

  return probability;
}

// ============================================================
// Feature Normalization
// ============================================================

/**
 * Normalize features using the scaler parameters from training.
 * z = (x - mean) / std
 */
function normalizeFeatures(
  features: BiometricFeatureVector,
  model: XGBoostModel
): Record<string, number> {
  const normalized: Record<string, number> = {};

  model.features.forEach((name, i) => {
    const raw = (features as any)[name] ?? 0;
    const mean = model.normalization.mean[name] ?? 0;
    const stdDev = model.normalization.std[name] ?? 1;

    let value: number;
    // Handle activityType encoding: sedentary=0, walking=1, active=2, sleeping=3
    if (name === 'activityType') {
      const actMap: Record<string, number> = { sedentary: 0, walking: 1, active: 2, sleeping: 3 };
      const encoded = typeof raw === 'string' ? (actMap[raw] ?? 0) : raw;
      value = stdDev > 0 ? (encoded - mean) / stdDev : 0;
    } else {
      value = stdDev > 0 ? (raw - mean) / stdDev : 0;
    }

    normalized[name] = value;
    // Alias by positional index: XGBoost dumps trees with split keys "f0".."fN"
    // (feature indices) when trained on a numpy array, so node.split is "f{i}", not the
    // feature name. Expose both keys -> tree traversal resolves regardless of dump format.
    normalized[`f${i}`] = value;
  });

  return normalized;
}

// ============================================================
// Public API: Stress Prediction
// ============================================================

/**
 * Predict stress level from a biometric feature vector.
 *
 * @param features - 29 biometric features for a 5-min window
 * @param baseline - Optional personal baseline for contextualized scoring
 * @returns StressPrediction with score 0-100 and contributing factors
 */
export function predictStress(
  features: BiometricFeatureVector,
  baseline?: PersonalBaseline | null,
): StressPrediction {
  if (!cachedModel) {
    // Fallback: rule-based estimation if model isn't loaded
    return ruleBasedStressEstimate(features, baseline);
  }

  // S7 — input quality gate: a window with no usable RR/HR signal would feed garbage
  // (all-zero features) into the model. Fall back to the rule-based estimate at low
  // confidence rather than emit a confident-looking but meaningless score.
  if (!hasUsableSignal(features)) {
    const fb = ruleBasedStressEstimate(features, baseline);
    return { ...fb, confidence: Math.min(fb.confidence, 0.3) };
  }

  // Normalize features using training scaler
  const normalizedFeatures = normalizeFeatures(features, cachedModel);

  // Run XGBoost inference
  const probability = xgboostPredict(cachedModel, normalizedFeatures);

  // Scale to 0-100 stress score
  const stressScore = Math.round(probability * 100);

  // S2 — use the model's calibrated decision threshold for the binary stressed call
  // and to anchor the severity bands (instead of a hardcoded 0.5 / 25-50-75).
  const threshold = cachedModel.decisionThreshold ?? 0.5;
  const isStressed = probability >= threshold;
  const stressLevel = getStressLevel(stressScore, threshold);

  // Identify top contributing factors (from the model's own importances)
  const contributors = identifyContributors(features, cachedModel, baseline);

  return {
    timestamp: features.timestamp,
    stressScore,
    stressLevel,
    isStressed,
    confidence: computeConfidence(features),
    topContributors: contributors,
  };
}



// ============================================================
// Rule-Based Fallback (when model isn't loaded)
// ============================================================

function ruleBasedStressEstimate(
  features: BiometricFeatureVector,
  baseline?: PersonalBaseline | null,
): StressPrediction {
  let score = 30; // Start at moderate baseline

  // RMSSD: lower = more stress
  if (features.rmssd < 25) score += 30;
  else if (features.rmssd < 40) score += 15;
  else if (features.rmssd > 60) score -= 15;

  // Heart rate: higher = more stress
  if (features.hrMean > 90) score += 20;
  else if (features.hrMean > 80) score += 10;
  else if (features.hrMean < 65) score -= 10;

  // LF/HF ratio: higher = sympathetic dominance = stress
  if (features.lfHfRatio > 4) score += 15;
  else if (features.lfHfRatio > 2.5) score += 8;

  // Compare against personal baseline if available
  if (baseline) {
    if (features.rmssd < baseline.rmssdMean - baseline.rmssdStd) score += 10;
    if (features.hrMean > baseline.restingHrMean + baseline.restingHrStd) score += 10;
  }

  score = Math.max(0, Math.min(100, Math.round(score)));

  return {
    timestamp: features.timestamp,
    stressScore: score,
    stressLevel: getStressLevel(score),
    isStressed: score >= 50,
    confidence: 0.6, // Lower confidence for rule-based
    topContributors: [],
  };
}

// ============================================================
// Helper Functions
// ============================================================

/**
 * Map a 0-100 score to a 4-band severity, anchored on the model's calibrated decision
 * threshold (S2): scores below the threshold are "not stressed" (low/moderate), at/above
 * are "stressed" (elevated/high). Falls back to 0.5 (the old 25/50/75 behaviour).
 */
function getStressLevel(score: number, threshold = 0.5): StressLevel {
  const t = threshold * 100;                          // stressed/not boundary
  if (score < t * 0.5) return 'low';
  if (score < t) return 'moderate';                   // below threshold -> not stressed
  if (score < t + (100 - t) / 2) return 'elevated';   // at/above threshold -> stressed
  return 'high';
}

/**
 * S7 — does this window carry a usable cardiac signal? Without real RR intervals
 * (rmssd/meanRR == 0) or with an implausible heart rate, the HRV features are zeros
 * and the model output is meaningless.
 */
function hasUsableSignal(f: BiometricFeatureVector): boolean {
  return f.rmssd > 0 && f.meanRR > 0 && f.hrMean >= 30 && f.hrMean <= 220;
}



function computeConfidence(features: BiometricFeatureVector): number {
  // Confidence based on data quality
  let confidence = 0.9;

  // Lower confidence if HR seems unrealistic
  if (features.hrMean < 40 || features.hrMean > 150) confidence -= 0.3;

  // Lower confidence if RMSSD is 0 (no RR data)
  if (features.rmssd === 0) confidence -= 0.4;

  // Lower confidence if no temperature data
  if (features.tempMean === 0) confidence -= 0.1;

  return Math.max(0.1, Math.min(1, confidence));
}

// Direction-aware, human-readable labels for the features the stress model uses.
// {high} = value at/above the typical (training-mean) level, {low} = below it.
const FEATURE_LABELS: Record<string, { high: string; low: string }> = {
  meanRR:        { high: 'Slower heart rate',              low: 'Faster heart rate' },
  hrMean:        { high: 'Elevated heart rate',            low: 'Lower heart rate' },
  hrStd:         { high: 'Variable heart rate',            low: 'Steady heart rate' },
  hrRange:       { high: 'Wide heart-rate swings',         low: 'Narrow heart-rate range' },
  sdnn:          { high: 'Higher heart-rate variability',  low: 'Reduced heart-rate variability' },
  rmssd:         { high: 'Higher vagal (recovery) tone',   low: 'Low HRV (RMSSD)' },
  pnn50:         { high: 'High beat-to-beat variability',  low: 'Low beat-to-beat variability' },
  pnn20:         { high: 'High beat-to-beat variability',  low: 'Low beat-to-beat variability' },
  cvRR:          { high: 'Higher RR variability',          low: 'Lower RR variability' },
  sd1:           { high: 'Higher short-term variability',  low: 'Lower short-term variability' },
  sd2:           { high: 'Higher long-term variability',   low: 'Lower long-term variability' },
  sd1sd2Ratio:   { high: 'Balanced autonomic activity',    low: 'Sympathetic dominance' },
  sampleEntropy: { high: 'Complex (healthy) rhythm',       low: 'Reduced signal complexity' },
  dfaAlpha1:     { high: 'More rigid heart rhythm',        low: 'More adaptive heart rhythm' },
};

/**
 * S4 — explain a prediction from the MODEL's own feature importances (not a fixed
 * rule list). Ranks the model's features by importance, then labels each by whether
 * the current value is above/below the training-mean reference.
 */
function identifyContributors(
  features: BiometricFeatureVector,
  model: XGBoostModel,
  baseline?: PersonalBaseline | null,
): StressContributor[] {
  const imp = model.importances || {};
  const mean = model.normalization?.mean || {};
  const total = model.features.reduce((s, f) => s + (imp[f] || 0), 0) || 1;

  const ranked = [...model.features].sort((a, b) => (imp[b] || 0) - (imp[a] || 0));
  const out: StressContributor[] = [];
  for (const f of ranked) {
    const val = (features as any)[f];
    const labels = FEATURE_LABELS[f];
    if (typeof val !== 'number' || !labels || (imp[f] || 0) <= 0) continue;
    // Reference = per-user baseline where available, else the model's training mean.
    const ref = baselineRef(f, baseline) ?? mean[f] ?? val;
    out.push({
      feature: f,
      label: val >= ref ? labels.high : labels.low,
      impact: (imp[f] || 0) / total,
    });
    if (out.length >= 3) break;
  }
  return out;
}

/** Per-user reference value for a feature, if the baseline carries it. */
function baselineRef(feature: string, baseline?: PersonalBaseline | null): number | null {
  if (!baseline) return null;
  if (feature === 'rmssd') return baseline.rmssdMean;
  if (feature === 'sdnn') return baseline.sdnnMean;
  if (feature === 'hrMean') return baseline.restingHrMean;
  return null;
}
