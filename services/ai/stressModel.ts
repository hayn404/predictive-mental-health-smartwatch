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
  AnxietyPrediction,
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

  for (const name of model.features) {
    const raw = (features as any)[name] ?? 0;
    const mean = model.normalization.mean[name] ?? 0;
    const stdDev = model.normalization.std[name] ?? 1;

    // Handle activityType encoding: sedentary=0, walking=1, active=2, sleeping=3
    if (name === 'activityType') {
      const actMap: Record<string, number> = { sedentary: 0, walking: 1, active: 2, sleeping: 3 };
      const encoded = typeof raw === 'string' ? (actMap[raw] ?? 0) : raw;
      normalized[name] = stdDev > 0 ? (encoded - mean) / stdDev : 0;
    } else {
      normalized[name] = stdDev > 0 ? (raw - mean) / stdDev : 0;
    }
  }

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

  // Normalize features using training scaler
  const normalizedFeatures = normalizeFeatures(features, cachedModel);

  // Run XGBoost inference
  const probability = xgboostPredict(cachedModel, normalizedFeatures);

  // Scale to 0-100 stress score
  const stressScore = Math.round(probability * 100);

  // Determine stress level
  const stressLevel = getStressLevel(stressScore);

  // Identify top contributing factors
  const contributors = identifyContributors(features, cachedModel, baseline);

  return {
    timestamp: features.timestamp,
    stressScore,
    stressLevel,
    confidence: computeConfidence(features),
    topContributors: contributors,
  };
}

/**
 * Predict anxiety index (combines stress score with temporal patterns).
 *
 * Anxiety differs from acute stress:
 * - It considers sustained HRV depression over time
 * - Weights recent sleep quality
 * - Considers time-of-day patterns
 */
export function predictAnxiety(
  stressPrediction: StressPrediction,
  features: BiometricFeatureVector,
  baseline: PersonalBaseline | null,
  recentSleepQuality: number | null,
): AnxietyPrediction {
  let anxietyIndex = stressPrediction.stressScore;

  if (baseline) {
    // Factor 1: HRV deviation from personal baseline
    const rmssdDeviation = baseline.rmssdMean > 0
      ? (features.rmssd - baseline.rmssdMean) / baseline.rmssdStd
      : 0;
    // Negative deviation (lower HRV than usual) increases anxiety
    if (rmssdDeviation < -1) {
      anxietyIndex += Math.min(15, Math.abs(rmssdDeviation) * 5);
    }

    // Factor 2: HR elevation above personal resting
    const hrDeviation = baseline.restingHrMean > 0
      ? (features.hrMean - baseline.restingHrMean) / baseline.restingHrStd
      : 0;
    if (hrDeviation > 1.5) {
      anxietyIndex += Math.min(10, hrDeviation * 3);
    }
  }

  // Factor 3: Poor recent sleep amplifies anxiety
  if (recentSleepQuality !== null && recentSleepQuality < 60) {
    anxietyIndex += (60 - recentSleepQuality) * 0.2; // Up to +12 for very poor sleep
  }

  // Factor 4: LF/HF ratio elevation (sympathetic dominance)
  if (features.lfHfRatio > 3) {
    anxietyIndex += Math.min(10, (features.lfHfRatio - 3) * 2);
  }

  anxietyIndex = Math.max(0, Math.min(100, Math.round(anxietyIndex)));

  const baselineDeviation = baseline
    ? (features.rmssd - baseline.rmssdMean) / (baseline.rmssdStd || 1)
    : 0;

  return {
    timestamp: features.timestamp,
    anxietyIndex,
    level: getAnxietyLevel(anxietyIndex),
    sustained: false, // Set by the pipeline when it tracks duration
    baselineDeviation: Math.max(-1, Math.min(1, baselineDeviation / 3)),
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
    confidence: 0.6, // Lower confidence for rule-based
    topContributors: [],
  };
}

// ============================================================
// Helper Functions
// ============================================================

function getStressLevel(score: number): StressLevel {
  if (score <= 25) return 'low';
  if (score <= 50) return 'moderate';
  if (score <= 75) return 'elevated';
  return 'high';
}

function getAnxietyLevel(index: number): AnxietyPrediction['level'] {
  if (index <= 20) return 'minimal';
  if (index <= 45) return 'mild';
  if (index <= 70) return 'moderate';
  return 'severe';
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

function identifyContributors(
  features: BiometricFeatureVector,
  model: XGBoostModel,
  baseline?: PersonalBaseline | null,
): StressContributor[] {
  const contributors: StressContributor[] = [];
  const importances = model.importances;

  // Get top features by importance and check their state
  const featureChecks: { feature: string; label: string; condition: boolean; impact: number }[] = [
    {
      feature: 'rmssd',
      label: features.rmssd < 30 ? 'Very low HRV' : 'Low HRV',
      condition: features.rmssd < 40,
      impact: importances['rmssd'] || 0.15,
    },
    {
      feature: 'lfHfRatio',
      label: 'High sympathetic activation',
      condition: features.lfHfRatio > 2.5,
      impact: importances['lfHfRatio'] || 0.12,
    },
    {
      feature: 'hrMean',
      label: 'Elevated heart rate',
      condition: features.hrMean > 80,
      impact: importances['hrMean'] || 0.10,
    },
    {
      feature: 'sdnn',
      label: 'Reduced heart rate variability',
      condition: features.sdnn < 40,
      impact: importances['sdnn'] || 0.10,
    },
    {
      feature: 'tempSlope',
      label: 'Dropping skin temperature',
      condition: features.tempSlope < -0.05,
      impact: importances['tempSlope'] || 0.05,
    },
    {
      feature: 'hfPower',
      label: 'Low vagal tone',
      condition: features.hfPower < (features.totalPower * 0.2),
      impact: importances['hfPower'] || 0.08,
    },
  ];

  for (const check of featureChecks) {
    if (check.condition) {
      contributors.push({
        feature: check.feature,
        label: check.label,
        impact: check.impact,
      });
    }
  }

  // Sort by impact and take top 3
  contributors.sort((a, b) => b.impact - a.impact);
  return contributors.slice(0, 3);
}
