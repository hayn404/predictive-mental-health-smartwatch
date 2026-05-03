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
  AnxietyPrediction,
  PersonalBaseline,
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
  stressLevels?: Record<string, { min: number; max: number; label: string; color: string }>;
  anxietyLevels?: Record<string, { min: number; max: number; label: string; color: string }>;
  importances: Record<string, number>;
}

// ============================================================
// Model Loading & Caching
// ============================================================

let cachedAnxietyModel: XGBoostModel | null = null;

export function loadAnxietyModel(modelJson: XGBoostModel): void {
  cachedAnxietyModel = modelJson;
}

export function loadAnxietyModelFromString(jsonString: string): void {
  cachedAnxietyModel = JSON.parse(jsonString);
}

export function isAnxietyModelLoaded(): boolean {
  return cachedAnxietyModel !== null;
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
 * Run full XGBoost inference for Regression (Anxiety Index)
 * raw_score = base_score + sum(tree_leaf_values)
 */
function xgboostPredict(model: XGBoostModel, features: Record<string, number>): number {
  let rawScore = model.baseScore;

  for (const tree of model.trees) {
    rawScore += traverseTree(tree, features);
  }

  // Regression does not use sigmoid. Bound the output to 0-100.
  return Math.max(0, Math.min(100, rawScore));
}

// ============================================================
// Feature Normalization
// ============================================================

/**
 * Extract raw feature values for tree traversal.
 * The anxiety model was trained on raw E4 sensor data — split conditions in
 * the trees are calibrated to that raw scale (e.g. accelMagnitudeMean ~58-66,
 * not z-scores). Do NOT normalize before traversal.
 * activityType is encoded: sedentary=0, walking=1, active=2, sleeping=3.
 */
function extractRawFeatures(
  features: BiometricFeatureVector,
  model: XGBoostModel,
): Record<string, number> {
  const raw: Record<string, number> = {};
  const actMap: Record<string, number> = { sedentary: 0, walking: 1, active: 2, sleeping: 3 };

  for (const name of model.features) {
    const val = (features as any)[name] ?? 0;
    if (name === 'activityType') {
      raw[name] = typeof val === 'string' ? (actMap[val] ?? 0) : val;
    } else {
      raw[name] = val;
    }
  }

  return raw;
}

// ============================================================
// Public API: Stress Prediction
// ============================================================

/**
 * Predict Anxiety Index from a biometric feature vector.
 */
export function predictAnxiety(
  features: BiometricFeatureVector,
  baseline?: PersonalBaseline | null,
): AnxietyPrediction {
  if (!cachedAnxietyModel) {
    return ruleBasedAnxietyEstimate(features, baseline);
  }

  const rawFeatures = extractRawFeatures(features, cachedAnxietyModel);
  const anxietyIndex = xgboostPredict(cachedAnxietyModel, rawFeatures);
  const baselineDeviation = baseline
    ? (features.rmssd - baseline.rmssdMean) / (baseline.rmssdStd || 1)
    : 0;

  return {
    timestamp: features.timestamp,
    anxietyIndex,
    level: getAnxietyLevel(anxietyIndex),
    sustained: false, 
    baselineDeviation: Math.max(-1, Math.min(1, baselineDeviation / 3)),
  };
}

// ============================================================
// Rule-Based Fallback (when model isn't loaded)
// ============================================================

function getAnxietyLevel(index: number): AnxietyPrediction['level'] {
  if (index <= 20) return 'minimal';
  if (index <= 45) return 'mild';
  if (index <= 70) return 'moderate';
  return 'severe';
}

function ruleBasedAnxietyEstimate(
  features: BiometricFeatureVector,
  baseline?: PersonalBaseline | null,
): AnxietyPrediction {
  let index = 30;

  if (features.rmssd < 20) index += 30;
  else if (features.rmssd < 35) index += 15;

  if (features.hrMean > 95) index += 25;
  
  if (features.lfHfRatio > 4) index += 15;

  if (baseline) {
    if (features.rmssd < baseline.rmssdMean - baseline.rmssdStd * 1.5) index += 10;
  }

  index = Math.max(0, Math.min(100, Math.round(index)));
  const baselineDeviation = baseline ? (features.rmssd - baseline.rmssdMean) / (baseline.rmssdStd || 1) : 0;

  return {
    timestamp: features.timestamp,
    anxietyIndex: index,
    level: getAnxietyLevel(index),
    sustained: false,
    baselineDeviation: Math.max(-1, Math.min(1, baselineDeviation / 3)),
  };
}

// ============================================================
// Helper Functions
// ============================================================


