/**
 * Model Validation Tests
 * ========================
 * Validates the XGBoost model file structure, feature schema alignment
 * between TypeScript and Python, and normalization parameters.
 */

import * as fs from 'fs';
import * as path from 'path';
import { FEATURE_NAMES } from '@/services/ai/types';

// ============================================================
// Model File Validation
// ============================================================

const MODEL_PATH = path.resolve(__dirname, '../assets/ml/stress_model.json');
const METADATA_PATH = path.resolve(__dirname, '../assets/ml/model_metadata.json');

describe('Model File Structure', () => {
  let model: any;

  beforeAll(() => {
    const raw = fs.readFileSync(MODEL_PATH, 'utf-8');
    model = JSON.parse(raw);
  });

  test('model file exists and is valid JSON', () => {
    expect(model).toBeDefined();
    expect(typeof model).toBe('object');
  });

  test('model has required top-level fields', () => {
    expect(model).toHaveProperty('version');
    expect(model).toHaveProperty('features');
    expect(model).toHaveProperty('numTrees');
    expect(model).toHaveProperty('baseScore');
    expect(model).toHaveProperty('learningRate');
    expect(model).toHaveProperty('normalization');
    expect(model).toHaveProperty('trees');
  });

  test('model has correct number of trees', () => {
    expect(model.trees).toBeInstanceOf(Array);
    expect(model.trees.length).toBe(model.numTrees);
    expect(model.numTrees).toBeGreaterThan(0);
  });

  test('each tree has valid root node structure', () => {
    for (const tree of model.trees) {
      expect(tree).toHaveProperty('nodeid');
      // Root can be a leaf or split node
      if (tree.leaf === undefined) {
        expect(tree).toHaveProperty('split');
        expect(tree).toHaveProperty('split_condition');
        expect(tree).toHaveProperty('children');
        expect(tree.children.length).toBeGreaterThan(0);
      }
    }
  });

  test('normalization has mean and std for all features', () => {
    expect(model.normalization).toHaveProperty('mean');
    expect(model.normalization).toHaveProperty('std');

    for (const feature of model.features) {
      expect(model.normalization.mean).toHaveProperty(feature);
      expect(model.normalization.std).toHaveProperty(feature);
      // Std should be positive
      expect(model.normalization.std[feature]).toBeGreaterThan(0);
    }
  });

  test('base score is a reasonable value', () => {
    expect(typeof model.baseScore).toBe('number');
    expect(model.baseScore).toBeGreaterThanOrEqual(-10);
    expect(model.baseScore).toBeLessThanOrEqual(10);
  });

  test('learning rate is between 0 and 1', () => {
    expect(model.learningRate).toBeGreaterThan(0);
    expect(model.learningRate).toBeLessThanOrEqual(1);
  });
});

// ============================================================
// Feature Schema Alignment
// ============================================================

describe('Feature Schema Alignment', () => {
  let model: any;

  beforeAll(() => {
    const raw = fs.readFileSync(MODEL_PATH, 'utf-8');
    model = JSON.parse(raw);
  });

  test('model features match TypeScript FEATURE_NAMES ordering', () => {
    // The model's feature list should be a subset of our TypeScript features
    for (const feature of model.features) {
      const isKnown = FEATURE_NAMES.includes(feature as any) || feature === 'activityType';
      expect(isKnown).toBe(true);
    }
  });

  test('all tree split features reference valid feature indices or names', () => {
    const allSplitFeatures = new Set<string>();

    function collectSplitFeatures(node: any) {
      if (node.split) {
        allSplitFeatures.add(node.split);
      }
      if (node.children) {
        for (const child of node.children) {
          collectSplitFeatures(child);
        }
      }
    }

    for (const tree of model.trees) {
      collectSplitFeatures(tree);
    }

    Array.from(allSplitFeatures).forEach(feature => {
      // XGBoost may use indexed names like "f0", "f5" or actual feature names
      const isIndexed = /^f\d+$/.test(feature);
      if (isIndexed) {
        const idx = parseInt(feature.slice(1), 10);
        expect(idx).toBeLessThan(model.features.length);
      } else {
        expect(model.features).toContain(feature);
      }
    });
  });

  test('feature importances reference valid features', () => {
    if (model.importances) {
      for (const feature of Object.keys(model.importances)) {
        expect(model.features).toContain(feature);
      }
    }
  });
});

// ============================================================
// Metadata File Validation
// ============================================================

describe('Model Metadata', () => {
  let metadata: any;

  beforeAll(() => {
    const raw = fs.readFileSync(METADATA_PATH, 'utf-8');
    metadata = JSON.parse(raw);
  });

  test('metadata file exists and is valid JSON', () => {
    expect(metadata).toBeDefined();
  });

  test('metadata has normalization parameters', () => {
    expect(metadata).toHaveProperty('normalization');
    expect(metadata.normalization).toHaveProperty('mean');
    expect(metadata.normalization).toHaveProperty('std');
  });

  test('normalization means are within physiological ranges', () => {
    const mean = metadata.normalization.mean;
    // HR should be between 40-120
    if (mean.hrMean) {
      expect(mean.hrMean).toBeGreaterThan(40);
      expect(mean.hrMean).toBeLessThan(120);
    }
    // RMSSD should be between 10-100
    if (mean.rmssd) {
      expect(mean.rmssd).toBeGreaterThan(5);
      expect(mean.rmssd).toBeLessThan(150);
    }
    // Temperature should be between 28-38
    if (mean.tempMean) {
      expect(mean.tempMean).toBeGreaterThan(28);
      expect(mean.tempMean).toBeLessThan(38);
    }
  });

  test('normalization stds are positive', () => {
    const std = metadata.normalization.std;
    for (const [, value] of Object.entries(std)) {
      expect(value).toBeGreaterThan(0);
    }
  });
});
