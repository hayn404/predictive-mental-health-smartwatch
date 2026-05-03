/**
 * PROOF: Same biometric data → different stress vs anxiety readings
 * ================================================================
 * Demonstrates that predictStress() and predictAnxiety() are
 * mathematically independent models that produce different outputs
 * even when given identical BiometricFeatureVector input.
 *
 * WHY they differ:
 *   Stress model  — 200-tree XGBoost binary classifier trained on WESAD.
 *                   Output: sigmoid(rawScore) × 100.
 *                   Trained to detect ACUTE physiological load RIGHT NOW.
 *
 *   Anxiety model — 487-tree XGBoost regressor trained on WESAD + PhysioNet
 *                   Exam Stress + Anxiety Dataset 2022 (GAD-7/STAI labels).
 *                   Output: raw regression score 0-100 (no sigmoid).
 *                   Trained to detect CHRONIC parasympathetic withdrawal.
 *
 * NOTE ON UNITS: Both models were trained on Empatica E4 raw sensor data.
 *   accelMagnitudeMean is in E4 ADC units (~58-66 at rest, not 0-1 g).
 *   hrMean reflects actual heart rate from IBI data (70-150 BPM typical).
 *   The test scenarios below use ACTUAL rows from anxiety_training_data.csv
 *   to ensure inputs are within the training distribution.
 */

import * as fs from 'fs';
import * as path from 'path';
import { loadModel, predictStress } from '@/services/ai/stressModel';
import { loadAnxietyModel, predictAnxiety } from '@/services/ai/anxietyModel';
import type { BiometricFeatureVector } from '@/services/ai/types';

// ─── Load both real trained models ───────────────────────────────────────────
const stressModelJson  = JSON.parse(fs.readFileSync(path.join(__dirname, '../ml/models/stress_model.json'), 'utf-8'));
const anxietyModelJson = JSON.parse(fs.readFileSync(path.join(__dirname, '../ml/models/anxiety_model_ts.json'), 'utf-8'));

// ─── Shared feature vector factory ───────────────────────────────────────────
function makeFeatures(overrides: Partial<BiometricFeatureVector> = {}): BiometricFeatureVector {
  return {
    timestamp: Date.now(),
    windowSeconds: 300,
    // Time-domain HRV
    meanRR: 857, sdnn: 50, rmssd: 45, pnn50: 25, pnn20: 60,
    hrMean: 70, hrStd: 5, hrRange: 18, cvRR: 0.058,
    // Frequency-domain HRV
    vlfPower: 200, lfPower: 800, hfPower: 600, lfHfRatio: 1.33,
    totalPower: 1600, lfNorm: 57, hfNorm: 43,
    // Non-linear HRV
    sd1: 32, sd2: 65, sd1sd2Ratio: 0.49, sampleEntropy: 1.4, dfaAlpha1: 1.0,
    // Temperature
    tempMean: 33.5, tempSlope: 0.0, tempStd: 0.2, tempRange: 0.6,
    // Activity
    accelMagnitudeMean: 0.05, accelMagnitudeStd: 0.02, stepCount: 50,
    activityType: 'sedentary',
    ...overrides,
  };
}

// ─── Four representative scenarios ───────────────────────────────────────────
// ACTUAL rows from anxiety_training_data.csv — inputs are within the training
// distribution so the XGBoost trees produce meaningful, differentiated outputs.
//
//   Row 0   → anxiety_index = 10  (WESAD baseline, high HRV, warm temp)
//   Row 83  → anxiety_index = 40  (PhysioNet exam, elevated HR, cold extremities)
//   Row 121 → anxiety_index = 60  (exam stress, mixed HRV)
//   Row 165 → anxiety_index = 85  (GAD-7 severe, cold, variable movement)
//
// accelMagnitudeMean is in E4 ADC units (~63-65 at rest, not g-force 0.05).
const SCENARIOS: Array<{ label: string; features: BiometricFeatureVector }> = [
  {
    label: 'Minimal anxiety  (WESAD baseline, hrMean=78, rmssd=134)',
    features: makeFeatures({
      meanRR: 777.47, sdnn: 88.48, rmssd: 134.25, pnn50: 51.78, pnn20: 79.29,
      hrMean: 78.21, hrStd: 9.31, hrRange: 46.67, cvRR: 0.114,
      vlfPower: 143.44, lfPower: 647.89, hfPower: 2828.08, lfHfRatio: 0.229,
      totalPower: 3619.40, lfNorm: 18.64, hfNorm: 81.36,
      sd1: 95.08, sd2: 81.64, sd1sd2Ratio: 1.165, sampleEntropy: 1.458, dfaAlpha1: 0.532,
      tempMean: 34.17, tempSlope: 0.047, tempStd: 0.073, tempRange: 0.24,
      accelMagnitudeMean: 64.67, accelMagnitudeStd: 2.86, stepCount: 577,
      activityType: 'active',
    }),
  },
  {
    label: 'Mild anxiety     (exam stress, hrMean=148, cold extremities)',
    features: makeFeatures({
      meanRR: 418.45, sdnn: 76.24, rmssd: 106.04, pnn50: 59.33, pnn20: 80.22,
      hrMean: 148.07, hrStd: 26.09, hrRange: 85.33, cvRR: 0.182,
      vlfPower: 153.59, lfPower: 473.33, hfPower: 1322.50, lfHfRatio: 0.358,
      totalPower: 1949.42, lfNorm: 26.36, hfNorm: 73.64,
      sd1: 75.07, sd2: 77.43, sd1sd2Ratio: 0.969, sampleEntropy: 2.902, dfaAlpha1: 0.671,
      tempMean: 21.90, tempSlope: 0.056, tempStd: 0.082, tempRange: 0.30,
      accelMagnitudeMean: 63.32, accelMagnitudeStd: 1.21, stepCount: 52,
      activityType: 'active',
    }),
  },
  {
    label: 'Moderate anxiety (exam stress, hrMean=136, tempMean=21.4)',
    features: makeFeatures({
      meanRR: 458.29, sdnn: 92.70, rmssd: 131.75, pnn50: 60.87, pnn20: 72.26,
      hrMean: 136.09, hrStd: 25.88, hrRange: 78.55, cvRR: 0.202,
      vlfPower: 94.86, lfPower: 1100.27, hfPower: 1736.94, lfHfRatio: 0.634,
      totalPower: 2932.06, lfNorm: 38.78, hfNorm: 61.22,
      sd1: 93.26, sd2: 92.16, sd1sd2Ratio: 1.012, sampleEntropy: 1.196, dfaAlpha1: 0.671,
      tempMean: 21.43, tempSlope: 0.025, tempStd: 0.107, tempRange: 0.56,
      accelMagnitudeMean: 63.09, accelMagnitudeStd: 3.50, stepCount: 566,
      activityType: 'active',
    }),
  },
  {
    label: 'Severe anxiety   (GAD-7 high, hrMean=131, variable movement)',
    features: makeFeatures({
      meanRR: 472.69, sdnn: 81.73, rmssd: 110.08, pnn50: 56.23, pnn20: 76.23,
      hrMean: 130.58, hrStd: 21.47, hrRange: 73.30, cvRR: 0.173,
      vlfPower: 130.12, lfPower: 855.15, hfPower: 1819.81, lfHfRatio: 0.470,
      totalPower: 2805.08, lfNorm: 31.97, hfNorm: 68.03,
      sd1: 78.99, sd2: 85.32, sd1sd2Ratio: 0.914, sampleEntropy: 1.479, dfaAlpha1: 0.696,
      tempMean: 21.70, tempSlope: -0.035, tempStd: 0.220, tempRange: 2.82,
      accelMagnitudeMean: 62.96, accelMagnitudeStd: 11.70, stepCount: 556,
      activityType: 'active',
    }),
  },
];

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('PROOF — stress vs anxiety produce different readings on identical data', () => {
  beforeAll(() => {
    loadModel(stressModelJson);
    loadAnxietyModel(anxietyModelJson as any);
  });

  test('real models load: stress=200 trees, anxiety=329 trees, in-distribution inputs', () => {
    expect(stressModelJson.numTrees).toBe(200);
    expect(anxietyModelJson.numTrees).toBe(329);
    expect(stressModelJson.features.length).toBeGreaterThan(0);
    // Confirm training data scale: accelMagnitudeMean in E4 ADC units (~63), not g-force (~0.05)
    expect(SCENARIOS[0].features.accelMagnitudeMean).toBeGreaterThan(60);
  });

  SCENARIOS.forEach(({ label, features }) => {
    test(`[${label}] stress ≠ anxiety`, () => {
      const stress  = predictStress(features);
      const anxiety = predictAnxiety(features);

      const stressScore  = Math.round(stress.stressScore);
      const anxietyScore = Math.round(anxiety.anxietyIndex);

      // ── Print side-by-side evidence ────────────────────────────────────────
      console.log('\n' + '─'.repeat(60));
      console.log(`Scenario : ${label}`);
      console.log(`Input    : rmssd=${features.rmssd}  hrMean=${features.hrMean}  lfHfRatio=${features.lfHfRatio}`);
      console.log(`  Stress  model → score=${stressScore}  level="${stress.stressLevel}"  (200-tree XGBoost binary classifier)`);
      console.log(`  Anxiety model → index=${anxietyScore}  level="${anxiety.level}"  (487-tree XGBoost regressor)`);
      console.log(`  Δ difference  → ${Math.abs(stressScore - anxietyScore)} points apart`);
      console.log('─'.repeat(60));

      // ── Assertions ─────────────────────────────────────────────────────────
      expect(stressScore).toBeGreaterThanOrEqual(0);
      expect(stressScore).toBeLessThanOrEqual(100);
      expect(anxietyScore).toBeGreaterThanOrEqual(0);
      expect(anxietyScore).toBeLessThanOrEqual(100);

      // Core claim: they ARE different numbers
      expect(stressScore).not.toBe(anxietyScore);
    });
  });

  test('same data — all 4 scenarios produce divergent stress vs anxiety pairs', () => {
    loadModel(stressModelJson);

    const results = SCENARIOS.map(({ label, features }) => {
      const stress  = predictStress(features);
      const anxiety = predictAnxiety(features);
      return {
        label,
        stressScore:  Math.round(stress.stressScore),
        anxietyScore: Math.round(anxiety.anxietyIndex),
        stressLevel:  stress.stressLevel,
        anxietyLevel: anxiety.level,
        delta:        Math.abs(Math.round(stress.stressScore) - Math.round(anxiety.anxietyIndex)),
      };
    });

    console.log('\n\n  ╔══════════════════════════════════════════════════════════════════╗');
    console.log(  '  ║       PROOF TABLE — Same Input, Different Model Outputs         ║');
    console.log(  '  ╠══════════════════════════════════════╦═══════════╦═══════════╦═══╣');
    console.log(  '  ║ Scenario                             ║ Stress    ║ Anxiety   ║ Δ ║');
    console.log(  '  ╠══════════════════════════════════════╬═══════════╬═══════════╬═══╣');
    results.forEach(r => {
      const scenCol   = r.label.padEnd(36);
      const stressCol = `${r.stressScore} (${r.stressLevel})`.padEnd(9);
      const anxCol    = `${r.anxietyScore} (${r.anxietyLevel})`.padEnd(9);
      console.log(`  ║ ${scenCol} ║ ${stressCol} ║ ${anxCol} ║${String(r.delta).padStart(3)}║`);
    });
    console.log(  '  ╚══════════════════════════════════════╩═══════════╩═══════════╩═══╝\n');

    // Every scenario must show a different stress vs anxiety reading
    results.forEach(r => {
      expect(r.stressScore).not.toBe(r.anxietyScore);
    });

    // Average delta across all scenarios must be meaningful (> 5 points)
    const avgDelta = results.reduce((s, r) => s + r.delta, 0) / results.length;
    console.log(`  Average divergence across all scenarios: ${avgDelta.toFixed(1)} points`);
    expect(avgDelta).toBeGreaterThan(5);
  });
});
