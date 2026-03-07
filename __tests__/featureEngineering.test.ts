/**
 * Unit Tests — Feature Engineering (HRV Metrics)
 * ================================================
 * Tests the 29-feature extraction pipeline: time-domain, frequency-domain,
 * non-linear HRV, temperature, and activity features.
 */

import {
  hrSamplesToRRIntervals,
  computeTimeDomain,
  computeFrequencyDomain,
  computeNonLinear,
  computeTemperatureFeatures,
  computeActivityFeatures,
  extractFeatures,
} from '@/services/ai/featureEngineering';
import type { RawHeartRateSample, RawTemperatureSample } from '@/services/ai/types';

// ============================================================
// Test Helpers
// ============================================================

function makeHRSamples(bpms: number[]): RawHeartRateSample[] {
  return bpms.map((bpm, i) => ({
    timestamp: Date.now() + i * 1000,
    bpm,
    source: 'test',
  }));
}

function makeTempSamples(temps: number[]): RawTemperatureSample[] {
  return temps.map((t, i) => ({
    timestamp: Date.now() + i * 60000,
    temperatureCelsius: t,
  }));
}

// Generate a steady 70 BPM signal (RR ~ 857ms)
function steadyHRSamples(n: number, bpm = 70): RawHeartRateSample[] {
  return makeHRSamples(Array(n).fill(bpm));
}

// Generate RR intervals with known properties
function steadyRR(n: number, meanRR = 857): number[] {
  return Array(n).fill(meanRR);
}

// ============================================================
// hrSamplesToRRIntervals
// ============================================================

describe('hrSamplesToRRIntervals', () => {
  test('returns empty for fewer than 2 samples', () => {
    expect(hrSamplesToRRIntervals([])).toEqual([]);
    expect(hrSamplesToRRIntervals(makeHRSamples([70]))).toEqual([]);
  });

  test('converts BPM to RR intervals correctly', () => {
    const samples = makeHRSamples([60, 60, 60, 60, 60]);
    const rr = hrSamplesToRRIntervals(samples);
    // 60 BPM => 60000/60 = 1000ms
    expect(rr.length).toBeGreaterThan(0);
    rr.forEach(interval => expect(interval).toBeCloseTo(1000, 0));
  });

  test('filters out physiologically impossible BPM values', () => {
    const samples = makeHRSamples([20, 70, 70, 70, 250, 70]);
    const rr = hrSamplesToRRIntervals(samples);
    // BPM 20 => RR 3000ms (>2000, filtered) and BPM 250 => RR 240ms (<300, filtered)
    // Only BPM 70 should survive
    rr.forEach(interval => {
      expect(interval).toBeGreaterThanOrEqual(300);
      expect(interval).toBeLessThanOrEqual(2000);
    });
  });

  test('removes outliers more than 30% from median', () => {
    // Mostly 70 BPM (857ms), one at 120 BPM (500ms) — >30% deviation
    const bpms = [70, 70, 70, 70, 70, 120, 70, 70, 70, 70];
    const rr = hrSamplesToRRIntervals(makeHRSamples(bpms));
    // The 500ms outlier should be filtered
    rr.forEach(interval => expect(interval).toBeCloseTo(857, -1));
  });
});

// ============================================================
// computeTimeDomain
// ============================================================

describe('computeTimeDomain', () => {
  test('returns zeros for fewer than 5 intervals', () => {
    const result = computeTimeDomain([800, 850, 900]);
    expect(result.meanRR).toBe(0);
    expect(result.sdnn).toBe(0);
    expect(result.rmssd).toBe(0);
  });

  test('computes correct mean RR', () => {
    const rr = [800, 850, 900, 850, 800];
    const result = computeTimeDomain(rr);
    expect(result.meanRR).toBeCloseTo(840, 0);
  });

  test('SDNN is zero for constant intervals', () => {
    const rr = steadyRR(20);
    const result = computeTimeDomain(rr);
    expect(result.sdnn).toBeCloseTo(0, 5);
    expect(result.rmssd).toBeCloseTo(0, 5);
  });

  test('pNN50 and pNN20 are within 0-100 range', () => {
    // Alternating intervals for high variability
    const rr = Array.from({ length: 30 }, (_, i) => i % 2 === 0 ? 800 : 900);
    const result = computeTimeDomain(rr);
    expect(result.pnn50).toBeGreaterThanOrEqual(0);
    expect(result.pnn50).toBeLessThanOrEqual(100);
    expect(result.pnn20).toBeGreaterThanOrEqual(0);
    expect(result.pnn20).toBeLessThanOrEqual(100);
  });

  test('heart rate metrics are derived from RR', () => {
    const rr = steadyRR(10, 1000); // 1000ms = 60 BPM
    const result = computeTimeDomain(rr);
    expect(result.hrMean).toBeCloseTo(60, 0);
    expect(result.hrRange).toBeCloseTo(0, 5);
  });

  test('coefficient of variation (cvRR) = SDNN / meanRR', () => {
    const rr = [800, 850, 900, 850, 800, 850, 900, 850];
    const result = computeTimeDomain(rr);
    expect(result.cvRR).toBeCloseTo(result.sdnn / result.meanRR, 5);
  });
});

// ============================================================
// computeFrequencyDomain
// ============================================================

describe('computeFrequencyDomain', () => {
  test('returns defaults for fewer than 30 intervals', () => {
    const result = computeFrequencyDomain(steadyRR(10));
    expect(result.lfNorm).toBe(50);
    expect(result.hfNorm).toBe(50);
    expect(result.totalPower).toBe(0);
  });

  test('produces non-negative power values for valid input', () => {
    // Simulate 5 minutes of HR data with some variability
    const rr = Array.from({ length: 300 }, (_, i) =>
      857 + 30 * Math.sin(2 * Math.PI * i / 50) + 10 * Math.sin(2 * Math.PI * i / 10)
    );
    const result = computeFrequencyDomain(rr);
    expect(result.vlfPower).toBeGreaterThanOrEqual(0);
    expect(result.lfPower).toBeGreaterThanOrEqual(0);
    expect(result.hfPower).toBeGreaterThanOrEqual(0);
    expect(result.totalPower).toBeGreaterThanOrEqual(0);
  });

  test('lfNorm + hfNorm approximately equals 100', () => {
    const rr = Array.from({ length: 300 }, (_, i) =>
      857 + 30 * Math.sin(2 * Math.PI * i / 50) + 20 * Math.sin(2 * Math.PI * i / 8)
    );
    const result = computeFrequencyDomain(rr);
    if (result.totalPower > 0) {
      expect(result.lfNorm + result.hfNorm).toBeCloseTo(100, 0);
    }
  });

  test('LF/HF ratio is non-negative', () => {
    const rr = Array.from({ length: 300 }, (_, i) => 857 + 20 * Math.sin(2 * Math.PI * i / 30));
    const result = computeFrequencyDomain(rr);
    expect(result.lfHfRatio).toBeGreaterThanOrEqual(0);
  });
});

// ============================================================
// computeNonLinear
// ============================================================

describe('computeNonLinear', () => {
  test('returns zeros for fewer than 10 intervals', () => {
    const result = computeNonLinear([800, 850, 900, 850, 800]);
    expect(result.sd1).toBe(0);
    expect(result.sd2).toBe(0);
    expect(result.sampleEntropy).toBe(0);
  });

  test('SD1 measures short-term variability (Poincare)', () => {
    // Low variability: constant intervals
    const steady = computeNonLinear(steadyRR(30));
    // High variability: alternating
    const alternating = Array.from({ length: 30 }, (_, i) => i % 2 === 0 ? 800 : 900);
    const variable = computeNonLinear(alternating);

    expect(variable.sd1).toBeGreaterThan(steady.sd1);
  });

  test('SD1/SD2 ratio is non-negative', () => {
    const rr = Array.from({ length: 50 }, (_, i) => 857 + 20 * Math.sin(i * 0.5));
    const result = computeNonLinear(rr);
    expect(result.sd1sd2Ratio).toBeGreaterThanOrEqual(0);
  });

  test('sample entropy is non-negative', () => {
    const rr = Array.from({ length: 50 }, (_, i) => 857 + 30 * Math.sin(i * 0.3) + Math.random() * 5);
    const result = computeNonLinear(rr);
    expect(result.sampleEntropy).toBeGreaterThanOrEqual(0);
  });

  test('DFA alpha1 is computed for sufficient data', () => {
    const rr = Array.from({ length: 100 }, (_, i) => 857 + 20 * Math.sin(i * 0.2) + Math.random() * 10);
    const result = computeNonLinear(rr);
    // DFA alpha1 should be a finite number for valid data
    expect(isFinite(result.dfaAlpha1)).toBe(true);
  });
});

// ============================================================
// computeTemperatureFeatures
// ============================================================

describe('computeTemperatureFeatures', () => {
  test('returns defaults for fewer than 2 samples', () => {
    const result = computeTemperatureFeatures([]);
    expect(result.tempMean).toBe(33.5);
    expect(result.tempSlope).toBe(0);
  });

  test('computes correct mean temperature', () => {
    const samples = makeTempSamples([33.0, 33.5, 34.0]);
    const result = computeTemperatureFeatures(samples);
    expect(result.tempMean).toBeCloseTo(33.5, 1);
  });

  test('detects positive temperature slope (warming)', () => {
    const samples = makeTempSamples([32.0, 33.0, 34.0]);
    const result = computeTemperatureFeatures(samples);
    expect(result.tempSlope).toBeGreaterThan(0);
  });

  test('detects negative temperature slope (cooling)', () => {
    const samples = makeTempSamples([34.0, 33.0, 32.0]);
    const result = computeTemperatureFeatures(samples);
    expect(result.tempSlope).toBeLessThan(0);
  });

  test('range equals max - min', () => {
    const samples = makeTempSamples([32.0, 33.0, 35.0, 34.0]);
    const result = computeTemperatureFeatures(samples);
    expect(result.tempRange).toBeCloseTo(3.0, 1);
  });
});

// ============================================================
// computeActivityFeatures
// ============================================================

describe('computeActivityFeatures', () => {
  test('classifies sedentary (< 1 step/min)', () => {
    const result = computeActivityFeatures(0, 300, 70);
    expect(result.activityType).toBe('sedentary');
    expect(result.stepCount).toBe(0);
  });

  test('classifies walking (1-40 steps/min)', () => {
    const result = computeActivityFeatures(100, 300, 80); // ~20 steps/min
    expect(result.activityType).toBe('walking');
  });

  test('classifies active (>40 steps/min)', () => {
    const result = computeActivityFeatures(1200, 300, 120); // 240 steps/min
    expect(result.activityType).toBe('active');
  });

  test('overrides to sleeping when HR < 55 and no steps', () => {
    const result = computeActivityFeatures(0, 300, 50);
    expect(result.activityType).toBe('sleeping');
  });

  test('accelerometer magnitude increases with activity level', () => {
    const sedentary = computeActivityFeatures(0, 300, 70);
    const walking = computeActivityFeatures(100, 300, 80);
    const active = computeActivityFeatures(1200, 300, 120);

    expect(walking.accelMagnitudeMean).toBeGreaterThan(sedentary.accelMagnitudeMean);
    expect(active.accelMagnitudeMean).toBeGreaterThan(walking.accelMagnitudeMean);
  });
});

// ============================================================
// extractFeatures (integration of all feature groups)
// ============================================================

describe('extractFeatures', () => {
  test('produces a complete 29-feature vector', () => {
    const result = extractFeatures({
      hrSamples: steadyHRSamples(60, 70),
      hrvSamples: [{ timestamp: Date.now(), rmssd: 45 }],
      tempSamples: makeTempSamples([33.0, 33.5, 34.0]),
      stepCount: 0,
      windowSeconds: 300,
      timestamp: Date.now(),
    });

    // Verify all 29 feature fields exist
    expect(result.meanRR).toBeDefined();
    expect(result.sdnn).toBeDefined();
    expect(result.rmssd).toBeDefined();
    expect(result.pnn50).toBeDefined();
    expect(result.pnn20).toBeDefined();
    expect(result.hrMean).toBeDefined();
    expect(result.hrStd).toBeDefined();
    expect(result.hrRange).toBeDefined();
    expect(result.cvRR).toBeDefined();
    expect(result.vlfPower).toBeDefined();
    expect(result.lfPower).toBeDefined();
    expect(result.hfPower).toBeDefined();
    expect(result.lfHfRatio).toBeDefined();
    expect(result.totalPower).toBeDefined();
    expect(result.lfNorm).toBeDefined();
    expect(result.hfNorm).toBeDefined();
    expect(result.sd1).toBeDefined();
    expect(result.sd2).toBeDefined();
    expect(result.sd1sd2Ratio).toBeDefined();
    expect(result.sampleEntropy).toBeDefined();
    expect(result.dfaAlpha1).toBeDefined();
    expect(result.tempMean).toBeDefined();
    expect(result.tempSlope).toBeDefined();
    expect(result.tempStd).toBeDefined();
    expect(result.tempRange).toBeDefined();
    expect(result.accelMagnitudeMean).toBeDefined();
    expect(result.accelMagnitudeStd).toBeDefined();
    expect(result.stepCount).toBeDefined();
    expect(result.activityType).toBeDefined();
  });

  test('uses direct RMSSD from Health Connect when available and divergent', () => {
    const directRmssd = 60;
    const result = extractFeatures({
      hrSamples: steadyHRSamples(60, 70), // constant => computed RMSSD ≈ 0
      hrvSamples: [{ timestamp: Date.now(), rmssd: directRmssd }],
      tempSamples: makeTempSamples([33.0, 33.5]),
      stepCount: 0,
      windowSeconds: 300,
      timestamp: Date.now(),
    });

    // When computed RMSSD differs from direct by >20ms, direct is used
    expect(result.rmssd).toBeCloseTo(directRmssd, 0);
  });

  test('timestamp and windowSeconds are preserved', () => {
    const now = Date.now();
    const result = extractFeatures({
      hrSamples: steadyHRSamples(20, 70),
      hrvSamples: [],
      tempSamples: [],
      stepCount: 50,
      windowSeconds: 300,
      timestamp: now,
    });

    expect(result.timestamp).toBe(now);
    expect(result.windowSeconds).toBe(300);
    expect(result.stepCount).toBe(50);
  });
});
