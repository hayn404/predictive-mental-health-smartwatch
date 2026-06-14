/**
 * Unit tests — per-user (per-subject) normalization (gap S1).
 */
import { computePerUserNorm } from '@/services/ai/perUserNorm';
import type { BiometricFeatureVector } from '@/services/ai/types';

function fv(rmssd: number, hrMean: number): BiometricFeatureVector {
  // Only the fields under test need to be present for these unit tests.
  return { rmssd, hrMean } as unknown as BiometricFeatureVector;
}

describe('computePerUserNorm', () => {
  test('returns null below minSamples (cold-start -> caller uses global)', () => {
    const windows = Array.from({ length: 10 }, () => fv(40, 70));
    expect(computePerUserNorm(windows, ['rmssd', 'hrMean'], 30)).toBeNull();
    expect(computePerUserNorm(null, ['rmssd'], 30)).toBeNull();
    expect(computePerUserNorm([], ['rmssd'], 30)).toBeNull();
  });

  test('computes per-feature mean/std at/above minSamples', () => {
    // rmssd = 10,20,30 repeated -> mean 20; hrMean constant 60 -> std 0
    const windows: BiometricFeatureVector[] = [];
    for (let i = 0; i < 30; i++) windows.push(fv([10, 20, 30][i % 3], 60));
    const n = computePerUserNorm(windows, ['rmssd', 'hrMean'], 30)!;
    expect(n).not.toBeNull();
    expect(n.sampleCount).toBe(30);
    expect(n.mean.rmssd).toBeCloseTo(20, 6);
    expect(n.std.rmssd).toBeGreaterThan(0);
    expect(n.mean.hrMean).toBeCloseTo(60, 6);
    expect(n.std.hrMean).toBeCloseTo(0, 6); // constant feature -> std 0 (caller falls back)
  });

  test('omits features with too few usable (numeric, finite) samples', () => {
    const windows: BiometricFeatureVector[] = [];
    for (let i = 0; i < 30; i++) {
      // rmssd always valid; hrMean only valid on a handful of windows
      const w = fv(40, NaN);
      if (i < 5) (w as any).hrMean = 70;
      windows.push(w);
    }
    const n = computePerUserNorm(windows, ['rmssd', 'hrMean'], 30)!;
    expect('rmssd' in n.mean).toBe(true);
    expect('hrMean' in n.mean).toBe(false); // <30 usable -> omitted -> global fallback
  });
});
