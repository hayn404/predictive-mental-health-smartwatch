/**
 * Unit tests — runtime model-health/drift monitor (gaps S6/L8).
 */
import {
  recordStressPrediction,
  getModelHealth,
  _resetModelMonitor,
} from '@/services/observability/modelMonitor';

beforeEach(() => _resetModelMonitor());

function feed(n: number, score: (i: number) => number, confidence: number) {
  for (let i = 0; i < n; i++) {
    recordStressPrediction({ stressScore: score(i), confidence, timestamp: i });
  }
}

describe('modelMonitor', () => {
  test('cold-start (< 20 samples) reports healthy / no drift', () => {
    feed(10, (i) => 30 + i, 0.9);
    const h = getModelHealth();
    expect(h.samples).toBe(10);
    expect(h.drift).toBe(false);
  });

  test('healthy: varied scores + high confidence -> no drift', () => {
    feed(40, (i) => 20 + (i % 30), 0.85);
    const h = getModelHealth();
    expect(h.drift).toBe(false);
    expect(h.meanConfidence).toBeCloseTo(0.85, 6);
    expect(h.degenerate).toBe(false);
  });

  test('sustained low confidence -> drift flagged', () => {
    feed(40, (i) => 20 + (i % 30), 0.2);
    const h = getModelHealth();
    expect(h.lowConfidenceRate).toBeCloseTo(1, 6);
    expect(h.drift).toBe(true);
    expect(h.reasons.join(' ')).toMatch(/low confidence/i);
  });

  test('degenerate (stuck) output -> drift flagged', () => {
    feed(40, () => 50, 0.9); // identical score every window
    const h = getModelHealth();
    expect(h.degenerate).toBe(true);
    expect(h.drift).toBe(true);
    expect(h.reasons.join(' ')).toMatch(/not varying/i);
  });

  test('rolling window caps at 200 samples', () => {
    feed(250, (i) => i % 50, 0.9);
    expect(getModelHealth().samples).toBe(200);
  });
});
