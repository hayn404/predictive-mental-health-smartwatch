/**
 * Unit Tests — Sleep Stage ML path (v3.2)
 * =======================================
 * Covers the deterministic, bug-prone preprocessing the on-device ONNX model
 * depends on: per-night robust normalization, feature-matrix assembly
 * (time_of_night), and the watch→phone binary wire decode. (The ONNX inference
 * itself needs the native runtime and is exercised on-device, not in jest.)
 */

// sleepReceiver imports expo-file-system only for I/O we don't call here; stub it
// so the module loads under ts-jest (node env).
jest.mock('expo-file-system/legacy', () => ({
  EncodingType: { Base64: 'base64' },
  readAsStringAsync: jest.fn(),
  getInfoAsync: jest.fn(),
  readDirectoryAsync: jest.fn(),
  deleteAsync: jest.fn(),
  documentDirectory: '/tmp/',
}));

import {
  robustNormalizePerNight,
  assembleFeatureMatrix,
  V32_NUM_FEATURES,
  type RawEpochFeatures,
} from '@/services/ai/sleepStageModel';
import { decodeBatch, base64ToArrayBuffer } from '@/services/ai/sleepReceiver';

const F = V32_NUM_FEATURES; // 12
const RAW_F = F - 1;        // 11 watch features

// ────────────────────────────────────────────────────────────────
// robustNormalizePerNight — (x - median) / (1.4826 * MAD) per column
// ────────────────────────────────────────────────────────────────
describe('robustNormalizePerNight', () => {
  function matrix(nEpochs: number, col0: number[]): Float32Array {
    const m = new Float32Array(nEpochs * F);
    for (let i = 0; i < nEpochs; i++) m[i * F + 0] = col0[i];
    return m;
  }

  test('robust z-scores a column by median and MAD', () => {
    const out = robustNormalizePerNight(matrix(5, [1, 2, 3, 4, 5]), 5);
    // median=3, absdev=[2,1,0,1,2], MAD=1, scale=1.4826
    expect(out[0 * F + 0]).toBeCloseTo((1 - 3) / 1.4826, 3);
    expect(out[2 * F + 0]).toBeCloseTo(0, 6);
    expect(out[4 * F + 0]).toBeCloseTo((5 - 3) / 1.4826, 3);
  });

  test('constant column (MAD=0) maps to 0, never NaN', () => {
    const out = robustNormalizePerNight(matrix(4, [7, 7, 7, 7]), 4);
    for (let i = 0; i < 4; i++) expect(out[i * F + 0]).toBe(0);
  });

  test('non-finite values become 0 (mirrors np.nan_to_num)', () => {
    const out = robustNormalizePerNight(matrix(3, [1, 2, Infinity]), 3);
    expect(Number.isFinite(out[2 * F + 0])).toBe(true);
    expect(out[2 * F + 0]).toBe(0);
  });

  test('throws on a shape mismatch', () => {
    expect(() => robustNormalizePerNight(new Float32Array(F + 1), 1)).toThrow(/expected/);
  });
});

// ────────────────────────────────────────────────────────────────
// assembleFeatureMatrix — appends time_of_night ∈ [-1, +1]
// ────────────────────────────────────────────────────────────────
describe('assembleFeatureMatrix', () => {
  function rawEpoch(start: number, fill: number): RawEpochFeatures {
    return { startTime: start, values: new Float32Array(RAW_F).fill(fill) };
  }

  test('appends time_of_night as a linear ramp from -1 to +1', () => {
    const out = assembleFeatureMatrix([rawEpoch(0, 1), rawEpoch(1, 2), rawEpoch(2, 3)]);
    expect(out[0 * F + (F - 1)]).toBeCloseTo(-1, 6);
    expect(out[1 * F + (F - 1)]).toBeCloseTo(0, 6);
    expect(out[2 * F + (F - 1)]).toBeCloseTo(1, 6);
    // raw features copied through
    expect(out[0 * F + 0]).toBe(1);
    expect(out[2 * F + 0]).toBe(3);
  });

  test('single epoch gets time_of_night = 0', () => {
    const out = assembleFeatureMatrix([rawEpoch(0, 5)]);
    expect(out[F - 1]).toBe(0);
  });

  test('throws when an epoch has the wrong feature count', () => {
    const bad = { startTime: 0, values: new Float32Array(RAW_F - 1) };
    expect(() => assembleFeatureMatrix([bad])).toThrow(/expected/);
  });
});

// ────────────────────────────────────────────────────────────────
// decodeBatch — watch→phone binary wire format (SRN1)
// ────────────────────────────────────────────────────────────────
describe('decodeBatch', () => {
  const MAGIC = 0x314e5253; // 'SRN1' little-endian
  const HEADER = 24;
  const EPOCH = 8 + RAW_F * 4;

  function buildBatch(epochs: { startMs: number; values: number[] }[], magic = MAGIC, feat = RAW_F): ArrayBuffer {
    const buf = new ArrayBuffer(HEADER + epochs.length * EPOCH);
    const dv = new DataView(buf);
    dv.setInt32(0, magic, true);
    dv.setInt32(16, epochs.length, true);
    dv.setUint16(20, feat, true);
    epochs.forEach((e, i) => {
      const off = HEADER + i * EPOCH;
      dv.setBigInt64(off, BigInt(e.startMs), true);
      for (let f = 0; f < RAW_F; f++) dv.setFloat32(off + 8 + f * 4, e.values[f], true);
    });
    return buf;
  }

  test('round-trips epoch start time and feature values', () => {
    const vals = Array.from({ length: RAW_F }, (_, k) => k + 0.5);
    const out = decodeBatch(buildBatch([{ startMs: 1_700_000_000_000, values: vals }]));
    expect(out).toHaveLength(1);
    expect(out[0].startTime).toBe(1_700_000_000_000);
    expect(out[0].values).toHaveLength(RAW_F);
    vals.forEach((v, k) => expect(out[0].values[k]).toBeCloseTo(v, 5));
  });

  test('throws on a bad magic number', () => {
    expect(() => decodeBatch(buildBatch([], 0xdeadbeef | 0))).toThrow(/magic/);
  });

  test('throws on a declared/actual size mismatch', () => {
    const good = buildBatch([{ startMs: 1, values: new Array(RAW_F).fill(0) }]);
    const dv = new DataView(good);
    dv.setInt32(16, 5, true); // claim 5 epochs but body has 1
    expect(() => decodeBatch(good)).toThrow(/size mismatch/);
  });
});

// ────────────────────────────────────────────────────────────────
// base64ToArrayBuffer — Hermes-safe base64 decode
// ────────────────────────────────────────────────────────────────
describe('base64ToArrayBuffer', () => {
  test('decodes bytes identically to a reference encoder', () => {
    const bytes = Uint8Array.from([0, 1, 2, 127, 128, 200, 253, 254, 255, 65, 66, 67]);
    const b64 = Buffer.from(bytes).toString('base64');
    const out = new Uint8Array(base64ToArrayBuffer(b64));
    expect(Array.from(out)).toEqual(Array.from(bytes));
  });
});
