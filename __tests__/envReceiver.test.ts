/**
 * Tests for the watch→phone environment pipeline:
 *  - wire-format round-trip (must match com.seren.watch.env.WearableEnvSender)
 *  - dwell segmentation of raw GPS fixes into visits
 */
// envReceiver transitively imports expo-file-system/legacy (ESM, not transformed by jest).
// The pure decoders under test don't touch the filesystem, so stub the module out.
jest.mock('expo-file-system/legacy', () => ({
  documentDirectory: 'file:///doc/',
  EncodingType: { Base64: 'base64' },
  getInfoAsync: jest.fn(),
  readDirectoryAsync: jest.fn(),
  readAsStringAsync: jest.fn(),
  deleteAsync: jest.fn(),
}));
// Native sensor modules pulled in transitively (sunlight/location tracking) — stub them out.
jest.mock('expo-sensors', () => ({
  LightSensor: { isAvailableAsync: jest.fn(), setUpdateInterval: jest.fn(), addListener: jest.fn() },
}));
jest.mock('expo-location', () => ({
  Accuracy: { Balanced: 3 },
  requestForegroundPermissionsAsync: jest.fn(),
  watchPositionAsync: jest.fn(),
}));
// sleepStageModel (pulled in via sleepReceiver) imports expo-constants for the
// Expo Go guard — stub it so the module loads under ts-jest.
jest.mock('expo-constants', () => ({ __esModule: true, default: { appOwnership: null } }));

import { decodeLight, decodeLocation } from '@/services/ai/envReceiver';
import { segmentVisitsFromPoints } from '@/services/ai/locationTracking';

const MAGIC_LIGHT = 0x314c5253; // 'SRL1'
const MAGIC_LOCATION = 0x31475253; // 'SRG1'

/** Build a light batch byte-for-byte like the Kotlin sender. */
function encodeLight(samples: { ts: number; lux: number }[]): ArrayBuffer {
  const buf = new ArrayBuffer(12 + samples.length * 12);
  const dv = new DataView(buf);
  dv.setInt32(0, MAGIC_LIGHT, true);
  dv.setUint16(4, 1, true); // version
  dv.setUint16(6, 0, true); // reserved
  dv.setUint32(8, samples.length, true);
  samples.forEach((s, i) => {
    const off = 12 + i * 12;
    dv.setBigInt64(off, BigInt(s.ts), true);
    dv.setFloat32(off + 8, s.lux, true);
  });
  return buf;
}

function encodeLocation(pts: { ts: number; lat: number; lon: number; acc: number }[]): ArrayBuffer {
  const buf = new ArrayBuffer(12 + pts.length * 28);
  const dv = new DataView(buf);
  dv.setInt32(0, MAGIC_LOCATION, true);
  dv.setUint16(4, 1, true);
  dv.setUint16(6, 0, true);
  dv.setUint32(8, pts.length, true);
  pts.forEach((p, i) => {
    const off = 12 + i * 28;
    dv.setBigInt64(off, BigInt(p.ts), true);
    dv.setFloat64(off + 8, p.lat, true);
    dv.setFloat64(off + 16, p.lon, true);
    dv.setFloat32(off + 24, p.acc, true);
  });
  return buf;
}

describe('env wire format', () => {
  it('round-trips a light batch with correct outdoor classification', () => {
    const buf = encodeLight([
      { ts: 1000, lux: 200 },     // indoor
      { ts: 2000, lux: 25000 },   // outdoor
    ]);
    const out = decodeLight(buf);
    expect(out).toHaveLength(2);
    expect(out[0].timestamp).toBe(1000);
    expect(out[0].luxValue).toBeCloseTo(200, 1);
    expect(out[0].isOutdoors).toBe(false);
    expect(out[1].isOutdoors).toBe(true);
  });

  it('round-trips a location batch preserving lat/lon precision', () => {
    const buf = encodeLocation([
      { ts: 5000, lat: 30.0444196, lon: 31.2357116, acc: 12.5 },
    ]);
    const out = decodeLocation(buf);
    expect(out).toHaveLength(1);
    expect(out[0].timestamp).toBe(5000);
    expect(out[0].latitude).toBeCloseTo(30.0444196, 6);
    expect(out[0].longitude).toBeCloseTo(31.2357116, 6);
    expect(out[0].accuracy).toBeCloseTo(12.5, 1);
  });
});

describe('segmentVisitsFromPoints', () => {
  it('returns no visits for empty input', () => {
    expect(segmentVisitsFromPoints([])).toEqual([]);
  });

  it('groups co-located fixes into one visit and tracks dwell time', () => {
    const home = { latitude: 30.0, longitude: 31.0 };
    const points = [
      { timestamp: 1000, ...home },
      { timestamp: 2000, latitude: 30.00001, longitude: 31.00001 }, // ~1.5 m away
      { timestamp: 3000, ...home },
    ];
    const visits = segmentVisitsFromPoints(points);
    expect(visits).toHaveLength(1);
    expect(visits[0].timestamp).toBe(1000);       // arrival
    expect(visits[0].departureTime).toBe(3000);   // departure
  });

  it('splits into separate visits when moving >100 m and reuses cluster index on return', () => {
    const home = { latitude: 30.0, longitude: 31.0 };
    const work = { latitude: 30.05, longitude: 31.05 }; // several km away
    const points = [
      { timestamp: 1000, ...home },
      { timestamp: 2000, ...work },
      { timestamp: 3000, ...home },
    ];
    const visits = segmentVisitsFromPoints(points);
    expect(visits).toHaveLength(3);
    // Returning home should reuse the same cluster index as the first visit.
    expect(visits[2].clusterIndex).toBe(visits[0].clusterIndex);
    expect(visits[1].clusterIndex).not.toBe(visits[0].clusterIndex);
  });
});
