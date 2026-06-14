/**
 * Seren — Environment batch receiver (phone side)
 * ================================================
 * Bridges the Wear OS app's ambient-light + GPS batches (persisted by the native
 * WearableEnvListenerService) into the sunlight-exposure and location-diversity
 * features. On app open this module:
 *
 *   1. Lists batch files under <documentDir>/seren_env/
 *   2. Decodes each by magic:
 *        - light  → insert sunlight samples
 *        - loc    → insert raw GPS points
 *   3. Deletes each file after decoding
 *   4. Recomputes today's sunlight summary (from all of today's samples) and
 *      today's location-diversity summary (by re-segmenting today's GPS points
 *      into dwell visits), then upserts both.
 *
 * Wire format (little-endian, version 1) — must match
 * com.seren.watch.env.WearableEnvSender:
 *   header : magic i32, version u16, reserved u16, count u32        (12 bytes)
 *   light  : count × (timestampMs i64 + lux f32)                    (12 bytes/sample)
 *   loc    : count × (timestampMs i64 + lat f64 + lon f64 + acc f32)(28 bytes/sample)
 */

import * as FileSystem from 'expo-file-system/legacy';
import { base64ToArrayBuffer } from './sleepReceiver';
import {
  insertSunlightSample,
  getSunlightSamples,
  upsertSunlightDaily,
  insertLocationPoint,
  getLocationPoints,
  deleteLocationVisitsInRange,
  insertLocationVisit,
  upsertLocationDiversity,
} from './db';
import { computeSunlightSummary, OUTDOOR_LUX_THRESHOLD } from './sunlightTracking';
import { segmentVisitsFromPoints, calculateDiversityScore } from './locationTracking';

const ENV_DIR = 'seren_env';
const MAGIC_LIGHT = 0x314c5253; // 'SRL1'
const MAGIC_LOCATION = 0x31475253; // 'SRG1'
const DAY_MS = 24 * 60 * 60 * 1000;

export interface EnvIngestResult {
  lightSamples: number;
  locationPoints: number;
}

function startOfToday(): number {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

/** Decode one light batch → SunlightReading-shaped rows. */
export function decodeLight(buf: ArrayBuffer): { timestamp: number; luxValue: number; isOutdoors: boolean }[] {
  const dv = new DataView(buf);
  const count = dv.getUint32(8, true);
  const headerSize = 12;
  const sampleSize = 12;
  if (buf.byteLength < headerSize + count * sampleSize) {
    throw new Error(`light size mismatch: ${count} samples vs ${buf.byteLength} bytes`);
  }
  const out: { timestamp: number; luxValue: number; isOutdoors: boolean }[] = new Array(count);
  for (let i = 0; i < count; i++) {
    const off = headerSize + i * sampleSize;
    const ts = Number(dv.getBigInt64(off, true));
    const lux = dv.getFloat32(off + 8, true);
    out[i] = { timestamp: ts, luxValue: lux, isOutdoors: lux >= OUTDOOR_LUX_THRESHOLD };
  }
  return out;
}

/** Decode one location batch → raw GPS points. */
export function decodeLocation(buf: ArrayBuffer): { timestamp: number; latitude: number; longitude: number; accuracy: number }[] {
  const dv = new DataView(buf);
  const count = dv.getUint32(8, true);
  const headerSize = 12;
  const sampleSize = 28;
  if (buf.byteLength < headerSize + count * sampleSize) {
    throw new Error(`location size mismatch: ${count} points vs ${buf.byteLength} bytes`);
  }
  const out: { timestamp: number; latitude: number; longitude: number; accuracy: number }[] = new Array(count);
  for (let i = 0; i < count; i++) {
    const off = headerSize + i * sampleSize;
    const ts = Number(dv.getBigInt64(off, true));
    const lat = dv.getFloat64(off + 8, true);
    const lon = dv.getFloat64(off + 16, true);
    const acc = dv.getFloat32(off + 24, true);
    out[i] = { timestamp: ts, latitude: lat, longitude: lon, accuracy: acc };
  }
  return out;
}

async function readBinary(uri: string): Promise<ArrayBuffer> {
  const b64 = await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 });
  return base64ToArrayBuffer(b64);
}

/**
 * Drain all pending environment batches from the watch, persist them, and
 * recompute today's summaries. Safe to call on every app open — a no-op when
 * the watch hasn't delivered anything.
 */
export async function processPendingEnv(): Promise<EnvIngestResult> {
  const root = `${FileSystem.documentDirectory}${ENV_DIR}/`;
  const info = await FileSystem.getInfoAsync(root);
  if (!info.exists) return { lightSamples: 0, locationPoints: 0 };

  const files = await FileSystem.readDirectoryAsync(root);
  let lightCount = 0;
  let locCount = 0;

  for (const f of files) {
    if (!f.endsWith('.bin')) continue;
    const uri = `${root}${f}`;
    try {
      const buf = await readBinary(uri);
      if (buf.byteLength < 12) throw new Error('batch too small');
      const magic = new DataView(buf).getInt32(0, true);
      if (magic === MAGIC_LIGHT) {
        const samples = decodeLight(buf);
        for (const s of samples) await insertSunlightSample(s);
        lightCount += samples.length;
      } else if (magic === MAGIC_LOCATION) {
        const points = decodeLocation(buf);
        for (const p of points) await insertLocationPoint(p);
        locCount += points.length;
      } else {
        console.warn(`[Seren] env: bad magic 0x${magic.toString(16)} in ${f}`);
      }
    } catch (e) {
      console.warn(`[Seren] env: failed to decode ${f}:`, e);
    }
    // Each file is consumed once; delete regardless so it isn't reprocessed.
    try {
      await FileSystem.deleteAsync(uri, { idempotent: true });
    } catch {
      /* ignore */
    }
  }

  const now = Date.now();
  const dayStart = startOfToday();

  if (lightCount > 0) {
    const samples = await getSunlightSamples(dayStart, now);
    const summary = computeSunlightSummary(samples);
    await upsertSunlightDaily(summary);
  }

  if (locCount > 0) {
    const todayPoints = await getLocationPoints(dayStart, now);
    const todayVisits = segmentVisitsFromPoints(todayPoints);
    // Re-derive today's visits idempotently from the full point set.
    await deleteLocationVisitsInRange(dayStart, now);
    for (const v of todayVisits) await insertLocationVisit(v);

    const weekPoints = await getLocationPoints(now - 7 * DAY_MS, now);
    const weekVisits = segmentVisitsFromPoints(weekPoints);
    const summary = calculateDiversityScore(todayVisits, weekVisits);
    await upsertLocationDiversity(summary);
  }

  return { lightSamples: lightCount, locationPoints: locCount };
}
