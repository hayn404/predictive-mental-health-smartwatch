/**
 * Seren — Sleep feature batch receiver (phone side)
 * ==================================================
 * Bridges the Wear OS companion's persisted batch files to the v3.2 ONNX
 * inference path. On app open (or on demand) this module:
 *
 *   1. Lists session directories under <documentDir>/seren_sleep/
 *   2. For each directory containing SESSION.final:
 *        a. Loads every batch_*.bin in chronological order
 *        b. Decodes the wire format (matches Kotlin WearableFeatureSender)
 *        c. Concatenates all epochs in time order
 *        d. Runs runV32Inference(...) → V32SleepOutput
 *        e. Returns the result to the caller (and deletes the session dir)
 *
 * Wire format (little-endian, version 1) — must match
 * com.seren.watch.sleep.WearableFeatureSender.encode():
 *   header  : magic 'SRN1' (i32), version u16, reserved u16
 *   metadata: captureStartMs i64, epochCount u32, featuresPerEpoch u16, reserved u16
 *   payload : epochCount × (startMs i64 + 11 × float32)
 */

import * as FileSystem from 'expo-file-system/legacy';
import {
  RawEpochFeatures,
  V32SleepOutput,
  runV32Inference,
  isV32SleepModelLoaded,
} from './sleepStageModel';

const MAGIC = 0x314e5253;  // 'SRN1' little-endian
const FEATURES_PER_EPOCH = 11;
const SLEEP_DIR = 'seren_sleep';

export interface SessionResult {
  captureStartMs: number;
  epochsReceived: number;
  output: V32SleepOutput | null;
  /** Error description if inference couldn't run (e.g. too few epochs). */
  skipped?: string;
}

/**
 * Scan persisted sessions and run v3.2 inference on every finalized one.
 * Returns one result per finalized session.
 */
export async function processPendingSessions(): Promise<SessionResult[]> {
  const root = `${FileSystem.documentDirectory}${SLEEP_DIR}/`;
  const info = await FileSystem.getInfoAsync(root);
  if (!info.exists) return [];

  const sessions = await FileSystem.readDirectoryAsync(root);
  const results: SessionResult[] = [];

  for (const sessionName of sessions) {
    const sessionDir = `${root}${sessionName}/`;
    const finalMarker = `${sessionDir}SESSION.final`;
    const markerInfo = await FileSystem.getInfoAsync(finalMarker);
    if (!markerInfo.exists) continue; // session still being captured

    // L5: if the model isn't loaded yet, KEEP the session and retry on a later open —
    // don't consume + delete the raw overnight data with no model available to process it.
    if (!isV32SleepModelLoaded()) {
      results.push({
        captureStartMs: parseInt(sessionName, 10),
        epochsReceived: 0,
        output: null,
        skipped: 'model_not_loaded',
      });
      continue;
    }

    const result = await processSession(sessionDir, parseInt(sessionName, 10));
    results.push(result);

    // Delete only now that the model has had its chance: a null result here means the data
    // is genuinely empty/too-short (won't improve on retry). Transient "model not loaded"
    // is handled above, so raw data is never lost on a retryable failure.
    try {
      await FileSystem.deleteAsync(sessionDir, { idempotent: true });
    } catch (e) {
      console.warn('[Seren] failed to delete session dir', sessionDir, e);
    }
  }
  return results;
}

async function processSession(
  sessionDir: string,
  captureStartMs: number,
): Promise<SessionResult> {
  const files = (await FileSystem.readDirectoryAsync(sessionDir))
    .filter((f) => f.endsWith('.bin'))
    // Sort so non-final batches come before the final one.
    // Filenames are batch_<seq>.bin or batch_final_<seq>.bin
    .sort((a, b) => {
      const af = a.includes('final');
      const bf = b.includes('final');
      if (af !== bf) return af ? 1 : -1;
      return a.localeCompare(b);
    });

  const allEpochs: RawEpochFeatures[] = [];
  for (const f of files) {
    const bytes = await readBinary(`${sessionDir}${f}`);
    try {
      const epochs = decodeBatch(bytes);
      allEpochs.push(...epochs);
    } catch (e) {
      console.warn(`[Seren] decode failed for ${f}:`, e);
    }
  }

  if (allEpochs.length === 0) {
    return { captureStartMs, epochsReceived: 0, output: null, skipped: 'empty' };
  }

  // Ensure chronological order (defence against out-of-order delivery)
  allEpochs.sort((a, b) => a.startTime - b.startTime);

  const output = await runV32Inference(allEpochs);
  return {
    captureStartMs,
    epochsReceived: allEpochs.length,
    output,
    skipped: output ? undefined : 'inference_failed_or_too_short',
  };
}

async function readBinary(uri: string): Promise<ArrayBuffer> {
  const b64 = await FileSystem.readAsStringAsync(uri, {
    encoding: FileSystem.EncodingType.Base64,
  });
  return base64ToArrayBuffer(b64);
}

export function base64ToArrayBuffer(b64: string): ArrayBuffer {
  // Lightweight base64 → bytes that works in Hermes without atob polyfill.
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const lookup = new Uint8Array(128);
  for (let i = 0; i < chars.length; i++) lookup[chars.charCodeAt(i)] = i;
  let bufLen = (b64.length * 3) >> 2;
  if (b64.endsWith('==')) bufLen -= 2;
  else if (b64.endsWith('=')) bufLen -= 1;
  const bytes = new Uint8Array(bufLen);
  let p = 0;
  for (let i = 0; i < b64.length; i += 4) {
    const c0 = lookup[b64.charCodeAt(i)];
    const c1 = lookup[b64.charCodeAt(i + 1)];
    const c2 = lookup[b64.charCodeAt(i + 2)];
    const c3 = lookup[b64.charCodeAt(i + 3)];
    if (p < bufLen) bytes[p++] = (c0 << 2) | (c1 >> 4);
    if (p < bufLen) bytes[p++] = ((c1 & 0x0f) << 4) | (c2 >> 2);
    if (p < bufLen) bytes[p++] = ((c2 & 0x03) << 6) | c3;
  }
  return bytes.buffer;
}

export function decodeBatch(buf: ArrayBuffer): RawEpochFeatures[] {
  const dv = new DataView(buf);
  if (buf.byteLength < 24) throw new Error(`batch too small: ${buf.byteLength} bytes`);
  const magic = dv.getInt32(0, true);
  if (magic !== MAGIC) throw new Error(`bad magic 0x${magic.toString(16)}`);
  // header bytes 4..16 (version, reserved, captureStartMs) are parsed from
  // the folder name instead; we only need epoch count + feature width here.
  const epochCount = dv.getInt32(16, true);
  const featuresPerEpoch = dv.getUint16(20, true);
  if (featuresPerEpoch !== FEATURES_PER_EPOCH) {
    throw new Error(`unexpected feature count: ${featuresPerEpoch}`);
  }
  const headerSize = 24;
  const epochSize = 8 + FEATURES_PER_EPOCH * 4;
  if (buf.byteLength !== headerSize + epochCount * epochSize) {
    throw new Error(
      `size mismatch: declared ${epochCount} epochs but body is ${buf.byteLength - headerSize} bytes`,
    );
  }

  const out: RawEpochFeatures[] = new Array(epochCount);
  for (let i = 0; i < epochCount; i++) {
    const off = headerSize + i * epochSize;
    const startMs = Number(dv.getBigInt64(off, true));
    const values = new Float32Array(FEATURES_PER_EPOCH);
    for (let f = 0; f < FEATURES_PER_EPOCH; f++) {
      values[f] = dv.getFloat32(off + 8 + f * 4, true);
    }
    out[i] = { startTime: startMs, values };
  }
  return out;
}
