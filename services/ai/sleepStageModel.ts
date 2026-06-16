/**
 * Seren AI — Sleep Stage Classification (v3.2, on-device ONNX inference)
 * =======================================================================
 * Model:   TCN + BiLSTM seq2seq, trained on BIDSleep (n=47/252 nights),
 *          held-out tested on Walch (n=31). 4-class: Wake/Light/Deep/REM.
 * Input:   features [batch, seq_len=41, n_features=12]
 *          12 features per 30-s epoch:
 *            hr_mean, hr_std, hr_min, hr_max, hr_range,
 *            hr_succdiff_std, hr_delta_prev,
 *            act_count, immobility_frac, act_max, act_std,
 *            time_of_night (linear [-1,+1] across the night)
 * Output:  logits [batch, seq_len=41, 4]  → argmax per epoch
 *
 * Data flow:
 *   Wear OS watch → 30-s epoch raw features (Float32Array[N, 12])
 *     → phone receives via WearableListenerService
 *     → robustNormalizePerNight()  (median + 1.4826*MAD per feature)
 *     → window into disjoint [1, 41, 12] tensors (stride 41)
 *     → ONNX inference per window
 *     → flatten + argmax + map to SleepStageType[]
 */

import type { TensorflowModel } from 'react-native-fast-tflite';
import Constants from 'expo-constants';
import { SleepStageType, RawSleepStage } from './types';

// ────────────────────────────────────────────────────────────────
// Contract — must match assets/ml/sleep/sleep_model_metadata.json
// ────────────────────────────────────────────────────────────────
export const V32_NUM_FEATURES = 12;
export const V32_SEQ_LEN = 41;
export const V32_EPOCH_SEC = 30;
export const V32_NUM_CLASSES = 4;

export const V32_FEATURE_NAMES = [
  'hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_range',
  'hr_succdiff_std', 'hr_delta_prev',
  'act_count', 'immobility_frac', 'act_max', 'act_std',
  'time_of_night',
] as const;

const STAGE_MAP: SleepStageType[] = ['awake', 'light', 'deep', 'rem'];

// ────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────

/** One epoch (30 s) of unnormalized features extracted on the watch. */
export interface RawEpochFeatures {
  /** Epoch start time (Unix ms). */
  startTime: number;
  /**
   * The 11 cache features in the order of V32_FEATURE_NAMES[0..11].
   * Do NOT pre-pend time_of_night — the phone fills that in once the
   * full night is known.
   */
  values: Float32Array; // length 11
}

export interface V32SleepPrediction {
  /** Epoch start (Unix ms). */
  startTime: number;
  /** Epoch end (Unix ms). */
  endTime: number;
  /** 0=Wake, 1=Light, 2=Deep, 3=REM. */
  classIndex: number;
  /** App-facing stage label. */
  stage: SleepStageType;
  /** Max softmax probability. */
  confidence: number;
  /** Per-class softmax. */
  probs: number[];
}

export interface V32SleepOutput {
  modelVersion: string;
  sessionStart: number;
  sessionEnd: number;
  epochs: V32SleepPrediction[];
  /** Stages collapsed into contiguous spans for the existing scoring pipeline. */
  mlStages: RawSleepStage[];
}

// ────────────────────────────────────────────────────────────────
// Lazy TFLite model (react-native-fast-tflite — New-Architecture compatible)
// ────────────────────────────────────────────────────────────────

let tfliteModel: TensorflowModel | null = null;
let modelMetaVersion = 'v3.2.0';

/**
 * Load the on-device TFLite sleep model. The .tflite was converted from the original
 * ONNX (onnx2tf with Erf/GELU replaced by native ops, so no Flex/Select-TF ops).
 * Parity vs ONNX: argmax-identical, max-abs-diff ~1e-6.
 * Tensor layout: input [1, FEAT(12), SEQ(41)] (feature-major), output [1, SEQ(41), CLASSES(4)].
 */
export async function loadV32SleepModel(): Promise<boolean> {
  // react-native-fast-tflite depends on NitroModules (a native module) which throws on
  // import in Expo Go. Skip it there so it never errors — sleep falls back to Health
  // Connect's pre-classified stages. It loads normally in a dev/production build.
  if (Constants.appOwnership === 'expo') {
    console.log('[Seren] v3.2 sleep model skipped in Expo Go (needs a dev build); using Health Connect stages.');
    return false;
  }
  try {
    // Dynamic import keeps the native module out of the JS bundle until needed
    // (and out of unit tests that import this file for the pure helpers).
    const { loadTensorflowModel } = await import('react-native-fast-tflite');
    // delegates: [] -> default CPU delegate (model uses native ops, no GPU needed)
    tfliteModel = await loadTensorflowModel(
      require('@/assets/ml/sleep/sleep_stage_model.tflite'),
      [],
    );
    console.log('[Seren] v3.2 sleep model (TFLite) loaded');
    return true;
  } catch (e) {
    console.warn('[Seren] v3.2 sleep model load failed:', e);
    tfliteModel = null;
    return false;
  }
}

export function isV32SleepModelLoaded(): boolean {
  return tfliteModel !== null;
}

// ────────────────────────────────────────────────────────────────
// Preprocessing — mirrors ml/sleep/prepare_features.py exactly
// ────────────────────────────────────────────────────────────────

/**
 * Per-night robust z-score: (x - median) / (1.4826 * MAD), per feature column.
 * Replaces NaN/Inf with 0 to mirror np.nan_to_num in the Python pipeline.
 * Input: matrix shape [N_epochs, 12].
 */
export function robustNormalizePerNight(features: Float32Array, nEpochs: number): Float32Array {
  const F = V32_NUM_FEATURES;
  if (features.length !== nEpochs * F) {
    throw new Error(`robustNormalize: expected ${nEpochs * F} values, got ${features.length}`);
  }
  const out = new Float32Array(features.length);
  const col = new Float32Array(nEpochs);

  for (let f = 0; f < F; f++) {
    for (let i = 0; i < nEpochs; i++) col[i] = features[i * F + f];

    const med = median(col);
    const absDev = new Float32Array(nEpochs);
    for (let i = 0; i < nEpochs; i++) absDev[i] = Math.abs(col[i] - med);
    const mad = median(absDev);
    const scale = mad > 1e-8 ? 1.4826 * mad : 1.0;

    for (let i = 0; i < nEpochs; i++) {
      const z = (col[i] - med) / scale;
      out[i * F + f] = Number.isFinite(z) ? z : 0;
    }
  }
  return out;
}

function median(arr: Float32Array): number {
  const sorted = Array.from(arr).sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) return 0;
  return n % 2 ? sorted[(n - 1) >> 1] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

/**
 * Fill in the time_of_night feature (linear [-1, +1] across the night)
 * and assemble the final [N_epochs, 12] matrix from raw 11-feature epochs.
 * Mirrors add_time_of_night() in ml/sleep/_build_notebook.py.
 */
export function assembleFeatureMatrix(rawEpochs: RawEpochFeatures[]): Float32Array {
  const N = rawEpochs.length;
  const F = V32_NUM_FEATURES;
  const out = new Float32Array(N * F);

  for (let i = 0; i < N; i++) {
    const e = rawEpochs[i];
    if (e.values.length !== F - 1) {
      throw new Error(`epoch ${i} has ${e.values.length} values, expected ${F - 1}`);
    }
    const base = i * F;
    for (let f = 0; f < F - 1; f++) out[base + f] = e.values[f];
    // time_of_night ∈ [-1, +1]; matches Python linspace(-1, 1, N)
    out[base + (F - 1)] = N > 1 ? -1 + (2 * i) / (N - 1) : 0;
  }
  return out;
}

// ────────────────────────────────────────────────────────────────
// Inference
// ────────────────────────────────────────────────────────────────

function softmax(logits: Float32Array, offset: number, n: number): number[] {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[offset + i] > max) max = logits[offset + i];
  const exps = new Array<number>(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    exps[i] = Math.exp(logits[offset + i] - max);
    sum += exps[i];
  }
  for (let i = 0; i < n; i++) exps[i] /= sum;
  return exps;
}

function argmax(probs: number[]): number {
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return best;
}

/**
 * Run inference on a full night of raw epoch features.
 *
 * @param rawEpochs - chronological 30-s epochs from the watch (11 values each)
 * @returns null if the model isn't loaded or there are too few epochs
 */
export async function runV32Inference(
  rawEpochs: RawEpochFeatures[],
): Promise<V32SleepOutput | null> {
  if (!tfliteModel) return null;
  const N = rawEpochs.length;
  if (N < V32_SEQ_LEN) {
    console.warn(`[Seren] v3.2 inference: need >= ${V32_SEQ_LEN} epochs, got ${N}`);
    return null;
  }

  // 1. Assemble + normalize the whole night
  const raw = assembleFeatureMatrix(rawEpochs);
  const norm = robustNormalizePerNight(raw, N);

  // 2. Slide disjoint windows of length SEQ_LEN (stride = SEQ_LEN, matching eval)
  const predictions: V32SleepPrediction[] = new Array(N);
  const F = V32_NUM_FEATURES;
  const C = V32_NUM_CLASSES;

  for (let start = 0; start < N; start += V32_SEQ_LEN) {
    const end = Math.min(start + V32_SEQ_LEN, N);
    const winLen = end - start;
    // Pad the last (possibly short) window by repeating the final epoch.
    // TFLite input layout is [1, F, SEQ] (feature-major) -> index = f*SEQ + i.
    const winInput = new Float32Array(F * V32_SEQ_LEN);
    for (let i = 0; i < V32_SEQ_LEN; i++) {
      const src = (i < winLen ? start + i : end - 1) * F;
      for (let f = 0; f < F; f++) winInput[f * V32_SEQ_LEN + i] = norm[src + f];
    }

    // fast-tflite takes/returns ArrayBuffer[]. Output [1, SEQ, C] row-major -> logits[i*C + c].
    const outputs = tfliteModel.runSync([winInput.buffer as ArrayBuffer]);
    const logits = new Float32Array(outputs[0]);

    for (let i = 0; i < winLen; i++) {
      const offset = i * C;
      const probs = softmax(logits, offset, C);
      const cls = argmax(probs);
      const epochStart = rawEpochs[start + i].startTime;
      predictions[start + i] = {
        startTime: epochStart,
        endTime: epochStart + V32_EPOCH_SEC * 1000,
        classIndex: cls,
        stage: STAGE_MAP[cls],
        confidence: probs[cls],
        probs,
      };
    }
  }

  // 3. Light smoothing: flip isolated 1-epoch islands surrounded by agreeing
  // neighbours when confidence is low — matches the existing legacy behaviour.
  for (let i = 1; i < N - 1; i++) {
    const p = predictions[i - 1].classIndex;
    const c = predictions[i].classIndex;
    const n = predictions[i + 1].classIndex;
    if (p === n && c !== p && predictions[i].confidence < 0.6) {
      predictions[i].classIndex = p;
      predictions[i].stage = STAGE_MAP[p];
    }
  }

  // 4. Collapse into contiguous spans for analyzeSleepSession()
  const mlStages: RawSleepStage[] = [];
  let spanStart = predictions[0].startTime;
  let spanStage = predictions[0].stage;
  for (let i = 1; i < N; i++) {
    if (predictions[i].stage !== spanStage) {
      mlStages.push({ startTime: spanStart, endTime: predictions[i].startTime, stage: spanStage });
      spanStart = predictions[i].startTime;
      spanStage = predictions[i].stage;
    }
  }
  mlStages.push({ startTime: spanStart, endTime: predictions[N - 1].endTime, stage: spanStage });

  return {
    modelVersion: modelMetaVersion,
    sessionStart: predictions[0].startTime,
    sessionEnd: predictions[N - 1].endTime,
    epochs: predictions,
    mlStages,
  };
}

// ────────────────────────────────────────────────────────────────
// Optional: setter for model version string (read from metadata.json)
// ────────────────────────────────────────────────────────────────
export function setV32ModelVersion(version: string): void {
  modelMetaVersion = version;
}
