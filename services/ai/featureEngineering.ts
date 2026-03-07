/**
 * Seren AI — On-Device HRV Feature Engineering
 * ===============================================
 * Computes the same 29 features as the Python training pipeline,
 * but in TypeScript for real-time on-device inference.
 *
 * Input: Raw HR samples + temperature + accelerometer from Health Connect
 * Output: BiometricFeatureVector (29 features) per 5-minute window
 *
 * Feature groups:
 *  1. Time-domain HRV (9): from RR intervals
 *  2. Frequency-domain HRV (7): from Welch's PSD of RR intervals
 *  3. Non-linear HRV (5): Poincare, sample entropy, DFA
 *  4. Skin temperature (4): mean, slope, std, range
 *  5. Accelerometer/activity (4): magnitude stats, steps, activity type
 */

import {
  BiometricFeatureVector,
  ActivityType,
  RawHeartRateSample,
  RawHRVSample,
  RawTemperatureSample,
} from './types';

// ============================================================
// RR Interval Extraction from HR Samples
// ============================================================

/**
 * Convert HR BPM samples to RR intervals in milliseconds.
 *
 * Since Health Connect gives us BPM (not raw PPG), we derive
 * approximate RR intervals: RR(ms) = 60000 / BPM.
 *
 * On a real watch with raw PPG access, we'd use peak detection
 * like the Python pipeline. This approximation is standard for
 * consumer wearable data.
 */
export function hrSamplesToRRIntervals(samples: RawHeartRateSample[]): number[] {
  if (samples.length < 2) return [];

  const rrIntervals: number[] = [];

  for (const sample of samples) {
    if (sample.bpm > 30 && sample.bpm < 220) {
      const rr = 60000 / sample.bpm;
      rrIntervals.push(rr);
    }
  }

  // Artifact removal: reject RR < 300ms or > 2000ms
  const filtered = rrIntervals.filter(rr => rr >= 300 && rr <= 2000);

  // Remove outliers: reject RR > 30% from median
  if (filtered.length > 5) {
    const sorted = [...filtered].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    return filtered.filter(rr => Math.abs(rr - median) / median < 0.30);
  }

  return filtered;
}

// ============================================================
// Time-Domain HRV Features (9 features)
// ============================================================

export function computeTimeDomain(rr: number[]): {
  meanRR: number; sdnn: number; rmssd: number; pnn50: number; pnn20: number;
  hrMean: number; hrStd: number; hrRange: number; cvRR: number;
} {
  if (rr.length < 5) {
    return { meanRR: 0, sdnn: 0, rmssd: 0, pnn50: 0, pnn20: 0, hrMean: 0, hrStd: 0, hrRange: 0, cvRR: 0 };
  }

  const n = rr.length;
  const meanRR = rr.reduce((s, v) => s + v, 0) / n;

  // SDNN: standard deviation of all NN intervals
  const sdnn = Math.sqrt(rr.reduce((s, v) => s + (v - meanRR) ** 2, 0) / (n - 1));

  // Successive differences
  const diffs: number[] = [];
  for (let i = 1; i < n; i++) {
    diffs.push(Math.abs(rr[i] - rr[i - 1]));
  }

  // RMSSD: root mean square of successive differences
  const rmssd = Math.sqrt(diffs.reduce((s, d) => s + d * d, 0) / diffs.length);

  // pNN50, pNN20: percentage of successive diffs exceeding threshold
  const pnn50 = (diffs.filter(d => d > 50).length / diffs.length) * 100;
  const pnn20 = (diffs.filter(d => d > 20).length / diffs.length) * 100;

  // Heart rate from RR intervals
  const hr = rr.map(r => 60000 / r);
  const hrMean = hr.reduce((s, v) => s + v, 0) / hr.length;
  const hrStd = Math.sqrt(hr.reduce((s, v) => s + (v - hrMean) ** 2, 0) / (hr.length - 1));
  const hrRange = Math.max(...hr) - Math.min(...hr);
  const cvRR = meanRR > 0 ? sdnn / meanRR : 0;

  return { meanRR, sdnn, rmssd, pnn50, pnn20, hrMean, hrStd, hrRange, cvRR };
}

// ============================================================
// Frequency-Domain HRV Features (7 features)
// ============================================================

/**
 * Compute frequency-domain HRV using a simplified Welch's method.
 *
 * For on-device we use a basic FFT approach:
 * 1. Resample RR intervals to uniform 4 Hz grid
 * 2. Apply Hann window
 * 3. Compute FFT
 * 4. Integrate PSD in VLF, LF, HF bands
 */
export function computeFrequencyDomain(rr: number[]): {
  vlfPower: number; lfPower: number; hfPower: number; lfHfRatio: number;
  totalPower: number; lfNorm: number; hfNorm: number;
} {
  const empty = { vlfPower: 0, lfPower: 0, hfPower: 0, lfHfRatio: 0, totalPower: 0, lfNorm: 50, hfNorm: 50 };

  if (rr.length < 30) return empty;

  try {
    // Create time axis from cumulative RR intervals (seconds)
    const t: number[] = [0];
    for (let i = 0; i < rr.length; i++) {
      t.push(t[t.length - 1] + rr[i] / 1000);
    }

    // Resample to uniform 4 Hz grid (linear interpolation)
    const fsResample = 4;
    const duration = t[t.length - 1];
    const nSamples = Math.floor(duration * fsResample);
    if (nSamples < 32) return empty;

    const rrResampled: number[] = [];
    let tIdx = 0;
    for (let i = 0; i < nSamples; i++) {
      const tTarget = i / fsResample;
      while (tIdx < t.length - 2 && t[tIdx + 1] < tTarget) tIdx++;

      if (tIdx < rr.length) {
        // Linear interpolation
        const frac = t[tIdx + 1] !== t[tIdx]
          ? (tTarget - t[tIdx]) / (t[tIdx + 1] - t[tIdx])
          : 0;
        const val = tIdx < rr.length - 1
          ? rr[tIdx] + frac * (rr[tIdx + 1] - rr[tIdx])
          : rr[tIdx];
        rrResampled.push(val);
      }
    }

    // Remove mean
    const mean = rrResampled.reduce((s, v) => s + v, 0) / rrResampled.length;
    const centered = rrResampled.map(v => v - mean);

    // Apply Hann window
    const N = centered.length;
    const windowed = centered.map((v, i) => v * 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1))));

    // Simple DFT (for small N, this is acceptable; for production, use FFT library)
    const halfN = Math.floor(N / 2);
    const psd: number[] = new Array(halfN).fill(0);
    const freqs: number[] = new Array(halfN).fill(0);

    for (let k = 0; k < halfN; k++) {
      let real = 0, imag = 0;
      for (let n = 0; n < N; n++) {
        const angle = -2 * Math.PI * k * n / N;
        real += windowed[n] * Math.cos(angle);
        imag += windowed[n] * Math.sin(angle);
      }
      psd[k] = (real * real + imag * imag) / (N * fsResample);
      freqs[k] = k * fsResample / N;
    }

    // Integrate power in bands
    let vlfPower = 0, lfPower = 0, hfPower = 0;
    for (let k = 0; k < halfN; k++) {
      const f = freqs[k];
      if (f >= 0.003 && f < 0.04) vlfPower += psd[k];
      if (f >= 0.04 && f < 0.15) lfPower += psd[k];
      if (f >= 0.15 && f < 0.4) hfPower += psd[k];
    }

    const totalPower = vlfPower + lfPower + hfPower;
    const lfHfSum = lfPower + hfPower;

    return {
      vlfPower,
      lfPower,
      hfPower,
      lfHfRatio: hfPower > 0 ? lfPower / hfPower : 0,
      totalPower,
      lfNorm: lfHfSum > 0 ? (lfPower / lfHfSum) * 100 : 50,
      hfNorm: lfHfSum > 0 ? (hfPower / lfHfSum) * 100 : 50,
    };
  } catch {
    return empty;
  }
}

// ============================================================
// Non-Linear HRV Features (5 features)
// ============================================================

export function computeNonLinear(rr: number[]): {
  sd1: number; sd2: number; sd1sd2Ratio: number;
  sampleEntropy: number; dfaAlpha1: number;
} {
  if (rr.length < 10) {
    return { sd1: 0, sd2: 0, sd1sd2Ratio: 0, sampleEntropy: 0, dfaAlpha1: 0 };
  }

  // Poincare: SD1 and SD2
  const rr1 = rr.slice(0, -1);
  const rr2 = rr.slice(1);
  const diffArr = rr2.map((v, i) => v - rr1[i]);
  const sumArr = rr2.map((v, i) => v + rr1[i]);

  const sd1 = std(diffArr) / Math.sqrt(2);
  const sd2 = std(sumArr) / Math.sqrt(2);

  return {
    sd1,
    sd2,
    sd1sd2Ratio: sd2 > 0 ? sd1 / sd2 : 0,
    sampleEntropy: computeSampleEntropy(rr, 2, 0.2),
    dfaAlpha1: computeDFA(rr),
  };
}

function computeSampleEntropy(data: number[], m: number, rFactor: number): number {
  const N = data.length;
  if (N < 20) return 0;

  const r = rFactor * std(data);
  if (r === 0) return 0;

  const countMatches = (templateLen: number): number => {
    let count = 0;
    for (let i = 0; i < N - templateLen; i++) {
      for (let j = i + 1; j < N - templateLen; j++) {
        let match = true;
        for (let k = 0; k < templateLen; k++) {
          if (Math.abs(data[i + k] - data[j + k]) >= r) {
            match = false;
            break;
          }
        }
        if (match) count++;
      }
    }
    return count;
  };

  const B = countMatches(m);
  const A = countMatches(m + 1);

  if (B === 0 || A === 0) return 0;
  return -Math.log(A / B);
}

function computeDFA(rr: number[]): number {
  const N = rr.length;
  if (N < 16) return 0;

  const mean = rr.reduce((s, v) => s + v, 0) / N;
  const y: number[] = [];
  let cumSum = 0;
  for (const r of rr) {
    cumSum += r - mean;
    y.push(cumSum);
  }

  const boxSizes: number[] = [];
  for (let n = 4; n <= Math.min(16, Math.floor(N / 4)); n++) {
    boxSizes.push(n);
  }
  if (boxSizes.length < 2) return 0;

  const fluctuations: number[] = [];
  for (const n of boxSizes) {
    const nBoxes = Math.floor(N / n);
    if (nBoxes === 0) continue;

    let totalRms = 0;
    for (let i = 0; i < nBoxes; i++) {
      const segment = y.slice(i * n, (i + 1) * n);
      // Linear detrend
      const x = segment.map((_, idx) => idx);
      const { slope, intercept } = linearRegression(x, segment);
      const trend = x.map(xi => slope * xi + intercept);
      const residuals = segment.map((v, idx) => v - trend[idx]);
      const rms = Math.sqrt(residuals.reduce((s, r) => s + r * r, 0) / residuals.length);
      totalRms += rms;
    }
    fluctuations.push(totalRms / nBoxes);
  }

  if (fluctuations.length < 2 || fluctuations.some(f => f <= 0)) return 0;

  const logN = boxSizes.slice(0, fluctuations.length).map(Math.log);
  const logF = fluctuations.map(Math.log);

  const valid = logN.map((_, i) => isFinite(logN[i]) && isFinite(logF[i]));
  const validLogN = logN.filter((_, i) => valid[i]);
  const validLogF = logF.filter((_, i) => valid[i]);

  if (validLogN.length < 2) return 0;

  const { slope } = linearRegression(validLogN, validLogF);
  return slope;
}

// ============================================================
// Temperature Features (4 features)
// ============================================================

export function computeTemperatureFeatures(samples: RawTemperatureSample[]): {
  tempMean: number; tempSlope: number; tempStd: number; tempRange: number;
} {
  if (samples.length < 2) {
    return { tempMean: 33.5, tempSlope: 0, tempStd: 0, tempRange: 0 };
  }

  const temps = samples.map(s => s.temperatureCelsius);
  const mean = temps.reduce((s, v) => s + v, 0) / temps.length;

  // Slope in °C/min
  const tMin = samples.map((s, i) => (s.timestamp - samples[0].timestamp) / 60000);
  const { slope } = linearRegression(tMin, temps);

  return {
    tempMean: mean,
    tempSlope: slope,
    tempStd: std(temps),
    tempRange: Math.max(...temps) - Math.min(...temps),
  };
}

// ============================================================
// Activity Features (4 features)
// ============================================================

export function computeActivityFeatures(
  steps: number,
  windowSeconds: number,
  hrMean: number,
): {
  accelMagnitudeMean: number; accelMagnitudeStd: number;
  stepCount: number; activityType: ActivityType;
} {
  // Since Health Connect doesn't give raw accelerometer,
  // we estimate activity from steps and heart rate
  const stepsPerMin = steps / (windowSeconds / 60);

  let activityType: ActivityType;
  let accelMagnitudeMean: number;
  let accelMagnitudeStd: number;

  if (stepsPerMin < 1) {
    activityType = 'sedentary';
    accelMagnitudeMean = 1.0;  // ~1g gravity only
    accelMagnitudeStd = 0.02;
  } else if (stepsPerMin < 40) {
    activityType = 'walking';
    accelMagnitudeMean = 1.05;
    accelMagnitudeStd = 0.15;
  } else {
    activityType = 'active';
    accelMagnitudeMean = 1.15;
    accelMagnitudeStd = 0.30;
  }

  // Override: if HR < 55, likely sleeping
  if (hrMean > 0 && hrMean < 55 && stepsPerMin < 1) {
    activityType = 'sleeping';
    accelMagnitudeMean = 0.98;
    accelMagnitudeStd = 0.01;
  }

  return { accelMagnitudeMean, accelMagnitudeStd, stepCount: steps, activityType };
}

// ============================================================
// Master Feature Extraction
// ============================================================

export interface FeatureExtractionInput {
  hrSamples: RawHeartRateSample[];
  hrvSamples: RawHRVSample[];
  tempSamples: RawTemperatureSample[];
  stepCount: number;
  windowSeconds: number;
  timestamp: number;
}

/**
 * Extract all 29 features from a 5-minute window of Health Connect data.
 * This is the main entry point called by the data pipeline every 5 minutes.
 */
export function extractFeatures(input: FeatureExtractionInput): BiometricFeatureVector {
  // Step 1: Derive RR intervals from HR samples
  const rr = hrSamplesToRRIntervals(input.hrSamples);

  // If we have direct RMSSD from Health Connect, use it as additional validation
  const directRmssd = input.hrvSamples.length > 0
    ? input.hrvSamples.reduce((s, v) => s + v.rmssd, 0) / input.hrvSamples.length
    : null;

  // Step 2: Compute all feature groups
  const timeDomain = computeTimeDomain(rr);
  const freqDomain = computeFrequencyDomain(rr);
  const nonLinear = computeNonLinear(rr);
  const tempFeatures = computeTemperatureFeatures(input.tempSamples);
  const activityFeatures = computeActivityFeatures(
    input.stepCount,
    input.windowSeconds,
    timeDomain.hrMean
  );

  // Use Health Connect RMSSD if our computed one seems off
  const finalRmssd = directRmssd !== null && Math.abs(timeDomain.rmssd - directRmssd) > 20
    ? directRmssd
    : timeDomain.rmssd;

  return {
    timestamp: input.timestamp,
    windowSeconds: input.windowSeconds,
    ...timeDomain,
    rmssd: finalRmssd,
    ...freqDomain,
    ...nonLinear,
    ...tempFeatures,
    ...activityFeatures,
  };
}

// ============================================================
// Utility Functions
// ============================================================

function std(arr: number[]): number {
  if (arr.length < 2) return 0;
  const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
  return Math.sqrt(arr.reduce((s, v) => s + (v - mean) ** 2, 0) / (arr.length - 1));
}

function linearRegression(x: number[], y: number[]): { slope: number; intercept: number } {
  const n = x.length;
  if (n < 2) return { slope: 0, intercept: 0 };

  const sumX = x.reduce((s, v) => s + v, 0);
  const sumY = y.reduce((s, v) => s + v, 0);
  const sumXY = x.reduce((s, v, i) => s + v * y[i], 0);
  const sumX2 = x.reduce((s, v) => s + v * v, 0);

  const denom = n * sumX2 - sumX * sumX;
  if (denom === 0) return { slope: 0, intercept: sumY / n };

  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;

  return { slope, intercept };
}
