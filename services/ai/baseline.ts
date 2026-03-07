/**
 * Seren AI — Personal Baseline Service
 * ======================================
 * Computes rolling 14-day personal baselines from stored feature windows
 * and sleep sessions. Used for personalized anomaly detection and
 * contextualizing stress/anxiety scores.
 *
 * Recomputed once daily (or on demand after first week of data).
 */

import {
  PersonalBaseline,
  BiometricFeatureVector,
  SleepAnalysis,
} from './types';

// ============================================================
// Configuration
// ============================================================

const BASELINE_PERIOD_DAYS = 14;
const MIN_SAMPLES_FOR_BASELINE = 50; // ~4 hours of 5-min windows
const RESTING_HR_PERCENTILE = 0.1;   // Bottom 10% of HR readings = resting
const NOCTURNAL_HOURS = { start: 0, end: 6 }; // Midnight to 6 AM

// ============================================================
// Public API
// ============================================================

/**
 * Compute a personal baseline from recent feature windows and sleep data.
 *
 * @param userId - User identifier
 * @param featureWindows - Recent feature windows (ideally 14 days)
 * @param sleepSessions - Recent sleep analyses
 * @returns PersonalBaseline or null if insufficient data
 */
export function computeBaseline(
  userId: string,
  featureWindows: BiometricFeatureVector[],
  sleepSessions: SleepAnalysis[],
): PersonalBaseline | null {
  if (featureWindows.length < MIN_SAMPLES_FOR_BASELINE) {
    return null;
  }

  // ---- Heart rate baselines ----
  const allHr = featureWindows.map(f => f.hrMean).filter(v => v > 0);
  const restingHr = computePercentileSubset(allHr, RESTING_HR_PERCENTILE);
  const nocturnalWindows = featureWindows.filter(f => isNocturnal(f.timestamp));

  const sleepingHrValues = nocturnalWindows
    .map(f => f.hrMean)
    .filter(v => v > 0);

  // ---- HRV baselines ----
  const allRmssd = featureWindows.map(f => f.rmssd).filter(v => v > 0);
  const allSdnn = featureWindows.map(f => f.sdnn).filter(v => v > 0);
  const nocturnalRmssd = nocturnalWindows
    .map(f => f.rmssd)
    .filter(v => v > 0);

  // ---- Sleep baselines ----
  const sleepStats = computeSleepBaselines(sleepSessions);

  // ---- Activity baselines ----
  const dailySteps = computeDailySteps(featureWindows);

  // ---- Temperature baselines ----
  const restingTemp = featureWindows
    .filter(f => f.activityType === 'sedentary' || f.activityType === 'sleeping')
    .map(f => f.tempMean)
    .filter(v => v > 0);

  return {
    userId,
    computedAt: Date.now(),
    periodDays: BASELINE_PERIOD_DAYS,
    sampleCount: featureWindows.length,

    restingHrMean: mean(restingHr),
    restingHrStd: std(restingHr),
    sleepingHrMean: sleepingHrValues.length > 0 ? mean(sleepingHrValues) : mean(restingHr),

    rmssdMean: mean(allRmssd),
    rmssdStd: std(allRmssd),
    sdnnMean: mean(allSdnn),
    sdnnStd: std(allSdnn),
    nocturnalRmssdMean: nocturnalRmssd.length > 0 ? mean(nocturnalRmssd) : mean(allRmssd),

    typicalBedtimeHour: sleepStats.typicalBedtimeHour,
    typicalWakeHour: sleepStats.typicalWakeHour,
    avgSleepDurationMin: sleepStats.avgSleepDurationMin,
    avgSleepEfficiency: sleepStats.avgSleepEfficiency,
    avgSleepQuality: sleepStats.avgSleepQuality,
    avgDeepSleepPct: sleepStats.avgDeepSleepPct,
    avgRemSleepPct: sleepStats.avgRemSleepPct,

    avgDailySteps: dailySteps,

    restingTempMean: restingTemp.length > 0 ? mean(restingTemp) : 0,
    restingTempStd: restingTemp.length > 0 ? std(restingTemp) : 0,
  };
}

/**
 * Check if a new baseline should be computed.
 * Recompute if last baseline is >24h old or doesn't exist.
 */
export function shouldRecomputeBaseline(current: PersonalBaseline | null): boolean {
  if (!current) return true;
  const ageMs = Date.now() - current.computedAt;
  return ageMs > 24 * 60 * 60 * 1000; // 24 hours
}

/**
 * Detect anomalies relative to the personal baseline.
 * Returns a list of flags for the current feature window.
 */
export function detectAnomalies(
  features: BiometricFeatureVector,
  baseline: PersonalBaseline,
): AnomalyFlag[] {
  const flags: AnomalyFlag[] = [];

  // RMSSD significantly below baseline
  if (baseline.rmssdStd > 0) {
    const zScore = (features.rmssd - baseline.rmssdMean) / baseline.rmssdStd;
    if (zScore < -2) {
      flags.push({
        type: 'hrv_low',
        severity: zScore < -3 ? 'high' : 'moderate',
        message: `HRV ${Math.abs(Math.round(zScore))} std devs below your baseline`,
        zScore,
      });
    }
  }

  // HR significantly above resting baseline
  if (baseline.restingHrStd > 0) {
    const zScore = (features.hrMean - baseline.restingHrMean) / baseline.restingHrStd;
    if (zScore > 2) {
      flags.push({
        type: 'hr_elevated',
        severity: zScore > 3 ? 'high' : 'moderate',
        message: `Heart rate ${Math.abs(Math.round(zScore))} std devs above your resting baseline`,
        zScore,
      });
    }
  }

  // Temperature drop (vasoconstriction under stress)
  if (baseline.restingTempStd > 0 && features.tempMean > 0) {
    const zScore = (features.tempMean - baseline.restingTempMean) / baseline.restingTempStd;
    if (zScore < -2) {
      flags.push({
        type: 'temp_drop',
        severity: 'moderate',
        message: 'Skin temperature below your baseline',
        zScore,
      });
    }
  }

  return flags;
}

// ============================================================
// Types
// ============================================================

export interface AnomalyFlag {
  type: 'hrv_low' | 'hr_elevated' | 'temp_drop' | 'sleep_deficit';
  severity: 'moderate' | 'high';
  message: string;
  zScore: number;
}

// ============================================================
// Internal Helpers
// ============================================================

function isNocturnal(timestampMs: number): boolean {
  const hour = new Date(timestampMs).getHours();
  return hour >= NOCTURNAL_HOURS.start && hour < NOCTURNAL_HOURS.end;
}

/**
 * Get the bottom `percentile` of values (e.g., bottom 10% for resting HR).
 */
function computePercentileSubset(values: number[], percentile: number): number[] {
  if (values.length === 0) return [0];
  const sorted = [...values].sort((a, b) => a - b);
  const cutoff = Math.max(1, Math.ceil(sorted.length * percentile));
  return sorted.slice(0, cutoff);
}

function computeSleepBaselines(sessions: SleepAnalysis[]) {
  if (sessions.length === 0) {
    return {
      typicalBedtimeHour: 23,
      typicalWakeHour: 7,
      avgSleepDurationMin: 0,
      avgSleepEfficiency: 0,
      avgSleepQuality: 0,
      avgDeepSleepPct: 0,
      avgRemSleepPct: 0,
    };
  }

  const bedtimeHours = sessions.map(s => {
    const d = new Date(s.sessionStart);
    let hour = d.getHours() + d.getMinutes() / 60;
    // Normalize: if bedtime is before noon, it's after midnight
    if (hour < 12) hour += 24;
    return hour;
  });

  const wakeHours = sessions.map(s => {
    const d = new Date(s.sessionEnd);
    return d.getHours() + d.getMinutes() / 60;
  });

  let avgBedtime = mean(bedtimeHours);
  if (avgBedtime >= 24) avgBedtime -= 24;

  return {
    typicalBedtimeHour: avgBedtime,
    typicalWakeHour: mean(wakeHours),
    avgSleepDurationMin: mean(sessions.map(s => s.totalSleepMin)),
    avgSleepEfficiency: mean(sessions.map(s => s.sleepEfficiency)),
    avgSleepQuality: mean(sessions.map(s => s.qualityScore)),
    avgDeepSleepPct: mean(sessions.map(s => s.deepSleepPct)),
    avgRemSleepPct: mean(sessions.map(s => s.remSleepPct)),
  };
}

function computeDailySteps(windows: BiometricFeatureVector[]): number {
  // Group by date, sum steps per day, then average
  const dailySteps: Record<string, number> = {};
  for (const w of windows) {
    const day = new Date(w.timestamp).toISOString().slice(0, 10);
    dailySteps[day] = (dailySteps[day] || 0) + w.stepCount;
  }
  const days = Object.values(dailySteps);
  return days.length > 0 ? mean(days) : 0;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

function std(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  const variance = values.reduce((s, v) => s + (v - m) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}
