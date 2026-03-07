/**
 * Seren AI — Sleep Analysis Service
 * ====================================
 * Computes sleep quality scores from Health Connect sleep sessions.
 *
 * Quality scoring uses a weighted formula based on clinical sleep research:
 *   - Sleep efficiency (time asleep / time in bed)
 *   - Deep sleep percentage (restorative slow-wave sleep)
 *   - REM sleep percentage (cognitive restoration)
 *   - Sleep onset latency (time to fall asleep)
 *   - Fragmentation (awakenings per hour)
 *   - Total duration adequacy
 */

import {
  RawSleepSession,
  RawSleepStage,
  SleepAnalysis,
  RawHeartRateSample,
  RawHRVSample,
  PersonalBaseline,
} from './types';

// ============================================================
// Clinical Thresholds
// ============================================================

const IDEAL_SLEEP_MIN = 420;          // 7 hours
const MAX_IDEAL_SLEEP_MIN = 540;      // 9 hours
const IDEAL_DEEP_SLEEP_PCT = 0.20;    // 20% of total sleep
const IDEAL_REM_SLEEP_PCT = 0.25;     // 25% of total sleep
const IDEAL_ONSET_LATENCY_MIN = 15;   // <15 min is good
const MAX_ONSET_LATENCY_MIN = 30;     // >30 min is poor
const IDEAL_EFFICIENCY = 0.85;        // >85% is good

// Quality score weights (sum to 1.0)
const WEIGHTS = {
  efficiency: 0.25,
  deepSleep: 0.20,
  remSleep: 0.15,
  duration: 0.20,
  fragmentation: 0.10,
  onsetLatency: 0.10,
};

// ============================================================
// Public API
// ============================================================

/**
 * Analyze a sleep session from Health Connect data.
 *
 * @param session - Raw sleep session with stages
 * @param hrDuringSleep - Heart rate samples during sleep window (optional)
 * @param hrvDuringSleep - HRV samples during sleep window (optional)
 * @param baseline - Personal baseline for consistency scoring (optional)
 */
export function analyzeSleepSession(
  session: RawSleepSession,
  hrDuringSleep?: RawHeartRateSample[],
  hrvDuringSleep?: RawHRVSample[],
  baseline?: PersonalBaseline | null,
): SleepAnalysis {
  const stages = session.stages;

  // ---- Compute stage durations ----
  const stageDurations = computeStageDurations(stages);
  const totalInBedMin = (session.endTime - session.startTime) / 60000;
  const totalSleepMin = stageDurations.light + stageDurations.deep + stageDurations.rem;

  // ---- Sleep onset latency ----
  const onsetLatencyMin = computeOnsetLatency(session.startTime, stages);

  // ---- WASO (Wake After Sleep Onset) ----
  const wasoMin = computeWASO(stages, session.startTime);

  // ---- Stage percentages ----
  const deepPct = totalSleepMin > 0 ? stageDurations.deep / totalSleepMin : 0;
  const remPct = totalSleepMin > 0 ? stageDurations.rem / totalSleepMin : 0;
  const lightPct = totalSleepMin > 0 ? stageDurations.light / totalSleepMin : 0;

  // ---- Sleep efficiency ----
  const efficiency = totalInBedMin > 0 ? totalSleepMin / totalInBedMin : 0;

  // ---- Fragmentation ----
  const awakeningCount = countAwakenings(stages);
  const sleepHours = totalSleepMin / 60;
  const fragmentationIndex = sleepHours > 0 ? awakeningCount / sleepHours : 0;

  // ---- Biometric summaries ----
  const hrStats = computeHRStats(hrDuringSleep);
  const hrvStats = computeHRVStats(hrvDuringSleep);

  // ---- Quality score ----
  const qualityScore = computeQualityScore({
    efficiency,
    deepPct,
    remPct,
    totalSleepMin,
    onsetLatencyMin,
    fragmentationIndex,
  });

  // ---- Recovery score (HRV-based) ----
  const recoveryScore = computeRecoveryScore(hrvStats.avgHrv, hrvStats.maxHrv, baseline);

  // ---- Consistency score ----
  const consistencyScore = computeConsistencyScore(session, baseline);

  const date = new Date(session.startTime).toISOString().slice(0, 10);

  return {
    date,
    sessionStart: session.startTime,
    sessionEnd: session.endTime,

    totalInBedMin: Math.round(totalInBedMin),
    totalSleepMin: Math.round(totalSleepMin),
    onsetLatencyMin: Math.round(onsetLatencyMin),
    wasoMin: Math.round(wasoMin),

    lightSleepMin: Math.round(stageDurations.light),
    deepSleepMin: Math.round(stageDurations.deep),
    remSleepMin: Math.round(stageDurations.rem),
    awakeMin: Math.round(stageDurations.awake),

    deepSleepPct: round2(deepPct),
    remSleepPct: round2(remPct),
    lightSleepPct: round2(lightPct),

    sleepEfficiency: round2(efficiency),
    fragmentationIndex: round2(fragmentationIndex),
    awakeningCount,

    avgHrSleep: hrStats.avgHr,
    minHrSleep: hrStats.minHr,
    avgHrvSleep: hrvStats.avgHrv,
    maxHrvSleep: hrvStats.maxHrv,

    qualityScore: Math.round(qualityScore),
    recoveryScore: Math.round(recoveryScore),
    consistencyScore: Math.round(consistencyScore),
  };
}

/**
 * Compute a trend summary for sleep over the last N days.
 */
export function computeSleepTrend(sessions: SleepAnalysis[], days: number = 7): SleepTrend {
  const recent = sessions.slice(-days);
  if (recent.length === 0) {
    return { avgQuality: 0, avgDuration: 0, trend: 'stable', daysAnalyzed: 0 };
  }

  const avgQuality = mean(recent.map(s => s.qualityScore));
  const avgDuration = mean(recent.map(s => s.totalSleepMin));

  // Trend: compare first half vs second half
  let trend: 'improving' | 'declining' | 'stable' = 'stable';
  if (recent.length >= 4) {
    const mid = Math.floor(recent.length / 2);
    const firstHalf = mean(recent.slice(0, mid).map(s => s.qualityScore));
    const secondHalf = mean(recent.slice(mid).map(s => s.qualityScore));
    const diff = secondHalf - firstHalf;
    if (diff > 5) trend = 'improving';
    else if (diff < -5) trend = 'declining';
  }

  return { avgQuality, avgDuration, trend, daysAnalyzed: recent.length };
}

export interface SleepTrend {
  avgQuality: number;
  avgDuration: number;
  trend: 'improving' | 'declining' | 'stable';
  daysAnalyzed: number;
}

// ============================================================
// Internal Helpers
// ============================================================

interface StageDurations {
  awake: number;
  light: number;
  deep: number;
  rem: number;
}

function computeStageDurations(stages: RawSleepStage[]): StageDurations {
  const durations: StageDurations = { awake: 0, light: 0, deep: 0, rem: 0 };

  for (const stage of stages) {
    const durationMin = (stage.endTime - stage.startTime) / 60000;
    switch (stage.stage) {
      case 'awake':
      case 'out_of_bed':
        durations.awake += durationMin;
        break;
      case 'light':
      case 'sleeping': // Generic "sleeping" treated as light
        durations.light += durationMin;
        break;
      case 'deep':
        durations.deep += durationMin;
        break;
      case 'rem':
        durations.rem += durationMin;
        break;
    }
  }

  return durations;
}

function computeOnsetLatency(sessionStart: number, stages: RawSleepStage[]): number {
  // Time from session start to first non-awake stage
  for (const stage of stages) {
    if (stage.stage !== 'awake' && stage.stage !== 'out_of_bed' && stage.stage !== 'unknown') {
      return Math.max(0, (stage.startTime - sessionStart) / 60000);
    }
  }
  return 0;
}

function computeWASO(stages: RawSleepStage[], sessionStart: number): number {
  // Sum of all awake time AFTER first sleep onset
  let sleepStarted = false;
  let wasoMin = 0;

  for (const stage of stages) {
    if (!sleepStarted) {
      if (stage.stage !== 'awake' && stage.stage !== 'out_of_bed' && stage.stage !== 'unknown') {
        sleepStarted = true;
      }
      continue;
    }
    if (stage.stage === 'awake') {
      wasoMin += (stage.endTime - stage.startTime) / 60000;
    }
  }

  return wasoMin;
}

function countAwakenings(stages: RawSleepStage[]): number {
  let count = 0;
  let wasSleeping = false;

  for (const stage of stages) {
    const isSleep = stage.stage !== 'awake' && stage.stage !== 'out_of_bed' && stage.stage !== 'unknown';
    if (wasSleeping && !isSleep) {
      count++;
    }
    wasSleeping = isSleep;
  }

  return count;
}

interface QualityInputs {
  efficiency: number;
  deepPct: number;
  remPct: number;
  totalSleepMin: number;
  onsetLatencyMin: number;
  fragmentationIndex: number;
}

function computeQualityScore(inputs: QualityInputs): number {
  // Each component scored 0-100, then weighted

  // Efficiency: 85%+ = 100, 70% = 50, <60% = 0
  const efficiencyScore = clamp(((inputs.efficiency - 0.6) / (IDEAL_EFFICIENCY - 0.6)) * 100, 0, 100);

  // Deep sleep: 20%+ = 100, 10% = 50, <5% = 0
  const deepScore = clamp((inputs.deepPct / IDEAL_DEEP_SLEEP_PCT) * 100, 0, 100);

  // REM: 25%+ = 100, 15% = 60, <10% = 0
  const remScore = clamp((inputs.remPct / IDEAL_REM_SLEEP_PCT) * 100, 0, 100);

  // Duration: 7-9h = 100, 6h = 70, <5h = 30, >10h penalized
  let durationScore: number;
  if (inputs.totalSleepMin >= IDEAL_SLEEP_MIN && inputs.totalSleepMin <= MAX_IDEAL_SLEEP_MIN) {
    durationScore = 100;
  } else if (inputs.totalSleepMin < IDEAL_SLEEP_MIN) {
    durationScore = clamp((inputs.totalSleepMin / IDEAL_SLEEP_MIN) * 100, 0, 100);
  } else {
    // Oversleeping penalty
    const excess = inputs.totalSleepMin - MAX_IDEAL_SLEEP_MIN;
    durationScore = clamp(100 - excess * 0.5, 50, 100);
  }

  // Onset latency: <15min = 100, 30min = 50, >45min = 0
  const latencyScore = inputs.onsetLatencyMin <= IDEAL_ONSET_LATENCY_MIN
    ? 100
    : clamp(100 - ((inputs.onsetLatencyMin - IDEAL_ONSET_LATENCY_MIN) / (MAX_ONSET_LATENCY_MIN - IDEAL_ONSET_LATENCY_MIN)) * 100, 0, 100);

  // Fragmentation: 0 = 100, 2/hr = 50, >4/hr = 0
  const fragScore = clamp(100 - inputs.fragmentationIndex * 25, 0, 100);

  return (
    WEIGHTS.efficiency * efficiencyScore +
    WEIGHTS.deepSleep * deepScore +
    WEIGHTS.remSleep * remScore +
    WEIGHTS.duration * durationScore +
    WEIGHTS.onsetLatency * latencyScore +
    WEIGHTS.fragmentation * fragScore
  );
}

function computeRecoveryScore(
  avgHrv: number,
  maxHrv: number,
  baseline?: PersonalBaseline | null,
): number {
  if (avgHrv === 0) return 50; // No data, neutral

  if (baseline && baseline.nocturnalRmssdMean > 0) {
    // Compare to personal nocturnal HRV baseline
    const ratio = avgHrv / baseline.nocturnalRmssdMean;
    // ratio > 1 = better than usual, < 1 = worse
    return clamp(ratio * 70, 0, 100);
  }

  // Population-based fallback: RMSSD 40-80ms = good range
  if (avgHrv >= 60) return 85;
  if (avgHrv >= 40) return 70;
  if (avgHrv >= 25) return 50;
  return 30;
}

function computeConsistencyScore(
  session: RawSleepSession,
  baseline?: PersonalBaseline | null,
): number {
  if (!baseline || baseline.avgSleepDurationMin === 0) return 50;

  const bedHour = (() => {
    const d = new Date(session.startTime);
    let h = d.getHours() + d.getMinutes() / 60;
    if (h < 12) h += 24;
    return h;
  })();

  let typicalBed = baseline.typicalBedtimeHour;
  if (typicalBed < 12) typicalBed += 24;

  // Bedtime deviation (hours)
  const bedtimeDeviation = Math.abs(bedHour - typicalBed);
  const bedtimeScore = clamp(100 - bedtimeDeviation * 20, 0, 100);

  // Duration deviation
  const durationMin = (session.endTime - session.startTime) / 60000;
  const durationDeviation = Math.abs(durationMin - baseline.avgSleepDurationMin);
  const durationScore = clamp(100 - (durationDeviation / 60) * 25, 0, 100);

  return (bedtimeScore + durationScore) / 2;
}

function computeHRStats(samples?: RawHeartRateSample[]) {
  if (!samples || samples.length === 0) {
    return { avgHr: 0, minHr: 0 };
  }
  const bpms = samples.map(s => s.bpm).filter(v => v > 0);
  return {
    avgHr: bpms.length > 0 ? Math.round(mean(bpms)) : 0,
    minHr: bpms.length > 0 ? Math.min(...bpms) : 0,
  };
}

function computeHRVStats(samples?: RawHRVSample[]) {
  if (!samples || samples.length === 0) {
    return { avgHrv: 0, maxHrv: 0 };
  }
  const values = samples.map(s => s.rmssd).filter(v => v > 0);
  return {
    avgHrv: values.length > 0 ? Math.round(mean(values)) : 0,
    maxHrv: values.length > 0 ? Math.max(...values) : 0,
  };
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function round2(value: number): number {
  return Math.round(value * 100) / 100;
}
