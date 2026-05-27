/**
 * services/ai/mockHealthData.ts
 *
 * Realistic mock biometric data generator for Seren.
 *
 * Produces physiologically plausible, smoothly time-varying HRV,
 * heart rate, sleep, and temperature data that mimics a real person's
 * 24-hour patterns. Used when Health Connect is unavailable (Expo Go)
 * or when the dev build is running without a paired watch.
 *
 * Key design decisions:
 *  - Circadian rhythm: HR and HRV follow a 24-hour sinusoidal pattern
 *    (lowest HR at ~4am, highest at ~3pm; HRV inversely correlated)
 *  - Natural variance: Gaussian noise layered on top of the base curve
 *  - Stress events: simulated stress spikes that look like real episodes
 *  - Continuity: state is persisted across calls so values evolve smoothly
 *    rather than jumping each render cycle
 *  - All values stay within clinically realistic ranges for a healthy adult
 */

// ─── Types (mirror the schema in db.ts) ──────────────────────────────────────

export interface MockBiometricSample {
  timestamp: number;       // Unix ms
  heartRate: number;       // BPM
  hrv_rmssd: number;       // ms
  hrv_sdnn: number;        // ms
  skinTemperature: number; // °C
  steps: number;           // steps in last 5 min
  spo2: number;            // %
  respiratoryRate: number; // breaths/min
  isRealData: false;
}

export interface MockSleepSession {
  startTime: number;
  endTime: number;
  stages: MockSleepStage[];
  isRealData: false;
}

export interface MockSleepStage {
  startTime: number;
  endTime: number;
  stage: 'awake' | 'light' | 'deep' | 'rem';
}

// ─── Constants ────────────────────────────────────────────────────────────────

const RESTING_HR_BASELINE = 68;        // BPM — healthy adult
const RESTING_HRV_RMSSD_BASELINE = 42; // ms  — healthy adult, moderate fitness
const RESTING_TEMP = 36.4;             // °C  — normal skin temperature
const RESTING_SPO2 = 97.5;             // %

// Circadian amplitude: how much HR rises from sleep trough to afternoon peak
const HR_CIRCADIAN_AMPLITUDE = 12;     // BPM
// HRV drops as HR rises (sympathovagal balance)
const HRV_CIRCADIAN_AMPLITUDE = 15;    // ms

// ─── Persistent state (survives across poll calls within a session) ───────────

interface MockState {
  lastTimestamp: number;
  currentHR: number;
  currentRMSSD: number;
  currentTemp: number;
  stressEventActive: boolean;
  stressEventProgress: number;  // 0–1, how far through the stress arc
  stressEventIntensity: number; // 0.3–1.0
  cumulativeSteps: number;
  lastStepReset: number;        // timestamp of last midnight step reset
}

let _state: MockState | null = null;

function getState(): MockState {
  if (!_state) {
    const now = Date.now();
    _state = {
      lastTimestamp: now,
      currentHR: RESTING_HR_BASELINE,
      currentRMSSD: RESTING_HRV_RMSSD_BASELINE,
      currentTemp: RESTING_TEMP,
      stressEventActive: false,
      stressEventProgress: 0,
      stressEventIntensity: 0,
      cumulativeSteps: 0,
      lastStepReset: startOfDay(now),
    };
  }
  return _state;
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

/** Box-Muller gaussian sample with given mean and std */
function gaussian(mean: number, std: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}

/** Clamp a value between min and max */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Get start-of-day timestamp for a given ms timestamp */
function startOfDay(ts: number): number {
  const d = new Date(ts);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

/**
 * Circadian HR offset: returns a value from -amplitude to +amplitude
 * based on time of day. Peak HR at 15:00 (3pm), trough at 04:00 (4am).
 */
function circadianHROffset(timestamp: number): number {
  const date = new Date(timestamp);
  const hourDecimal = date.getHours() + date.getMinutes() / 60;
  // Phase shift: peak at hour 15, so shift by (15 - 6) = 9 hours from standard cosine
  const phaseShift = 15;
  const radians = (2 * Math.PI * (hourDecimal - phaseShift)) / 24;
  return HR_CIRCADIAN_AMPLITUDE * Math.sin(radians);
}

/** Expected resting HR at a given time of day */
function expectedHR(timestamp: number): number {
  return RESTING_HR_BASELINE + circadianHROffset(timestamp);
}

/** Expected RMSSD at a given time of day (inversely correlated with HR) */
function expectedRMSSD(timestamp: number): number {
  return RESTING_HRV_RMSSD_BASELINE - (circadianHROffset(timestamp) / HR_CIRCADIAN_AMPLITUDE) * HRV_CIRCADIAN_AMPLITUDE;
}

// ─── Stress event simulation ──────────────────────────────────────────────────

/**
 * Randomly trigger stress events.
 * Probability ~8% per 5-min window during working hours (9–18),
 * ~1% outside working hours. Each event lasts 15–45 minutes.
 */
function maybeStartStressEvent(timestamp: number): void {
  const state = getState();
  if (state.stressEventActive) return;

  const hour = new Date(timestamp).getHours();
  const inWorkHours = hour >= 9 && hour <= 18;
  const probability = inWorkHours ? 0.08 : 0.01;

  if (Math.random() < probability) {
    state.stressEventActive = true;
    state.stressEventProgress = 0;
    state.stressEventIntensity = 0.4 + Math.random() * 0.6; // 0.4–1.0
  }
}

/**
 * Advance the stress event state by one 5-minute window.
 * Stress arc: ramp up over 10 min, hold for 10–20 min, decay over 15 min.
 * Returns { hrDelta, hrv_rmssd_delta } to add to base values.
 */
function advanceStressEvent(): { hrDelta: number; hrvDelta: number } {
  const state = getState();
  if (!state.stressEventActive) return { hrDelta: 0, hrvDelta: 0 };

  // Progress increments by ~0.067 per 5-min window (full arc ~75 min)
  state.stressEventProgress += 0.067;

  if (state.stressEventProgress >= 1.0) {
    state.stressEventActive = false;
    state.stressEventProgress = 0;
    return { hrDelta: 0, hrvDelta: 0 };
  }

  // Shape: sin curve peaking at progress=0.4
  const intensity = state.stressEventIntensity;
  const arc = Math.sin(state.stressEventProgress * Math.PI);

  const hrDelta = intensity * 18 * arc;     // up to +18 BPM at peak
  const hrvDelta = -(intensity * 20 * arc); // up to -20 ms RMSSD at peak

  return { hrDelta, hrvDelta };
}

// ─── Main data generation ─────────────────────────────────────────────────────

/**
 * Generate a single 5-minute biometric window at the given timestamp.
 * Each call smoothly evolves state from the previous call.
 */
export function generateMockBiometricSample(
  timestamp: number = Date.now()
): MockBiometricSample {
  const state = getState();

  // Check for stress events
  maybeStartStressEvent(timestamp);
  const { hrDelta, hrvDelta } = advanceStressEvent();

  // Target values for this window
  const targetHR = expectedHR(timestamp) + hrDelta;
  const targetRMSSD = clamp(expectedRMSSD(timestamp) + hrvDelta, 15, 90);

  // Smooth toward target (exponential moving average, alpha=0.3)
  // This prevents jumps between readings
  const alpha = 0.3;
  state.currentHR = alpha * targetHR + (1 - alpha) * state.currentHR;
  state.currentRMSSD = alpha * targetRMSSD + (1 - alpha) * state.currentRMSSD;

  // Add per-sample noise
  const hr = clamp(gaussian(state.currentHR, 1.5), 45, 140);
  const rmssd = clamp(gaussian(state.currentRMSSD, 3.0), 12, 95);

  // SDNN is typically ~1.6× RMSSD for healthy adults at rest
  const sdnn = clamp(rmssd * (1.4 + Math.random() * 0.4), 10, 130);

  // Skin temperature: varies ±0.5°C across the day, noise ±0.1°C
  const tempCircadian = 0.3 * Math.sin((2 * Math.PI * (new Date(timestamp).getHours() - 14)) / 24);
  state.currentTemp = clamp(
    RESTING_TEMP + tempCircadian + gaussian(0, 0.08),
    35.0,
    38.0
  );

  // SpO2: stable 96–99%, very slight HR correlation
  const spo2 = clamp(gaussian(RESTING_SPO2 - (hr - 65) * 0.02, 0.4), 94, 99.5);

  // Respiratory rate: loosely correlated with HR
  const respRate = clamp(gaussian(12 + (hr - 65) * 0.06, 0.8), 10, 22);

  // Steps in this 5-min window: depends on time of day and activity
  const hour = new Date(timestamp).getHours();
  const isActiveHour = (hour >= 7 && hour <= 8) || (hour >= 12 && hour <= 13) || (hour >= 17 && hour <= 19);
  const stepRate = isActiveHour
    ? gaussian(450, 120)     // ~90 steps/min during active periods
    : gaussian(80, 60);      // ~16 steps/min otherwise (walking around)
  const steps = clamp(Math.round(Math.max(0, stepRate)), 0, 800);

  // Cumulative step reset at midnight
  if (startOfDay(timestamp) > state.lastStepReset) {
    state.cumulativeSteps = 0;
    state.lastStepReset = startOfDay(timestamp);
  }
  state.cumulativeSteps += steps;
  state.lastTimestamp = timestamp;

  return {
    timestamp,
    heartRate: Math.round(hr),
    hrv_rmssd: Math.round(rmssd * 10) / 10,
    hrv_sdnn: Math.round(sdnn * 10) / 10,
    skinTemperature: Math.round(state.currentTemp * 10) / 10,
    steps,
    spo2: Math.round(spo2 * 10) / 10,
    respiratoryRate: Math.round(respRate * 10) / 10,
    isRealData: false,
  };
}

/**
 * Generate a sequence of biometric samples for the past N hours.
 * Used to populate the home dashboard and trend charts on first launch.
 */
export function generateMockHistory(hoursBack: number = 24): MockBiometricSample[] {
  const now = Date.now();
  const windowMs = 5 * 60 * 1000; // 5-minute windows
  const samples: MockBiometricSample[] = [];

  // Reset state for a clean history generation
  _state = null;

  const startTs = now - hoursBack * 60 * 60 * 1000;
  let ts = startTs;

  while (ts <= now) {
    samples.push(generateMockBiometricSample(ts));
    ts += windowMs;
  }

  // Reset state again so live polling starts fresh from now
  _state = null;

  return samples;
}

/**
 * Generate a realistic sleep session for last night.
 * Produces a ~7.5h session with proper NREM/REM cycling.
 */
export function generateMockSleepSession(): MockSleepSession {
  const now = new Date();
  // Sleep onset: 11pm–12:30am yesterday
  const sleepOnsetHour = 23 + Math.random() * 1.5;
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  yesterday.setHours(Math.floor(sleepOnsetHour), Math.round((sleepOnsetHour % 1) * 60), 0, 0);

  const sleepOnset = yesterday.getTime();
  // Total sleep time: 7–8.5 hours
  const totalSleepMs = (7 + Math.random() * 1.5) * 60 * 60 * 1000;
  const wakeTime = sleepOnset + totalSleepMs;

  // Build realistic sleep architecture: ~4–5 NREM/REM cycles of ~90 min each
  const stages: MockSleepStage[] = [];
  let cursor = sleepOnset;

  // Sleep onset latency: 5–20 min of light/awake
  const onsetLatency = (5 + Math.random() * 15) * 60 * 1000;
  stages.push({ startTime: cursor, endTime: cursor + onsetLatency, stage: 'light' });
  cursor += onsetLatency;

  const cycleCount = Math.floor(totalSleepMs / (90 * 60 * 1000));

  for (let cycle = 0; cycle < cycleCount && cursor < wakeTime; cycle++) {
    const isEarlyCycle = cycle < 2;

    // N2 light sleep: 15–25 min
    const lightDuration = (15 + Math.random() * 10) * 60 * 1000;
    stages.push({ startTime: cursor, endTime: Math.min(cursor + lightDuration, wakeTime), stage: 'light' });
    cursor += lightDuration;
    if (cursor >= wakeTime) break;

    // N3 deep sleep: more in early cycles, less later
    const deepDuration = isEarlyCycle
      ? (20 + Math.random() * 20) * 60 * 1000  // 20–40 min early
      : (5 + Math.random() * 10) * 60 * 1000;  // 5–15 min late
    stages.push({ startTime: cursor, endTime: Math.min(cursor + deepDuration, wakeTime), stage: 'deep' });
    cursor += deepDuration;
    if (cursor >= wakeTime) break;

    // REM: more in later cycles
    const remDuration = isEarlyCycle
      ? (10 + Math.random() * 10) * 60 * 1000  // 10–20 min early
      : (20 + Math.random() * 25) * 60 * 1000; // 20–45 min late
    stages.push({ startTime: cursor, endTime: Math.min(cursor + remDuration, wakeTime), stage: 'rem' });
    cursor += remDuration;
    if (cursor >= wakeTime) break;

    // Brief micro-arousal ~10% chance
    if (Math.random() < 0.1) {
      const awakeDuration = (1 + Math.random() * 3) * 60 * 1000;
      stages.push({ startTime: cursor, endTime: Math.min(cursor + awakeDuration, wakeTime), stage: 'awake' });
      cursor += awakeDuration;
    }
  }

  // Final light sleep before wake
  if (cursor < wakeTime) {
    stages.push({ startTime: cursor, endTime: wakeTime, stage: 'light' });
  }

  return {
    startTime: sleepOnset,
    endTime: wakeTime,
    stages,
    isRealData: false,
  };
}

/**
 * Trigger a simulated stress event immediately.
 * Useful for demo purposes — call this from a hidden dev button
 * to show the stress score rising in real time.
 *
 * Usage (in your component):
 *   import { triggerDemoStressEvent } from './mockHealthData';
 *   <Pressable onLongPress={triggerDemoStressEvent}>...</Pressable>
 */
export function triggerDemoStressEvent(intensity: number = 0.8): void {
  const state = getState();
  state.stressEventActive = true;
  state.stressEventProgress = 0;
  state.stressEventIntensity = clamp(intensity, 0.3, 1.0);
  console.log(`[MockData] Demo stress event triggered (intensity=${intensity})`);
}

/**
 * Reset all mock state. Useful in tests or when switching between
 * mock and real data modes.
 */
export function resetMockState(): void {
  _state = null;
}
