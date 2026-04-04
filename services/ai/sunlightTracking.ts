/**
 * Seren AI — Sunlight Exposure Tracking
 * =======================================
 * Monitors ambient light levels to estimate time spent outdoors.
 * Recommends sunlight exposure during optimal vitamin D windows (10am-3pm).
 *
 * Uses expo-sensors LightSensor on supported devices.
 * Falls back to mock data on unsupported platforms (iOS, web, Expo Go).
 *
 * Clinical basis:
 * - 15-30 min daily outdoor light helps regulate circadian rhythm
 * - Vitamin D synthesis requires UVB exposure (10am-3pm optimal)
 * - Low sunlight exposure correlates with seasonal affective disorder
 */

import { LightSensor } from 'expo-sensors';
import { SunlightReading, SunlightExposureSummary } from './types';

// ============================================================
// Constants
// ============================================================

/** Lux threshold to classify as outdoors (indoor: 100-500, outdoor shade: 1000+) */
export const OUTDOOR_LUX_THRESHOLD = 1000;

/** Default daily sunlight goal in minutes */
export const DEFAULT_GOAL_MINUTES = 30;

/** Vitamin D optimal window */
export const VITAMIN_D_START_HOUR = 10;
export const VITAMIN_D_END_HOUR = 15;

/** Sensor polling interval (1 minute — battery friendly) */
const SAMPLE_INTERVAL_MS = 60000;

// ============================================================
// Sensor Monitoring
// ============================================================

/**
 * Check if the ambient light sensor is available on this device.
 */
export async function isLightSensorAvailable(): Promise<boolean> {
  try {
    return await LightSensor.isAvailableAsync();
  } catch {
    return false;
  }
}

/**
 * Start monitoring ambient light levels.
 * Calls onReading every ~60 seconds with the latest lux value.
 * Returns an unsubscribe function.
 */
export async function startSunlightMonitoring(
  onReading: (reading: SunlightReading) => void,
): Promise<(() => void) | null> {
  const available = await isLightSensorAvailable();
  if (!available) return null;

  LightSensor.setUpdateInterval(SAMPLE_INTERVAL_MS);

  const subscription = LightSensor.addListener((data) => {
    const reading: SunlightReading = {
      timestamp: Date.now(),
      luxValue: data.illuminance,
      isOutdoors: data.illuminance >= OUTDOOR_LUX_THRESHOLD,
    };
    onReading(reading);
  });

  return () => subscription.remove();
}

// ============================================================
// Summary Computation
// ============================================================

/**
 * Check if the current time is within the vitamin D optimal window.
 */
export function isVitaminDWindow(): boolean {
  const hour = new Date().getHours();
  return hour >= VITAMIN_D_START_HOUR && hour < VITAMIN_D_END_HOUR;
}

/**
 * Compute daily sunlight summary from a list of sensor readings.
 * Each reading represents ~1 minute of sampling.
 */
export function computeSunlightSummary(
  samples: SunlightReading[],
  goalMinutes: number = DEFAULT_GOAL_MINUTES,
): SunlightExposureSummary {
  const today = new Date().toISOString().slice(0, 10);

  if (samples.length === 0) {
    return {
      date: today,
      totalOutdoorMinutes: 0,
      optimalWindowMinutes: 0,
      peakLux: 0,
      avgOutdoorLux: 0,
      goalMinutes,
      goalProgress: 0,
      isVitaminDWindow: isVitaminDWindow(),
    };
  }

  const outdoorSamples = samples.filter(s => s.isOutdoors);
  const totalOutdoorMinutes = outdoorSamples.length; // Each sample ≈ 1 minute

  // Count minutes in the vitamin D optimal window (10am-3pm)
  const optimalSamples = outdoorSamples.filter(s => {
    const hour = new Date(s.timestamp).getHours();
    return hour >= VITAMIN_D_START_HOUR && hour < VITAMIN_D_END_HOUR;
  });
  const optimalWindowMinutes = optimalSamples.length;

  const peakLux = Math.max(...samples.map(s => s.luxValue), 0);
  const avgOutdoorLux = outdoorSamples.length > 0
    ? outdoorSamples.reduce((sum, s) => sum + s.luxValue, 0) / outdoorSamples.length
    : 0;

  return {
    date: today,
    totalOutdoorMinutes,
    optimalWindowMinutes,
    peakLux,
    avgOutdoorLux,
    goalMinutes,
    goalProgress: goalMinutes > 0 ? Math.min(1, totalOutdoorMinutes / goalMinutes) : 0,
    isVitaminDWindow: isVitaminDWindow(),
  };
}

// ============================================================
// Mock Data (for development in Expo Go / unsupported devices)
// ============================================================

/**
 * Generate plausible mock sunlight exposure data.
 * Simulates partial outdoor time based on current hour.
 */
export function getMockSunlightExposure(): SunlightExposureSummary {
  const today = new Date().toISOString().slice(0, 10);
  const hour = new Date().getHours();

  // Simulate accumulating outdoor time through the day
  let outdoorMin = 0;
  if (hour >= 7 && hour < 9) outdoorMin = 5 + Math.round(Math.random() * 5);
  else if (hour >= 9 && hour < 12) outdoorMin = 10 + Math.round(Math.random() * 10);
  else if (hour >= 12 && hour < 15) outdoorMin = 18 + Math.round(Math.random() * 12);
  else if (hour >= 15 && hour < 18) outdoorMin = 22 + Math.round(Math.random() * 15);
  else if (hour >= 18) outdoorMin = 25 + Math.round(Math.random() * 15);

  const optimalMin = Math.min(outdoorMin, Math.round(outdoorMin * 0.6));

  return {
    date: today,
    totalOutdoorMinutes: outdoorMin,
    optimalWindowMinutes: optimalMin,
    peakLux: 15000 + Math.round(Math.random() * 30000),
    avgOutdoorLux: 5000 + Math.round(Math.random() * 10000),
    goalMinutes: DEFAULT_GOAL_MINUTES,
    goalProgress: Math.min(1, outdoorMin / DEFAULT_GOAL_MINUTES),
    isVitaminDWindow: isVitaminDWindow(),
  };
}

/**
 * Generate mock weekly sunlight history.
 */
export function getMockWeeklySunlight(): { date: string; value: number }[] {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  return days.map(d => ({
    date: d,
    value: Math.round(10 + Math.random() * 35),
  }));
}
