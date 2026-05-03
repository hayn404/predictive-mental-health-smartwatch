/**
 * Seren AI — Health Connect Data Ingestion Service
 * ==================================================
 * Reads biometric data from Google Health Connect (Samsung Galaxy Watch via Wear OS).
 *
 * Data flow:
 *   Samsung Galaxy Watch → Samsung Health → Google Health Connect → This service → SQLite
 *
 * Health Connect data types used:
 *   - HeartRateRecord           → HR BPM samples
 *   - HeartRateVariabilityRmssdRecord → RMSSD values
 *   - SleepSessionRecord        → Sleep stages & duration
 *   - StepsRecord               → Step count
 *   - OxygenSaturationRecord    → SpO2
 *   - SkinTemperatureRecord     → Wrist temperature
 *   - RespiratoryRateRecord     → Breathing rate
 *
 * NOTE: Requires `react-native-health-connect` package and expo-dev-client.
 *       This file provides the interface. When running in dev/mock mode,
 *       the mock implementation is used instead.
 */

import {
  RawHeartRateSample,
  RawHRVSample,
  RawSleepSession,
  RawStepsSample,
  RawTemperatureSample,
  RawSpO2Sample,
  RawRespiratoryRateSample,
} from './types';

// ============================================================
// Health Connect Permission Types
// ============================================================

export const REQUIRED_PERMISSIONS = [
  'android.permission.health.READ_HEART_RATE',
  'android.permission.health.READ_HEART_RATE_VARIABILITY',
  'android.permission.health.READ_SLEEP',
  'android.permission.health.READ_STEPS',
  'android.permission.health.READ_OXYGEN_SATURATION',
  'android.permission.health.READ_SKIN_TEMPERATURE',
  'android.permission.health.READ_RESPIRATORY_RATE',
] as const;

// ============================================================
// Health Connect Service Interface
// ============================================================

export interface HealthConnectService {
  /** Check if Health Connect is available on this device */
  isAvailable(): Promise<boolean>;

  /** Request required permissions */
  requestPermissions(): Promise<boolean>;

  /** Check if all permissions are granted */
  hasPermissions(): Promise<boolean>;

  /** Read heart rate samples in a time range */
  readHeartRate(startTime: number, endTime: number): Promise<RawHeartRateSample[]>;

  /** Read HRV RMSSD samples in a time range */
  readHRV(startTime: number, endTime: number): Promise<RawHRVSample[]>;

  /** Read sleep sessions in a time range */
  readSleepSessions(startTime: number, endTime: number): Promise<RawSleepSession[]>;

  /** Read step records in a time range */
  readSteps(startTime: number, endTime: number): Promise<RawStepsSample[]>;

  /** Read skin temperature in a time range */
  readTemperature(startTime: number, endTime: number): Promise<RawTemperatureSample[]>;

  /** Read SpO2 in a time range */
  readSpO2(startTime: number, endTime: number): Promise<RawSpO2Sample[]>;

  /** Read respiratory rate in a time range */
  readRespiratoryRate(startTime: number, endTime: number): Promise<RawRespiratoryRateSample[]>;
}

// ============================================================
// Real Health Connect Implementation
// ============================================================
// Uses react-native-health-connect package
// Only works on Android with expo-dev-client (not Expo Go)

export function createHealthConnectService(): HealthConnectService {
  // Dynamic import to avoid crash when package isn't installed
  let HC: typeof import('react-native-health-connect') | null = null;
  let initialized = false;

  const getHC = async () => {
    if (!HC) {
      try {
        HC = await import('react-native-health-connect');
      } catch {
        throw new Error('react-native-health-connect not installed. Use mock service instead.');
      }
    }
    // Must call initialize() before any other Health Connect API call
    if (!initialized) {
      await HC.initialize();
      initialized = true;
      console.log('[Seren] Health Connect SDK initialized');
    }
    return HC;
  };

  return {
    async isAvailable() {
      try {
        const hc = await getHC();
        const status = await hc.getSdkStatus();
        console.log('[Seren] Health Connect SDK status:', status);
        return status === hc.SdkAvailabilityStatus.SDK_AVAILABLE;
      } catch (e) {
        console.warn('[Seren] Health Connect availability check failed:', e);
        return false;
      }
    },

    async requestPermissions() {
      try {
        const hc = await getHC();
        // Request permissions via the native popup.
        // Requires MainActivity to call HealthConnectPermissionDelegate.setPermissionDelegate(this) in onCreate.
        const granted = await hc.requestPermission([
          { accessType: 'read', recordType: 'HeartRate' },
          { accessType: 'read', recordType: 'HeartRateVariabilityRmssd' },
          { accessType: 'read', recordType: 'SleepSession' },
          { accessType: 'read', recordType: 'Steps' },
          { accessType: 'read', recordType: 'OxygenSaturation' },
        ]);
        console.log('[Seren] Health Connect permissions result:', JSON.stringify(granted));
        return Array.isArray(granted) && granted.length > 0;
      } catch (e) {
        console.error('[Seren] Health Connect permission request failed:', e);
        // Fallback: open Health Connect settings so user can grant manually
        try {
          const hc = await getHC();
          if (typeof (hc as any).openHealthConnectSettings === 'function') {
            await (hc as any).openHealthConnectSettings();
            return true;
          }
        } catch {}
        return false;
      }
    },

    async hasPermissions() {
      // Attempt a small read to check if we have permissions
      try {
        const hc = await getHC();
        const now = new Date();
        await hc.readRecords('HeartRate', {
          timeRangeFilter: {
            operator: 'between',
            startTime: new Date(now.getTime() - 60000).toISOString(),
            endTime: now.toISOString(),
          },
        });
        return true;
      } catch (e) {
        console.log('[Seren] hasPermissions check failed (likely not granted yet):', e);
        return false;
      }
    },

    async readHeartRate(startTime, endTime) {
      const hc = await getHC();
      const result = await hc.readRecords('HeartRate', {
        timeRangeFilter: {
          operator: 'between',
          startTime: new Date(startTime).toISOString(),
          endTime: new Date(endTime).toISOString(),
        },
      });

      const samples: RawHeartRateSample[] = [];
      for (const record of result.records) {
        for (const sample of (record as any).samples || []) {
          samples.push({
            timestamp: new Date(sample.time).getTime(),
            bpm: sample.beatsPerMinute,
            source: (record as any).metadata?.dataOrigin?.packageName || 'unknown',
          });
        }
      }
      return samples;
    },

    async readHRV(startTime, endTime) {
      const hc = await getHC();
      const result = await hc.readRecords('HeartRateVariabilityRmssd', {
        timeRangeFilter: {
          operator: 'between',
          startTime: new Date(startTime).toISOString(),
          endTime: new Date(endTime).toISOString(),
        },
      });

      return result.records.map((record: any) => ({
        timestamp: new Date(record.time).getTime(),
        rmssd: record.heartRateVariabilityMillis,
      }));
    },

    async readSleepSessions(startTime, endTime) {
      const hc = await getHC();
      const result = await hc.readRecords('SleepSession', {
        timeRangeFilter: {
          operator: 'between',
          startTime: new Date(startTime).toISOString(),
          endTime: new Date(endTime).toISOString(),
        },
      });

      return result.records.map((record: any) => ({
        startTime: new Date(record.startTime).getTime(),
        endTime: new Date(record.endTime).getTime(),
        stages: (record.stages || []).map((stage: any) => ({
          startTime: new Date(stage.startTime).getTime(),
          endTime: new Date(stage.endTime).getTime(),
          stage: mapHealthConnectSleepStage(stage.stage),
        })),
        source: record.metadata?.dataOrigin?.packageName || 'unknown',
      }));
    },

    async readSteps(startTime, endTime) {
      const hc = await getHC();
      const result = await hc.readRecords('Steps', {
        timeRangeFilter: {
          operator: 'between',
          startTime: new Date(startTime).toISOString(),
          endTime: new Date(endTime).toISOString(),
        },
      });

      return result.records.map((record: any) => ({
        startTime: new Date(record.startTime).getTime(),
        endTime: new Date(record.endTime).getTime(),
        count: record.count,
      }));
    },

    async readTemperature(startTime, endTime) {
      // SkinTemperature not supported in all react-native-health-connect versions
      try {
        const hc = await getHC();
        const result = await hc.readRecords('SkinTemperature' as any, {
          timeRangeFilter: {
            operator: 'between',
            startTime: new Date(startTime).toISOString(),
            endTime: new Date(endTime).toISOString(),
          },
        });
        return result.records.map((record: any) => ({
          timestamp: new Date(record.time).getTime(),
          temperatureCelsius: record.temperature?.inCelsius || 0,
        }));
      } catch {
        return [];
      }
    },

    async readSpO2(startTime, endTime) {
      try {
        const hc = await getHC();
        const result = await hc.readRecords('OxygenSaturation', {
          timeRangeFilter: {
            operator: 'between',
            startTime: new Date(startTime).toISOString(),
            endTime: new Date(endTime).toISOString(),
          },
        });
        return result.records.map((record: any) => ({
          timestamp: new Date(record.time).getTime(),
          percentage: record.percentage,
        }));
      } catch {
        return [];
      }
    },

    async readRespiratoryRate(startTime, endTime) {
      // RespiratoryRate not supported in all react-native-health-connect versions
      try {
        const hc = await getHC();
        const result = await hc.readRecords('RespiratoryRate' as any, {
          timeRangeFilter: {
            operator: 'between',
            startTime: new Date(startTime).toISOString(),
            endTime: new Date(endTime).toISOString(),
          },
        });
        return result.records.map((record: any) => ({
          timestamp: new Date(record.time).getTime(),
          breathsPerMinute: record.rate,
        }));
      } catch {
        return [];
      }
    },
  };
}

// ============================================================
// Sleep Stage Mapping
// ============================================================

/** Map Health Connect numeric sleep stage to our enum */
function mapHealthConnectSleepStage(stage: number): RawSleepSession['stages'][0]['stage'] {
  // Health Connect sleep stage constants:
  // 0 = UNKNOWN, 1 = AWAKE, 2 = SLEEPING, 3 = OUT_OF_BED,
  // 4 = AWAKE_IN_BED, 5 = LIGHT, 6 = DEEP, 7 = REM
  const map: Record<number, RawSleepSession['stages'][0]['stage']> = {
    0: 'unknown',
    1: 'awake',
    2: 'sleeping',
    3: 'out_of_bed',
    4: 'awake',
    5: 'light',
    6: 'deep',
    7: 'rem',
  };
  return map[stage] || 'unknown';
}

// ============================================================
// Mock Health Connect Service (for development)
// ============================================================

export function createMockHealthConnectService(): HealthConnectService {
  // Simulate exam stress conditions
  const isExamTime = true; // Mock data simulates someone in an exam

  const generateHRSamples = (start: number, end: number): RawHeartRateSample[] => {
    const samples: RawHeartRateSample[] = [];
    const interval = 5000; // Every 5 seconds
    for (let t = start; t < end; t += interval) {
      // Exam stress: HR 90-110 BPM | Normal: 65-85 BPM
      const bpm = isExamTime ? 95 + Math.random() * 20 : 65 + Math.random() * 20;
      samples.push({
        timestamp: t,
        bpm,
        source: 'mock.samsung.health',
      });
    }
    return samples;
  };

  const generateHRVSamples = (start: number, end: number): RawHRVSample[] => {
    const samples: RawHRVSample[] = [];
    const interval = 300000; // Every 5 minutes
    for (let t = start; t < end; t += interval) {
      // Exam stress: HRV 15-35 ms (low) | Normal: 35-75 ms
      const rmssd = isExamTime ? 20 + Math.random() * 20 : 35 + Math.random() * 40;
      samples.push({
        timestamp: t,
        rmssd,
      });
    }
    return samples;
  };

  return {
    async isAvailable() { return true; },
    async requestPermissions() { return true; },
    async hasPermissions() { return true; },
    async readHeartRate(start, end) { return generateHRSamples(start, end); },
    async readHRV(start, end) { return generateHRVSamples(start, end); },
    async readSleepSessions(start, end) {
      const bedtime = new Date(start);
      bedtime.setHours(23, 0, 0, 0);
      const wake = new Date(start);
      wake.setDate(wake.getDate() + 1);
      wake.setHours(7, 0, 0, 0);

      return [{
        startTime: bedtime.getTime(),
        endTime: wake.getTime(),
        stages: [
          { startTime: bedtime.getTime(), endTime: bedtime.getTime() + 30 * 60000, stage: 'light' as const },
          { startTime: bedtime.getTime() + 30 * 60000, endTime: bedtime.getTime() + 120 * 60000, stage: 'deep' as const },
          { startTime: bedtime.getTime() + 120 * 60000, endTime: bedtime.getTime() + 180 * 60000, stage: 'rem' as const },
          { startTime: bedtime.getTime() + 180 * 60000, endTime: bedtime.getTime() + 300 * 60000, stage: 'light' as const },
          { startTime: bedtime.getTime() + 300 * 60000, endTime: wake.getTime(), stage: 'deep' as const },
        ],
        source: 'mock.samsung.health',
      }];
    },
    async readSteps(start, end) {
      return [{ startTime: start, endTime: end, count: 4500 + Math.floor(Math.random() * 3000) }];
    },
    async readTemperature(start, end) {
      const samples: RawTemperatureSample[] = [];
      for (let t = start; t < end; t += 60000) {
        // Exam stress: 34-36°C | Normal: 33-35°C
        const temp = isExamTime ? 34.5 + Math.random() * 1.5 : 33 + Math.random() * 2;
        samples.push({ timestamp: t, temperatureCelsius: temp });
      }
      return samples;
    },
    async readSpO2(start, end) {
      return [{ timestamp: (start + end) / 2, percentage: 96 + Math.random() * 3 }];
    },
    async readRespiratoryRate(start, end) {
      // Exam stress: 20-24 breaths/min | Normal: 14-18
      const rr = isExamTime ? 22 + Math.random() * 4 : 14 + Math.random() * 4;
      return [{ timestamp: (start + end) / 2, breathsPerMinute: rr }];
    },
  };
}
