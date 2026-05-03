/**
 * Seren AI — Data Export Service
 * =================================
 * Exports all local health data as JSON for privacy/portability.
 * Uses expo-file-system to write and expo-sharing to share.
 */

import * as FileSystem from 'expo-file-system';
import { Paths, File as EXFile } from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import {
  getBiometricSamples,
  getFeatureWindows,
  getSleepSessions,
  getRecentCheckins,
  getLatestBaseline,
} from './db';

// ============================================================
// Export Types
// ============================================================

export interface ExportData {
  exportDate: string;
  appVersion: string;
  format: 'seren-health-export-v1';
  data: {
    biometricSamples: any[];
    featureWindows: any[];
    sleepSessions: any[];
    checkins: any[];
    baseline: any | null;
  };
  metadata: {
    totalSamples: number;
    dateRange: { start: string; end: string } | null;
  };
}

// ============================================================
// Export Function
// ============================================================

/**
 * Export all user health data as a JSON file and open the share dialog.
 *
 * @param days Number of days of data to export (default: all available = 365)
 * @returns Path to the exported file, or null if failed
 */
export async function exportHealthData(days: number = 365): Promise<string | null> {
  try {
    const now = Date.now();
    const startTime = now - days * 24 * 60 * 60 * 1000;

    // Gather all data in parallel
    const [biometrics, features, sleepSessions, checkins, baseline] = await Promise.all([
      getBiometricSamples(startTime, now).catch(() => []),
      getFeatureWindows(startTime, now).catch(() => []),
      getSleepSessions(days).catch(() => []),
      getRecentCheckins(days).catch(() => []),
      getLatestBaseline('default_user').catch(() => null),
    ]);

    // Build export object
    const exportData: ExportData = {
      exportDate: new Date().toISOString(),
      appVersion: '1.0.0',
      format: 'seren-health-export-v1',
      data: {
        biometricSamples: biometrics,
        featureWindows: features,
        sleepSessions: sleepSessions,
        checkins: checkins.map(c => ({
          ...c,
          // Strip raw transcript for privacy — keep only analysis
          transcript: '[redacted for privacy]',
        })),
        baseline: baseline,
      },
      metadata: {
        totalSamples: biometrics.length + features.length + sleepSessions.length + checkins.length,
        dateRange: biometrics.length > 0
          ? {
              start: new Date(startTime).toISOString(),
              end: new Date(now).toISOString(),
            }
          : null,
      },
    };

    // Write to file
    const filename = `seren-health-export-${new Date().toISOString().split('T')[0]}.json`;
    const file = new EXFile(Paths.document, filename);
    file.write(JSON.stringify(exportData, null, 2));
    const filePath = file.uri;

    // Share
    const canShare = await Sharing.isAvailableAsync();
    if (canShare) {
      await Sharing.shareAsync(filePath, {
        mimeType: 'application/json',
        dialogTitle: 'Export Seren Health Data',
        UTI: 'public.json',
      });
    }

    return filePath;
  } catch (e) {
    console.warn('Data export failed:', e);
    return null;
  }
}

/**
 * Export data with full transcripts (no redaction).
 * Use only when user explicitly requests unredacted export.
 */
export async function exportHealthDataFull(days: number = 365): Promise<string | null> {
  try {
    const now = Date.now();
    const startTime = now - days * 24 * 60 * 60 * 1000;

    const [biometrics, features, sleepSessions, checkins, baseline] = await Promise.all([
      getBiometricSamples(startTime, now).catch(() => []),
      getFeatureWindows(startTime, now).catch(() => []),
      getSleepSessions(days).catch(() => []),
      getRecentCheckins(days).catch(() => []),
      getLatestBaseline('default_user').catch(() => null),
    ]);

    const exportData: ExportData = {
      exportDate: new Date().toISOString(),
      appVersion: '1.0.0',
      format: 'seren-health-export-v1',
      data: {
        biometricSamples: biometrics,
        featureWindows: features,
        sleepSessions: sleepSessions,
        checkins: checkins,
        baseline: baseline,
      },
      metadata: {
        totalSamples: biometrics.length + features.length + sleepSessions.length + checkins.length,
        dateRange: biometrics.length > 0
          ? {
              start: new Date(startTime).toISOString(),
              end: new Date(now).toISOString(),
            }
          : null,
      },
    };

    const filename = `seren-health-export-full-${new Date().toISOString().split('T')[0]}.json`;
    const file = new EXFile(Paths.document, filename);
    file.write(JSON.stringify(exportData, null, 2));
    const filePath = file.uri;

    const canShare = await Sharing.isAvailableAsync();
    if (canShare) {
      await Sharing.shareAsync(filePath, {
        mimeType: 'application/json',
        dialogTitle: 'Export Seren Health Data (Full)',
        UTI: 'public.json',
      });
    }

    return filePath;
  } catch (e) {
    console.warn('Full data export failed:', e);
    return null;
  }
}
