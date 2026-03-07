/**
 * Seren AI — Local Database Layer (expo-sqlite)
 * ================================================
 * All biometric data stays on-device. This module handles:
 *   - Schema creation and migrations
 *   - CRUD for biometric samples, feature windows, sleep sessions
 *   - Baseline and recommendation storage
 *   - Data retention and cleanup
 *
 * Uses expo-sqlite (synchronous API available in SDK 54+).
 */

import {
  DBBiometricSample,
  DBFeatureWindow,
  DBSleepSession,
  BiometricFeatureVector,
  PersonalBaseline,
  SleepAnalysis,
  CheckinAnalysis,
  Recommendation,
} from './types';

// ============================================================
// Database Interface (abstraction for testing/mocking)
// ============================================================

/**
 * Minimal SQLite interface matching expo-sqlite's API.
 * This lets us inject a mock DB for development.
 */
export interface SQLiteDatabase {
  execAsync(sql: string): Promise<void>;
  runAsync(sql: string, ...params: any[]): Promise<{ lastInsertRowId: number; changes: number }>;
  getFirstAsync<T>(sql: string, ...params: any[]): Promise<T | null>;
  getAllAsync<T>(sql: string, ...params: any[]): Promise<T[]>;
}

// ============================================================
// Schema
// ============================================================

const SCHEMA_VERSION = 1;

const CREATE_TABLES = `
  CREATE TABLE IF NOT EXISTS biometric_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    hr_bpm REAL,
    rr_interval_ms REAL,
    skin_temp_c REAL,
    spo2_pct REAL,
    resp_rate REAL,
    accel_magnitude REAL,
    step_count INTEGER,
    source TEXT DEFAULT 'health_connect'
  );

  CREATE TABLE IF NOT EXISTS feature_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    window_seconds INTEGER NOT NULL DEFAULT 300,
    features_json TEXT NOT NULL,
    stress_score REAL,
    anxiety_index REAL
  );

  CREATE TABLE IF NOT EXISTS sleep_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    session_start INTEGER NOT NULL,
    session_end INTEGER NOT NULL,
    total_sleep_min REAL,
    deep_sleep_min REAL,
    rem_sleep_min REAL,
    light_sleep_min REAL,
    awake_min REAL,
    quality_score REAL,
    recovery_score REAL,
    stages_json TEXT
  );

  CREATE TABLE IF NOT EXISTS baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    computed_at INTEGER NOT NULL,
    data_json TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS checkins (
    id TEXT PRIMARY KEY,
    timestamp INTEGER NOT NULL,
    transcript TEXT NOT NULL,
    sentiment TEXT,
    sentiment_score REAL,
    emotions_json TEXT,
    insights_json TEXT,
    hr_at_checkin REAL,
    hrv_at_checkin REAL,
    stress_at_checkin REAL
  );

  CREATE TABLE IF NOT EXISTS recommendations_log (
    id TEXT PRIMARY KEY,
    timestamp INTEGER NOT NULL,
    category TEXT,
    title TEXT,
    trigger_type TEXT,
    trigger_reason TEXT,
    status TEXT DEFAULT 'pending',
    pre_stress REAL,
    post_stress REAL,
    pre_hrv REAL,
    post_hrv REAL,
    effectiveness REAL,
    data_json TEXT
  );

  CREATE INDEX IF NOT EXISTS idx_biometric_ts ON biometric_samples(timestamp);
  CREATE INDEX IF NOT EXISTS idx_features_ts ON feature_windows(timestamp);
  CREATE INDEX IF NOT EXISTS idx_sleep_date ON sleep_sessions(date);
  CREATE INDEX IF NOT EXISTS idx_checkins_ts ON checkins(timestamp);
`;

// ============================================================
// Database Manager
// ============================================================

let db: SQLiteDatabase | null = null;

/**
 * Initialize the database with schema.
 * Call this once at app startup.
 */
export async function initDatabase(database: SQLiteDatabase): Promise<void> {
  db = database;
  await db.execAsync(CREATE_TABLES);
}

/**
 * Get the active database instance.
 */
export function getDb(): SQLiteDatabase {
  if (!db) throw new Error('Database not initialized. Call initDatabase() first.');
  return db;
}

// ============================================================
// Biometric Samples
// ============================================================

export async function insertBiometricSample(sample: Omit<DBBiometricSample, 'id'>): Promise<number> {
  const result = await getDb().runAsync(
    `INSERT INTO biometric_samples (timestamp, hr_bpm, rr_interval_ms, skin_temp_c, spo2_pct, resp_rate, accel_magnitude, step_count, source)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    sample.timestamp, sample.hr_bpm, sample.rr_interval_ms,
    sample.skin_temp_c, sample.spo2_pct, sample.resp_rate,
    sample.accel_magnitude, sample.step_count, sample.source,
  );
  return result.lastInsertRowId;
}

export async function insertBiometricSamplesBatch(
  samples: Omit<DBBiometricSample, 'id'>[],
): Promise<void> {
  for (const sample of samples) {
    await insertBiometricSample(sample);
  }
}

export async function getBiometricSamples(
  startTime: number,
  endTime: number,
): Promise<DBBiometricSample[]> {
  return getDb().getAllAsync<DBBiometricSample>(
    'SELECT * FROM biometric_samples WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp',
    startTime, endTime,
  );
}

export async function getLatestBiometricSample(): Promise<DBBiometricSample | null> {
  return getDb().getFirstAsync<DBBiometricSample>(
    'SELECT * FROM biometric_samples ORDER BY timestamp DESC LIMIT 1',
  );
}

// ============================================================
// Feature Windows
// ============================================================

export async function insertFeatureWindow(
  features: BiometricFeatureVector,
  stressScore: number | null,
  anxietyIndex: number | null,
): Promise<number> {
  const result = await getDb().runAsync(
    `INSERT INTO feature_windows (timestamp, window_seconds, features_json, stress_score, anxiety_index)
     VALUES (?, ?, ?, ?, ?)`,
    features.timestamp, features.windowSeconds,
    JSON.stringify(features), stressScore, anxietyIndex,
  );
  return result.lastInsertRowId;
}

export async function getFeatureWindows(
  startTime: number,
  endTime: number,
): Promise<BiometricFeatureVector[]> {
  const rows = await getDb().getAllAsync<DBFeatureWindow>(
    'SELECT * FROM feature_windows WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp',
    startTime, endTime,
  );
  return rows.map(r => JSON.parse(r.features_json));
}

export async function getRecentFeatureWindows(days: number): Promise<BiometricFeatureVector[]> {
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
  return getFeatureWindows(cutoff, Date.now());
}

// ============================================================
// Sleep Sessions
// ============================================================

export async function insertSleepSession(analysis: SleepAnalysis): Promise<number> {
  const result = await getDb().runAsync(
    `INSERT INTO sleep_sessions (date, session_start, session_end, total_sleep_min, deep_sleep_min, rem_sleep_min, light_sleep_min, awake_min, quality_score, recovery_score, stages_json)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    analysis.date, analysis.sessionStart, analysis.sessionEnd,
    analysis.totalSleepMin, analysis.deepSleepMin, analysis.remSleepMin,
    analysis.lightSleepMin, analysis.awakeMin,
    analysis.qualityScore, analysis.recoveryScore,
    '[]', // stages_json — we store the analysis, raw stages can be re-fetched
  );
  return result.lastInsertRowId;
}

export async function getSleepSessions(days: number): Promise<SleepAnalysis[]> {
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
  const rows = await getDb().getAllAsync<DBSleepSession>(
    'SELECT * FROM sleep_sessions WHERE session_start > ? ORDER BY session_start DESC',
    cutoff,
  );
  return rows.map(rowToSleepAnalysis);
}

export async function getLatestSleepSession(): Promise<SleepAnalysis | null> {
  const row = await getDb().getFirstAsync<DBSleepSession>(
    'SELECT * FROM sleep_sessions ORDER BY session_start DESC LIMIT 1',
  );
  return row ? rowToSleepAnalysis(row) : null;
}

function rowToSleepAnalysis(row: DBSleepSession): SleepAnalysis {
  return {
    date: row.date,
    sessionStart: row.session_start,
    sessionEnd: row.session_end,
    totalInBedMin: Math.round((row.session_end - row.session_start) / 60000),
    totalSleepMin: row.total_sleep_min,
    onsetLatencyMin: 0,
    wasoMin: 0,
    lightSleepMin: row.light_sleep_min,
    deepSleepMin: row.deep_sleep_min,
    remSleepMin: row.rem_sleep_min,
    awakeMin: row.awake_min,
    deepSleepPct: row.total_sleep_min > 0 ? row.deep_sleep_min / row.total_sleep_min : 0,
    remSleepPct: row.total_sleep_min > 0 ? row.rem_sleep_min / row.total_sleep_min : 0,
    lightSleepPct: row.total_sleep_min > 0 ? row.light_sleep_min / row.total_sleep_min : 0,
    sleepEfficiency: 0,
    fragmentationIndex: 0,
    awakeningCount: 0,
    avgHrSleep: 0,
    minHrSleep: 0,
    avgHrvSleep: 0,
    maxHrvSleep: 0,
    qualityScore: row.quality_score,
    recoveryScore: row.recovery_score,
    consistencyScore: 50,
  };
}

// ============================================================
// Baselines
// ============================================================

export async function saveBaseline(baseline: PersonalBaseline): Promise<void> {
  await getDb().runAsync(
    `INSERT INTO baselines (user_id, computed_at, data_json) VALUES (?, ?, ?)`,
    baseline.userId, baseline.computedAt, JSON.stringify(baseline),
  );
}

export async function getLatestBaseline(userId: string): Promise<PersonalBaseline | null> {
  const row = await getDb().getFirstAsync<{ data_json: string }>(
    'SELECT data_json FROM baselines WHERE user_id = ? ORDER BY computed_at DESC LIMIT 1',
    userId,
  );
  return row ? JSON.parse(row.data_json) : null;
}

// ============================================================
// Check-ins
// ============================================================

export async function saveCheckin(checkin: CheckinAnalysis): Promise<void> {
  await getDb().runAsync(
    `INSERT OR REPLACE INTO checkins (id, timestamp, transcript, sentiment, sentiment_score, emotions_json, insights_json, hr_at_checkin, hrv_at_checkin, stress_at_checkin)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    checkin.id, checkin.timestamp, checkin.transcript,
    checkin.sentiment, checkin.sentimentScore,
    JSON.stringify(checkin.emotionScores), JSON.stringify(checkin.keyInsights),
    checkin.hrAtCheckin, checkin.hrvAtCheckin, checkin.stressAtCheckin,
  );
}

export async function getRecentCheckins(days: number): Promise<CheckinAnalysis[]> {
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
  const rows = await getDb().getAllAsync<any>(
    'SELECT * FROM checkins WHERE timestamp > ? ORDER BY timestamp DESC',
    cutoff,
  );
  return rows.map((r: any) => ({
    id: r.id,
    timestamp: r.timestamp,
    transcript: r.transcript,
    sentiment: r.sentiment,
    sentimentScore: r.sentiment_score,
    emotionScores: JSON.parse(r.emotions_json || '{}'),
    keyInsights: JSON.parse(r.insights_json || '[]'),
    hrAtCheckin: r.hr_at_checkin,
    hrvAtCheckin: r.hrv_at_checkin,
    stressAtCheckin: r.stress_at_checkin,
  }));
}

// ============================================================
// Recommendations Log
// ============================================================

export async function saveRecommendation(rec: Recommendation): Promise<void> {
  await getDb().runAsync(
    `INSERT OR REPLACE INTO recommendations_log (id, timestamp, category, title, trigger_type, trigger_reason, status, pre_stress, post_stress, pre_hrv, post_hrv, effectiveness, data_json)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    rec.id, Date.now(), rec.category, rec.title,
    rec.trigger, rec.triggerReason, rec.status,
    rec.preStress ?? null, rec.postStress ?? null,
    rec.preHrv ?? null, rec.postHrv ?? null,
    rec.effectivenessScore ?? null,
    JSON.stringify(rec),
  );
}

export async function getRecentRecommendationIds(hours: number = 4): Promise<string[]> {
  const cutoff = Date.now() - hours * 60 * 60 * 1000;
  const rows = await getDb().getAllAsync<{ id: string }>(
    'SELECT id FROM recommendations_log WHERE timestamp > ?',
    cutoff,
  );
  return rows.map(r => r.id);
}

// ============================================================
// Data Retention & Cleanup
// ============================================================

/**
 * Delete data older than `days` to manage storage.
 * Biometric samples: keep 30 days
 * Feature windows: keep 90 days
 * Sleep sessions: keep 365 days
 */
export async function cleanupOldData(
  biometricDays: number = 30,
  featureDays: number = 90,
  sleepDays: number = 365,
): Promise<{ biometric: number; features: number; sleep: number }> {
  const now = Date.now();

  const r1 = await getDb().runAsync(
    'DELETE FROM biometric_samples WHERE timestamp < ?',
    now - biometricDays * 86400000,
  );
  const r2 = await getDb().runAsync(
    'DELETE FROM feature_windows WHERE timestamp < ?',
    now - featureDays * 86400000,
  );
  const r3 = await getDb().runAsync(
    'DELETE FROM sleep_sessions WHERE session_start < ?',
    now - sleepDays * 86400000,
  );

  return {
    biometric: r1.changes,
    features: r2.changes,
    sleep: r3.changes,
  };
}

/**
 * Delete ALL user data (for privacy / account deletion).
 */
export async function deleteAllData(): Promise<void> {
  const tables = [
    'biometric_samples', 'feature_windows', 'sleep_sessions',
    'baselines', 'checkins', 'recommendations_log',
  ];
  for (const table of tables) {
    await getDb().runAsync(`DELETE FROM ${table}`);
  }
}
