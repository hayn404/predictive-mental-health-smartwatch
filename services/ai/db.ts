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
  DBLocationVisit,
  DBLocationDiversity,
  DBSunlightSample,
  DBSunlightDaily,
  BiometricFeatureVector,
  PersonalBaseline,
  SleepAnalysis,
  CheckinAnalysis,
  Recommendation,
  LocationVisit,
  LocationDiversitySummary,
  SunlightReading,
  SunlightExposureSummary,
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

  CREATE TABLE IF NOT EXISTS location_visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    departure_time INTEGER NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    label TEXT DEFAULT 'unknown',
    cluster_index INTEGER DEFAULT -1
  );

  CREATE TABLE IF NOT EXISTS location_diversity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    unique_places INTEGER,
    total_transitions INTEGER,
    diversity_score REAL,
    home_time_pct REAL,
    work_time_pct REAL,
    novel_places INTEGER,
    is_monotonous INTEGER DEFAULT 0
  );

  CREATE TABLE IF NOT EXISTS sunlight_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    lux_value REAL NOT NULL,
    is_outdoors INTEGER DEFAULT 0
  );

  CREATE TABLE IF NOT EXISTS sunlight_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    total_outdoor_min REAL,
    optimal_window_min REAL,
    peak_lux REAL,
    avg_outdoor_lux REAL,
    goal_minutes REAL DEFAULT 30
  );

  CREATE INDEX IF NOT EXISTS idx_biometric_ts ON biometric_samples(timestamp);
  CREATE INDEX IF NOT EXISTS idx_features_ts ON feature_windows(timestamp);
  CREATE INDEX IF NOT EXISTS idx_sleep_date ON sleep_sessions(date);
  CREATE INDEX IF NOT EXISTS idx_checkins_ts ON checkins(timestamp);
  CREATE INDEX IF NOT EXISTS idx_location_visits_ts ON location_visits(timestamp);
  CREATE INDEX IF NOT EXISTS idx_sunlight_ts ON sunlight_samples(timestamp);
  CREATE INDEX IF NOT EXISTS idx_sunlight_daily_date ON sunlight_daily(date);
  CREATE INDEX IF NOT EXISTS idx_location_div_date ON location_diversity(date);
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
    themes: [],
    emotionalIntensity: 0,
    empathyResponse: '',
    suggestedFollowUp: null,
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
// Location Visits
// ============================================================

export async function insertLocationVisit(visit: LocationVisit): Promise<number> {
  const result = await getDb().runAsync(
    `INSERT INTO location_visits (visit_id, timestamp, departure_time, latitude, longitude, label, cluster_index)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
    visit.id, visit.timestamp, visit.departureTime,
    visit.latitude, visit.longitude, visit.label, visit.clusterIndex,
  );
  return result.lastInsertRowId;
}

export async function getLocationVisits(startTime: number, endTime: number): Promise<LocationVisit[]> {
  const rows = await getDb().getAllAsync<DBLocationVisit>(
    'SELECT * FROM location_visits WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp',
    startTime, endTime,
  );
  return rows.map(r => ({
    id: r.visit_id,
    timestamp: r.timestamp,
    departureTime: r.departure_time,
    latitude: r.latitude,
    longitude: r.longitude,
    label: r.label,
    clusterIndex: r.cluster_index,
  }));
}

export async function upsertLocationDiversity(summary: LocationDiversitySummary): Promise<void> {
  await getDb().runAsync(
    `INSERT OR REPLACE INTO location_diversity (date, unique_places, total_transitions, diversity_score, home_time_pct, work_time_pct, novel_places, is_monotonous)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    summary.date, summary.uniquePlacesVisited, summary.totalTransitions,
    summary.diversityScore, summary.homeTimePercent, summary.workTimePercent,
    summary.novelPlaces, summary.isMonotonous ? 1 : 0,
  );
}

export async function getTodayLocationDiversity(): Promise<LocationDiversitySummary | null> {
  const today = new Date().toISOString().slice(0, 10);
  const row = await getDb().getFirstAsync<DBLocationDiversity>(
    'SELECT * FROM location_diversity WHERE date = ?', today,
  );
  return row ? rowToLocationDiversity(row) : null;
}

export async function getLocationDiversityHistory(days: number): Promise<LocationDiversitySummary[]> {
  const cutoff = new Date(Date.now() - days * 86400000).toISOString().slice(0, 10);
  const rows = await getDb().getAllAsync<DBLocationDiversity>(
    'SELECT * FROM location_diversity WHERE date >= ? ORDER BY date', cutoff,
  );
  return rows.map(rowToLocationDiversity);
}

function rowToLocationDiversity(row: DBLocationDiversity): LocationDiversitySummary {
  return {
    date: row.date,
    uniquePlacesVisited: row.unique_places,
    totalTransitions: row.total_transitions,
    diversityScore: row.diversity_score,
    homeTimePercent: row.home_time_pct,
    workTimePercent: row.work_time_pct,
    novelPlaces: row.novel_places,
    isMonotonous: row.is_monotonous === 1,
  };
}

// ============================================================
// Sunlight Samples & Daily Summaries
// ============================================================

export async function insertSunlightSample(reading: SunlightReading): Promise<number> {
  const result = await getDb().runAsync(
    `INSERT INTO sunlight_samples (timestamp, lux_value, is_outdoors) VALUES (?, ?, ?)`,
    reading.timestamp, reading.luxValue, reading.isOutdoors ? 1 : 0,
  );
  return result.lastInsertRowId;
}

export async function getSunlightSamples(startTime: number, endTime: number): Promise<SunlightReading[]> {
  const rows = await getDb().getAllAsync<DBSunlightSample>(
    'SELECT * FROM sunlight_samples WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp',
    startTime, endTime,
  );
  return rows.map(r => ({
    timestamp: r.timestamp,
    luxValue: r.lux_value,
    isOutdoors: r.is_outdoors === 1,
  }));
}

export async function upsertSunlightDaily(summary: SunlightExposureSummary): Promise<void> {
  await getDb().runAsync(
    `INSERT OR REPLACE INTO sunlight_daily (date, total_outdoor_min, optimal_window_min, peak_lux, avg_outdoor_lux, goal_minutes)
     VALUES (?, ?, ?, ?, ?, ?)`,
    summary.date, summary.totalOutdoorMinutes, summary.optimalWindowMinutes,
    summary.peakLux, summary.avgOutdoorLux, summary.goalMinutes,
  );
}

export async function getTodaySunlightDaily(): Promise<SunlightExposureSummary | null> {
  const today = new Date().toISOString().slice(0, 10);
  const row = await getDb().getFirstAsync<DBSunlightDaily>(
    'SELECT * FROM sunlight_daily WHERE date = ?', today,
  );
  return row ? rowToSunlightSummary(row) : null;
}

export async function getSunlightDailyHistory(days: number): Promise<SunlightExposureSummary[]> {
  const cutoff = new Date(Date.now() - days * 86400000).toISOString().slice(0, 10);
  const rows = await getDb().getAllAsync<DBSunlightDaily>(
    'SELECT * FROM sunlight_daily WHERE date >= ? ORDER BY date', cutoff,
  );
  return rows.map(rowToSunlightSummary);
}

function rowToSunlightSummary(row: DBSunlightDaily): SunlightExposureSummary {
  const now = new Date();
  const hour = now.getHours();
  return {
    date: row.date,
    totalOutdoorMinutes: row.total_outdoor_min,
    optimalWindowMinutes: row.optimal_window_min,
    peakLux: row.peak_lux,
    avgOutdoorLux: row.avg_outdoor_lux,
    goalMinutes: row.goal_minutes,
    goalProgress: row.goal_minutes > 0 ? Math.min(1, row.total_outdoor_min / row.goal_minutes) : 0,
    isVitaminDWindow: hour >= 10 && hour < 15,
  };
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

  // Clean up location and sunlight data (keep 30 days for samples, 365 for summaries)
  await getDb().runAsync('DELETE FROM location_visits WHERE timestamp < ?', now - biometricDays * 86400000);
  await getDb().runAsync('DELETE FROM sunlight_samples WHERE timestamp < ?', now - biometricDays * 86400000);

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
    'location_visits', 'location_diversity', 'sunlight_samples', 'sunlight_daily',
  ];
  for (const table of tables) {
    await getDb().runAsync(`DELETE FROM ${table}`);
  }
}
