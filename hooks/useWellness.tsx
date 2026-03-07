/**
 * Seren — Wellness Context & Hooks
 * ===================================
 * Central orchestrator that connects all AI services to the UI.
 *
 * Provides:
 *   - Real-time stress/anxiety from the XGBoost model
 *   - Sleep analysis from Health Connect (real or mock)
 *   - Personal baseline & anomaly detection
 *   - Recommendation engine output
 *   - Voice check-in analysis
 *   - Database persistence (expo-sqlite)
 *   - Anomaly push notifications
 *   - Data export
 *
 * Falls back to mock data when Health Connect isn't available (e.g., Expo Go).
 */

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  ReactNode,
} from 'react';
import { Platform } from 'react-native';
import * as FileSystem from 'expo-file-system';
import { Asset } from 'expo-asset';
import * as SQLite from 'expo-sqlite';

import {
  StressPrediction,
  AnxietyPrediction,
  SleepAnalysis,
  PersonalBaseline,
  Recommendation,
  CheckinAnalysis,
  BiometricFeatureVector,
} from '@/services/ai/types';

import {
  loadModel,
  predictStress,
  predictAnxiety,
} from '@/services/ai/stressModel';
import { extractFeatures } from '@/services/ai/featureEngineering';
import { computeBaseline, shouldRecomputeBaseline, detectAnomalies, AnomalyFlag } from '@/services/ai/baseline';
import { analyzeSleepSession, computeSleepTrend, SleepTrend } from '@/services/ai/sleepAnalysis';
import { analyzeCheckin, computeCheckinTrend, CheckinTrend } from '@/services/ai/voiceAnalysis';
import { generateRecommendations } from '@/services/ai/recommendations';
import {
  HealthConnectService,
  createHealthConnectService,
  createMockHealthConnectService,
} from '@/services/ai/healthConnect';
import {
  initDatabase,
  insertFeatureWindow,
  insertSleepSession,
  saveBaseline,
  getLatestBaseline,
  saveCheckin,
  getRecentCheckins,
  getSleepSessions,
  getRecentFeatureWindows,
  deleteAllData as dbDeleteAllData,
  cleanupOldData,
  SQLiteDatabase,
} from '@/services/ai/db';
import {
  requestNotificationPermission,
  notifyAnomalies,
  notifyHighStress,
  notifySustainedAnxiety,
  scheduleDailyCheckinReminder,
  cancelAllNotifications,
} from '@/services/ai/notifications';
import { exportHealthData } from '@/services/ai/dataExport';
import { configureLLMPreset, isLLMConfigured, getLLMConfig } from '@/services/ai/llmService';
import { configureWhisper, isWhisperConfigured } from '@/services/ai/whisperService';
import { initializeAIServices } from '@/services/ai/aiConfig';

// ============================================================
// Types
// ============================================================

interface WellnessContextValue {
  // Current state
  stress: StressPrediction;
  anxiety: AnxietyPrediction;
  lastSleep: SleepAnalysis | null;
  sleepTrend: SleepTrend;
  baseline: PersonalBaseline | null;
  anomalies: AnomalyFlag[];
  recommendations: Recommendation[];
  features: BiometricFeatureVector | null;

  // Real-time vitals (for display)
  heartRate: number;
  hrv: number;

  // Check-in
  lastCheckin: CheckinAnalysis | null;
  checkinHistory: CheckinAnalysis[];
  checkinTrend: CheckinTrend;
  performCheckin: (transcript: string) => Promise<CheckinAnalysis>;

  // Weekly data for charts
  weeklyStress: { date: string; value: number }[];
  weeklyHrv: { date: string; value: number }[];
  weeklySleep: { date: string; value: number }[];

  // Insights chart data
  monthlySleepGrid: number[][];
  hrvTrendData: { time: string; value: number }[];

  // Status
  modelLoaded: boolean;
  watchConnected: boolean;
  healthConnectAvailable: boolean;
  lastSyncTime: Date;
  isLive: boolean;
  dbReady: boolean;

  // Actions
  toggleLive: () => void;
  refreshData: () => Promise<void>;
  deleteAllData: () => Promise<void>;
  exportData: () => Promise<string | null>;
  requestHealthConnectPermissions: () => Promise<boolean>;

  // LLM configuration
  llmConfigured: boolean;
  llmProvider: string | null;
  configureLLM: (provider: 'openai' | 'groq' | 'ollama', apiKey: string) => void;

  // Whisper (speech-to-text) configuration
  whisperConfigured: boolean;
  configureWhisperSTT: (provider: 'openai' | 'groq', apiKey: string) => void;
}

// ============================================================
// Default values
// ============================================================

const DEFAULT_STRESS: StressPrediction = {
  timestamp: Date.now(),
  stressScore: 24,
  stressLevel: 'low',
  confidence: 0.6,
  topContributors: [],
};

const DEFAULT_ANXIETY: AnxietyPrediction = {
  timestamp: Date.now(),
  anxietyIndex: 18,
  level: 'minimal',
  sustained: false,
  baselineDeviation: 0,
};

const DEFAULT_SLEEP_TREND: SleepTrend = { avgQuality: 0, avgDuration: 0, trend: 'stable', daysAnalyzed: 0 };
const DEFAULT_CHECKIN_TREND: CheckinTrend = { avgSentiment: 0, dominantEmotion: 'calm', trend: 'stable', count: 0 };

// ============================================================
// Context
// ============================================================

const WellnessContext = createContext<WellnessContextValue | null>(null);

export function useWellness(): WellnessContextValue {
  const ctx = useContext(WellnessContext);
  if (!ctx) throw new Error('useWellness must be used within WellnessProvider');
  return ctx;
}

// ============================================================
// Provider
// ============================================================

export function WellnessProvider({ children }: { children: ReactNode }) {
  // Core state
  const [stress, setStress] = useState<StressPrediction>(DEFAULT_STRESS);
  const [anxiety, setAnxiety] = useState<AnxietyPrediction>(DEFAULT_ANXIETY);
  const [lastSleep, setLastSleep] = useState<SleepAnalysis | null>(null);
  const [sleepTrend, setSleepTrend] = useState<SleepTrend>(DEFAULT_SLEEP_TREND);
  const [baseline, setBaseline] = useState<PersonalBaseline | null>(null);
  const [anomalies, setAnomalies] = useState<AnomalyFlag[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [features, setFeatures] = useState<BiometricFeatureVector | null>(null);
  const [heartRate, setHeartRate] = useState(68);
  const [hrv, setHrv] = useState(58);

  // Check-in state
  const [lastCheckin, setLastCheckin] = useState<CheckinAnalysis | null>(null);
  const [checkinHistory, setCheckinHistory] = useState<CheckinAnalysis[]>([]);
  const [checkinTrend, setCheckinTrend] = useState<CheckinTrend>(DEFAULT_CHECKIN_TREND);

  // Weekly chart data
  const [weeklyStress, setWeeklyStress] = useState<{ date: string; value: number }[]>([]);
  const [weeklyHrv, setWeeklyHrv] = useState<{ date: string; value: number }[]>([]);
  const [weeklySleep, setWeeklySleep] = useState<{ date: string; value: number }[]>([]);

  // Chart-ready derived data for Insights screen
  const [monthlySleepGrid, setMonthlySleepGrid] = useState<number[][]>([]);
  const [hrvTrendData, setHrvTrendData] = useState<{ time: string; value: number }[]>([]);

  // Status
  const [modelLoaded, setModelLoaded] = useState(false);
  const [watchConnected, setWatchConnected] = useState(false);
  const [healthConnectAvailable, setHealthConnectAvailable] = useState(false);
  const [lastSyncTime, setLastSyncTime] = useState(new Date());
  const [isLive, setIsLive] = useState(true);
  const [dbReady, setDbReady] = useState(false);

  // Refs
  const healthService = useRef<HealthConnectService | null>(null);
  const featureHistory = useRef<BiometricFeatureVector[]>([]);
  const sleepHistory = useRef<SleepAnalysis[]>([]);
  const dbReadyRef = useRef(false);

  // ============================================================
  // Initialization
  // ============================================================

  useEffect(() => {
    initializeApp();
  }, []);

  async function initializeApp() {
    // Initialize AI services from config (API keys)
    initializeAIServices();

    await Promise.all([
      loadStressModel(),
      initDb(),
      initHealthConnect(),
      initNotifications(),
    ]);
    generateInitialChartData();
  }

  async function loadStressModel() {
    try {
      // Import JSON directly — Metro bundler handles JSON requires natively
      const modelJson = require('@/assets/ml/stress_model.json');
      loadModel(modelJson);
      setModelLoaded(true);
    } catch (e) {
      console.warn('[Seren] Failed to load stress model:', e);
    }
  }

  async function initDb() {
    try {
      const database = SQLite.openDatabaseSync('seren.db');
      const dbWrapper: SQLiteDatabase = {
        async execAsync(sql: string) {
          database.execSync(sql);
        },
        async runAsync(sql: string, ...params: any[]) {
          const result = database.runSync(sql, params);
          return { lastInsertRowId: result.lastInsertRowId, changes: result.changes };
        },
        async getFirstAsync<T>(sql: string, ...params: any[]): Promise<T | null> {
          const result = database.getFirstSync(sql, params);
          return (result as T) || null;
        },
        async getAllAsync<T>(sql: string, ...params: any[]): Promise<T[]> {
          const result = database.getAllSync(sql, params);
          return result as T[];
        },
      };

      await initDatabase(dbWrapper);
      dbReadyRef.current = true;
      setDbReady(true);
      await loadPersistedData();
    } catch (e) {
      console.warn('[Seren] Database init failed:', e);
    }
  }

  async function loadPersistedData() {
    try {
      const savedBaseline = await getLatestBaseline('default_user');
      if (savedBaseline) setBaseline(savedBaseline);

      const savedCheckins = await getRecentCheckins(30);
      if (savedCheckins.length > 0) {
        setCheckinHistory(savedCheckins);
        setLastCheckin(savedCheckins[0]);
        setCheckinTrend(computeCheckinTrend(savedCheckins));
      }

      const savedSleep = await getSleepSessions(30);
      if (savedSleep.length > 0) {
        sleepHistory.current = savedSleep;
        setLastSleep(savedSleep[0]);
        setSleepTrend(computeSleepTrend(savedSleep));
      }

      const savedFeatures = await getRecentFeatureWindows(14);
      if (savedFeatures.length > 0) {
        featureHistory.current = savedFeatures;
      }

      await cleanupOldData();
      updateInsightsChartData();
    } catch (e) {
      console.warn('[Seren] Failed to load persisted data:', e);
    }
  }

  async function initHealthConnect() {
    try {
      if (Platform.OS !== 'android') {
        healthService.current = createMockHealthConnectService();
        setWatchConnected(true);
        return;
      }

      const realService = createHealthConnectService();
      const available = await realService.isAvailable();

      if (available) {
        setHealthConnectAvailable(true);
        const hasPerms = await realService.hasPermissions();
        if (hasPerms) {
          healthService.current = realService;
          setWatchConnected(true);
        } else {
          healthService.current = createMockHealthConnectService();
          setWatchConnected(true);
        }
      } else {
        healthService.current = createMockHealthConnectService();
        setWatchConnected(true);
      }
    } catch (e) {
      console.warn('[Seren] Health Connect init failed, using mock:', e);
      healthService.current = createMockHealthConnectService();
      setWatchConnected(true);
    }
  }

  async function initNotifications() {
    try {
      await requestNotificationPermission();
      await scheduleDailyCheckinReminder(20, 0);
    } catch (e) {
      console.warn('[Seren] Notification init failed:', e);
    }
  }

  function generateInitialChartData() {
    // Try to build from persisted data; fall back to plausible defaults
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const now = new Date();
    const dayOfWeek = now.getDay(); // 0=Sun

    // Map each weekday slot to its feature windows
    const stressPerDay: Record<string, number[]> = {};
    const hrvPerDay: Record<string, number[]> = {};
    days.forEach(d => { stressPerDay[d] = []; hrvPerDay[d] = []; });

    featureHistory.current.forEach(fv => {
      const d = new Date(fv.timestamp);
      const diff = Math.floor((now.getTime() - d.getTime()) / (24 * 60 * 60 * 1000));
      if (diff >= 0 && diff < 7) {
        const idx = (d.getDay() + 6) % 7; // Mon=0
        const dayLabel = days[idx];
        // Derive stress from HR std (higher = more stressed)
        const stressEst = Math.min(100, Math.round(fv.hrStd * 4 + fv.hrMean * 0.3));
        stressPerDay[dayLabel].push(stressEst);
        hrvPerDay[dayLabel].push(fv.meanRR > 0 ? Math.round(fv.rmssd) : 55);
      }
    });

    const sleepPerDay: Record<string, number> = {};
    sleepHistory.current.forEach(sa => {
      const d = new Date(sa.startTime);
      const diff = Math.floor((now.getTime() - d.getTime()) / (24 * 60 * 60 * 1000));
      if (diff >= 0 && diff < 7) {
        const idx = (d.getDay() + 6) % 7;
        sleepPerDay[days[idx]] = sa.qualityScore;
      }
    });

    const hasData = featureHistory.current.length > 0 || sleepHistory.current.length > 0;

    setWeeklyStress(days.map(d => ({
      date: d,
      value: stressPerDay[d].length > 0
        ? Math.round(stressPerDay[d].reduce((a, b) => a + b, 0) / stressPerDay[d].length)
        : hasData ? 0 : Math.round(25 + Math.random() * 40),
    })));
    setWeeklyHrv(days.map(d => ({
      date: d,
      value: hrvPerDay[d].length > 0
        ? Math.round(hrvPerDay[d].reduce((a, b) => a + b, 0) / hrvPerDay[d].length)
        : hasData ? 0 : Math.round(40 + Math.random() * 30),
    })));
    setWeeklySleep(days.map(d => ({
      date: d,
      value: sleepPerDay[d] !== undefined
        ? Math.round(sleepPerDay[d])
        : hasData ? 0 : Math.round(60 + Math.random() * 30),
    })));
  }

  // ============================================================
  // Derived chart data for Insights screen
  // ============================================================

  function updateInsightsChartData() {
    // Monthly sleep grid: 4 rows x 7 cols of quality scores (0-100)
    const grid: number[][] = [[], [], [], []];
    const now = Date.now();
    if (sleepHistory.current.length > 0) {
      // Fill from most recent 28 days
      for (let day = 0; day < 28; day++) {
        const row = Math.floor(day / 7);
        const target = now - day * 24 * 60 * 60 * 1000;
        const match = sleepHistory.current.find(s => {
          const diff = Math.abs(s.startTime - target);
          return diff < 24 * 60 * 60 * 1000;
        });
        grid[row].push(match ? Math.round(match.qualityScore) : 0);
      }
    }
    setMonthlySleepGrid(grid.length > 0 && grid[0].length > 0 ? grid : []);

    // HRV intraday trend from today's feature windows
    const todayStart = new Date();
    todayStart.setHours(0, 0, 0, 0);
    const todayFeatures = featureHistory.current.filter(f => f.timestamp >= todayStart.getTime());
    if (todayFeatures.length > 0) {
      const trend = todayFeatures.map(f => {
        const d = new Date(f.timestamp);
        const h = d.getHours().toString().padStart(2, '0');
        const m = d.getMinutes().toString().padStart(2, '0');
        return { time: `${h}:${m}`, value: Math.round(f.rmssd) };
      });
      setHrvTrendData(trend);
    }
  }

  // ============================================================
  // Live Inference Cycle (every 5 seconds)
  // ============================================================

  useEffect(() => {
    if (!isLive || !healthService.current) return;

    const interval = setInterval(() => { runInferenceCycle(); }, 5000);
    runInferenceCycle();
    return () => clearInterval(interval);
  }, [isLive, modelLoaded, baseline]);

  async function runInferenceCycle() {
    try {
      const service = healthService.current;
      if (!service) return;

      const now = Date.now();
      const fiveMinAgo = now - 5 * 60 * 1000;

      const [hrSamples, hrvSamples, stepsSamples, tempSamples] = await Promise.all([
        service.readHeartRate(fiveMinAgo, now),
        service.readHRV(fiveMinAgo, now),
        service.readSteps(fiveMinAgo, now),
        service.readTemperature(fiveMinAgo, now),
      ]);

      if (hrSamples.length > 0) {
        setHeartRate(hrSamples[hrSamples.length - 1].bpm);
      }
      if (hrvSamples.length > 0) {
        setHrv(Math.round(hrvSamples[hrvSamples.length - 1].rmssd));
      }

      const totalSteps = stepsSamples.reduce((sum: number, s: { count: number }) => sum + s.count, 0);
      const featureVector = extractFeatures({
        hrSamples,
        hrvSamples,
        tempSamples,
        stepCount: totalSteps,
        windowSeconds: 300,
        timestamp: now,
      });
      setFeatures(featureVector);
      featureHistory.current.push(featureVector);
      if (featureHistory.current.length > 4032) {
        featureHistory.current = featureHistory.current.slice(-4032);
      }

      const stressPred = predictStress(featureVector, baseline);
      setStress(stressPred);

      const lastSleepQuality = lastSleep?.qualityScore ?? null;
      const anxietyPred = predictAnxiety(stressPred, featureVector, baseline, lastSleepQuality);
      setAnxiety(anxietyPred);

      if (baseline) {
        const flags = detectAnomalies(featureVector, baseline);
        setAnomalies(flags);
        if (flags.length > 0) notifyAnomalies(flags);
      }

      notifyHighStress(stressPred);
      notifySustainedAnxiety(anxietyPred);

      const recs = generateRecommendations(stressPred, anxietyPred, lastSleep, lastCheckin, baseline, []);
      setRecommendations(recs);

      if (dbReadyRef.current) {
        insertFeatureWindow(featureVector, stressPred.stressScore, anxietyPred.anxietyIndex).catch(() => {});
      }

      setLastSyncTime(new Date());
    } catch (e) {
      console.warn('[Seren] Inference cycle error:', e);
    }
  }

  // ============================================================
  // Baseline Recomputation
  // ============================================================

  useEffect(() => {
    if (shouldRecomputeBaseline(baseline) && featureHistory.current.length >= 50) {
      const newBaseline = computeBaseline('default_user', featureHistory.current, sleepHistory.current);
      if (newBaseline) {
        setBaseline(newBaseline);
        if (dbReadyRef.current) saveBaseline(newBaseline).catch(() => {});
      }
    }
  }, [stress.timestamp]);

  // ============================================================
  // Sleep Data (hourly)
  // ============================================================

  useEffect(() => {
    fetchSleepData();
    const interval = setInterval(fetchSleepData, 60 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  async function fetchSleepData() {
    try {
      const service = healthService.current;
      if (!service) return;

      const now = Date.now();
      const yesterday = now - 24 * 60 * 60 * 1000;
      const sessions = await service.readSleepSessions(yesterday, now);

      if (sessions.length > 0) {
        const latest = sessions[sessions.length - 1];
        const analysis = analyzeSleepSession(latest);
        setLastSleep(analysis);
        sleepHistory.current.push(analysis);
        setSleepTrend(computeSleepTrend(sleepHistory.current));

        if (dbReadyRef.current) {
          insertSleepSession(analysis).catch(() => {});
        }
      }
    } catch (e) {
      console.warn('[Seren] Sleep fetch error:', e);
    }
  }

  // ============================================================
  // Check-in Handler
  // ============================================================

  const performCheckin = useCallback(async (transcript: string): Promise<CheckinAnalysis> => {
    // Build full biometric snapshot for deep AI cross-referencing
    const weeklyStressValues = weeklyStress.map(w => w.value).filter(v => v > 0);
    let stressTrendWeek: string | undefined;
    if (weeklyStressValues.length >= 4) {
      const firstHalf = weeklyStressValues.slice(0, Math.floor(weeklyStressValues.length / 2));
      const secondHalf = weeklyStressValues.slice(Math.floor(weeklyStressValues.length / 2));
      const avg1 = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
      const avg2 = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
      if (avg2 - avg1 > 10) stressTrendWeek = 'rising';
      else if (avg1 - avg2 > 10) stressTrendWeek = 'falling';
      else stressTrendWeek = 'stable';
    }

    const result = await analyzeCheckin(transcript, {
      hr: heartRate,
      hrv: hrv,
      stress: stress.stressScore,
      anxietyIndex: anxiety.anxietyIndex,
      sleepQuality: lastSleep?.qualityScore ?? null,
      sleepDurationHrs: lastSleep ? lastSleep.totalSleepMin / 60 : null,
      recentMood: lastCheckin?.sentiment,
      stressTrendWeek,
    });
    setLastCheckin(result);
    setCheckinHistory((prev: CheckinAnalysis[]) => {
      const updated = [result, ...prev].slice(0, 50);
      setCheckinTrend(computeCheckinTrend(updated));
      return updated;
    });

    if (dbReadyRef.current) saveCheckin(result).catch(() => {});
    return result;
  }, [heartRate, hrv, stress.stressScore]);

  // ============================================================
  // Actions
  // ============================================================

  const toggleLive = useCallback(() => setIsLive((v: boolean) => !v), []);

  const refreshData = useCallback(async () => {
    await runInferenceCycle();
    await fetchSleepData();
  }, []);

  const deleteAllUserData = useCallback(async () => {
    featureHistory.current = [];
    sleepHistory.current = [];
    setBaseline(null);
    setCheckinHistory([]);
    setLastCheckin(null);
    setLastSleep(null);
    setStress(DEFAULT_STRESS);
    setAnxiety(DEFAULT_ANXIETY);
    setRecommendations([]);
    setAnomalies([]);
    setSleepTrend(DEFAULT_SLEEP_TREND);
    setCheckinTrend(DEFAULT_CHECKIN_TREND);

    if (dbReadyRef.current) {
      try { await dbDeleteAllData(); } catch { /* ignore */ }
    }
    await cancelAllNotifications();
  }, []);

  const doExportData = useCallback(async (): Promise<string | null> => {
    if (!dbReadyRef.current) return null;
    return exportHealthData(365);
  }, []);

  const requestHCPermissions = useCallback(async (): Promise<boolean> => {
    try {
      const realService = createHealthConnectService();
      const granted = await realService.requestPermissions();
      if (granted) {
        healthService.current = realService;
        setWatchConnected(true);
        setHealthConnectAvailable(true);
      }
      return granted;
    } catch {
      return false;
    }
  }, []);

  // ============================================================
  // Context Value
  // ============================================================

  const value: WellnessContextValue = {
    stress, anxiety, lastSleep, sleepTrend, baseline, anomalies,
    recommendations, features, heartRate, hrv,
    lastCheckin, checkinHistory, checkinTrend, performCheckin,
    weeklyStress, weeklyHrv, weeklySleep,
    monthlySleepGrid, hrvTrendData,
    modelLoaded, watchConnected, healthConnectAvailable,
    lastSyncTime, isLive, dbReady,
    toggleLive, refreshData,
    deleteAllData: deleteAllUserData,
    exportData: doExportData,
    requestHealthConnectPermissions: requestHCPermissions,
    llmConfigured: isLLMConfigured(),
    llmProvider: getLLMConfig()?.provider ?? null,
    configureLLM: (provider: 'openai' | 'groq' | 'ollama', apiKey: string) => {
      configureLLMPreset(provider, apiKey);
    },
    whisperConfigured: isWhisperConfigured(),
    configureWhisperSTT: (provider: 'openai' | 'groq', apiKey: string) => {
      configureWhisper(provider, apiKey);
    },
  };

  return (
    <WellnessContext.Provider value={value}>
      {children}
    </WellnessContext.Provider>
  );
}
