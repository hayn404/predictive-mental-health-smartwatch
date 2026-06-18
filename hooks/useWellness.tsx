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
import * as SQLite from 'expo-sqlite';

import {
  StressPrediction,
  AnxietyPrediction,
  FocusPrediction,
  SleepAnalysis,
  PersonalBaseline,
  Recommendation,
  CheckinAnalysis,
  BiometricFeatureVector,
  LocationDiversitySummary,
  SunlightExposureSummary,
  DepressionPrediction,
} from '@/services/ai/types';

import {
  loadModel,
  predictStress,
} from '@/services/ai/stressModel';
import {
  loadAnxietyModel,
  predictAnxiety,
} from '@/services/ai/anxietyModel';
import {
  loadFocusModel,
  predictFocus,
  getFocusLevel,
} from '@/services/ai/focusModel';
import { loadBioAgeModel, predictBioAge, type BioAgePrediction } from '@/services/ai/bioAgeModel';
import {
  loadDepressionModel,
  predictDepression,
  buildDailyActivityFeatures,
} from '@/services/ai/depressionModel';
import { sendDepressionToWatch } from '@/services/wearDepressionBridge';
import { getFocusTips } from '@/services/ai/focusRecommendations';
import { extractFeatures } from '@/services/ai/featureEngineering';
import { computeBaseline, shouldRecomputeBaseline, detectAnomalies, AnomalyFlag } from '@/services/ai/baseline';
import { analyzeSleepSession, computeSleepTrend, SleepTrend } from '@/services/ai/sleepAnalysis';
import { loadV32SleepModel, isV32SleepModelLoaded } from '@/services/ai/sleepStageModel';
import { processPendingSessions } from '@/services/ai/sleepReceiver';
import { processPendingEnv } from '@/services/ai/envReceiver';
import { recordStressPrediction } from '@/services/observability/modelMonitor';
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
  getSetting,
  setSetting,
  upsertLocationDiversity,
  getTodayLocationDiversity,
  getLocationDiversityHistory,
  upsertSunlightDaily,
  getTodaySunlightDaily,
  insertSunlightSample,
  getSunlightDailyHistory,
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
import {
  getMockLocationDiversity,
  getMockWeeklyLocationDiversity,
  startLocationTracking as startLocationWatch,
  calculateDiversityScore,
} from '@/services/ai/locationTracking';
import {
  getMockSunlightExposure,
  getMockWeeklySunlight,
  startSunlightMonitoring,
  isLightSensorAvailable,
} from '@/services/ai/sunlightTracking';

// ============================================================
// Types
// ============================================================

interface WellnessContextValue {
  // Current state
  stress: StressPrediction;
  anxiety: AnxietyPrediction;
  focus: FocusPrediction;
  bioAge: BioAgePrediction | null;
  depression: DepressionPrediction | null;
  chronologicalAge: number | null;
  setChronologicalAge: (age: number | null) => void;
  gender: number | null;
  setGender: (g: number | null) => void;
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

  // Location & Sunlight
  locationDiversity: LocationDiversitySummary | null;
  sunlightExposure: SunlightExposureSummary | null;

  // Weekly data for charts
  weeklyStress: { date: string; value: number }[];
  weeklyHrv: { date: string; value: number }[];
  weeklySleep: { date: string; value: number }[];
  weeklyLocationDiversity: { date: string; value: number }[];
  weeklySunlight: { date: string; value: number }[];

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

const DEFAULT_FOCUS: FocusPrediction = {
  timestamp: Date.now(),
  focusScore: 65,
  focusLevel: 'steady',
  elevatedFeatures: [],
  groqTips: [],
  groqLoading: false,
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
  const [focus, setFocus] = useState<FocusPrediction>(DEFAULT_FOCUS);
  const [bioAge, setBioAge] = useState<BioAgePrediction | null>(null);
  const [depression, setDepression] = useState<DepressionPrediction | null>(null);
  const [chronologicalAge, setChronoState] = useState<number | null>(25);
  const chronoRef = useRef<number | null>(25);
  const [gender, setGenderState] = useState<number | null>(null);
  const genderRef = useRef<number | null>(null);
  const lastDepressionDateRef = useRef<string | null>(null);
  const [lastSleep, setLastSleep] = useState<SleepAnalysis | null>(null);
  const [sleepTrend, setSleepTrend] = useState<SleepTrend>(DEFAULT_SLEEP_TREND);
  const [baseline, setBaseline] = useState<PersonalBaseline | null>(null);
  const [anomalies, setAnomalies] = useState<AnomalyFlag[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [features, setFeatures] = useState<BiometricFeatureVector | null>(null);
  const [heartRate, setHeartRate] = useState(68);
  const [hrv, setHrv] = useState(58);

  // Location & Sunlight state
  const [locationDiversity, setLocationDiversity] = useState<LocationDiversitySummary | null>(null);
  const [sunlightExposure, setSunlightExposure] = useState<SunlightExposureSummary | null>(null);
  const [weeklyLocationDiversity, setWeeklyLocationDiversity] = useState<{ date: string; value: number }[]>([]);
  const [weeklySunlight, setWeeklySunlight] = useState<{ date: string; value: number }[]>([]);

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
  const [sleepModelReady, setSleepModelReady] = useState(false);
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
  const lastFocusScoreRef = useRef(0);
  // Rolling buffer of recent focus scores for temporal smoothing (sustained-state read).
  // ~24 cycles × 5s ≈ 2 min, matching the window aggregation that lifts LOSO AUC 0.82→0.95.
  const focusScoreBufferRef = useRef<number[]>([]);
  // Tracks where biometrics are coming from: a real watch (Health Connect) or mock data.
  const dataSourceRef = useRef<'watch' | 'mock'>('mock');

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
      loadModels(),
      loadSleepStageModel(),
      initDb(),
      initHealthConnect(),
      initNotifications(),
    ]);
    // Drain any sleep batches the Wear OS companion persisted while the
    // phone app was closed. Runs once per app open; subsequent batches
    // arrive live via the WearableSleepListenerService and are processed
    // here on the next open.
    processWearSleepSessions();
    generateInitialChartData();
    initLocationAndSunlight();
  }

  async function loadModels() {
    try {
      // Import JSON directly — Metro bundler handles JSON requires natively
      const stressModelJson = require('@/assets/ml/stress/stress_model.json');
      const anxietyModelJson = require('@/assets/ml/anxiety/anxiety_model.json');
      const focusModelJson = require('@/assets/ml/focus/focus_model.json');
      loadModel(stressModelJson);
      loadAnxietyModel(anxietyModelJson);
      loadFocusModel(focusModelJson);
      loadBioAgeModel(require('@/assets/ml/bioage/bioage_model.json'));
      loadDepressionModel(require('@/assets/ml/depression/depression_model.json'));
      setModelLoaded(true);
    } catch (e) {
      console.warn('[Seren] Failed to load ML models:', e);
    }
  }

  async function loadSleepStageModel() {
    try {
      // TFLite model is required inside loadV32SleepModel (react-native-fast-tflite).
      await loadV32SleepModel();
      setSleepModelReady(isV32SleepModelLoaded());
    } catch (e) {
      console.warn('[Seren] v3.2 sleep model not available (will use Health Connect stages):', e);
      setSleepModelReady(false);
    }
  }

  /**
   * Drain any sleep feature batches the Wear OS companion persisted while the
   * phone was closed. Each finalized session becomes a SleepAnalysis record.
   */
  async function processWearSleepSessions() {
    try {
      if (!isV32SleepModelLoaded()) return;
      const results = await processPendingSessions();
      for (const r of results) {
        if (!r.output) {
          console.log(`[Seren] Skipped watch session ${r.captureStartMs}: ${r.skipped}`);
          continue;
        }
        const fakeSession = {
          startTime: r.output.sessionStart,
          endTime: r.output.sessionEnd,
          stages: r.output.mlStages,
          source: 'com.seren.watch.v3.2',
        };
        const analysis = analyzeSleepSession(
          fakeSession as any,
          undefined,
          undefined,
          baseline,
          r.output.mlStages,
        );
        // L4: mean per-epoch confidence from the on-device model -> surfaced in the UI.
        const epochs = r.output.epochs;
        const mlConfidence = epochs.length
          ? epochs.reduce((s, e) => s + e.confidence, 0) / epochs.length
          : undefined;
        const enriched = { ...analysis, mlConfidence };
        setLastSleep(enriched);
        sleepHistory.current.push(enriched);
        setSleepTrend(computeSleepTrend(sleepHistory.current));
        if (dbReadyRef.current) {
          insertSleepSession(analysis).catch(() => {});
        }
        console.log(
          `[Seren] Wear OS sleep session ${r.captureStartMs}: ${r.epochsReceived} epochs → v3.2 classified`,
        );
      }
    } catch (e) {
      console.warn('[Seren] processWearSleepSessions failed:', e);
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

      // Restore the user's chronological age (for the bio-age age-gap), if set.
      const savedAge = await getSetting('chronological_age');
      if (savedAge != null && savedAge !== '') {
        const a = parseInt(savedAge, 10);
        if (!isNaN(a)) { chronoRef.current = a; setChronoState(a); }
      }

      // Restore gender (for the depression model), if set.
      const savedGender = await getSetting('gender');
      if (savedGender != null && savedGender !== '') {
        const g = parseFloat(savedGender);
        if (!isNaN(g)) { genderRef.current = g; setGenderState(g); }
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
        console.log('[Seren] Not Android — using mock Health Connect');
        healthService.current = createMockHealthConnectService();
        dataSourceRef.current = 'mock';
        setWatchConnected(true);
        return;
      }

      const realService = createHealthConnectService();
      const available = await realService.isAvailable();

      if (available) {
        setHealthConnectAvailable(true);
        console.log('[Seren] Health Connect available, checking permissions...');

        // Only check if we already have permissions — don't request here.
        // The permission dialog must be triggered by user interaction (Settings or Onboarding).
        const hasPerms = await realService.hasPermissions();

        if (hasPerms) {
          console.log('[Seren] Health Connect permissions granted — using REAL watch data');
          healthService.current = realService;
          dataSourceRef.current = 'watch';
          setWatchConnected(true);
        } else {
          console.log('[Seren] Permissions not yet granted — using mock data. Grant via Settings.');
          healthService.current = createMockHealthConnectService();
          dataSourceRef.current = 'mock';
          setWatchConnected(false);
        }
      } else {
        console.log('[Seren] Health Connect not available — using mock data');
        healthService.current = createMockHealthConnectService();
        dataSourceRef.current = 'mock';
        setWatchConnected(true);
      }
    } catch (e) {
      console.warn('[Seren] Health Connect init failed, using mock:', e);
      healthService.current = createMockHealthConnectService();
      dataSourceRef.current = 'mock';
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

  async function initLocationAndSunlight() {
    try {
      // Drain any ambient-light + GPS batches the watch delivered, then recompute
      // today's summaries so the reads below pick up real watch data over mock.
      if (dbReadyRef.current) {
        try {
          const env = await processPendingEnv();
          if (env.lightSamples || env.locationPoints) {
            console.log(`[Seren] env from watch: ${env.lightSamples} light, ${env.locationPoints} GPS`);
          }
        } catch (e) {
          console.warn('[Seren] processPendingEnv failed:', e);
        }
      }

      // Load persisted data (already recomputed above from watch batches when present).
      let haveLocation = false;
      let haveSunlight = false;
      if (dbReadyRef.current) {
        const savedLocation = await getTodayLocationDiversity();
        if (savedLocation) { setLocationDiversity(savedLocation); haveLocation = true; }

        const savedSunlight = await getTodaySunlightDaily();
        if (savedSunlight) { setSunlightExposure(savedSunlight); haveSunlight = true; }

        const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

        const locationHistory = await getLocationDiversityHistory(7);
        if (locationHistory.length > 0) {
          setWeeklyLocationDiversity(locationHistory.map(h => {
            const dayIdx = (new Date(h.date).getDay() + 6) % 7;
            return { date: days[dayIdx], value: Math.round(h.diversityScore) };
          }));
        }

        const sunlightHistory = await getSunlightDailyHistory(7);
        if (sunlightHistory.length > 0) {
          setWeeklySunlight(sunlightHistory.map(h => {
            const dayIdx = (new Date(h.date).getDay() + 6) % 7;
            return { date: days[dayIdx], value: Math.round(h.totalOutdoorMinutes) };
          }));
        }
      }

      // Fall back to mock only when the watch hasn't delivered real data yet
      // (sunlight + location are sourced from the watch via processPendingEnv).
      if (!haveLocation) {
        setLocationDiversity(getMockLocationDiversity());
        setWeeklyLocationDiversity(getMockWeeklyLocationDiversity());
      }
      if (!haveSunlight) {
        setSunlightExposure(getMockSunlightExposure());
        setWeeklySunlight(getMockWeeklySunlight());
      }
    } catch (e) {
      console.warn('[Seren] Location/Sunlight init failed:', e);
      // Fallback to mock
      setLocationDiversity(getMockLocationDiversity());
      setSunlightExposure(getMockSunlightExposure());
      setWeeklyLocationDiversity(getMockWeeklyLocationDiversity());
      setWeeklySunlight(getMockWeeklySunlight());
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
      const d = new Date(sa.sessionStart);
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
          const diff = Math.abs(s.sessionStart - target);
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

      // S1: pass the user's recent windows so the model can normalize per-subject
      // (its headline accuracy lever) instead of the global training scaler.
      const stressPred = predictStress(featureVector, baseline, featureHistory.current);
      setStress(stressPred);
      recordStressPrediction(stressPred); // S6/L8: runtime model-health/drift monitoring

      const anxietyPred = predictAnxiety(featureVector, baseline);
      setAnxiety(anxietyPred);

      // Focus: trained CogWear model + per-user normalization (pass recent windows,
      // same history the stress model uses for its per-subject scaler).
      const focusPred = predictFocus(featureVector, featureHistory.current);

      // Temporal smoothing: focus/engagement is a sustained state, not a 5-second blip.
      // Average the model's recent scores (~2 min) — the deployment analogue of the
      // per-session window aggregation that raised LOSO AUC 0.82→0.95.
      const buf = focusScoreBufferRef.current;
      buf.push(focusPred.focusScore);
      if (buf.length > 24) buf.shift();
      const smoothedScore = Math.round(buf.reduce((s, v) => s + v, 0) / buf.length);
      const focusSmoothed = {
        ...focusPred,
        focusScore: smoothedScore,
        focusLevel: getFocusLevel(smoothedScore),
      };
      setFocus({ ...focusSmoothed, groqLoading: false });

      // Depression: daily actigraphy signal — compute once per calendar day.
      const todayStr = new Date().toISOString().slice(0, 10);
      if (lastDepressionDateRef.current !== todayStr) {
        try {
          const dayAgo = now - 24 * 60 * 60 * 1000;
          const daySteps = await service.readSteps(dayAgo, now);
          const daysAvailable = Math.min(30, Math.ceil(featureHistory.current.length / 288));
          const dailyFeats = buildDailyActivityFeatures(
            daySteps, genderRef.current ?? 1.5, chronoRef.current, daysAvailable,
          );
          const depPred = predictDepression(dailyFeats);
          setDepression(depPred);
          lastDepressionDateRef.current = todayStr;
          sendDepressionToWatch(depPred).catch(() => {});
        } catch (e) {
          console.warn('[Seren][Depression] daily prediction failed:', e);
        }
      }

      // Biological age: a stable trait → aggregate the user's recent windows (median),
      // global-normalize, predict physiological age, derive the age gap vs chronological age.
      const bioAgePred = predictBioAge(featureHistory.current, chronoRef.current);
      setBioAge(bioAgePred);
      console.log(
        `[Seren][BioAge] ready=${bioAgePred.ready} predictedAge=${bioAgePred.predictedAge} ` +
        `chrono=${bioAgePred.chronologicalAge} gap=${bioAgePred.ageGap} ` +
        `windows=${featureHistory.current.length} conf=${bioAgePred.confidence.toFixed(2)}`,
      );

      // [Focus] live-signal trace — shows the watch data feeding the focus model each cycle.
      // source=watch → real Health Connect data; source=mock → synthetic (no watch connected).
      const normMode = featureHistory.current.length >= 20 ? 'per-user' : 'cold-start/global';
      console.log(
        `[Seren][Focus] source=${dataSourceRef.current} norm=${normMode} ` +
        `HR=${featureVector.hrMean.toFixed(0)}bpm ` +
        `HRV=${featureVector.rmssd.toFixed(0)}ms ` +
        `lf/hf=${featureVector.lfHfRatio.toFixed(2)} ` +
        `→ raw=${focusPred.focusScore} smoothed=${smoothedScore} (${focusSmoothed.focusLevel}) ` +
        `elevated=[${focusPred.elevatedFeatures.map(f => f.feature).join(',') || 'none'}]`,
      );

      // Only trigger Groq if the (smoothed) focus score changed by 7+ points.
      const scoreDiff = Math.abs(smoothedScore - lastFocusScoreRef.current);
      if (scoreDiff >= 7) {
        console.log(`[Seren] Focus score changed ${scoreDiff.toFixed(0)} points (${lastFocusScoreRef.current} → ${smoothedScore}) - Triggering Groq`);
        lastFocusScoreRef.current = smoothedScore;
        setFocus(prev => ({ ...prev, groqLoading: true }));

        getFocusTips(smoothedScore, focusSmoothed.focusLevel, focusPred.elevatedFeatures, false)
          .then(tips => {
            console.log(`[Seren] Groq tips received: ${tips.length} tips`);
            setFocus(prev => ({ ...prev, groqTips: tips, groqLoading: false }));
          })
          .catch(err => {
            console.warn(`[Seren] Groq failed: ${err}`);
            setFocus(prev => ({ ...prev, groqLoading: false }));
          });
      } else {
        console.log(`[Seren] Focus score change too small (${scoreDiff.toFixed(1)}/7) - No Groq generation`);
        setFocus(prev => ({ ...prev, groqTips: [], groqLoading: false }));
      }

      if (baseline) {
        const flags = detectAnomalies(featureVector, baseline);
        setAnomalies(flags);
        if (flags.length > 0) notifyAnomalies(flags);
      }

      notifyHighStress(stressPred);
      notifySustainedAnxiety(anxietyPred);

      const recs = generateRecommendations(
        stressPred, anxietyPred, lastSleep, lastCheckin, baseline, [],
        locationDiversity, sunlightExposure,
      );
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

        // v3.2 sleep classification runs only from Wear OS feature batches
        // (Health Connect doesn't expose the raw accel the model needs).
        // When no watch session is available for this period we fall back to
        // Health Connect's pre-classified stages.
        const analysis = analyzeSleepSession(latest, undefined, undefined, baseline);
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

  const setChronologicalAge = useCallback((age: number | null) => {
    chronoRef.current = age;
    setChronoState(age);
    setBioAge(predictBioAge(featureHistory.current, age));
    if (dbReadyRef.current) {
      setSetting('chronological_age', age == null ? '' : String(age)).catch(() => {});
    }
  }, []);

  const setGender = useCallback((g: number | null) => {
    genderRef.current = g;
    setGenderState(g);
    // Reset the daily gate so the depression model re-runs with the new gender
    lastDepressionDateRef.current = null;
    if (dbReadyRef.current) {
      setSetting('gender', g == null ? '' : String(g)).catch(() => {});
    }
  }, []);

  const refreshData = useCallback(async () => {
    await runInferenceCycle();
    await fetchSleepData();
  }, []);

  const deleteAllUserData = useCallback(async () => {
    featureHistory.current = [];
    sleepHistory.current = [];
    focusScoreBufferRef.current = [];
    setBaseline(null);
    setCheckinHistory([]);
    setLastCheckin(null);
    setLastSleep(null);
    setDepression(null);
    lastDepressionDateRef.current = null;
    setStress(DEFAULT_STRESS);
    setAnxiety(DEFAULT_ANXIETY);
    setFocus(DEFAULT_FOCUS);
    setRecommendations([]);
    setAnomalies([]);
    setSleepTrend(DEFAULT_SLEEP_TREND);
    setCheckinTrend(DEFAULT_CHECKIN_TREND);
    setLocationDiversity(null);
    setSunlightExposure(null);
    setWeeklyLocationDiversity([]);
    setWeeklySunlight([]);

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
      const available = await realService.isAvailable();
      if (!available) {
        healthService.current = createMockHealthConnectService();
        dataSourceRef.current = 'mock';
        setHealthConnectAvailable(false);
        setWatchConnected(true);
        return false;
      }

      const granted = await realService.requestPermissions();
      if (granted) {
        healthService.current = realService;
        dataSourceRef.current = 'watch';
        console.log('[Seren] Health Connect permissions granted via Settings — switching to REAL watch data');
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
    stress, anxiety, focus, bioAge, depression, chronologicalAge, setChronologicalAge,
    gender, setGender,
    lastSleep, sleepTrend, baseline, anomalies,
    recommendations, features, heartRate, hrv,
    locationDiversity, sunlightExposure,
    lastCheckin, checkinHistory, checkinTrend, performCheckin,
    weeklyStress, weeklyHrv, weeklySleep,
    weeklyLocationDiversity, weeklySunlight,
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
