// ============================================================
// Seren AI Services — Core Type Definitions
// ============================================================
// These types define the data contracts between:
// 1. Samsung Galaxy Watch (via Health Connect) -> App
// 2. App -> Feature Engineering -> ML Models
// 3. ML Models -> UI Layer
// ============================================================

// ----------------------------------------------------------
// RAW DATA FROM HEALTH CONNECT (Samsung Galaxy Watch)
// ----------------------------------------------------------

/** Raw heart rate sample from Health Connect HeartRateRecord */
export interface RawHeartRateSample {
  timestamp: number;        // Unix ms
  bpm: number;              // Beats per minute
  source: string;           // e.g. 'com.samsung.health'
}

/** Raw HRV sample from Health Connect HeartRateVariabilityRmssdRecord */
export interface RawHRVSample {
  timestamp: number;
  rmssd: number;            // RMSSD in milliseconds
}

/** Raw inter-beat interval (RR interval) derived from HR data */
export interface RawRRInterval {
  timestamp: number;
  intervalMs: number;       // Milliseconds between beats (e.g., 857ms = ~70 BPM)
}

/** Sleep stage from Health Connect SleepSessionRecord */
export type SleepStageType = 'awake' | 'light' | 'deep' | 'rem' | 'sleeping' | 'out_of_bed' | 'unknown';

export interface RawSleepStage {
  startTime: number;        // Unix ms
  endTime: number;
  stage: SleepStageType;
}

export interface RawSleepSession {
  startTime: number;
  endTime: number;
  stages: RawSleepStage[];
  source: string;
}

/** Steps from Health Connect StepsRecord */
export interface RawStepsSample {
  startTime: number;
  endTime: number;
  count: number;
}

/** Skin temperature from Health Connect SkinTemperatureRecord */
export interface RawTemperatureSample {
  timestamp: number;
  temperatureCelsius: number;
}

/** SpO2 from Health Connect OxygenSaturationRecord */
export interface RawSpO2Sample {
  timestamp: number;
  percentage: number;       // 0-100
}

/** Respiratory rate from Health Connect RespiratoryRateRecord */
export interface RawRespiratoryRateSample {
  timestamp: number;
  breathsPerMinute: number;
}

// ----------------------------------------------------------
// FEATURE ENGINEERING OUTPUT (computed on-device)
// ----------------------------------------------------------

/**
 * 29 HRV + biometric features computed per 5-minute window.
 * This is the input vector for the stress prediction model.
 *
 * Feature groups:
 * - Time-domain HRV (9 features)
 * - Frequency-domain HRV (7 features)
 * - Non-linear HRV (5 features)
 * - Skin temperature (4 features)
 * - Accelerometer / Activity (4 features)
 */
export interface BiometricFeatureVector {
  timestamp: number;
  windowSeconds: number;    // Typically 300 (5 min)

  // ---- Time-domain HRV (9) ----
  meanRR: number;           // Mean RR interval in ms
  sdnn: number;             // Std dev of NN intervals (ms)
  rmssd: number;            // Root mean square successive diff (ms)
  pnn50: number;            // % successive diffs > 50ms (0-100)
  pnn20: number;            // % successive diffs > 20ms (0-100)
  hrMean: number;           // Mean heart rate (BPM)
  hrStd: number;            // Heart rate std dev
  hrRange: number;          // Max HR - Min HR in window
  cvRR: number;             // Coefficient of variation (SDNN / meanRR)

  // ---- Frequency-domain HRV (7) ----
  vlfPower: number;         // Very low freq power (0.003-0.04 Hz)
  lfPower: number;          // Low freq power (0.04-0.15 Hz)
  hfPower: number;          // High freq power (0.15-0.4 Hz)
  lfHfRatio: number;        // LF/HF ratio (sympathovagal balance)
  totalPower: number;       // Sum of all frequency bands
  lfNorm: number;           // LF / (LF + HF) * 100
  hfNorm: number;           // HF / (LF + HF) * 100

  // ---- Non-linear HRV (5) ----
  sd1: number;              // Poincare plot short-term variability
  sd2: number;              // Poincare plot long-term variability
  sd1sd2Ratio: number;      // SD1 / SD2
  sampleEntropy: number;    // Signal complexity
  dfaAlpha1: number;        // Detrended fluctuation analysis (4-16 beats)

  // ---- Skin temperature (4) ----
  tempMean: number;         // Mean wrist temperature (°C)
  tempSlope: number;        // Rate of change (°C/min)
  tempStd: number;          // Temperature variability
  tempRange: number;        // Max - Min in window

  // ---- Activity / Accelerometer (4) ----
  accelMagnitudeMean: number;   // Mean movement intensity
  accelMagnitudeStd: number;    // Movement variability
  stepCount: number;            // Steps in this window
  activityType: ActivityType;   // Classified activity
}

export type ActivityType = 'sedentary' | 'walking' | 'active' | 'sleeping';

/**
 * Ordered feature names matching the XGBoost model's expected input.
 * MUST match the order used during Python training.
 */
export const FEATURE_NAMES: (keyof BiometricFeatureVector)[] = [
  'meanRR', 'sdnn', 'rmssd', 'pnn50', 'pnn20',
  'hrMean', 'hrStd', 'hrRange', 'cvRR',
  'vlfPower', 'lfPower', 'hfPower', 'lfHfRatio',
  'totalPower', 'lfNorm', 'hfNorm',
  'sd1', 'sd2', 'sd1sd2Ratio', 'sampleEntropy', 'dfaAlpha1',
  'tempMean', 'tempSlope', 'tempStd', 'tempRange',
  'accelMagnitudeMean', 'accelMagnitudeStd', 'stepCount',
  // activityType is encoded as number: sedentary=0, walking=1, active=2, sleeping=3
];

// ----------------------------------------------------------
// PERSONAL BASELINE
// ----------------------------------------------------------

/** Rolling 14-day personal baseline for normalization & anomaly detection */
export interface PersonalBaseline {
  userId: string;
  computedAt: number;       // Unix ms
  periodDays: number;       // Typically 14
  sampleCount: number;      // How many windows were used

  // Heart rate baselines
  restingHrMean: number;
  restingHrStd: number;
  sleepingHrMean: number;

  // HRV baselines
  rmssdMean: number;
  rmssdStd: number;
  sdnnMean: number;
  sdnnStd: number;
  nocturnalRmssdMean: number;

  // Sleep baselines
  typicalBedtimeHour: number;   // 0-23 fractional (e.g., 23.5 = 11:30 PM)
  typicalWakeHour: number;
  avgSleepDurationMin: number;
  avgSleepEfficiency: number;   // 0-1
  avgSleepQuality: number;      // 0-100
  avgDeepSleepPct: number;      // 0-1
  avgRemSleepPct: number;       // 0-1

  // Activity baselines
  avgDailySteps: number;

  // Temperature baselines
  restingTempMean: number;
  restingTempStd: number;
}

// ----------------------------------------------------------
// MODEL OUTPUTS
// ----------------------------------------------------------

/** Stress model prediction result */
export interface StressPrediction {
  timestamp: number;
  stressScore: number;          // 0-100
  stressLevel: StressLevel;
  confidence: number;           // 0-1
  topContributors: StressContributor[];
}

export type StressLevel = 'low' | 'moderate' | 'elevated' | 'high';

export interface StressContributor {
  feature: string;              // e.g., 'rmssd', 'lfHfRatio'
  label: string;                // Human-readable: 'Low HRV'
  impact: number;               // How much this feature contributed (0-1)
}

/** Anxiety index (derived from stress + context + patterns) */
export interface AnxietyPrediction {
  timestamp: number;
  anxietyIndex: number;         // 0-100
  level: 'minimal' | 'mild' | 'moderate' | 'severe';
  sustained: boolean;           // True if elevated for >10 min
  baselineDeviation: number;    // How far from personal baseline (-1 to +1)
}

// ----------------------------------------------------------
// SLEEP ANALYSIS
// ----------------------------------------------------------

export interface SleepAnalysis {
  date: string;                 // YYYY-MM-DD
  sessionStart: number;         // Unix ms
  sessionEnd: number;

  // Durations (minutes)
  totalInBedMin: number;
  totalSleepMin: number;
  onsetLatencyMin: number;      // Time to fall asleep
  wasoMin: number;              // Wake after sleep onset

  // Stage durations (minutes)
  lightSleepMin: number;
  deepSleepMin: number;
  remSleepMin: number;
  awakeMin: number;

  // Stage percentages (0-1)
  deepSleepPct: number;
  remSleepPct: number;
  lightSleepPct: number;

  // Quality metrics
  sleepEfficiency: number;      // 0-1 (totalSleep / totalInBed)
  fragmentationIndex: number;   // awakenings per hour
  awakeningCount: number;

  // Biometric summaries during sleep
  avgHrSleep: number;
  minHrSleep: number;
  avgHrvSleep: number;
  maxHrvSleep: number;

  // Computed scores
  qualityScore: number;         // 0-100
  recoveryScore: number;        // 0-100 (HRV-based)
  consistencyScore: number;     // 0-100 (vs usual schedule)
}

// ----------------------------------------------------------
// VOICE CHECK-IN
// ----------------------------------------------------------

export type Sentiment = 'positive' | 'neutral' | 'concerned' | 'distressed';

export interface EmotionScores {
  joy: number;
  sadness: number;
  anxiety: number;
  anger: number;
  calm: number;
  fear: number;
  gratitude: number;
  fatigue: number;
}

/** Life context themes detected in check-in */
export type LifeTheme =
  | 'work' | 'relationships' | 'health' | 'sleep' | 'finances'
  | 'family' | 'social' | 'academic' | 'self_image' | 'existential'
  | 'general';

export interface CheckinAnalysis {
  id: string;
  timestamp: number;
  transcript: string;
  sentiment: Sentiment;
  sentimentScore: number;       // -1.0 (very negative) to +1.0 (very positive)
  emotionScores: EmotionScores;
  keyInsights: string[];

  // Deep understanding (Phase 2)
  themes: LifeTheme[];                // What life areas are involved
  emotionalIntensity: number;         // 0-1 how intense the feelings are
  empathyResponse: string;            // AI's empathetic, human-like reply
  suggestedFollowUp: string | null;   // A gentle follow-up question or null

  // Biometric context at time of check-in
  hrAtCheckin: number;
  hrvAtCheckin: number;
  stressAtCheckin: number;
}

// ----------------------------------------------------------
// RECOMMENDATION ENGINE
// ----------------------------------------------------------

export type InterventionCategory =
  | 'breathing'
  | 'physical'
  | 'meditation'
  | 'journaling'
  | 'sleep_hygiene'
  | 'social'
  | 'outdoor'
  | 'exploration';

export type RecommendationTrigger = 'biometric' | 'schedule' | 'pattern' | 'checkin' | 'user_request';

export interface Recommendation {
  id: string;
  category: InterventionCategory;
  title: string;
  description: string;
  durationMin: number;
  instructions: string[];

  // Why this was recommended
  trigger: RecommendationTrigger;
  triggerReason: string;        // Human-readable reason
  priorityScore: number;        // 0-1

  // Clinical basis
  evidenceLevel: 'strong' | 'moderate' | 'emerging';
  citation: string;

  // Tracking
  status: 'pending' | 'accepted' | 'completed' | 'dismissed';
  preStress?: number;
  postStress?: number;
  preHrv?: number;
  postHrv?: number;
  effectivenessScore?: number;
}

// ----------------------------------------------------------
// AGGREGATE STATE (drives the UI)
// ----------------------------------------------------------

/** The real-time computed state that the UI subscribes to */
export interface CurrentWellnessState {
  timestamp: number;
  stressPrediction: StressPrediction;
  anxietyPrediction: AnxietyPrediction;
  lastSleep: SleepAnalysis | null;
  latestFeatures: BiometricFeatureVector | null;
  baseline: PersonalBaseline | null;
  activeRecommendations: Recommendation[];
  watchConnected: boolean;
  lastSyncTime: number;
}

// ----------------------------------------------------------
// DATABASE SCHEMA TYPES (for expo-sqlite)
// ----------------------------------------------------------

/** Stored biometric sample row in SQLite */
export interface DBBiometricSample {
  id: number;
  timestamp: number;
  hr_bpm: number;
  rr_interval_ms: number | null;
  skin_temp_c: number | null;
  spo2_pct: number | null;
  resp_rate: number | null;
  accel_magnitude: number | null;
  step_count: number | null;
  source: string;
}

/** Stored HRV feature window row in SQLite */
export interface DBFeatureWindow {
  id: number;
  timestamp: number;
  window_seconds: number;
  features_json: string;    // Serialized BiometricFeatureVector
  stress_score: number | null;
  anxiety_index: number | null;
}

/** Stored sleep session row in SQLite */
export interface DBSleepSession {
  id: number;
  date: string;
  session_start: number;
  session_end: number;
  total_sleep_min: number;
  deep_sleep_min: number;
  rem_sleep_min: number;
  light_sleep_min: number;
  awake_min: number;
  quality_score: number;
  recovery_score: number;
  stages_json: string;      // Serialized RawSleepStage[]
}

// ----------------------------------------------------------
// LOCATION DIVERSITY TRACKING
// ----------------------------------------------------------

/** A detected location visit (clustered from raw GPS) */
export interface LocationVisit {
  id: string;
  timestamp: number;          // Arrival time (Unix ms)
  departureTime: number;      // Departure time (Unix ms)
  latitude: number;
  longitude: number;
  label: string;              // 'home' | 'work' | 'unknown_1' | etc.
  clusterIndex: number;       // Which cluster this belongs to
}

/** Daily location diversity summary */
export interface LocationDiversitySummary {
  date: string;               // YYYY-MM-DD
  uniquePlacesVisited: number;
  totalTransitions: number;   // Number of place changes
  diversityScore: number;     // 0-100 (0 = one place, 100 = highly varied)
  homeTimePercent: number;    // 0-1
  workTimePercent: number;    // 0-1
  novelPlaces: number;        // Places visited for the first time this week
  isMonotonous: boolean;      // True if only home <-> work pattern
}

/** DB row for location visits */
export interface DBLocationVisit {
  id: number;
  visit_id: string;
  timestamp: number;
  departure_time: number;
  latitude: number;
  longitude: number;
  label: string;
  cluster_index: number;
}

/** DB row for daily location diversity */
export interface DBLocationDiversity {
  id: number;
  date: string;
  unique_places: number;
  total_transitions: number;
  diversity_score: number;
  home_time_pct: number;
  work_time_pct: number;
  novel_places: number;
  is_monotonous: number;      // 0 or 1
}

// ----------------------------------------------------------
// SUNLIGHT EXPOSURE TRACKING
// ----------------------------------------------------------

/** A single sunlight sensor reading */
export interface SunlightReading {
  timestamp: number;          // Unix ms
  luxValue: number;           // Raw lux from light sensor
  isOutdoors: boolean;        // lux > threshold (1000 lux)
}

/** Daily sunlight exposure summary */
export interface SunlightExposureSummary {
  date: string;               // YYYY-MM-DD
  totalOutdoorMinutes: number;
  optimalWindowMinutes: number;  // Minutes outdoors during 10am-3pm (vitamin D window)
  peakLux: number;
  avgOutdoorLux: number;
  goalMinutes: number;        // Default 30
  goalProgress: number;       // 0-1
  isVitaminDWindow: boolean;  // Whether current time is 10am-3pm
}

/** DB row for sunlight samples */
export interface DBSunlightSample {
  id: number;
  timestamp: number;
  lux_value: number;
  is_outdoors: number;        // 0 or 1
}

/** DB row for daily sunlight summary */
export interface DBSunlightDaily {
  id: number;
  date: string;
  total_outdoor_min: number;
  optimal_window_min: number;
  peak_lux: number;
  avg_outdoor_lux: number;
  goal_minutes: number;
}
