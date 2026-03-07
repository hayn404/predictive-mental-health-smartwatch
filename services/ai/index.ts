/**
 * Seren AI Services — Barrel Export
 * ====================================
 * Single import point for all AI services.
 *
 * Usage:
 *   import { predictStress, analyzeSleepSession, generateRecommendations } from '@/services/ai';
 */

// Core types
export * from './types';

// Health Connect data ingestion
export { createHealthConnectService, createMockHealthConnectService } from './healthConnect';

// Feature engineering (on-device)
export { extractFeatures } from './featureEngineering';

// Stress & anxiety prediction
export { loadModel, loadModelFromString, isModelLoaded, predictStress, predictAnxiety } from './stressModel';

// Personal baseline
export { computeBaseline, shouldRecomputeBaseline, detectAnomalies } from './baseline';
export type { AnomalyFlag } from './baseline';

// Sleep analysis
export { analyzeSleepSession, computeSleepTrend } from './sleepAnalysis';
export type { SleepTrend } from './sleepAnalysis';

// Voice check-in
export { analyzeCheckin, analyzeCheckinLocal, computeCheckinTrend } from './voiceAnalysis';
export type { CheckinTrend } from './voiceAnalysis';

// LLM service (external API with privacy encryption)
export {
  configureLLM,
  configureLLMPreset,
  isLLMConfigured,
  getLLMConfig,
  analyzeCheckinWithLLM,
  encryptForStorage,
  decryptFromStorage,
} from './llmService';
export type { LLMConfig, FullBiometricContext } from './llmService';

// Recommendation engine
export { generateRecommendations, recordOutcome } from './recommendations';

// Local database
export {
  initDatabase,
  insertBiometricSample,
  insertBiometricSamplesBatch,
  getBiometricSamples,
  getLatestBiometricSample,
  insertFeatureWindow,
  getFeatureWindows,
  getRecentFeatureWindows,
  insertSleepSession,
  getSleepSessions,
  getLatestSleepSession,
  saveBaseline,
  getLatestBaseline,
  saveCheckin,
  getRecentCheckins,
  saveRecommendation,
  getRecentRecommendationIds,
  cleanupOldData,
  deleteAllData,
} from './db';
export type { SQLiteDatabase } from './db';

// Speech recognition (legacy simulated)
export { createSpeechService } from './speechRecognition';

// Real audio recording (expo-av)
export { createRecordingSession, readAudioAsBase64, deleteAudioFile } from './audioRecorder';
export type { RecordingSession, RecordingStatus } from './audioRecorder';

// Whisper speech-to-text
export { configureWhisper, configureWhisperFull, isWhisperConfigured, getWhisperConfig, transcribeAudio } from './whisperService';
export type { WhisperConfig } from './whisperService';

// Push notifications
export {
  requestNotificationPermission,
  notifyAnomalies,
  notifyHighStress,
  notifySustainedAnxiety,
  scheduleDailyCheckinReminder,
  cancelAllNotifications,
} from './notifications';

// Data export
export { exportHealthData, exportHealthDataFull } from './dataExport';

// AI config (central API key management)
export { initializeAIServices, isAIConfigured, getAIProvider } from './aiConfig';
