/**
 * Seren AI — Central Configuration
 * ===================================
 * Put your API keys here. Both LLM (emotional analysis) and
 * Whisper (voice transcription) auto-configure on app startup.
 *
 * Supported providers:
 *   - 'groq'   — Free tier, fast (recommended for development)
 *   - 'openai'  — GPT-4o-mini + Whisper
 *   - 'ollama'  — Local LLM only (no voice transcription)
 */

import { configureLLMPreset } from './llmService';
import { configureWhisper } from './whisperService';

// ============================================================
// YOUR API KEYS — edit these values
// ============================================================

const AI_CONFIG = {
  /** Which provider to use: 'groq' | 'openai' | 'ollama' */
  provider: 'groq' as 'openai' | 'groq' | 'ollama',

  /** Your API key (get one free at https://console.groq.com or https://platform.openai.com) */
  apiKey: '',  // Set your API key here or use environment variable

  /**
   * Language for voice transcription (ISO 639-1).
   * Examples: 'en' for English, 'ar' for Arabic
   * Leave empty for auto-detect.
   */
  language: 'en',
};

// ============================================================
// Auto-initialization (called once on app startup)
// ============================================================

let _initialized = false;

/**
 * Initialize AI services from the config above.
 * Call this once in the app entry point (e.g., WellnessProvider).
 * Safe to call multiple times — only runs once.
 */
export function initializeAIServices(): void {
  if (_initialized) return;
  _initialized = true;

  const { provider, apiKey } = AI_CONFIG;

  if (!apiKey) {
    console.log('[Seren AI] No API key configured — using local fallback analysis.');
    console.log('[Seren AI] To enable real AI, add your key in services/ai/aiConfig.ts');
    return;
  }

  // Configure LLM (emotional analysis)
  configureLLMPreset(provider, apiKey);
  console.log(`[Seren AI] LLM configured: ${provider}`);

  // Configure Whisper (voice transcription) — only for cloud providers
  if (provider !== 'ollama') {
    configureWhisper(provider, apiKey, AI_CONFIG.language || undefined);
    console.log(`[Seren AI] Whisper configured: ${provider}`);
  }
}

/**
 * Check if AI services have been configured with a valid key.
 */
export function isAIConfigured(): boolean {
  return AI_CONFIG.apiKey.length > 0;
}

/**
 * Get the current provider name (for display in UI).
 */
export function getAIProvider(): string {
  return AI_CONFIG.provider;
}
