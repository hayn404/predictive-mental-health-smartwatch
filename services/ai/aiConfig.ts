/**
 * Seren AI — Central Configuration
 * ===================================
 * API keys are loaded from environment variables (never committed to source).
 * Configure them in your .env file:
 *   EXPO_PUBLIC_OPENROUTER_API_KEY=sk-or-v1-...
 *   EXPO_PUBLIC_GROQ_API_KEY=gsk_...  (optional, for voice transcription)
 */

import { configureLLMPreset } from './llmService';
import { configureWhisper } from './whisperService';

// ============================================================
// Internal configuration — API keys from env vars (secrets)
// ============================================================

const OPENROUTER_API_KEY = process.env.EXPO_PUBLIC_OPENROUTER_API_KEY || '';
const GROQ_API_KEY = process.env.EXPO_PUBLIC_GROQ_API_KEY || '';

const LANGUAGE = 'en';

// ============================================================
// Auto-initialization (called once on app startup)
// ============================================================

let _initialized = false;

/**
 * Initialize AI services from environment variables.
 * Call this once in the app entry point (e.g., WellnessProvider).
 * Safe to call multiple times — only runs once.
 */
export function initializeAIServices(): void {
  if (_initialized) return;
  _initialized = true;

  // Configure LLM with OpenRouter (emotional analysis)
  if (OPENROUTER_API_KEY) {
    configureLLMPreset('openrouter', OPENROUTER_API_KEY);
    console.log('[Seren AI] LLM configured: OpenRouter');
  } else {
    console.log('[Seren AI] OpenRouter key not set — using local fallback analysis.');
  }

  // Configure Whisper via Groq (optional — voice transcription)
  if (GROQ_API_KEY) {
    configureWhisper('groq', GROQ_API_KEY, LANGUAGE);
    console.log('[Seren AI] Whisper configured: Groq');
  }
}

/**
 * Check if AI services have been configured.
 */
export function isAIConfigured(): boolean {
  return OPENROUTER_API_KEY.length > 0;
}

/**
 * Get the current provider name (for internal logging).
 */
export function getAIProvider(): string {
  return 'openrouter';
}
