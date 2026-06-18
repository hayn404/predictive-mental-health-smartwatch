/**
 * Seren AI — Central Configuration
 * ===================================
 * API keys are loaded from environment variables (never committed to source).
 * Configure them in your .env file:
 *   EXPO_PUBLIC_GROQ_API_KEY=gsk_...  (for focus tips & voice transcription)
 *   EXPO_PUBLIC_OPENROUTER_API_KEY=sk-or-v1-... (optional, for emotional analysis)
 */

import { configureLLMPreset } from './llmService';
import { configureWhisper } from './whisperService';
import { configureNemotron } from './nemotronAsrService';
import { configureRemoteTTS } from './ttsService';

// ============================================================
// Internal configuration — API keys from env vars (secrets)
// ============================================================

const GROQ_API_KEY = process.env.EXPO_PUBLIC_GROQ_API_KEY || '';
const OPENROUTER_API_KEY = process.env.EXPO_PUBLIC_OPENROUTER_API_KEY || '';
const ELEVENLABS_API_KEY = process.env.EXPO_PUBLIC_ELEVENLABS_API_KEY || '';
const NEMOTRON_RELAY_URL = process.env.EXPO_PUBLIC_NEMOTRON_RELAY_URL || '';
const NEMOTRON_RELAY_KEY = process.env.EXPO_PUBLIC_NEMOTRON_RELAY_KEY || '';

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

  // Configure LLM with Groq (focus tips & emotional analysis)
  if (GROQ_API_KEY) {
    configureLLMPreset('groq', GROQ_API_KEY);
    console.log('[Seren AI] LLM configured: Groq');
  } else if (OPENROUTER_API_KEY) {
    configureLLMPreset('openrouter', OPENROUTER_API_KEY);
    console.log('[Seren AI] LLM configured: OpenRouter');
  } else {
    console.log('[Seren AI] No LLM key set — using local fallback analysis.');
  }

  // Configure Whisper via Groq (optional — voice transcription)
  if (GROQ_API_KEY) {
    configureWhisper('groq', GROQ_API_KEY, LANGUAGE);
    console.log('[Seren AI] Whisper configured: Groq');
  }

  // Configure Nemotron ASR relay (optional — see server/nemotron-relay/).
  // Whisper above stays configured regardless, so voiceAssistant.ts's
  // Nemotron-first/Whisper-fallback has something to fall back to.
  if (NEMOTRON_RELAY_URL) {
    configureNemotron(NEMOTRON_RELAY_URL, NEMOTRON_RELAY_KEY || undefined, 'en-US');
    console.log('[Seren AI] Nemotron ASR relay configured:', NEMOTRON_RELAY_URL);
  }

  // Configure remote TTS (optional — falls back to on-device voice when unset)
  if (ELEVENLABS_API_KEY) {
    configureRemoteTTS({ provider: 'elevenlabs', apiKey: ELEVENLABS_API_KEY });
    console.log('[Seren AI] Remote TTS configured: ElevenLabs');
  }
}

/**
 * Check if AI services have been configured.
 */
export function isAIConfigured(): boolean {
  return GROQ_API_KEY.length > 0 || OPENROUTER_API_KEY.length > 0;
}

/**
 * Get the current provider name (for internal logging).
 */
export function getAIProvider(): string {
  if (GROQ_API_KEY) return 'groq';
  if (OPENROUTER_API_KEY) return 'openrouter';
  return 'none';
}
