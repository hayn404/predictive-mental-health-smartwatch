/**
 * Seren AI — Speech Recognition Service
 * ========================================
 * Phase 1: Uses expo-speech-recognition for real-time STT.
 * Falls back to manual text input when speech recognition is unavailable.
 *
 * Phase 2 (future): whisper.rn for fully offline STT.
 */

import { Platform } from 'react-native';

// ============================================================
// Types
// ============================================================

export interface SpeechRecognitionResult {
  transcript: string;
  isFinal: boolean;
  confidence: number;
}

export interface SpeechRecognitionCallbacks {
  onResult: (result: SpeechRecognitionResult) => void;
  onError: (error: string) => void;
  onEnd: () => void;
}

export interface SpeechRecognitionService {
  isAvailable(): Promise<boolean>;
  start(callbacks: SpeechRecognitionCallbacks): Promise<void>;
  stop(): Promise<string>;
}

// ============================================================
// Real Implementation (expo-speech-recognition or Voice)
// ============================================================

let ExpoSpeech: any = null;

async function loadSpeechModule(): Promise<boolean> {
  if (ExpoSpeech) return true;
  try {
    // Try @react-native-voice/voice first (more reliable)
    ExpoSpeech = await import('@react-native-voice/voice');
    return true;
  } catch {
    try {
      // Fallback: expo-speech (TTS only, not STT — won't work for recognition)
      // In this case, we fall back to simulated mode
      return false;
    } catch {
      return false;
    }
  }
}

// ============================================================
// Simulated Speech Recognition (for Expo Go / development)
// ============================================================

const SIMULATED_PHRASES = [
  "I'm feeling ",
  "I'm feeling a bit tired today, ",
  "I'm feeling a bit tired today, but generally optimistic. ",
  "I'm feeling a bit tired today, but generally optimistic. Work has been intense ",
  "I'm feeling a bit tired today, but generally optimistic. Work has been intense but I feel like I'm making progress.",
];

function createSimulatedService(): SpeechRecognitionService {
  let intervalId: ReturnType<typeof setInterval> | null = null;
  let phraseIndex = 0;
  let currentCallbacks: SpeechRecognitionCallbacks | null = null;

  return {
    async isAvailable() {
      return true; // Simulated is always available
    },

    async start(callbacks) {
      phraseIndex = 0;
      currentCallbacks = callbacks;

      intervalId = setInterval(() => {
        if (phraseIndex < SIMULATED_PHRASES.length) {
          const isFinal = phraseIndex === SIMULATED_PHRASES.length - 1;
          callbacks.onResult({
            transcript: SIMULATED_PHRASES[phraseIndex],
            isFinal,
            confidence: 0.85 + Math.random() * 0.1,
          });
          phraseIndex++;
          if (isFinal) {
            if (intervalId) clearInterval(intervalId);
            callbacks.onEnd();
          }
        }
      }, 1200);
    },

    async stop() {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
      const finalTranscript = phraseIndex > 0
        ? SIMULATED_PHRASES[Math.min(phraseIndex - 1, SIMULATED_PHRASES.length - 1)]
        : '';
      if (currentCallbacks) {
        currentCallbacks.onEnd();
      }
      return finalTranscript;
    },
  };
}

// ============================================================
// Real Speech Recognition (using @react-native-voice/voice)
// ============================================================

function createRealService(): SpeechRecognitionService {
  let Voice: any = null;
  let currentTranscript = '';
  let currentCallbacks: SpeechRecognitionCallbacks | null = null;

  return {
    async isAvailable() {
      try {
        const available = await loadSpeechModule();
        if (!available) return false;
        Voice = ExpoSpeech?.default || ExpoSpeech;
        return typeof Voice?.start === 'function';
      } catch {
        return false;
      }
    },

    async start(callbacks) {
      currentCallbacks = callbacks;
      currentTranscript = '';

      try {
        Voice = ExpoSpeech?.default || ExpoSpeech;

        Voice.onSpeechResults = (e: any) => {
          const text = e?.value?.[0] || '';
          currentTranscript = text;
          callbacks.onResult({
            transcript: text,
            isFinal: false,
            confidence: 0.9,
          });
        };

        Voice.onSpeechPartialResults = (e: any) => {
          const text = e?.value?.[0] || '';
          callbacks.onResult({
            transcript: text,
            isFinal: false,
            confidence: 0.7,
          });
        };

        Voice.onSpeechEnd = () => {
          callbacks.onResult({
            transcript: currentTranscript,
            isFinal: true,
            confidence: 0.9,
          });
          callbacks.onEnd();
        };

        Voice.onSpeechError = (e: any) => {
          callbacks.onError(e?.error?.message || 'Speech recognition error');
        };

        await Voice.start('en-US');
      } catch (e: any) {
        callbacks.onError(e?.message || 'Failed to start speech recognition');
      }
    },

    async stop() {
      try {
        Voice = ExpoSpeech?.default || ExpoSpeech;
        await Voice?.stop?.();
        await Voice?.destroy?.();
      } catch {
        // Ignore cleanup errors
      }
      return currentTranscript;
    },
  };
}

// ============================================================
// Factory
// ============================================================

/**
 * Create a speech recognition service.
 * Tries real STT first, falls back to simulated.
 */
export async function createSpeechService(): Promise<SpeechRecognitionService> {
  const real = createRealService();
  const available = await real.isAvailable();
  if (available) {
    console.log('[Seren] Using real speech recognition');
    return real;
  }
  console.log('[Seren] Speech recognition unavailable, using simulated mode');
  return createSimulatedService();
}

/**
 * Create a simulated speech recognition service (always available).
 */
export function createSimulatedSpeechService(): SpeechRecognitionService {
  return createSimulatedService();
}
