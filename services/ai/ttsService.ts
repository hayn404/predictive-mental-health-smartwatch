/**
 * Seren AI — Text-to-Speech Service
 * ====================================
 * The "talk back" half of the voice duet (see voiceAssistant.ts).
 *
 * v1 (default): wraps expo-speech — already a dependency, zero extra infra,
 * works offline, sounds robotic but is instant and free.
 *
 * v2 (optional): a remote TTS hook (ElevenLabs-style, or NVIDIA NIM/Riva TTS
 * via the same relay used for nemotronAsrService) for higher-quality voice.
 * Not wired up by default — configureRemoteTTS() must be called explicitly.
 *
 * Design note: speak() is Promise-based so voiceAssistant.ts's state machine
 * can `await` a turn finishing, and stop() is synchronous and safe to call
 * at any time so it can be used for tap-to-interrupt (barge-in).
 */

import * as Speech from 'expo-speech';

// ============================================================
// Types
// ============================================================

export interface TTSOptions {
  rate?: number;     // 0.1 - 2.0, default 1.0
  pitch?: number;     // 0.5 - 2.0, default 1.0
  voice?: string;     // platform voice identifier, optional
  language?: string;  // e.g. 'en-US'
}

export interface RemoteTTSConfig {
  provider: 'elevenlabs' | 'nim';
  apiKey: string;
  baseUrl?: string;   // defaults per provider
  voiceId?: string;
}

const DEFAULT_OPTIONS: TTSOptions = {
  rate: 1.0,
  pitch: 1.0,
  language: 'en-US',
};

let defaultOptions: TTSOptions = { ...DEFAULT_OPTIONS };
let remoteConfig: RemoteTTSConfig | null = null;
let _isSpeaking = false;

// expo-av Sound instance for remote-audio playback (lazily imported — only
// needed if remote TTS is actually configured, to avoid pulling expo-av into
// this module's load path for the common device-TTS-only case).
let remoteSound: any = null;

// ============================================================
// Configuration
// ============================================================

/** Set default rate/pitch/voice/language for all speak() calls. */
export function configureTTS(options: TTSOptions): void {
  defaultOptions = { ...DEFAULT_OPTIONS, ...options };
}

/**
 * Configure a remote TTS provider (ElevenLabs or NVIDIA NIM/Riva TTS).
 * Optional — when unset, speak() always uses the on-device voice.
 */
export function configureRemoteTTS(config: RemoteTTSConfig): void {
  remoteConfig = config;
}

export function isRemoteTTSConfigured(): boolean {
  return remoteConfig !== null && remoteConfig.apiKey.length > 0;
}

// ============================================================
// Public API
// ============================================================

/**
 * Speak text aloud. Resolves when playback finishes naturally, or when
 * stop() is called early (does NOT reject on interrupt — an interrupted
 * turn isn't an error, it's a barge-in).
 *
 * Tries the remote provider first when configured (better quality), and
 * falls back to the on-device voice on any remote failure — same
 * fail-soft shape as the ASR fallback in voiceAssistant.ts.
 */
export async function speak(text: string, opts?: TTSOptions): Promise<void> {
  const trimmed = text?.trim();
  if (!trimmed) return;

  if (isRemoteTTSConfigured()) {
    try {
      await speakRemote(trimmed);
      return;
    } catch (e) {
      console.warn('[Seren TTS] Remote TTS failed, falling back to device voice:', e);
      // fall through to device TTS below
    }
  }

  await speakDevice(trimmed, opts);
}

/** Stop any in-progress speech immediately. Safe to call when idle. */
export function stop(): void {
  _isSpeaking = false;
  try {
    Speech.stop();
  } catch {
    // ignore — nothing was speaking
  }
  if (remoteSound) {
    try {
      remoteSound.stopAsync?.();
    } catch {
      // ignore
    }
  }
}

export function isSpeaking(): boolean {
  return _isSpeaking;
}

// ============================================================
// Device TTS (expo-speech)
// ============================================================

function speakDevice(text: string, opts?: TTSOptions): Promise<void> {
  const merged = { ...defaultOptions, ...opts };

  return new Promise<void>((resolve) => {
    _isSpeaking = true;
    Speech.speak(text, {
      rate: merged.rate,
      pitch: merged.pitch,
      voice: merged.voice,
      language: merged.language,
      onDone: () => {
        _isSpeaking = false;
        resolve();
      },
      onStopped: () => {
        _isSpeaking = false;
        resolve(); // interrupted (barge-in) — not an error
      },
      onError: (error) => {
        _isSpeaking = false;
        console.warn('[Seren TTS] Device speech error:', error);
        resolve(); // never reject — a failed "talk back" shouldn't break the duet
      },
    });
  });
}

// ============================================================
// Remote TTS (optional — ElevenLabs / NIM Riva TTS)
// ============================================================

const REMOTE_PRESETS: Record<string, string> = {
  elevenlabs: 'https://api.elevenlabs.io/v1',
  // If you extend server/nemotron-relay with a POST /speak endpoint backed
  // by Riva TTS, point baseUrl at that relay instead of NIM directly.
  nim: '',
};

async function speakRemote(text: string): Promise<void> {
  if (!remoteConfig) throw new Error('Remote TTS not configured');

  const baseUrl = remoteConfig.baseUrl || REMOTE_PRESETS[remoteConfig.provider];
  if (!baseUrl) throw new Error(`No baseUrl configured for provider "${remoteConfig.provider}"`);

  let audioBase64: string;

  if (remoteConfig.provider === 'elevenlabs') {
    const voiceId = remoteConfig.voiceId || '21m00Tcm4TlvDq8ikWAM'; // ElevenLabs default voice
    const response = await fetch(`${baseUrl}/text-to-speech/${voiceId}`, {
      method: 'POST',
      headers: {
        'xi-api-key': remoteConfig.apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_turbo_v2_5',
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      }),
    });
    if (!response.ok) {
      throw new Error(`ElevenLabs TTS failed: ${response.status}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    audioBase64 = arrayBufferToBase64(arrayBuffer);
  } else {
    // 'nim' — expects the relay's POST /speak contract:
    // { text } -> { audioBase64, mimeType }
    const response = await fetch(`${baseUrl}/speak`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(remoteConfig.apiKey ? { Authorization: `Bearer ${remoteConfig.apiKey}` } : {}),
      },
      body: JSON.stringify({ text, voiceId: remoteConfig.voiceId }),
    });
    if (!response.ok) {
      throw new Error(`NIM relay TTS failed: ${response.status}`);
    }
    const data = await response.json();
    audioBase64 = data.audioBase64;
  }

  await playBase64Audio(audioBase64);
}

async function playBase64Audio(base64: string): Promise<void> {
  // Lazy import — only pulled in when remote TTS is actually used.
  const { Audio } = await import('expo-av');

  _isSpeaking = true;
  const { sound } = await Audio.Sound.createAsync(
    { uri: `data:audio/mpeg;base64,${base64}` },
    { shouldPlay: true },
  );
  remoteSound = sound;

  return new Promise<void>((resolve) => {
    sound.setOnPlaybackStatusUpdate((status: any) => {
      if (status.didJustFinish || !status.isLoaded) {
        _isSpeaking = false;
        sound.unloadAsync().catch(() => {});
        remoteSound = null;
        resolve();
      }
    });
  });
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  // btoa is available in the Hermes/RN JS runtime via a polyfill in Expo SDK 54+.
  return btoa(binary);
}
