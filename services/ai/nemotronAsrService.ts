/**
 * Seren AI — Nemotron ASR Service (via relay)
 * ===============================================
 * Browser/RN JS can't speak gRPC to NVIDIA's NIM/Riva ASR endpoint directly,
 * so this talks HTTP to server/nemotron-relay/ — a small Node service that
 * does the actual gRPC call (see server/nemotron-relay/README.md for setup).
 *
 * Same shape as whisperService.ts (configure*, isXConfigured, transcribeAudio)
 * plus what Whisper doesn't need: a cached health check, since "is the relay
 * even reachable right now" is a real question a REST API doesn't have.
 *
 * This file is provider-only — it does NOT fall back to Whisper itself.
 * The Nemotron-first/Whisper-fallback decision lives in voiceAssistant.ts's
 * transcribeSpeech(), so each ASR provider file stays single-purpose and
 * useCheckin() (hooks/useHealthData.ts) can also use that combinator
 * directly without going through the full voice-duet orchestrator.
 */

import * as FileSystem from 'expo-file-system/legacy';

// ============================================================
// Configuration
// ============================================================

export interface NemotronConfig {
  /** Base URL of server/nemotron-relay/, e.g. https://your-relay.example.com */
  relayUrl: string;
  /** Shared secret sent as `Authorization: Bearer <key>` — matches the relay's RELAY_SHARED_SECRET */
  apiKey?: string;
  language?: string; // BCP-47, e.g. 'en-US' — passed through to the relay
  timeoutMs?: number;
}

let nemotronConfig: NemotronConfig | null = null;

// Health check cache — avoids eating a timeout on every transcription
// attempt when the relay is down or not deployed.
const HEALTH_CACHE_TTL_MS = 60_000;
let cachedHealth: { ok: boolean; checkedAt: number } | null = null;

// ============================================================
// Public API
// ============================================================

export function configureNemotron(
  relayUrl: string,
  apiKey?: string,
  language?: string,
): void {
  nemotronConfig = {
    relayUrl: relayUrl.replace(/\/+$/, ''), // strip trailing slash
    apiKey,
    language,
    timeoutMs: 15_000,
  };
  cachedHealth = null; // config changed — don't trust the old cache
}

export function isNemotronConfigured(): boolean {
  return nemotronConfig !== null && nemotronConfig.relayUrl.length > 0;
}

export function getNemotronConfig(): Omit<NemotronConfig, 'apiKey'> | null {
  if (!nemotronConfig) return null;
  const { apiKey, ...rest } = nemotronConfig;
  return rest;
}

/**
 * Is the relay reachable right now? Cached for HEALTH_CACHE_TTL_MS so the
 * common case (relay configured but a Whisper-only dev environment, or a
 * temporarily-down relay) doesn't add a timeout to every single check-in.
 */
export async function checkNemotronHealth(force = false): Promise<boolean> {
  if (!isNemotronConfigured()) return false;

  if (!force && cachedHealth && Date.now() - cachedHealth.checkedAt < HEALTH_CACHE_TTL_MS) {
    return cachedHealth.ok;
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2_500);
    const response = await fetch(`${nemotronConfig!.relayUrl}/health`, {
      method: 'GET',
      signal: controller.signal,
    });
    clearTimeout(timeout);

    const ok = response.ok;
    cachedHealth = { ok, checkedAt: Date.now() };
    return ok;
  } catch {
    cachedHealth = { ok: false, checkedAt: Date.now() };
    return false;
  }
}

/**
 * Transcribe an audio file via the Nemotron relay.
 * Mirrors whisperService.ts's transcribeAudio signature/behavior exactly
 * (same input, same `string | null` contract) so callers can swap providers
 * with no other code changes.
 */
export async function transcribeAudio(audioUri: string): Promise<string | null> {
  if (!nemotronConfig) {
    console.warn('[Seren Nemotron] Not configured');
    return null;
  }

  try {
    const { relayUrl, apiKey, language, timeoutMs } = nemotronConfig;

    const headers: Record<string, string> = {};
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

    const parameters: Record<string, string> = {};
    if (language) parameters.language = language;

    // FileSystem.uploadAsync — same multipart approach as whisperService.ts,
    // since it's the most reliable way to send files from RN. It has no
    // built-in timeout option, so race it against one manually — a slow/dead
    // relay shouldn't hang a check-in indefinitely.
    const response = await raceWithTimeout(
      FileSystem.uploadAsync(`${relayUrl}/transcribe`, audioUri, {
        httpMethod: 'POST',
        uploadType: FileSystem.FileSystemUploadType.MULTIPART,
        fieldName: 'file',
        mimeType: 'audio/mp4', // matches audioRecorder.ts's .m4a output
        headers,
        parameters,
      }),
      timeoutMs ?? 15_000,
    );

    if (response.status !== 200) {
      console.warn('[Seren Nemotron] Relay error:', response.status, response.body);
      return null;
    }

    const data = JSON.parse(response.body);
    const text = data.text?.trim();

    if (!text) {
      console.warn('[Seren Nemotron] Empty transcription');
      return null;
    }

    console.log('[Seren Nemotron] Transcribed:', text.substring(0, 80) + '...');
    return text;
  } catch (error) {
    console.warn('[Seren Nemotron] Transcription failed:', error);
    return null;
  }
}

// ============================================================
// Utilities
// ============================================================

function raceWithTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Nemotron relay timed out after ${ms}ms`)), ms);
    promise.then(
      (value) => { clearTimeout(timer); resolve(value); },
      (err) => { clearTimeout(timer); reject(err); },
    );
  });
}
