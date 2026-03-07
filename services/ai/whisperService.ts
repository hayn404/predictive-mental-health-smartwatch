/**
 * Seren AI — Whisper Speech-to-Text Service
 * ============================================
 * Sends recorded audio to OpenAI's Whisper API (or compatible)
 * for real transcription. No hardcoded phrases — real speech understanding.
 *
 * Supports:
 * - OpenAI Whisper API (whisper-1)
 * - Groq Whisper (whisper-large-v3-turbo) — faster, free tier
 * - Any OpenAI-compatible STT endpoint
 *
 * Privacy: Audio is sent over HTTPS and is NOT stored by the API
 * when using Groq or when OpenAI's data retention is disabled.
 */

import * as FileSystem from 'expo-file-system/legacy';

// ============================================================
// Configuration
// ============================================================

export interface WhisperConfig {
  provider: 'openai' | 'groq' | 'custom';
  apiKey: string;
  baseUrl: string;
  model: string;
  language?: string; // ISO 639-1, e.g. 'en', 'ar'
}

const WHISPER_PRESETS: Record<string, Omit<WhisperConfig, 'apiKey'>> = {
  openai: {
    provider: 'openai',
    baseUrl: 'https://api.openai.com/v1',
    model: 'whisper-1',
  },
  groq: {
    provider: 'groq',
    baseUrl: 'https://api.groq.com/openai/v1',
    model: 'whisper-large-v3-turbo',
  },
};

let whisperConfig: WhisperConfig | null = null;

// ============================================================
// Public API
// ============================================================

/**
 * Configure Whisper using a preset provider.
 */
export function configureWhisper(
  provider: 'openai' | 'groq',
  apiKey: string,
  language?: string,
): void {
  const preset = WHISPER_PRESETS[provider];
  whisperConfig = { ...preset, apiKey, language };
}

/**
 * Configure Whisper with a full custom config.
 */
export function configureWhisperFull(config: WhisperConfig): void {
  whisperConfig = config;
}

/**
 * Check if Whisper is configured and ready.
 */
export function isWhisperConfigured(): boolean {
  return whisperConfig !== null && whisperConfig.apiKey.length > 0;
}

/**
 * Get current Whisper config (without API key).
 */
export function getWhisperConfig(): Omit<WhisperConfig, 'apiKey'> | null {
  if (!whisperConfig) return null;
  const { apiKey, ...rest } = whisperConfig;
  return rest;
}

/**
 * Transcribe an audio file using the Whisper API.
 *
 * @param audioUri - Local file URI of the recorded audio (m4a)
 * @returns Transcribed text, or null if transcription failed
 */
export async function transcribeAudio(audioUri: string): Promise<string | null> {
  if (!whisperConfig) {
    console.warn('[Seren Whisper] Not configured');
    return null;
  }

  try {
    const { baseUrl, apiKey, model, language } = whisperConfig;

    // Use FileSystem.uploadAsync for multipart form upload
    // This is the most reliable way to send files in React Native
    const uploadUrl = `${baseUrl}/audio/transcriptions`;

    const headers: Record<string, string> = {
      Authorization: `Bearer ${apiKey}`,
    };

    // Build multipart form fields
    const parameters: Record<string, string> = {
      model,
      response_format: 'json',
    };

    if (language) {
      parameters.language = language;
    }

    // Add prompt to improve accuracy for mental health context
    parameters.prompt =
      'This is a mental health check-in. The speaker is describing how they feel, ' +
      'their emotions, stress levels, sleep quality, and general wellbeing.';

    const response = await FileSystem.uploadAsync(uploadUrl, audioUri, {
      httpMethod: 'POST',
      uploadType: FileSystem.FileSystemUploadType.MULTIPART,
      fieldName: 'file',
      mimeType: 'audio/mp4',
      headers,
      parameters,
    });

    if (response.status !== 200) {
      console.warn('[Seren Whisper] API error:', response.status, response.body);
      return null;
    }

    const data = JSON.parse(response.body);
    const text = data.text?.trim();

    if (!text) {
      console.warn('[Seren Whisper] Empty transcription');
      return null;
    }

    console.log('[Seren Whisper] Transcribed:', text.substring(0, 80) + '...');
    return text;
  } catch (error) {
    console.warn('[Seren Whisper] Transcription failed:', error);
    return null;
  }
}
