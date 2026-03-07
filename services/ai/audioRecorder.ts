/**
 * Seren AI — Audio Recorder (expo-av)
 * =====================================
 * Records real audio from the device microphone using expo-av.
 * Outputs an audio file (m4a) that can be sent to Whisper API for transcription.
 *
 * Works in Expo Go — no native build required.
 */

import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system/legacy';

// ============================================================
// Types
// ============================================================

export interface RecordingSession {
  /** Start recording. Returns immediately. */
  start(): Promise<void>;
  /** Stop recording and return the audio file URI. */
  stop(): Promise<string | null>;
  /** Get current recording status */
  getStatus(): Promise<RecordingStatus>;
  /** Whether currently recording */
  isRecording: boolean;
  /** Get real-time metering levels (for waveform visualization) */
  onMeteringUpdate?: (db: number) => void;
}

export interface RecordingStatus {
  isRecording: boolean;
  durationMs: number;
  meteringDb: number; // -160 to 0, where 0 is loudest
}

// ============================================================
// Recording Settings — optimized for speech
// ============================================================

const RECORDING_OPTIONS: Audio.RecordingOptions = {
  android: {
    extension: '.m4a',
    outputFormat: Audio.AndroidOutputFormat.MPEG_4,
    audioEncoder: Audio.AndroidAudioEncoder.AAC,
    sampleRate: 16000, // Whisper optimal sample rate
    numberOfChannels: 1, // Mono — speech doesn't need stereo
    bitRate: 64000,
  },
  ios: {
    extension: '.m4a',
    audioQuality: Audio.IOSAudioQuality.MEDIUM,
    sampleRate: 16000,
    numberOfChannels: 1,
    bitRate: 64000,
    outputFormat: Audio.IOSOutputFormat.MPEG4AAC,
  },
  web: {
    mimeType: 'audio/webm',
    bitsPerSecond: 64000,
  },
};

// ============================================================
// Create a recording session
// ============================================================

export async function createRecordingSession(): Promise<RecordingSession> {
  let recording: Audio.Recording | null = null;
  let _isRecording = false;
  let meteringInterval: ReturnType<typeof setInterval> | null = null;
  let _onMeteringUpdate: ((db: number) => void) | undefined;

  // Request microphone permissions
  const { granted } = await Audio.requestPermissionsAsync();
  if (!granted) {
    throw new Error('Microphone permission not granted');
  }

  // Set audio mode for recording
  await Audio.setAudioModeAsync({
    allowsRecordingIOS: true,
    playsInSilentModeIOS: true,
  });

  const session: RecordingSession = {
    get isRecording() {
      return _isRecording;
    },

    set onMeteringUpdate(cb: ((db: number) => void) | undefined) {
      _onMeteringUpdate = cb;
    },

    get onMeteringUpdate() {
      return _onMeteringUpdate;
    },

    async start() {
      if (_isRecording) return;

      recording = new Audio.Recording();
      await recording.prepareToRecordAsync({
        ...RECORDING_OPTIONS,
        isMeteringEnabled: true,
      });
      await recording.startAsync();
      _isRecording = true;

      // Poll metering for waveform visualization
      meteringInterval = setInterval(async () => {
        if (!recording || !_isRecording) return;
        try {
          const status = await recording.getStatusAsync();
          if (status.isRecording && status.metering !== undefined) {
            _onMeteringUpdate?.(status.metering);
          }
        } catch {
          // Ignore status errors during recording
        }
      }, 100);
    },

    async stop(): Promise<string | null> {
      if (!recording || !_isRecording) return null;

      _isRecording = false;
      if (meteringInterval) {
        clearInterval(meteringInterval);
        meteringInterval = null;
      }

      try {
        await recording.stopAndUnloadAsync();
        const uri = recording.getURI();
        recording = null;

        // Reset audio mode
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: false,
        });

        return uri;
      } catch (e) {
        console.warn('[Seren] Recording stop error:', e);
        recording = null;
        return null;
      }
    },

    async getStatus(): Promise<RecordingStatus> {
      if (!recording) {
        return { isRecording: false, durationMs: 0, meteringDb: -160 };
      }
      try {
        const status = await recording.getStatusAsync();
        return {
          isRecording: status.isRecording,
          durationMs: status.durationMillis,
          meteringDb: status.metering ?? -160,
        };
      } catch {
        return { isRecording: false, durationMs: 0, meteringDb: -160 };
      }
    },
  };

  return session;
}

/**
 * Read an audio file as base64 (for sending to Whisper API).
 */
export async function readAudioAsBase64(uri: string): Promise<string> {
  return FileSystem.readAsStringAsync(uri, {
    encoding: FileSystem.EncodingType.Base64,
  });
}

/**
 * Delete a temporary audio file after transcription.
 */
export async function deleteAudioFile(uri: string): Promise<void> {
  try {
    await FileSystem.deleteAsync(uri, { idempotent: true });
  } catch {
    // Ignore cleanup errors
  }
}
