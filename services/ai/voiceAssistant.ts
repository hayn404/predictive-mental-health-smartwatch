/**
 * Seren AI — Voice Assistant Orchestrator (the "duet")
 * ========================================================
 * Composes the existing pieces into one stateful loop:
 *
 *   record → transcribe (Nemotron-first, Whisper-fallback) → analyze → speak
 *                                                                         │
 *                                                  tap-to-interrupt ──────┘
 *                                                  (barge-in) → record
 *
 * Deliberately does NOT reimplement biometric-context-building or DB
 * persistence — `analyze` is injected by the caller (useVoiceAssistant() in
 * hooks/useHealthData.ts passes useWellness()'s performCheckin), so this
 * file stays a framework-agnostic state machine, same shape as
 * createRecordingSession() (audioRecorder.ts) and createSpeechService()
 * (speechRecognition.ts) already in this codebase.
 *
 * Barge-in scope (v1): tap-to-interrupt only. True hands-free barge-in
 * (the mic picking up the user talking over device-speaker TTS) needs echo
 * handling Expo Go doesn't give cleanly — without headphones the mic mostly
 * just hears Seren's own voice played back. interrupt() is built so a v2
 * amplitude-threshold auto-barge-in could plug in later without changing
 * this public API; it just isn't trustworthy enough to ship as automatic.
 */

import { CheckinAnalysis, CheckinMeta } from './types';
import {
  createRecordingSession,
  deleteAudioFile,
  RecordingSession,
} from './audioRecorder';
import {
  isNemotronConfigured,
  checkNemotronHealth,
  transcribeAudio as transcribeWithNemotron,
} from './nemotronAsrService';
import { isWhisperConfigured, transcribeAudio as transcribeWithWhisper } from './whisperService';
import { speak, stop as stopSpeaking } from './ttsService';

// ============================================================
// Types
// ============================================================

export type VoiceAssistantState =
  | 'idle'
  | 'recording'
  | 'transcribing'
  | 'analyzing'
  | 'speaking'
  | 'error';

export type AsrProvider = 'nemotron' | 'whisper';

export interface VoiceAssistantDeps {
  /**
   * Runs full analysis on a transcript (biometric cross-referencing, LLM
   * call, DB save) — pass useWellness()'s performCheckin so this file
   * doesn't need its own copy of that logic.
   */
  analyze: (transcript: string, meta: Partial<CheckinMeta>) => Promise<CheckinAnalysis>;
  /**
   * Patch a check-in's meta after the fact (used to record latencyMs/
   * ttsUsed once a turn finishes speaking — analyze() doesn't know that
   * yet when it first saves). Pass useWellness()'s recordVoiceTurnMeta.
   * Optional — omit it and latency just won't be tracked.
   */
  recordTurnMeta?: (checkinId: string, patch: Partial<CheckinMeta>) => void;
}

export interface VoiceAssistantCallbacks {
  onStateChange?: (state: VoiceAssistantState) => void;
  onTranscript?: (text: string, provider: AsrProvider) => void;
  onAnalysis?: (analysis: CheckinAnalysis) => void;
  onError?: (error: Error) => void;
  /** Real-time mic level (dB, -160 to 0) while recording — for waveform UI, same data audioRecorder.ts already exposes. */
  onMetering?: (db: number) => void;
}

export interface VoiceAssistant {
  /** Begin recording a new turn. No-op if not currently idle. */
  start: () => Promise<void>;
  /** Stop recording and run the rest of the pipeline (transcribe → analyze → speak). */
  stopRecording: () => Promise<void>;
  /**
   * Tap-to-interrupt: if Seren is currently speaking, stop the TTS playback
   * and immediately begin recording the next turn. No-op in any other state.
   */
  interrupt: () => Promise<void>;
  getState: () => VoiceAssistantState;
  /** Tear down any in-progress recording/speech (e.g. on unmount). */
  dispose: () => Promise<void>;
}

// ============================================================
// ASR routing — Nemotron-first, Whisper-fallback
// ============================================================
// Exported on its own so useCheckin()'s simpler record→transcribe→analyze
// flow (hooks/useHealthData.ts) can also benefit from Nemotron without
// going through the full duet state machine below.

export interface TranscriptionResult {
  text: string;
  provider: AsrProvider;
}

export async function transcribeSpeech(audioUri: string): Promise<TranscriptionResult | null> {
  if (isNemotronConfigured() && (await checkNemotronHealth())) {
    const text = await transcribeWithNemotron(audioUri);
    if (text) return { text, provider: 'nemotron' };
    console.warn('[Seren VoiceAssistant] Nemotron call failed, falling back to Whisper');
  }

  if (isWhisperConfigured()) {
    const text = await transcribeWithWhisper(audioUri);
    if (text) return { text, provider: 'whisper' };
  }

  return null;
}

// ============================================================
// Orchestrator
// ============================================================

export function createVoiceAssistant(
  deps: VoiceAssistantDeps,
  callbacks: VoiceAssistantCallbacks = {},
): VoiceAssistant {
  let state: VoiceAssistantState = 'idle';
  let recorder: RecordingSession | null = null;
  let turnStartedAt = 0;

  function setState(next: VoiceAssistantState) {
    state = next;
    callbacks.onStateChange?.(next);
  }

  function fail(error: unknown) {
    const err = error instanceof Error ? error : new Error(String(error));
    console.warn('[Seren VoiceAssistant] Error:', err);
    setState('error');
    callbacks.onError?.(err);
    setState('idle');
  }

  async function start(): Promise<void> {
    if (state !== 'idle') return; // already mid-turn; stopRecording()/interrupt() handle other states
    turnStartedAt = Date.now();
    try {
      const session = await createRecordingSession();
      if (callbacks.onMetering) {
        session.onMeteringUpdate = callbacks.onMetering;
      }
      await session.start();
      recorder = session;
      setState('recording');
    } catch (e) {
      fail(e);
    }
  }

  async function stopRecording(): Promise<void> {
    if (state !== 'recording' || !recorder) return;

    const activeRecorder = recorder;
    recorder = null;

    let audioUri: string | null = null;
    try {
      audioUri = await activeRecorder.stop();
    } catch (e) {
      fail(e);
      return;
    }

    if (!audioUri) {
      fail(new Error('No audio captured'));
      return;
    }

    setState('transcribing');
    let transcription: TranscriptionResult | null = null;
    try {
      transcription = await transcribeSpeech(audioUri);
    } finally {
      deleteAudioFile(audioUri);
    }

    if (!transcription) {
      fail(new Error('Transcription failed — no ASR provider configured/reachable'));
      return;
    }
    callbacks.onTranscript?.(transcription.text, transcription.provider);

    setState('analyzing');
    let analysis: CheckinAnalysis;
    try {
      analysis = await deps.analyze(transcription.text, {
        inputMode: 'voice',
        asrProvider: transcription.provider,
        ttsUsed: true,
      });
    } catch (e) {
      fail(e);
      return;
    }
    callbacks.onAnalysis?.(analysis);

    // "Latency" here is record-start → response-ready (ASR + analysis time),
    // not including TTS playback duration — playback length scales with how
    // much Seren has to say, not with how fast the pipeline is, so it isn't
    // a fair "is this snappy" signal the way time-to-first-word is.
    const latencyMs = Date.now() - turnStartedAt;
    deps.recordTurnMeta?.(analysis.id, { latencyMs, ttsUsed: true });

    setState('speaking');
    try {
      await speak(analysis.empathyResponse);
    } catch (e) {
      // A failed "talk back" shouldn't strand the user mid-conversation —
      // log it and return to idle rather than surfacing an error state.
      console.warn('[Seren VoiceAssistant] TTS failed:', e);
    }

    // interrupt() may have already kicked off the next recording while we
    // were speaking — don't stomp on that by forcing 'idle'.
    //
    // Routed through getState() rather than comparing the closured `state`
    // variable directly: TypeScript narrowed `state` to the literal
    // 'recording' from the guard near the top of this function, and doesn't
    // widen that narrowing back across the setState() calls in between (it
    // has no way to know setState mutates the outer variable) — so a direct
    // `state === 'speaking'` here is a false TS2367 "no overlap" error
    // despite being meaningful at runtime. getState()'s declared return
    // type isn't narrowed, so this comparison type-checks correctly.
    if (getState() === 'speaking') setState('idle');
  }

  async function interrupt(): Promise<void> {
    if (state !== 'speaking') return;
    stopSpeaking();
    setState('idle');
    await start();
  }

  function getState(): VoiceAssistantState {
    return state;
  }

  async function dispose(): Promise<void> {
    stopSpeaking();
    if (recorder) {
      try {
        await recorder.stop();
      } catch {
        // ignore — tearing down anyway
      }
      recorder = null;
    }
    setState('idle');
  }

  return { start, stopRecording, interrupt, getState, dispose };
}
