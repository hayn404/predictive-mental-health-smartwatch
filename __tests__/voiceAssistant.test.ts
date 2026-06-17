/**
 * voiceAssistant.ts — orchestrator tests
 * ==========================================
 * Mocks audioRecorder/nemotronAsrService/whisperService/ttsService (the
 * native-dependent leaves) so the state machine itself — the part with
 * actual logic worth testing — runs under plain Jest/Node, no phone,
 * simulator, or Expo dev server required.
 *
 * Uses explicit jest.mock(path, factory) rather than bare jest.mock(path)
 * automocking, so Jest never has to load the real expo-av/expo-speech/
 * expo-file-system-backed modules at all.
 */

import { CheckinAnalysis } from '@/services/ai/types';

jest.mock('@/services/ai/audioRecorder', () => ({
  createRecordingSession: jest.fn(),
  deleteAudioFile: jest.fn(),
}));
jest.mock('@/services/ai/nemotronAsrService', () => ({
  isNemotronConfigured: jest.fn(),
  checkNemotronHealth: jest.fn(),
  transcribeAudio: jest.fn(),
}));
jest.mock('@/services/ai/whisperService', () => ({
  isWhisperConfigured: jest.fn(),
  transcribeAudio: jest.fn(),
}));
jest.mock('@/services/ai/ttsService', () => ({
  speak: jest.fn(),
  stop: jest.fn(),
}));

import * as audioRecorder from '@/services/ai/audioRecorder';
import * as nemotron from '@/services/ai/nemotronAsrService';
import * as whisper from '@/services/ai/whisperService';
import * as tts from '@/services/ai/ttsService';
import { createVoiceAssistant, transcribeSpeech, VoiceAssistantState } from '@/services/ai/voiceAssistant';

function fakeRecorder(stopReturns: string | null = 'file://fake-clip.m4a') {
  return {
    isRecording: true,
    start: jest.fn().mockResolvedValue(undefined),
    stop: jest.fn().mockResolvedValue(stopReturns),
    getStatus: jest.fn(),
  };
}

/** Poll until a condition is true — used when a promise is deliberately held open (e.g. simulating "still speaking") so we can't just await it. */
async function waitUntil(predicate: () => boolean, maxTicks = 50): Promise<void> {
  for (let i = 0; i < maxTicks; i++) {
    if (predicate()) return;
    await new Promise((r) => setTimeout(r, 0));
  }
  throw new Error('waitUntil: condition not met in time');
}

function fakeAnalysis(overrides: Partial<CheckinAnalysis> = {}): CheckinAnalysis {
  return {
    id: 'checkin_test_1',
    timestamp: Date.now(),
    transcript: "I'm feeling good today",
    sentiment: 'positive',
    sentimentScore: 0.6,
    emotionScores: { joy: 0.6, sadness: 0, anxiety: 0, anger: 0, calm: 0.3, fear: 0, gratitude: 0, fatigue: 0 },
    keyInsights: ['Your words reflect a sense of joy'],
    themes: ['general'],
    emotionalIntensity: 0.5,
    empathyResponse: "That's wonderful to hear!",
    suggestedFollowUp: null,
    hrAtCheckin: 70,
    hrvAtCheckin: 55,
    stressAtCheckin: 20,
    ...overrides,
  };
}

beforeEach(() => {
  jest.clearAllMocks();
});

describe('transcribeSpeech (ASR routing)', () => {
  test('uses Nemotron when configured and healthy', async () => {
    (nemotron.isNemotronConfigured as jest.Mock).mockReturnValue(true);
    (nemotron.checkNemotronHealth as jest.Mock).mockResolvedValue(true);
    (nemotron.transcribeAudio as jest.Mock).mockResolvedValue('hello from nemotron');

    const result = await transcribeSpeech('file://clip.m4a');

    expect(result).toEqual({ text: 'hello from nemotron', provider: 'nemotron' });
    expect(whisper.transcribeAudio).not.toHaveBeenCalled();
  });

  test('falls back to Whisper when Nemotron is unreachable', async () => {
    (nemotron.isNemotronConfigured as jest.Mock).mockReturnValue(true);
    (nemotron.checkNemotronHealth as jest.Mock).mockResolvedValue(false); // relay down
    (whisper.isWhisperConfigured as jest.Mock).mockReturnValue(true);
    (whisper.transcribeAudio as jest.Mock).mockResolvedValue('hello from whisper');

    const result = await transcribeSpeech('file://clip.m4a');

    expect(result).toEqual({ text: 'hello from whisper', provider: 'whisper' });
    expect(nemotron.transcribeAudio).not.toHaveBeenCalled(); // never even tried — health check failed first
  });

  test('falls back to Whisper when Nemotron call itself fails', async () => {
    (nemotron.isNemotronConfigured as jest.Mock).mockReturnValue(true);
    (nemotron.checkNemotronHealth as jest.Mock).mockResolvedValue(true);
    (nemotron.transcribeAudio as jest.Mock).mockResolvedValue(null); // relay reachable but transcription failed
    (whisper.isWhisperConfigured as jest.Mock).mockReturnValue(true);
    (whisper.transcribeAudio as jest.Mock).mockResolvedValue('hello from whisper');

    const result = await transcribeSpeech('file://clip.m4a');

    expect(result).toEqual({ text: 'hello from whisper', provider: 'whisper' });
  });

  test('returns null when nothing is configured', async () => {
    (nemotron.isNemotronConfigured as jest.Mock).mockReturnValue(false);
    (whisper.isWhisperConfigured as jest.Mock).mockReturnValue(false);

    const result = await transcribeSpeech('file://clip.m4a');

    expect(result).toBeNull();
  });
});

describe('createVoiceAssistant (state machine)', () => {
  test('happy path: idle → recording → transcribing → analyzing → speaking → idle', async () => {
    const recorder = fakeRecorder();
    (audioRecorder.createRecordingSession as jest.Mock).mockResolvedValue(recorder);
    (nemotron.isNemotronConfigured as jest.Mock).mockReturnValue(true);
    (nemotron.checkNemotronHealth as jest.Mock).mockResolvedValue(true);
    (nemotron.transcribeAudio as jest.Mock).mockResolvedValue("I'm feeling good today");
    (tts.speak as jest.Mock).mockResolvedValue(undefined);

    const analysis = fakeAnalysis();
    const analyze = jest.fn().mockResolvedValue(analysis);
    const recordTurnMeta = jest.fn();
    const states: VoiceAssistantState[] = [];

    const va = createVoiceAssistant(
      { analyze, recordTurnMeta },
      { onStateChange: (s) => states.push(s) },
    );

    expect(va.getState()).toBe('idle');

    await va.start();
    expect(va.getState()).toBe('recording');
    expect(recorder.start).toHaveBeenCalledTimes(1);

    await va.stopRecording();

    expect(recorder.stop).toHaveBeenCalledTimes(1);
    expect(audioRecorder.deleteAudioFile).toHaveBeenCalledWith('file://fake-clip.m4a');
    expect(nemotron.transcribeAudio).toHaveBeenCalledWith('file://fake-clip.m4a');
    expect(analyze).toHaveBeenCalledWith("I'm feeling good today", {
      inputMode: 'voice',
      asrProvider: 'nemotron',
      ttsUsed: true,
    });
    expect(tts.speak).toHaveBeenCalledWith("That's wonderful to hear!");
    expect(recordTurnMeta).toHaveBeenCalledWith(
      'checkin_test_1',
      expect.objectContaining({ ttsUsed: true, latencyMs: expect.any(Number) }),
    );
    expect(va.getState()).toBe('idle');
    expect(states).toEqual(['recording', 'transcribing', 'analyzing', 'speaking', 'idle']);
  });

  test('interrupt(): stops speech and immediately starts recording the next turn', async () => {
    const firstRecorder = fakeRecorder();
    const secondRecorder = fakeRecorder('file://second-clip.m4a');
    (audioRecorder.createRecordingSession as jest.Mock)
      .mockResolvedValueOnce(firstRecorder)
      .mockResolvedValueOnce(secondRecorder);
    (nemotron.isNemotronConfigured as jest.Mock).mockReturnValue(true);
    (nemotron.checkNemotronHealth as jest.Mock).mockResolvedValue(true);
    (nemotron.transcribeAudio as jest.Mock).mockResolvedValue('turn one');

    // speak() resolves on its own timer in real life; here we just never let
    // it resolve during the test, to simulate "Seren is mid-sentence" when
    // interrupt() is called.
    let resolveSpeak: () => void = () => {};
    (tts.speak as jest.Mock).mockReturnValue(new Promise<void>((resolve) => { resolveSpeak = resolve; }));

    const analyze = jest.fn().mockResolvedValue(fakeAnalysis());
    const va = createVoiceAssistant({ analyze, recordTurnMeta: jest.fn() });

    await va.start();
    const turnPromise = va.stopRecording(); // NOT awaited — it can't resolve until speak() does, which we're holding open
    await waitUntil(() => va.getState() === 'speaking');

    await va.interrupt();

    expect(tts.stop).toHaveBeenCalledTimes(1);
    expect(audioRecorder.createRecordingSession).toHaveBeenCalledTimes(2); // started a second turn
    expect(secondRecorder.start).toHaveBeenCalledTimes(1);
    expect(va.getState()).toBe('recording');

    resolveSpeak(); // let the original (now-stale) stopRecording() call finish
    await turnPromise;

    // The original call's tail end checks getState() === 'speaking' before
    // forcing 'idle' — by now interrupt() has already moved us to
    // 'recording', so it must NOT stomp that back to 'idle'. This is the
    // exact race the comment in voiceAssistant.ts's stopRecording() guards
    // against.
    expect(va.getState()).toBe('recording');
  });

  test('no ASR available: surfaces onError and returns to idle (not stuck)', async () => {
    const recorder = fakeRecorder();
    (audioRecorder.createRecordingSession as jest.Mock).mockResolvedValue(recorder);
    (nemotron.isNemotronConfigured as jest.Mock).mockReturnValue(false);
    (whisper.isWhisperConfigured as jest.Mock).mockReturnValue(false);

    const onError = jest.fn();
    const analyze = jest.fn();
    const va = createVoiceAssistant({ analyze, recordTurnMeta: jest.fn() }, { onError });

    await va.start();
    await va.stopRecording();

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0].message).toMatch(/no ASR provider/i);
    expect(analyze).not.toHaveBeenCalled();
    expect(va.getState()).toBe('idle'); // doesn't get stuck in 'error'
  });

  test('no audio captured (recorder.stop() returns null): surfaces onError', async () => {
    const recorder = fakeRecorder(null);
    (audioRecorder.createRecordingSession as jest.Mock).mockResolvedValue(recorder);

    const onError = jest.fn();
    const va = createVoiceAssistant({ analyze: jest.fn(), recordTurnMeta: jest.fn() }, { onError });

    await va.start();
    await va.stopRecording();

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0].message).toMatch(/no audio captured/i);
  });

  test('dispose(): tears down an in-progress recording and stops any speech', async () => {
    const recorder = fakeRecorder();
    (audioRecorder.createRecordingSession as jest.Mock).mockResolvedValue(recorder);

    const va = createVoiceAssistant({ analyze: jest.fn(), recordTurnMeta: jest.fn() });
    await va.start();
    expect(va.getState()).toBe('recording');

    await va.dispose();

    expect(recorder.stop).toHaveBeenCalledTimes(1);
    expect(tts.stop).toHaveBeenCalledTimes(1);
    expect(va.getState()).toBe('idle');
  });
});
