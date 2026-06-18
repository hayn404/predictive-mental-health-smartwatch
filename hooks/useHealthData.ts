/**
 * Bridge hooks — maps WellnessProvider (real AI) data
 * into the shapes the existing screens already consume.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWellness } from './useWellness';
import {
  getMonthlySleepData as getDefaultMonthlySleep,
  getHRVTrend as getDefaultHRVTrend,
  HealthSnapshot,
  DailyEntry,
  WatchStatus,
  Recommendation as MockRecommendation,
  CheckinEntry,
} from '@/services/mockData';
import {
  createRecordingSession,
  RecordingSession,
  deleteAudioFile,
} from '@/services/ai/audioRecorder';
import {
  isWhisperConfigured,
} from '@/services/ai/whisperService';
import { isNemotronConfigured } from '@/services/ai/nemotronAsrService';
import {
  createVoiceAssistant,
  transcribeSpeech,
  VoiceAssistantState,
} from '@/services/ai/voiceAssistant';

export function useHealthData() {
  const w = useWellness();

  const health: HealthSnapshot = {
    stressLevel: w.stress.stressScore,
    heartRate: w.heartRate,
    hrv: w.hrv,
    sleepQuality: w.lastSleep?.qualityScore ?? 82,
    anxietyIndex: w.anxiety.anxietyIndex,
    timestamp: new Date(w.stress.timestamp),
  };

  const weeklyData: DailyEntry[] = w.weeklyStress.map((s, i) => ({
    date: s.date,
    stress: s.value,
    anxiety: Math.round(s.value * 0.8),
    sleep: w.weeklySleep[i]?.value ?? 75,
    hrv: w.weeklyHrv[i]?.value ?? 55,
    heartRate: 60 + Math.round(s.value * 0.2),
    mood: s.value > 60 ? 'Stressed' : s.value > 40 ? 'Okay' : 'Good',
  }));

  // Use real data from WellnessProvider when available, fall back to mock defaults
  const monthlySleep = w.monthlySleepGrid.length > 0 ? w.monthlySleepGrid : getDefaultMonthlySleep();
  const hrvTrend = w.hrvTrendData.length > 0 ? w.hrvTrendData : getDefaultHRVTrend();

  const watchStatus: WatchStatus = {
    connected: w.watchConnected,
    model: 'Smartwatch',
    lastSync: w.lastSyncTime,
    batteryLevel: 85,
  };

  const recommendations: MockRecommendation[] = w.recommendations.map(r => ({
    id: r.id,
    title: r.title,
    description: r.description,
    category: r.category as MockRecommendation['category'],
    duration: `${r.durationMin}m`,
    aiReason: r.triggerReason,
  }));

  const checkinHistory: CheckinEntry[] = w.checkinHistory.map(c => ({
    id: c.id,
    timestamp: new Date(c.timestamp),
    transcript: c.transcript,
    sentiment: c.sentiment === 'distressed' ? 'concerned' : c.sentiment,
    keyInsights: c.keyInsights,
  }));

  // Location & Sunlight
  const locationDiversity = w.locationDiversity;
  const sunlightExposure = w.sunlightExposure;
  const weeklyLocationDiversity = w.weeklyLocationDiversity;
  const weeklySunlight = w.weeklySunlight;

  return {
    health,
    weeklyData,
    monthlySleep,
    hrvTrend,
    watchStatus,
    recommendations,
    checkinHistory,
    emotionalMaturity: w.emotionalMaturity,
    locationDiversity,
    sunlightExposure,
    weeklyLocationDiversity,
    weeklySunlight,
    isLive: w.isLive,
    toggleLive: w.toggleLive,
  };
}

export function useCheckin() {
  const w = useWellness();
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [result, setResult] = useState<null | { sentiment: string; insights: string[]; empathyResponse?: string; followUp?: string | null }>(null);
  const FLAT_WAVE: number[] = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4];
  const [waveAmplitudes, setWaveAmplitudes] = useState<number[]>(FLAT_WAVE);

  const recorderRef = useRef<RecordingSession | null>(null);
  const durationInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // Convert metering dB (-160..0) to visual amplitude (4..40)
  const dbToAmplitude = (db: number): number => {
    const normalized = (db + 160) / 160; // 0..1
    return 4 + normalized * 36;
  };

  // Animate wave from real microphone metering while recording
  useEffect(() => {
    if (isRecording) {
      // Use real metering data when available, random as fallback
      const interval = setInterval(() => {
        setWaveAmplitudes(prev => {
          const out: number[] = [];
          for (let i = 0; i < 20; i++) out.push(Math.random() * 36 + 4);
          return out;
        });
      }, 100);
      return () => clearInterval(interval);
    } else {
      setWaveAmplitudes(FLAT_WAVE);
    }
  }, [isRecording]);

  const startRecording = useCallback(async () => {
    setTranscript('');
    setResult(null);
    setRecordingDuration(0);

    try {
      const session = await createRecordingSession();

      // Set up real metering callback for waveform
      session.onMeteringUpdate = (db: number) => {
        setWaveAmplitudes(prev => {
          const next = [...prev.slice(1), dbToAmplitude(db)];
          return next;
        });
      };

      await session.start();
      recorderRef.current = session;
      setIsRecording(true);

      // Track duration
      const startTime = Date.now();
      durationInterval.current = setInterval(() => {
        setRecordingDuration(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    } catch (e) {
      console.warn('[Seren] Failed to start recording:', e);
    }
  }, []);

  const stopAndAnalyze = useCallback(async () => {
    setIsRecording(false);
    if (durationInterval.current) {
      clearInterval(durationInterval.current);
      durationInterval.current = null;
    }

    const recorder = recorderRef.current;
    if (!recorder) return;

    // Step 1: Stop recording, get audio file
    const audioUri = await recorder.stop();
    recorderRef.current = null;

    if (!audioUri) {
      console.warn('[Seren] No audio file produced');
      return;
    }

    // Step 2: Transcribe — Nemotron first (if configured/reachable), Whisper fallback
    setIsTranscribing(true);

    const transcription = await transcribeSpeech(audioUri);

    // Clean up the audio file
    deleteAudioFile(audioUri);

    if (!transcription) {
      setIsTranscribing(false);
      const anyAsrConfigured = isWhisperConfigured() || isNemotronConfigured();
      setTranscript(anyAsrConfigured
        ? '[Transcription failed — check your API key(s) and network connection, then try again]'
        : '[No speech-to-text configured — add a Whisper API key in services/ai/aiConfig.ts, or set up server/nemotron-relay/]');
      return;
    }

    const finalTranscript = transcription.text;
    setTranscript(finalTranscript);
    setIsTranscribing(false);

    // Step 3: Analyze with LLM
    setIsAnalyzing(true);
    const [analysis] = await Promise.all([
      w.performCheckin(finalTranscript, { inputMode: 'voice', asrProvider: transcription.provider }),
      new Promise(resolve => setTimeout(resolve, 1500)),
    ]);

    setIsAnalyzing(false);
    setResult({
      sentiment: analysis.sentiment === 'distressed' ? 'concerned' : analysis.sentiment,
      insights: analysis.keyInsights,
      empathyResponse: analysis.empathyResponse,
      followUp: analysis.suggestedFollowUp,
    });
  }, [w.performCheckin]);

  const submitText = useCallback(async (text: string) => {
    if (!text.trim()) return;
    setTranscript(text);
    setIsAnalyzing(true);

    const [analysis] = await Promise.all([
      w.performCheckin(text.trim()),
      new Promise(resolve => setTimeout(resolve, 1500)),
    ]);

    setIsAnalyzing(false);
    setResult({
      sentiment: analysis.sentiment === 'distressed' ? 'concerned' : analysis.sentiment,
      insights: analysis.keyInsights,
      empathyResponse: analysis.empathyResponse,
      followUp: analysis.suggestedFollowUp,
    });
  }, [w.performCheckin]);

  const reset = useCallback(() => {
    setTranscript('');
    setResult(null);
    setIsRecording(false);
    setIsTranscribing(false);
    setIsAnalyzing(false);
    setRecordingDuration(0);
  }, []);

  return {
    isRecording, isTranscribing, transcript, isAnalyzing, result,
    waveAmplitudes, recordingDuration,
    startRecording, stopAndAnalyze, submitText, reset,
  };
}

// ============================================================
// useVoiceAssistant — the full duet (record → ASR → analyze → speak)
// ============================================================
// A separate, opt-in hook rather than a replacement for useCheckin() above,
// so the existing simple check-in flow keeps working unchanged. Pass this
// to a UI that wants Seren to talk back (see voiceAssistant.ts for the
// state machine and barge-in scope notes).

export interface VoiceAssistantUIResult {
  sentiment: string;
  insights: string[];
  empathyResponse: string;
  followUp: string | null;
}

export function useVoiceAssistant() {
  const w = useWellness();
  const [state, setVAState] = useState<VoiceAssistantState>('idle');
  const [transcript, setTranscript] = useState('');
  const [result, setResult] = useState<VoiceAssistantUIResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(0);

  const FLAT_WAVE: number[] = Array(20).fill(4);
  const [waveAmplitudes, setWaveAmplitudes] = useState<number[]>(FLAT_WAVE);

  const durationInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  const dbToAmplitude = (db: number): number => {
    const normalized = (db + 160) / 160;
    return 4 + normalized * 36;
  };

  // Lazy-initialized once per mount (not on every render) — the standard
  // ref pattern for "construct an object once, even though the constructor
  // expression itself would otherwise re-run on every render since it's a
  // plain function call, not wrapped in useState's lazy initializer form).
  const assistantRef = useRef<ReturnType<typeof createVoiceAssistant> | null>(null);
  if (!assistantRef.current) {
    assistantRef.current = createVoiceAssistant(
      {
        analyze: w.performCheckin,
        recordTurnMeta: w.recordVoiceTurnMeta,
      },
      {
        onStateChange: (next) => {
          setVAState(next);
          if (next === 'recording') {
            setTranscript('');
            setResult(null);
            setError(null);
          }
        },
        onTranscript: (text) => setTranscript(text),
        onAnalysis: (analysis) => setResult({
          sentiment: analysis.sentiment === 'distressed' ? 'concerned' : analysis.sentiment,
          insights: analysis.keyInsights,
          empathyResponse: analysis.empathyResponse,
          followUp: analysis.suggestedFollowUp,
        }),
        onError: (e) => setError(e.message),
        onMetering: (db) => {
          setWaveAmplitudes(prev => [...prev.slice(1), dbToAmplitude(db)]);
        },
      },
    );
  }

  useEffect(() => {
    return () => { assistantRef.current!.dispose(); };
  }, []);

  useEffect(() => {
    if (state === 'recording') {
      const startTime = Date.now();
      durationInterval.current = setInterval(() => {
        setRecordingDuration(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    } else {
      if (durationInterval.current) clearInterval(durationInterval.current);
      durationInterval.current = null;
      setRecordingDuration(0);
      setWaveAmplitudes(FLAT_WAVE);
    }
    return () => { if (durationInterval.current) clearInterval(durationInterval.current); };
  }, [state === 'recording']);

  const start = useCallback(() => assistantRef.current!.start(), []);
  const stopRecording = useCallback(() => assistantRef.current!.stopRecording(), []);
  /** While Seren is speaking, tapping the mic interrupts her and starts listening again. */
  const interrupt = useCallback(() => assistantRef.current!.interrupt(), []);
  const reset = useCallback(() => {
    setTranscript('');
    setResult(null);
    setError(null);
  }, []);

  return {
    state,
    transcript,
    result,
    error,
    waveAmplitudes,
    recordingDuration,
    isRecording: state === 'recording',
    isTranscribing: state === 'transcribing',
    isAnalyzing: state === 'analyzing',
    isSpeaking: state === 'speaking',
    start,
    stopRecording,
    interrupt,
    reset,
  };
}
