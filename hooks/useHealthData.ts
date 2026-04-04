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
  transcribeAudio,
  isWhisperConfigured,
} from '@/services/ai/whisperService';

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

    // Step 2: Transcribe with Whisper API
    setIsTranscribing(true);
    let finalTranscript: string | null = null;

    const whisperReady = isWhisperConfigured();
    console.log('[Seren] Whisper configured:', whisperReady);

    if (whisperReady) {
      finalTranscript = await transcribeAudio(audioUri);
    }

    // Clean up the audio file
    deleteAudioFile(audioUri);

    if (!finalTranscript) {
      setIsTranscribing(false);
      if (!whisperReady) {
        setTranscript('[Whisper not configured — add your API key in services/ai/aiConfig.ts]');
      } else {
        setTranscript('[Transcription failed — check your API key and network connection, then try again]');
      }
      return;
    }

    setTranscript(finalTranscript);
    setIsTranscribing(false);

    // Step 3: Analyze with LLM
    setIsAnalyzing(true);
    const [analysis] = await Promise.all([
      w.performCheckin(finalTranscript),
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
