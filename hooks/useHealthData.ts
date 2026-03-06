import { useState, useEffect, useCallback } from 'react';
import {
  getCurrentHealth,
  getWeeklyData,
  getMonthlySleepData,
  getHRVTrend,
  getWatchStatus,
  getRecommendations,
  getCheckinHistory,
  HealthSnapshot,
  DailyEntry,
  WatchStatus,
  Recommendation,
  CheckinEntry,
} from '@/services/mockData';

export function useHealthData() {
  const [health, setHealth] = useState<HealthSnapshot>(getCurrentHealth());
  const [weeklyData] = useState<DailyEntry[]>(getWeeklyData());
  const [monthlySleep] = useState<number[][]>(getMonthlySleepData());
  const [hrvTrend] = useState(getHRVTrend());
  const [watchStatus, setWatchStatus] = useState<WatchStatus>(getWatchStatus());
  const [recommendations] = useState<Recommendation[]>(getRecommendations());
  const [checkinHistory] = useState<CheckinEntry[]>(getCheckinHistory());
  const [isLive, setIsLive] = useState(true);

  // Simulate live data updates
  useEffect(() => {
    if (!isLive) return;
    const interval = setInterval(() => {
      setHealth(prev => ({
        ...prev,
        stressLevel: Math.max(0, Math.min(100, prev.stressLevel + (Math.random() - 0.5) * 3)),
        heartRate: Math.max(55, Math.min(95, prev.heartRate + (Math.random() - 0.5) * 2)),
        hrv: Math.max(30, Math.min(85, prev.hrv + (Math.random() - 0.5) * 1.5)),
        timestamp: new Date(),
      }));
    }, 4000);
    return () => clearInterval(interval);
  }, [isLive]);

  // Watch sync simulation
  useEffect(() => {
    const syncInterval = setInterval(() => {
      setWatchStatus(prev => ({
        ...prev,
        lastSync: new Date(),
        batteryLevel: Math.max(0, prev.batteryLevel - 0.01),
      }));
    }, 30000);
    return () => clearInterval(syncInterval);
  }, []);

  const toggleLive = useCallback(() => setIsLive(v => !v), []);

  return {
    health,
    weeklyData,
    monthlySleep,
    hrvTrend,
    watchStatus,
    recommendations,
    checkinHistory,
    isLive,
    toggleLive,
  };
}

export function useCheckin() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<null | { sentiment: string; insights: string[] }>(null);
  const [waveAmplitudes, setWaveAmplitudes] = useState<number[]>(Array(20).fill(4));

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isRecording) {
      interval = setInterval(() => {
        setWaveAmplitudes(Array(20).fill(0).map(() => Math.random() * 36 + 4));
      }, 100);

      // Simulate transcript building
      const phrases = [
        "I'm feeling... ",
        "I'm feeling a bit tired today, ",
        "I'm feeling a bit tired today, but generally optimistic. ",
        "I'm feeling a bit tired today, but generally optimistic. Work has been intense ",
        "I'm feeling a bit tired today, but generally optimistic. Work has been intense but I feel like I'm making progress.",
      ];
      let idx = 0;
      const transcriptInterval = setInterval(() => {
        if (idx < phrases.length) {
          setTranscript(phrases[idx]);
          idx++;
        }
      }, 1200);

      return () => {
        clearInterval(interval);
        clearInterval(transcriptInterval);
      };
    } else {
      setWaveAmplitudes(Array(20).fill(4));
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const startRecording = useCallback(() => {
    setIsRecording(true);
    setTranscript('');
    setResult(null);
  }, []);

  const stopAndAnalyze = useCallback(() => {
    setIsRecording(false);
    setIsAnalyzing(true);
    setTimeout(() => {
      setIsAnalyzing(false);
      setResult({
        sentiment: 'neutral',
        insights: ['Mild fatigue noted', 'Work stress acknowledged', 'Optimistic outlook maintained'],
      });
    }, 2500);
  }, []);

  const reset = useCallback(() => {
    setTranscript('');
    setResult(null);
    setIsRecording(false);
    setIsAnalyzing(false);
  }, []);

  return { isRecording, transcript, isAnalyzing, result, waveAmplitudes, startRecording, stopAndAnalyze, reset };
}
