// Seren - Mock Health Data Service

export interface HealthSnapshot {
  stressLevel: number;
  anxietyIndex: number;
  sleepQuality: number;
  hrv: number; // ms
  heartRate: number;
  timestamp: Date;
}

export interface DailyEntry {
  date: string;
  stress: number;
  anxiety: number;
  sleep: number;
  hrv: number;
  heartRate: number;
  mood: string;
}

export interface WatchStatus {
  connected: boolean;
  batteryLevel: number;
  model: string;
  lastSync: Date;
}

export interface Recommendation {
  id: string;
  category: 'breathing' | 'physical' | 'journaling' | 'meditation' | 'sleep_hygiene' | 'social' | 'outdoor' | 'exploration';
  title: string;
  description: string;
  duration: string;
  aiReason: string;
}

export interface CheckinEntry {
  id: string;
  timestamp: Date;
  transcript: string;
  sentiment: 'positive' | 'neutral' | 'concerned';
  keyInsights: string[];
}

// Current health snapshot
export const getCurrentHealth = (): HealthSnapshot => ({
  stressLevel: 24,
  anxietyIndex: 18,
  sleepQuality: 82,
  hrv: 58,
  heartRate: 68,
  timestamp: new Date(),
});

// Weekly trend data
export const getWeeklyData = (): DailyEntry[] => [
  { date: 'Mon', stress: 45, anxiety: 38, sleep: 75, hrv: 52, heartRate: 72, mood: 'Okay' },
  { date: 'Tue', stress: 62, anxiety: 55, sleep: 68, hrv: 44, heartRate: 78, mood: 'Stressed' },
  { date: 'Wed', stress: 38, anxiety: 30, sleep: 85, hrv: 62, heartRate: 65, mood: 'Good' },
  { date: 'Thu', stress: 71, anxiety: 65, sleep: 60, hrv: 38, heartRate: 82, mood: 'Anxious' },
  { date: 'Fri', stress: 50, anxiety: 42, sleep: 78, hrv: 55, heartRate: 70, mood: 'Neutral' },
  { date: 'Sat', stress: 28, anxiety: 22, sleep: 90, hrv: 68, heartRate: 62, mood: 'Calm' },
  { date: 'Sun', stress: 24, anxiety: 18, sleep: 82, hrv: 58, heartRate: 68, mood: 'Good' },
];

// Monthly sleep heatmap (4 weeks x 7 days)
export const getMonthlySleepData = (): number[][] => [
  [72, 68, 85, 60, 78, 90, 82],
  [65, 55, 88, 75, 70, 92, 85],
  [80, 72, 78, 65, 82, 88, 90],
  [70, 60, 82, 78, 75, 85, 82],
];

// HRV trend (24 hours)
export const getHRVTrend = (): { time: string; value: number }[] => [
  { time: '12am', value: 62 },
  { time: '3am', value: 68 },
  { time: '6am', value: 58 },
  { time: '9am', value: 52 },
  { time: '12pm', value: 45 },
  { time: '3pm', value: 38 },
  { time: '6pm', value: 44 },
  { time: '9pm', value: 55 },
  { time: 'Now', value: 58 },
];

// Watch status
export const getWatchStatus = (): WatchStatus => ({
  connected: true,
  batteryLevel: 85,
  model: 'Smartwatch',
  lastSync: new Date(Date.now() - 3 * 60 * 1000),
});

// Recommendations
export const getRecommendations = (): Recommendation[] => [
  {
    id: '1',
    category: 'breathing',
    title: 'Box Breathing',
    description: '4-4-4-4 breathing pattern to activate your parasympathetic nervous system.',
    duration: '4 min',
    aiReason: 'Your HRV dropped below 45ms at 3 PM — this can help restore balance.',
  },
  {
    id: '2',
    category: 'physical',
    title: 'Afternoon Walk',
    description: 'A 15-minute outdoor walk significantly reduces cortisol levels.',
    duration: '15 min',
    aiReason: 'You show elevated stress daily between 2–4 PM. Movement interrupts the cycle.',
  },
  {
    id: '3',
    category: 'journaling',
    title: 'Gratitude Journal',
    description: 'Write 3 things you appreciate today. Rewires neural pathways over time.',
    duration: '5 min',
    aiReason: 'Post-evening voice check-ins show ruminative thought patterns.',
  },
  {
    id: '4',
    category: 'breathing',
    title: '4-7-8 Breathing',
    description: 'Inhale 4s, hold 7s, exhale 8s. Powerful sleep onset technique.',
    duration: '3 min',
    aiReason: 'Sleep onset is taking 35+ minutes this week. This technique helps.',
  },
  {
    id: '5',
    category: 'meditation',
    title: 'Body Scan Meditation',
    description: 'Progressive relaxation from head to toe. Reduces tension held in muscles.',
    duration: '10 min',
    aiReason: 'Your anxiety index is highest before bed. Grounding helps break the loop.',
  },
];

// Check-in history
export const getCheckinHistory = (): CheckinEntry[] => [
  {
    id: '1',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
    transcript: "I'm feeling a bit overwhelmed with work deadlines but overall managing okay. Had a good lunch which helped.",
    sentiment: 'neutral',
    keyInsights: ['Work stress identified', 'Self-care awareness positive', 'Emotional regulation intact'],
  },
  {
    id: '2',
    timestamp: new Date(Date.now() - 26 * 60 * 60 * 1000),
    transcript: "Feeling really good today. The morning run made a huge difference. I feel clear-headed.",
    sentiment: 'positive',
    keyInsights: ['Exercise positive impact noted', 'Cognitive clarity high', 'Energy levels elevated'],
  },
];

export const getStressLabel = (level: number): { label: string; color: string; description: string } => {
  if (level <= 25) return { label: 'Low', color: '#35e27e', description: 'You\'re in a calm, relaxed state.' };
  if (level <= 50) return { label: 'Moderate', color: '#2DBD6A', description: 'Some tension present. Monitor and breathe.' };
  if (level <= 75) return { label: 'Elevated', color: '#1F9952', description: 'Stress is building. Consider a break.' };
  return { label: 'High', color: '#157A3E', description: 'High stress detected. Please take action.' };
};

export const getAnxietyColor = (level: number): { label: string; color: string; description: string } => {
  if (level <= 25) return { label: 'Minimal', color: '#7AC59E', description: 'Anxiety is low. You\'re feeling calm.' };
  if (level <= 50) return { label: 'Mild', color: '#A3A6D4', description: 'Mild anxiety present. Consider grounding.' };
  if (level <= 75) return { label: 'Moderate', color: '#837AB5', description: 'Moderate anxiety. Take a breathing break.' };
  return { label: 'Severe', color: '#5B4A82', description: 'Elevated anxiety detected. Try a breathing exercise.' };
};
