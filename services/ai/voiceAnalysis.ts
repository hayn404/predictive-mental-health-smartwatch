/**
 * Seren AI — Voice Check-in Analysis Service
 * =============================================
 * Primary: LLM-powered deep emotional understanding via external API.
 * Fallback: VADER-style lexicon sentiment for offline/unconfigured mode.
 *
 * The check-in flow:
 *   1. User taps "How are you feeling?" FAB
 *   2. STT captures their spoken response
 *   3. LLM analyzes transcript for deep emotional understanding
 *   4. Results are combined with current biometric context
 *   5. Stored (encrypted) for trend analysis and recommendation input
 */

import {
  CheckinAnalysis,
  Sentiment,
  EmotionScores,
} from './types';
import { analyzeCheckinWithLLM, isLLMConfigured, FullBiometricContext } from './llmService';

// ============================================================
// VADER-Style Sentiment Lexicon (Phase 1)
// ============================================================
// A curated subset of VADER (Valence Aware Dictionary) focused on
// mental health context. Each word maps to a valence score (-4 to +4).

const LEXICON: Record<string, number> = {
  // Positive
  good: 1.9, great: 3.1, amazing: 3.5, wonderful: 3.2, happy: 2.7,
  calm: 2.0, relaxed: 2.1, peaceful: 2.5, grateful: 2.8, thankful: 2.5,
  energetic: 2.0, motivated: 2.3, confident: 2.5, hopeful: 2.3, better: 1.5,
  fine: 0.8, okay: 0.5, love: 3.0, excited: 2.8, proud: 2.3,
  joyful: 3.0, content: 2.0, refreshed: 2.0, rested: 1.5, strong: 1.8,

  // Negative
  bad: -2.5, terrible: -3.4, awful: -3.3, stressed: -2.5, anxious: -2.7,
  worried: -2.3, sad: -2.5, depressed: -3.0, tired: -1.8, exhausted: -2.5,
  angry: -2.8, frustrated: -2.4, overwhelmed: -2.7, scared: -2.8, afraid: -2.6,
  nervous: -2.1, upset: -2.2, miserable: -3.3, lonely: -2.8, hopeless: -3.4,
  worthless: -3.5, numb: -2.0, irritable: -2.2, restless: -1.8, panicking: -3.2,
  panic: -3.0, crying: -2.5, cry: -2.0, hurt: -2.4, pain: -2.3,
  struggling: -2.3, suffering: -3.0, insomnia: -2.2, nightmare: -2.5,

  // Intensifiers
  very: 0, really: 0, so: 0, extremely: 0, quite: 0,
  // (handled separately as multipliers)

  // Negators
  not: 0, never: 0, no: 0, "don't": 0, "can't": 0, "couldn't": 0,
  // (handled separately as sign flippers)
};

const INTENSIFIERS: Record<string, number> = {
  very: 1.3, really: 1.3, so: 1.2, extremely: 1.5,
  quite: 1.1, absolutely: 1.4, incredibly: 1.4, super: 1.3,
  totally: 1.2,
};

const NEGATORS = new Set([
  'not', 'never', 'no', "don't", "can't", "couldn't", "won't",
  "wouldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
  'neither', 'nor', 'hardly', 'barely',
]);

// Emotion keyword mapping for multi-label emotion detection
const EMOTION_KEYWORDS: Record<keyof EmotionScores, string[]> = {
  joy: ['happy', 'great', 'amazing', 'wonderful', 'joyful', 'excited', 'love', 'grateful', 'thankful', 'proud'],
  sadness: ['sad', 'depressed', 'lonely', 'hopeless', 'crying', 'miserable', 'hurt', 'worthless', 'numb', 'lost'],
  anxiety: ['anxious', 'worried', 'nervous', 'scared', 'panicking', 'panic', 'afraid', 'restless', 'overwhelmed', 'stressed'],
  anger: ['angry', 'frustrated', 'irritable', 'furious', 'upset', 'annoyed', 'mad', 'rage'],
  calm: ['calm', 'relaxed', 'peaceful', 'content', 'serene', 'tranquil', 'rested', 'refreshed'],
  fear: ['afraid', 'scared', 'terrified', 'frightened', 'panicking', 'panic', 'fearful'],
  gratitude: ['grateful', 'thankful', 'blessed', 'appreciate', 'appreciated'],
  fatigue: ['tired', 'exhausted', 'sleepy', 'drained', 'fatigued', 'weary', 'worn'],
};

// ============================================================
// Public API
// ============================================================

/**
 * Analyze a voice check-in transcript.
 * Uses LLM for deep emotional understanding when configured,
 * falls back to local VADER-style analysis otherwise.
 *
 * @param transcript - STT output text
 * @param biometricContext - Current HR, HRV, stress at time of check-in
 * @returns CheckinAnalysis with sentiment, emotions, and insights
 */
export async function analyzeCheckin(
  transcript: string,
  biometricContext: FullBiometricContext,
): Promise<CheckinAnalysis> {
  // Use LLM when configured for deep understanding
  if (isLLMConfigured()) {
    return analyzeCheckinWithLLM(transcript, biometricContext);
  }

  // Fallback: local VADER-style analysis
  return analyzeCheckinLocal(transcript, biometricContext);
}

/**
 * Local (offline) check-in analysis using VADER-style sentiment.
 */
export function analyzeCheckinLocal(
  transcript: string,
  biometricContext: FullBiometricContext,
): CheckinAnalysis {
  const sentimentResult = analyzeSentiment(transcript);
  const emotions = detectEmotions(transcript);
  const insights = generateInsights(transcript, sentimentResult, emotions, biometricContext);

  return {
    id: generateId(),
    timestamp: Date.now(),
    transcript,
    sentiment: sentimentResult.sentiment,
    sentimentScore: sentimentResult.score,
    emotionScores: emotions,
    keyInsights: insights,
    themes: ['general'],
    emotionalIntensity: Math.min(1, Math.abs(sentimentResult.score) + 0.2),
    empathyResponse: generateLocalEmpathy(sentimentResult.sentiment),
    suggestedFollowUp: null,
    hrAtCheckin: biometricContext.hr,
    hrvAtCheckin: biometricContext.hrv,
    stressAtCheckin: biometricContext.stress,
  };
}

function generateLocalEmpathy(sentiment: Sentiment): string {
  if (sentiment === 'distressed') {
    return "I can hear that you're going through a really tough time. Your feelings are valid, and it's okay to not be okay right now.";
  }
  if (sentiment === 'concerned') {
    return "It sounds like something is weighing on you. It's completely normal to feel this way, and acknowledging it is an important step.";
  }
  if (sentiment === 'positive') {
    return "It's really great to hear that! Those positive moments matter, and I'm glad you're taking the time to recognize them.";
  }
  return "Thank you for checking in. Taking a moment to reflect on how you're feeling is a valuable practice.";
}

/**
 * Compute sentiment trend across multiple check-ins.
 */
export function computeCheckinTrend(
  checkins: CheckinAnalysis[],
  days: number = 7,
): CheckinTrend {
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
  const recent = checkins.filter(c => c.timestamp > cutoff);

  if (recent.length === 0) {
    return { avgSentiment: 0, dominantEmotion: 'calm', trend: 'stable', count: 0 };
  }

  const avgSentiment = recent.reduce((s, c) => s + c.sentimentScore, 0) / recent.length;

  // Find dominant emotion across all check-ins
  const emotionTotals: Record<string, number> = {};
  for (const c of recent) {
    for (const [emotion, score] of Object.entries(c.emotionScores)) {
      emotionTotals[emotion] = (emotionTotals[emotion] || 0) + score;
    }
  }
  const dominantEmotion = Object.entries(emotionTotals)
    .sort((a, b) => b[1] - a[1])[0]?.[0] as keyof EmotionScores || 'calm';

  // Trend
  let trend: 'improving' | 'declining' | 'stable' = 'stable';
  if (recent.length >= 3) {
    const mid = Math.floor(recent.length / 2);
    const firstAvg = recent.slice(0, mid).reduce((s, c) => s + c.sentimentScore, 0) / mid;
    const secondAvg = recent.slice(mid).reduce((s, c) => s + c.sentimentScore, 0) / (recent.length - mid);
    if (secondAvg - firstAvg > 0.2) trend = 'improving';
    else if (firstAvg - secondAvg > 0.2) trend = 'declining';
  }

  return { avgSentiment, dominantEmotion, trend, count: recent.length };
}

export interface CheckinTrend {
  avgSentiment: number;
  dominantEmotion: keyof EmotionScores;
  trend: 'improving' | 'declining' | 'stable';
  count: number;
}

// ============================================================
// VADER-Style Sentiment Analysis
// ============================================================

interface SentimentResult {
  score: number;       // -1.0 to +1.0
  sentiment: Sentiment;
}

function analyzeSentiment(text: string): SentimentResult {
  const words = tokenize(text);
  let totalValence = 0;
  let wordCount = 0;

  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    let valence = LEXICON[word];

    if (valence === undefined || valence === 0) continue;

    // Check for preceding negator (flip sign)
    if (i > 0 && NEGATORS.has(words[i - 1])) {
      valence *= -0.75;
    }

    // Check for preceding intensifier (amplify)
    if (i > 0 && INTENSIFIERS[words[i - 1]] !== undefined) {
      valence *= INTENSIFIERS[words[i - 1]];
    }

    totalValence += valence;
    wordCount++;
  }

  // Normalize to -1 to +1 range using VADER-style normalization
  // compound = sum / sqrt(sum^2 + alpha) where alpha = 15
  const compound = wordCount > 0
    ? totalValence / Math.sqrt(totalValence * totalValence + 15)
    : 0;

  const score = Math.max(-1, Math.min(1, compound));

  return {
    score,
    sentiment: scoreToSentiment(score),
  };
}

function scoreToSentiment(score: number): Sentiment {
  if (score >= 0.3) return 'positive';
  if (score >= -0.1) return 'neutral';
  if (score >= -0.5) return 'concerned';
  return 'distressed';
}

// ============================================================
// Emotion Detection
// ============================================================

function detectEmotions(text: string): EmotionScores {
  const words = new Set(tokenize(text));
  const scores: EmotionScores = {
    joy: 0, sadness: 0, anxiety: 0, anger: 0,
    calm: 0, fear: 0, gratitude: 0, fatigue: 0,
  };

  for (const [emotion, keywords] of Object.entries(EMOTION_KEYWORDS) as [keyof EmotionScores, string[]][]) {
    let hits = 0;
    for (const keyword of keywords) {
      if (words.has(keyword)) hits++;
    }
    // Score: proportion of keywords matched, scaled 0-1
    scores[emotion] = keywords.length > 0 ? Math.min(1, hits / 2) : 0;
  }

  // Normalize so at least one emotion has some signal
  const maxScore = Math.max(...Object.values(scores));
  if (maxScore === 0) {
    scores.calm = 0.3; // Default to mild calm if no keywords matched
  }

  return scores;
}

// ============================================================
// Insight Generation
// ============================================================

function generateInsights(
  _transcript: string,
  sentiment: SentimentResult,
  emotions: EmotionScores,
  biometrics: FullBiometricContext,
): string[] {
  const insights: string[] = [];

  // Biometric-verbal alignment check
  const verballyStressed = sentiment.score < -0.3;
  const biometricallyStressed = biometrics.stress > 60;

  if (verballyStressed && biometricallyStressed) {
    insights.push('Both your words and body signals suggest elevated stress');
  } else if (verballyStressed && !biometricallyStressed) {
    insights.push('You expressed some concern, but your body metrics look steady');
  } else if (!verballyStressed && biometricallyStressed) {
    insights.push('Your words sound positive, but your body may be carrying some tension');
  }

  // Dominant emotion insight
  const topEmotions = Object.entries(emotions)
    .filter(([, score]) => score > 0.3)
    .sort((a, b) => b[1] - a[1]);

  if (topEmotions.length > 0) {
    const [topEmotion] = topEmotions[0];
    const emotionLabels: Record<string, string> = {
      joy: 'a sense of joy',
      sadness: 'some sadness',
      anxiety: 'feelings of anxiety',
      anger: 'frustration',
      calm: 'a sense of calm',
      fear: 'some worry',
      gratitude: 'gratitude',
      fatigue: 'tiredness',
    };
    insights.push(`Your words reflect ${emotionLabels[topEmotion] || topEmotion}`);
  }

  // HRV context
  if (biometrics.hrv > 0 && biometrics.hrv < 25) {
    insights.push('Your HRV is quite low right now — a breathing exercise might help');
  }

  return insights.slice(0, 3);
}

// ============================================================
// Utilities
// ============================================================

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z'\s-]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 0);
}

function generateId(): string {
  return `checkin_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}
