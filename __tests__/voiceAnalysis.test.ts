/**
 * Unit Tests — Voice Analysis & Sentiment
 * =========================================
 * Tests VADER-style sentiment analysis, emotion detection,
 * insight generation, and check-in trend computation.
 */

import {
  analyzeCheckinLocal,
  computeCheckinTrend,
} from '@/services/ai/voiceAnalysis';
import type { CheckinAnalysis, EmotionScores } from '@/services/ai/types';
import type { FullBiometricContext } from '@/services/ai/llmService';

// ============================================================
// Test Helpers
// ============================================================

const defaultContext: FullBiometricContext = {
  hr: 72, hrv: 45, stress: 35,
  sleepQuality: 75, anxietyIndex: 25,
  sleepDurationHrs: 7.5,
};

const stressedContext: FullBiometricContext = {
  hr: 95, hrv: 20, stress: 75,
  sleepQuality: 40, anxietyIndex: 65,
  sleepDurationHrs: 5,
};

function makeCheckin(
  sentiment: 'positive' | 'neutral' | 'concerned' | 'distressed',
  score: number,
  emotions: Partial<EmotionScores> = {},
  timestamp?: number,
): CheckinAnalysis {
  return {
    id: `test_${Math.random().toString(36).slice(2)}`,
    timestamp: timestamp ?? Date.now(),
    transcript: 'test',
    sentiment,
    sentimentScore: score,
    emotionScores: {
      joy: 0, sadness: 0, anxiety: 0, anger: 0,
      calm: 0.3, fear: 0, gratitude: 0, fatigue: 0,
      ...emotions,
    },
    keyInsights: [],
    themes: ['general'],
    emotionalIntensity: Math.abs(score),
    empathyResponse: '',
    suggestedFollowUp: null,
    hrAtCheckin: 72,
    hrvAtCheckin: 45,
    stressAtCheckin: 35,
  };
}

// ============================================================
// analyzeCheckinLocal — Sentiment Classification
// ============================================================

describe('analyzeCheckinLocal — Sentiment', () => {
  test('positive text returns positive sentiment', () => {
    const result = analyzeCheckinLocal('I feel great and happy today', defaultContext);
    expect(result.sentiment).toBe('positive');
    expect(result.sentimentScore).toBeGreaterThan(0.3);
  });

  test('negative text returns concerned or distressed', () => {
    const result = analyzeCheckinLocal('I feel terrible and stressed', defaultContext);
    expect(['concerned', 'distressed']).toContain(result.sentiment);
    expect(result.sentimentScore).toBeLessThan(-0.1);
  });

  test('neutral text returns neutral sentiment', () => {
    const result = analyzeCheckinLocal('I went to the store today', defaultContext);
    expect(result.sentiment).toBe('neutral');
  });

  test('strongly negative text returns distressed', () => {
    const result = analyzeCheckinLocal(
      'I feel hopeless and worthless, everything is terrible and I am miserable',
      defaultContext,
    );
    expect(result.sentiment).toBe('distressed');
    expect(result.sentimentScore).toBeLessThan(-0.5);
  });

  test('sentiment score is bounded between -1 and 1', () => {
    const positive = analyzeCheckinLocal('Amazing wonderful great happy', defaultContext);
    const negative = analyzeCheckinLocal('Terrible awful hopeless miserable', defaultContext);
    expect(positive.sentimentScore).toBeLessThanOrEqual(1);
    expect(positive.sentimentScore).toBeGreaterThanOrEqual(-1);
    expect(negative.sentimentScore).toBeLessThanOrEqual(1);
    expect(negative.sentimentScore).toBeGreaterThanOrEqual(-1);
  });
});

// ============================================================
// analyzeCheckinLocal — Negation Handling
// ============================================================

describe('analyzeCheckinLocal — Negation', () => {
  test('negation immediately before sentiment word flips valence', () => {
    const positive = analyzeCheckinLocal('I feel good', defaultContext);
    // "not good" — negator directly precedes the sentiment word
    const negated = analyzeCheckinLocal('I feel not good today', defaultContext);
    expect(negated.sentimentScore).toBeLessThan(positive.sentimentScore);
  });

  test("negator immediately before sentiment word flips it", () => {
    const positive = analyzeCheckinLocal('I feel happy', defaultContext);
    const negated = analyzeCheckinLocal('not happy at all', defaultContext);
    expect(negated.sentimentScore).toBeLessThan(positive.sentimentScore);
  });
});

// ============================================================
// analyzeCheckinLocal — Intensifiers
// ============================================================

describe('analyzeCheckinLocal — Intensifiers', () => {
  test('intensifiers amplify sentiment', () => {
    const normal = analyzeCheckinLocal('I feel good', defaultContext);
    const intensified = analyzeCheckinLocal('I feel very good', defaultContext);
    expect(Math.abs(intensified.sentimentScore)).toBeGreaterThanOrEqual(
      Math.abs(normal.sentimentScore),
    );
  });

  test('extremely is a stronger intensifier', () => {
    const veryGood = analyzeCheckinLocal('I feel very good', defaultContext);
    const extremelyGood = analyzeCheckinLocal('I feel extremely good', defaultContext);
    expect(Math.abs(extremelyGood.sentimentScore)).toBeGreaterThanOrEqual(
      Math.abs(veryGood.sentimentScore),
    );
  });
});

// ============================================================
// analyzeCheckinLocal — Emotion Detection
// ============================================================

describe('analyzeCheckinLocal — Emotions', () => {
  test('detects joy from positive keywords', () => {
    const result = analyzeCheckinLocal('I am so happy and excited today', defaultContext);
    expect(result.emotionScores.joy).toBeGreaterThan(0);
  });

  test('detects anxiety from anxiety keywords', () => {
    const result = analyzeCheckinLocal('I feel anxious and worried', defaultContext);
    expect(result.emotionScores.anxiety).toBeGreaterThan(0);
  });

  test('detects sadness from sadness keywords', () => {
    const result = analyzeCheckinLocal('I feel sad and lonely', defaultContext);
    expect(result.emotionScores.sadness).toBeGreaterThan(0);
  });

  test('detects fatigue from tiredness keywords', () => {
    const result = analyzeCheckinLocal('I am so tired and exhausted', defaultContext);
    expect(result.emotionScores.fatigue).toBeGreaterThan(0);
  });

  test('detects gratitude keywords', () => {
    const result = analyzeCheckinLocal('I feel grateful and thankful for my family', defaultContext);
    expect(result.emotionScores.gratitude).toBeGreaterThan(0);
  });

  test('defaults to mild calm when no keywords match', () => {
    const result = analyzeCheckinLocal('I went to the park today', defaultContext);
    expect(result.emotionScores.calm).toBeGreaterThan(0);
  });

  test('all emotion scores are between 0 and 1', () => {
    const result = analyzeCheckinLocal('I am very happy and grateful but also a bit anxious', defaultContext);
    for (const [, score] of Object.entries(result.emotionScores)) {
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });
});

// ============================================================
// analyzeCheckinLocal — Biometric-Verbal Alignment Insights
// ============================================================

describe('analyzeCheckinLocal — Insights', () => {
  test('detects verbal+biometric stress alignment', () => {
    const result = analyzeCheckinLocal('I feel terrible and stressed', stressedContext);
    const aligned = result.keyInsights.some(i => i.includes('words and body'));
    expect(aligned).toBe(true);
  });

  test('detects verbal concern with calm biometrics', () => {
    const result = analyzeCheckinLocal('I feel stressed and anxious', defaultContext);
    const mismatch = result.keyInsights.some(i => i.includes('metrics look steady'));
    expect(mismatch).toBe(true);
  });

  test('detects positive words with stressed biometrics', () => {
    const result = analyzeCheckinLocal('I feel great and happy', stressedContext);
    const hidden = result.keyInsights.some(i => i.includes('body may be carrying'));
    expect(hidden).toBe(true);
  });

  test('suggests breathing exercise when HRV is very low', () => {
    const lowHRVContext = { ...defaultContext, hrv: 15 };
    const result = analyzeCheckinLocal('I feel okay', lowHRVContext);
    const hrvInsight = result.keyInsights.some(i => i.includes('HRV'));
    expect(hrvInsight).toBe(true);
  });

  test('limits insights to max 3', () => {
    const result = analyzeCheckinLocal('I feel stressed and anxious', stressedContext);
    expect(result.keyInsights.length).toBeLessThanOrEqual(3);
  });
});

// ============================================================
// analyzeCheckinLocal — Output Shape
// ============================================================

describe('analyzeCheckinLocal — Output', () => {
  test('returns a complete CheckinAnalysis object', () => {
    const result = analyzeCheckinLocal('I feel good today', defaultContext);
    expect(result.id).toBeTruthy();
    expect(result.id.startsWith('checkin_')).toBe(true);
    expect(result.transcript).toBe('I feel good today');
    expect(result.hrAtCheckin).toBe(72);
    expect(result.hrvAtCheckin).toBe(45);
    expect(result.stressAtCheckin).toBe(35);
    expect(result.themes).toEqual(['general']);
    expect(result.suggestedFollowUp).toBeNull();
    expect(result.empathyResponse).toBeTruthy();
  });

  test('emotional intensity is proportional to sentiment magnitude', () => {
    const neutral = analyzeCheckinLocal('I went for a walk', defaultContext);
    const strong = analyzeCheckinLocal('I feel absolutely terrible and hopeless', defaultContext);
    expect(strong.emotionalIntensity).toBeGreaterThan(neutral.emotionalIntensity);
  });

  test('empathy response varies by sentiment', () => {
    const positive = analyzeCheckinLocal('I feel amazing', defaultContext);
    const distressed = analyzeCheckinLocal('I feel hopeless and miserable', defaultContext);
    expect(positive.empathyResponse).not.toBe(distressed.empathyResponse);
  });
});

// ============================================================
// computeCheckinTrend
// ============================================================

describe('computeCheckinTrend', () => {
  test('returns stable with empty checkins', () => {
    const result = computeCheckinTrend([]);
    expect(result.trend).toBe('stable');
    expect(result.count).toBe(0);
  });

  test('computes average sentiment correctly', () => {
    const checkins = [
      makeCheckin('positive', 0.5),
      makeCheckin('neutral', 0.0),
      makeCheckin('concerned', -0.3),
    ];
    const result = computeCheckinTrend(checkins);
    expect(result.avgSentiment).toBeCloseTo((0.5 + 0.0 + -0.3) / 3, 2);
  });

  test('identifies dominant emotion', () => {
    const checkins = [
      makeCheckin('concerned', -0.3, { anxiety: 0.8 }),
      makeCheckin('concerned', -0.4, { anxiety: 0.7 }),
      makeCheckin('neutral', 0.0, { calm: 0.5 }),
    ];
    const result = computeCheckinTrend(checkins);
    expect(result.dominantEmotion).toBe('anxiety');
  });

  test('detects improving trend', () => {
    const now = Date.now();
    const checkins = [
      makeCheckin('distressed', -0.7, {}, now - 6 * 86400000),
      makeCheckin('concerned', -0.4, {}, now - 4 * 86400000),
      makeCheckin('neutral', 0.0, {}, now - 2 * 86400000),
      makeCheckin('positive', 0.5, {}, now),
    ];
    const result = computeCheckinTrend(checkins);
    expect(result.trend).toBe('improving');
  });

  test('detects declining trend', () => {
    const now = Date.now();
    const checkins = [
      makeCheckin('positive', 0.6, {}, now - 6 * 86400000),
      makeCheckin('neutral', 0.1, {}, now - 4 * 86400000),
      makeCheckin('concerned', -0.3, {}, now - 2 * 86400000),
      makeCheckin('distressed', -0.7, {}, now),
    ];
    const result = computeCheckinTrend(checkins);
    expect(result.trend).toBe('declining');
  });

  test('filters out old checkins beyond the window', () => {
    const now = Date.now();
    const checkins = [
      makeCheckin('distressed', -0.8, {}, now - 30 * 86400000), // 30 days ago
      makeCheckin('positive', 0.5, {}, now),
    ];
    const result = computeCheckinTrend(checkins, 7);
    expect(result.count).toBe(1);
  });
});
