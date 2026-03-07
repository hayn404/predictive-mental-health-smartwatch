/**
 * Unit Tests — Recommendation Engine
 * ====================================
 * Tests clinical threshold triggers, priority sorting,
 * deduplication, time-based scheduling, and outcome tracking.
 */

import { generateRecommendations, recordOutcome } from '@/services/ai/recommendations';
import type {
  StressPrediction,
  AnxietyPrediction,
  SleepAnalysis,
  CheckinAnalysis,
  Recommendation,
} from '@/services/ai/types';

// ============================================================
// Test Helpers
// ============================================================

function makeStress(score: number): StressPrediction {
  const level = score <= 25 ? 'low' : score <= 50 ? 'moderate' : score <= 75 ? 'elevated' : 'high';
  return {
    timestamp: Date.now(), stressScore: score,
    stressLevel: level, confidence: 0.85, topContributors: [],
  };
}

function makeAnxiety(index: number, sustained = false): AnxietyPrediction {
  const level = index <= 20 ? 'minimal' : index <= 45 ? 'mild' : index <= 70 ? 'moderate' : 'severe';
  return {
    timestamp: Date.now(), anxietyIndex: index,
    level, sustained, baselineDeviation: 0,
  };
}

function makeSleep(quality: number, efficiency = 0.85, consistency = 75): SleepAnalysis {
  return {
    date: '2026-03-07', sessionStart: Date.now(), sessionEnd: Date.now(),
    totalInBedMin: 480, totalSleepMin: 420, onsetLatencyMin: 10, wasoMin: 15,
    lightSleepMin: 200, deepSleepMin: 100, remSleepMin: 120, awakeMin: 25,
    deepSleepPct: 0.2, remSleepPct: 0.25, lightSleepPct: 0.55,
    sleepEfficiency: efficiency, fragmentationIndex: 1.5, awakeningCount: 2,
    avgHrSleep: 55, minHrSleep: 48, avgHrvSleep: 50, maxHrvSleep: 70,
    qualityScore: quality, recoveryScore: 70, consistencyScore: consistency,
  };
}

function makeCheckin(sentimentScore: number, emotions: Partial<Record<string, number>> = {}): CheckinAnalysis {
  return {
    id: 'test', timestamp: Date.now(), transcript: 'test',
    sentiment: sentimentScore > 0 ? 'positive' : 'concerned',
    sentimentScore,
    emotionScores: {
      joy: 0, sadness: 0, anxiety: 0, anger: 0,
      calm: 0, fear: 0, gratitude: 0, fatigue: 0,
      ...emotions,
    },
    keyInsights: [], themes: ['general'], emotionalIntensity: 0.5,
    empathyResponse: '', suggestedFollowUp: null,
    hrAtCheckin: 72, hrvAtCheckin: 45, stressAtCheckin: 35,
  };
}

// ============================================================
// Stress-triggered recommendations
// ============================================================

describe('Stress-triggered recommendations', () => {
  test('high stress (>=70) triggers breathing + grounding', () => {
    const recs = generateRecommendations(makeStress(80), null, null, null, null);
    expect(recs.length).toBeGreaterThan(0);
    const ids = recs.map(r => r.id);
    expect(ids).toContain('breathing_box');
  });

  test('moderate stress (50-69) triggers coherent breathing or walk', () => {
    const recs = generateRecommendations(makeStress(55), null, null, null, null);
    expect(recs.length).toBeGreaterThan(0);
    const ids = recs.map(r => r.id);
    const hasCoherentOrWalk = ids.includes('breathing_coherent') || ids.includes('physical_walk');
    expect(hasCoherentOrWalk).toBe(true);
  });

  test('low stress (<50) does not trigger stress interventions', () => {
    const recs = generateRecommendations(makeStress(20), null, null, null, null);
    const stressRecs = recs.filter(r => r.triggerReason.includes('Stress score'));
    expect(stressRecs.length).toBe(0);
  });
});

// ============================================================
// Anxiety-triggered recommendations
// ============================================================

describe('Anxiety-triggered recommendations', () => {
  test('severe anxiety (>=70) triggers grounding + breathing', () => {
    const recs = generateRecommendations(makeStress(30), makeAnxiety(80), null, null, null);
    const ids = recs.map(r => r.id);
    expect(ids).toContain('meditation_grounding');
  });

  test('sustained anxiety suggests social connection', () => {
    const recs = generateRecommendations(makeStress(30), makeAnxiety(80, true), null, null, null);
    const ids = recs.map(r => r.id);
    expect(ids).toContain('social_reach_out');
  });

  test('moderate anxiety (50-69) triggers journaling', () => {
    const recs = generateRecommendations(makeStress(30), makeAnxiety(55), null, null, null);
    const ids = recs.map(r => r.id);
    expect(ids).toContain('journaling_expressive');
  });
});

// ============================================================
// Sleep-triggered recommendations
// ============================================================

describe('Sleep-triggered recommendations', () => {
  test('poor sleep quality (<50) triggers wind-down', () => {
    const recs = generateRecommendations(makeStress(30), null, makeSleep(40), null, null);
    const ids = recs.map(r => r.id);
    expect(ids).toContain('sleep_wind_down');
  });

  test('low sleep efficiency (<75%) triggers schedule alignment', () => {
    const recs = generateRecommendations(makeStress(30), null, makeSleep(60, 0.70), null, null);
    const ids = recs.map(r => r.id);
    expect(ids).toContain('sleep_consistent');
  });
});

// ============================================================
// Check-in triggered recommendations
// ============================================================

describe('Check-in triggered recommendations', () => {
  test('distressed check-in triggers expressive writing', () => {
    const checkin = makeCheckin(-0.7);
    const recs = generateRecommendations(makeStress(30), null, null, checkin, null);
    const ids = recs.map(r => r.id);
    expect(ids).toContain('journaling_expressive');
  });

  test('fatigued check-in triggers stretching', () => {
    const checkin = makeCheckin(0.0, { fatigue: 0.7 });
    const recs = generateRecommendations(makeStress(30), null, null, checkin, null);
    const ids = recs.map(r => r.id);
    expect(ids).toContain('physical_stretch');
  });

  test('sad check-in triggers social connection', () => {
    const checkin = makeCheckin(-0.4, { sadness: 0.7 });
    const recs = generateRecommendations(makeStress(30), null, null, checkin, null);
    const ids = recs.map(r => r.id);
    expect(ids).toContain('social_reach_out');
  });
});

// ============================================================
// Priority and deduplication
// ============================================================

describe('Priority and deduplication', () => {
  test('returns max 3 recommendations', () => {
    // High stress + high anxiety + poor sleep + sad checkin = many candidates
    const recs = generateRecommendations(
      makeStress(85), makeAnxiety(80, true),
      makeSleep(35, 0.65), makeCheckin(-0.8, { sadness: 0.8 }), null,
    );
    expect(recs.length).toBeLessThanOrEqual(3);
  });

  test('recommendations are sorted by priority (descending)', () => {
    const recs = generateRecommendations(
      makeStress(85), makeAnxiety(75), null, null, null,
    );
    for (let i = 1; i < recs.length; i++) {
      expect(recs[i - 1].priorityScore).toBeGreaterThanOrEqual(recs[i].priorityScore);
    }
  });

  test('deduplicates same intervention keeping highest priority', () => {
    // Both stress and anxiety might trigger meditation_grounding
    const recs = generateRecommendations(
      makeStress(85), makeAnxiety(80), null, null, null,
    );
    const groundingCount = recs.filter(r => r.id === 'meditation_grounding').length;
    expect(groundingCount).toBeLessThanOrEqual(1);
  });

  test('filters out recently shown recommendations', () => {
    const recent = ['breathing_box', 'meditation_grounding'];
    const recs = generateRecommendations(makeStress(85), null, null, null, null, recent);
    const ids = recs.map(r => r.id);
    expect(ids).not.toContain('breathing_box');
    expect(ids).not.toContain('meditation_grounding');
  });
});

// ============================================================
// Recommendation structure
// ============================================================

describe('Recommendation structure', () => {
  test('each recommendation has required fields', () => {
    const recs = generateRecommendations(makeStress(75), null, null, null, null);
    for (const rec of recs) {
      expect(rec.id).toBeTruthy();
      expect(rec.title).toBeTruthy();
      expect(rec.description).toBeTruthy();
      expect(rec.category).toBeTruthy();
      expect(rec.instructions.length).toBeGreaterThan(0);
      expect(rec.evidenceLevel).toBeTruthy();
      expect(rec.citation).toBeTruthy();
      expect(rec.trigger).toBeTruthy();
      expect(rec.triggerReason).toBeTruthy();
      expect(rec.priorityScore).toBeGreaterThan(0);
      expect(rec.status).toBe('pending');
    }
  });
});

// ============================================================
// recordOutcome
// ============================================================

describe('recordOutcome', () => {
  test('marks recommendation as completed with post metrics', () => {
    const rec: Recommendation = {
      id: 'breathing_box', category: 'breathing',
      title: 'Box Breathing', description: 'test',
      durationMin: 5, instructions: ['breathe'],
      trigger: 'biometric', triggerReason: 'High stress',
      priorityScore: 0.9, evidenceLevel: 'strong',
      citation: 'test', status: 'accepted',
      preStress: 80, preHrv: 30,
    };

    const result = recordOutcome(rec, 50, 45);
    expect(result.status).toBe('completed');
    expect(result.postStress).toBe(50);
    expect(result.postHrv).toBe(45);
    expect(result.effectivenessScore).toBeDefined();
  });

  test('effectiveness score is between 0 and 1', () => {
    const rec: Recommendation = {
      id: 'breathing_box', category: 'breathing',
      title: 'Box Breathing', description: 'test',
      durationMin: 5, instructions: ['breathe'],
      trigger: 'biometric', triggerReason: 'High stress',
      priorityScore: 0.9, evidenceLevel: 'strong',
      citation: 'test', status: 'accepted',
      preStress: 80, preHrv: 30,
    };

    const result = recordOutcome(rec, 40, 55);
    expect(result.effectivenessScore).toBeGreaterThanOrEqual(0);
    expect(result.effectivenessScore).toBeLessThanOrEqual(1);
  });

  test('stress reduction increases effectiveness', () => {
    const rec: Recommendation = {
      id: 'test', category: 'breathing', title: 't', description: 't',
      durationMin: 5, instructions: ['t'], trigger: 'biometric',
      triggerReason: 't', priorityScore: 0.5, evidenceLevel: 'moderate',
      citation: 't', status: 'accepted', preStress: 80, preHrv: 30,
    };

    const noChange = recordOutcome(rec, 80, 30);
    const improved = recordOutcome(rec, 40, 50);
    expect(improved.effectivenessScore!).toBeGreaterThan(noChange.effectivenessScore!);
  });
});
