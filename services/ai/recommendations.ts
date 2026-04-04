/**
 * Seren AI — Recommendation Engine
 * ===================================
 * Rule-based intervention recommendations using clinical thresholds.
 *
 * Inputs: current stress, sleep quality, anxiety, time of day, check-in data.
 * Outputs: prioritized list of evidence-based interventions.
 *
 * Each recommendation includes:
 *   - Clinical evidence level and citation
 *   - Why it was triggered (biometric, schedule, pattern, etc.)
 *   - Pre/post tracking hooks for effectiveness measurement
 */

import {
  Recommendation,
  InterventionCategory,
  RecommendationTrigger,
  StressPrediction,
  AnxietyPrediction,
  SleepAnalysis,
  CheckinAnalysis,
  PersonalBaseline,
  LocationDiversitySummary,
  SunlightExposureSummary,
} from './types';

// ============================================================
// Intervention Library
// ============================================================

interface InterventionTemplate {
  id: string;
  category: InterventionCategory;
  title: string;
  description: string;
  durationMin: number;
  instructions: string[];
  evidenceLevel: 'strong' | 'moderate' | 'emerging';
  citation: string;
}

const INTERVENTIONS: InterventionTemplate[] = [
  // ---- Breathing exercises ----
  {
    id: 'breathing_478',
    category: 'breathing',
    title: '4-7-8 Breathing',
    description: 'A calming breath pattern that activates your parasympathetic nervous system.',
    durationMin: 4,
    instructions: [
      'Find a comfortable position',
      'Breathe in through your nose for 4 seconds',
      'Hold your breath for 7 seconds',
      'Exhale slowly through your mouth for 8 seconds',
      'Repeat 4 cycles',
    ],
    evidenceLevel: 'strong',
    citation: 'Weil, A. (2015). Spontaneous Healing. Ballantine Books.',
  },
  {
    id: 'breathing_box',
    category: 'breathing',
    title: 'Box Breathing',
    description: 'Used by Navy SEALs to manage acute stress. Equal-length inhale, hold, exhale, hold.',
    durationMin: 5,
    instructions: [
      'Sit upright and relax your shoulders',
      'Breathe in for 4 seconds',
      'Hold for 4 seconds',
      'Breathe out for 4 seconds',
      'Hold for 4 seconds',
      'Repeat for 5 minutes',
    ],
    evidenceLevel: 'strong',
    citation: 'Ma et al. (2017). Effect of diaphragmatic breathing on attention. Front Psychol.',
  },
  {
    id: 'breathing_coherent',
    category: 'breathing',
    title: 'Coherent Breathing',
    description: 'Breathe at 5 breaths per minute to maximize heart rate variability.',
    durationMin: 10,
    instructions: [
      'Set a timer for 10 minutes',
      'Breathe in for 6 seconds',
      'Breathe out for 6 seconds',
      'Keep the rhythm gentle and steady',
      'Focus on the smooth transition between inhale and exhale',
    ],
    evidenceLevel: 'strong',
    citation: 'Lehrer & Gevirtz (2014). Heart rate variability biofeedback. Front Public Health.',
  },

  // ---- Physical exercises ----
  {
    id: 'physical_walk',
    category: 'physical',
    title: 'Mindful Walk',
    description: 'A short walk with attention to your surroundings and body sensations.',
    durationMin: 10,
    instructions: [
      'Step outside or find a quiet hallway',
      'Walk at a comfortable pace',
      'Notice 5 things you can see',
      'Notice 3 things you can hear',
      'Focus on the sensation of your feet touching the ground',
    ],
    evidenceLevel: 'strong',
    citation: 'Oppezzo & Schwartz (2014). Give your ideas some legs. J Exp Psych: Learning, Memory, Cognition.',
  },
  {
    id: 'physical_stretch',
    category: 'physical',
    title: 'Desk Stretch Sequence',
    description: 'Release physical tension stored in neck, shoulders, and back.',
    durationMin: 5,
    instructions: [
      'Roll your neck slowly in circles (30 sec each direction)',
      'Shrug shoulders up to ears, hold 5 sec, release (5 reps)',
      'Clasp hands behind back, open chest, hold 15 sec',
      'Reach arms overhead and lean to each side (15 sec each)',
      'Stand and touch your toes, hanging relaxed for 15 sec',
    ],
    evidenceLevel: 'moderate',
    citation: 'Cowen & Adams (2005). Physical activity is associated with reduced anxiety. J Psychosom Res.',
  },

  // ---- Meditation ----
  {
    id: 'meditation_body_scan',
    category: 'meditation',
    title: 'Body Scan Meditation',
    description: 'Systematically relax each part of your body to release stored tension.',
    durationMin: 10,
    instructions: [
      'Lie down or sit comfortably',
      'Close your eyes and take 3 deep breaths',
      'Focus attention on your toes, noticing any tension',
      'Slowly move attention up through feet, legs, torso, arms, neck, head',
      'At each area, breathe into any tension and let it soften',
    ],
    evidenceLevel: 'strong',
    citation: 'Kabat-Zinn (1990). Full Catastrophe Living. Bantam Books.',
  },
  {
    id: 'meditation_grounding',
    category: 'meditation',
    title: '5-4-3-2-1 Grounding',
    description: 'A sensory grounding technique for moments of acute anxiety.',
    durationMin: 3,
    instructions: [
      'Name 5 things you can see',
      'Name 4 things you can touch',
      'Name 3 things you can hear',
      'Name 2 things you can smell',
      'Name 1 thing you can taste',
    ],
    evidenceLevel: 'moderate',
    citation: 'Bremner et al. (2017). Grounding techniques for PTSD. J Trauma Stress.',
  },

  // ---- Journaling ----
  {
    id: 'journaling_gratitude',
    category: 'journaling',
    title: 'Gratitude Reflection',
    description: 'Write or think about 3 things you are grateful for today.',
    durationMin: 5,
    instructions: [
      'Take a moment to pause',
      'Think of 3 specific things from today you appreciate',
      'For each one, think about WHY it matters to you',
      'Notice how reflecting on these makes you feel',
    ],
    evidenceLevel: 'strong',
    citation: 'Emmons & McCullough (2003). Counting blessings vs burdens. J Pers Soc Psychol.',
  },
  {
    id: 'journaling_expressive',
    category: 'journaling',
    title: 'Expressive Writing',
    description: 'Write freely about what is on your mind for 10 minutes without editing.',
    durationMin: 10,
    instructions: [
      'Set a timer for 10 minutes',
      'Write continuously about your thoughts and feelings',
      'Do not worry about grammar or spelling',
      'If you get stuck, repeat the last word until a new thought comes',
      'When done, you can keep or discard what you wrote',
    ],
    evidenceLevel: 'strong',
    citation: 'Pennebaker & Beall (1986). Confronting a traumatic event. J Abnormal Psych.',
  },

  // ---- Sleep hygiene ----
  {
    id: 'sleep_wind_down',
    category: 'sleep_hygiene',
    title: 'Wind-Down Routine',
    description: 'Prepare your body and mind for restful sleep.',
    durationMin: 30,
    instructions: [
      'Dim the lights 1 hour before bed',
      'Put away screens (phone, laptop, TV)',
      'Do a gentle activity: reading, stretching, or light journaling',
      'Keep the room cool (65-68°F / 18-20°C)',
      'Try the 4-7-8 breathing as you lie down',
    ],
    evidenceLevel: 'strong',
    citation: 'Irish et al. (2015). Role of sleep hygiene in promoting public health. Sleep Med Rev.',
  },
  {
    id: 'sleep_consistent',
    category: 'sleep_hygiene',
    title: 'Schedule Alignment',
    description: 'Your bedtime has been inconsistent. A regular schedule improves sleep quality.',
    durationMin: 0,
    instructions: [
      'Choose a target bedtime and wake time',
      'Stick to it within 30 minutes, even on weekends',
      'Set a "bedtime alarm" to remind you to start winding down',
    ],
    evidenceLevel: 'strong',
    citation: 'Walker, M. (2017). Why We Sleep. Scribner.',
  },

  // ---- Social ----
  {
    id: 'social_reach_out',
    category: 'social',
    title: 'Connect With Someone',
    description: 'Social connection is one of the strongest buffers against stress.',
    durationMin: 10,
    instructions: [
      'Think of someone you trust — a friend, family member, or colleague',
      'Send them a brief message or give them a call',
      'Share how you are feeling, even briefly',
      'Connection does not need to be deep to be helpful',
    ],
    evidenceLevel: 'strong',
    citation: 'Holt-Lunstad et al. (2010). Social relationships and mortality risk. PLoS Med.',
  },

  // ---- Outdoor / Sunlight ----
  {
    id: 'outdoor_sunlight',
    category: 'outdoor',
    title: 'Get Some Sunlight',
    description: 'Sunlight exposure helps regulate your circadian rhythm and boosts serotonin production.',
    durationMin: 15,
    instructions: [
      'Step outside for 15 minutes',
      'Leave your sunglasses off if possible (safe for short periods)',
      'Walk or sit in a sunny area',
      'The best time is between 10am and 3pm for vitamin D synthesis',
    ],
    evidenceLevel: 'strong',
    citation: 'Holick, M. (2007). Vitamin D deficiency. N Engl J Med.',
  },
  {
    id: 'outdoor_walk_sunshine',
    category: 'outdoor',
    title: 'Sunshine Walk',
    description: 'A walk in daylight combines physical activity with light exposure — a double mood booster.',
    durationMin: 20,
    instructions: [
      'Head outside during daylight hours',
      'Walk at a comfortable pace for 20 minutes',
      'Try to stay in sunlit areas',
      'Pay attention to how the warmth feels on your skin',
    ],
    evidenceLevel: 'strong',
    citation: 'Mead et al. (2009). Exercise for depression. Cochrane Database Syst Rev.',
  },

  // ---- Exploration / Location Diversity ----
  {
    id: 'explore_new_place',
    category: 'exploration',
    title: 'Explore Somewhere New',
    description: 'Breaking routine by visiting a new place can lift mood and reduce depressive patterns.',
    durationMin: 30,
    instructions: [
      'Think of a place you have not been to recently — a park, cafe, or neighborhood',
      'Take a different route than usual',
      'Notice your surroundings with curiosity',
      'Even a short detour from your routine counts',
    ],
    evidenceLevel: 'moderate',
    citation: 'Heller et al. (2020). Association between real-world experiential diversity and positive affect. Nature Neuroscience.',
  },
  {
    id: 'explore_nature',
    category: 'exploration',
    title: 'Visit a Green Space',
    description: 'Time in nature reduces cortisol and improves mood. Even 20 minutes helps.',
    durationMin: 20,
    instructions: [
      'Find a nearby park, garden, or green area',
      'Walk slowly and notice the natural surroundings',
      'Sit for a few minutes and listen to the ambient sounds',
      'Leave your phone on silent if you can',
    ],
    evidenceLevel: 'strong',
    citation: 'Hunter et al. (2019). Urban nature experiences reduce stress. Front Psychol.',
  },
];

// ============================================================
// Clinical Thresholds (triggers)
// ============================================================

const THRESHOLDS = {
  stressHigh: 70,
  stressElevated: 50,
  anxietyModerate: 50,
  anxietySevere: 70,
  sleepQualityPoor: 50,
  sleepEfficiencyLow: 0.75,
  hrvCriticallyLow: 20,      // RMSSD in ms
  hrElevated: 100,            // BPM
};

// ============================================================
// Public API
// ============================================================

/**
 * Generate recommendations based on current state.
 *
 * @param stress - Latest stress prediction
 * @param anxiety - Latest anxiety prediction
 * @param lastSleep - Most recent sleep analysis (if available)
 * @param lastCheckin - Most recent voice check-in (if available)
 * @param baseline - Personal baseline (if available)
 * @param recentRecommendations - Recently shown recommendations (to avoid repetition)
 * @returns Sorted list of recommendations (highest priority first)
 */
export function generateRecommendations(
  stress: StressPrediction,
  anxiety: AnxietyPrediction | null,
  lastSleep: SleepAnalysis | null,
  lastCheckin: CheckinAnalysis | null,
  baseline: PersonalBaseline | null,
  recentRecommendations: string[] = [],
  locationDiversity: LocationDiversitySummary | null = null,
  sunlightExposure: SunlightExposureSummary | null = null,
): Recommendation[] {
  const candidates: Recommendation[] = [];
  const now = new Date();
  const hour = now.getHours();

  // ---- Stress-triggered recommendations ----
  if (stress.stressScore >= THRESHOLDS.stressHigh) {
    // Acute high stress — immediate intervention
    candidates.push(
      makeRecommendation('breathing_box', 'biometric',
        `Stress score is ${stress.stressScore}/100 — elevated`, 0.95),
      makeRecommendation('meditation_grounding', 'biometric',
        'High stress detected — grounding can help', 0.85),
    );
  } else if (stress.stressScore >= THRESHOLDS.stressElevated) {
    candidates.push(
      makeRecommendation('breathing_coherent', 'biometric',
        'Moderate stress detected — coherent breathing can boost HRV', 0.7),
      makeRecommendation('physical_walk', 'biometric',
        'A short walk can help lower stress hormones', 0.6),
    );
  }

  // ---- Anxiety-triggered recommendations ----
  if (anxiety && anxiety.anxietyIndex >= THRESHOLDS.anxietySevere) {
    candidates.push(
      makeRecommendation('meditation_grounding', 'biometric',
        'Anxiety is elevated — 5-4-3-2-1 grounding is recommended', 0.9),
      makeRecommendation('breathing_478', 'biometric',
        '4-7-8 breathing activates your calming nervous system', 0.85),
    );
    if (anxiety.sustained) {
      candidates.push(
        makeRecommendation('social_reach_out', 'pattern',
          'Sustained anxiety detected — reaching out can help', 0.8),
      );
    }
  } else if (anxiety && anxiety.anxietyIndex >= THRESHOLDS.anxietyModerate) {
    candidates.push(
      makeRecommendation('journaling_expressive', 'biometric',
        'Writing about your feelings can help process moderate anxiety', 0.6),
    );
  }

  // ---- Sleep-triggered recommendations ----
  if (lastSleep) {
    if (lastSleep.qualityScore < THRESHOLDS.sleepQualityPoor) {
      candidates.push(
        makeRecommendation('sleep_wind_down', 'pattern',
          `Last night's sleep quality was ${lastSleep.qualityScore}/100`, 0.7),
      );
    }
    if (lastSleep.sleepEfficiency < THRESHOLDS.sleepEfficiencyLow) {
      candidates.push(
        makeRecommendation('sleep_consistent', 'pattern',
          `Sleep efficiency was ${Math.round(lastSleep.sleepEfficiency * 100)}%`, 0.65),
      );
    }
    if (lastSleep.consistencyScore < 50 && baseline) {
      candidates.push(
        makeRecommendation('sleep_consistent', 'pattern',
          'Your sleep schedule has been inconsistent', 0.6),
      );
    }
  }

  // ---- Check-in triggered recommendations ----
  if (lastCheckin) {
    if (lastCheckin.sentimentScore < -0.5) {
      candidates.push(
        makeRecommendation('journaling_expressive', 'checkin',
          'Your check-in expressed distress — writing can help process these feelings', 0.75),
      );
    }
    if (lastCheckin.emotionScores.fatigue > 0.5) {
      candidates.push(
        makeRecommendation('physical_stretch', 'checkin',
          'You mentioned feeling tired — gentle stretching can boost energy', 0.5),
      );
    }
    if (lastCheckin.emotionScores.sadness > 0.5) {
      candidates.push(
        makeRecommendation('social_reach_out', 'checkin',
          'It sounds like you could use some connection right now', 0.7),
      );
    }
  }

  // ---- Time-based recommendations ----
  if (hour >= 20 && hour <= 23) {
    // Evening: suggest wind-down if stress is elevated
    if (stress.stressScore > 40) {
      candidates.push(
        makeRecommendation('sleep_wind_down', 'schedule',
          'It\'s evening and your stress is elevated — time to wind down', 0.55),
      );
    }
  }
  if (hour >= 6 && hour <= 9) {
    // Morning: gratitude
    candidates.push(
      makeRecommendation('journaling_gratitude', 'schedule',
        'Start your morning with a brief gratitude reflection', 0.3),
    );
  }

  // ---- Location diversity-triggered recommendations ----
  if (locationDiversity) {
    if (locationDiversity.isMonotonous) {
      candidates.push(
        makeRecommendation('explore_new_place', 'pattern',
          'Your routine has been limited to the same places — try somewhere new', 0.7),
      );
    }
    if (locationDiversity.diversityScore < 20) {
      candidates.push(
        makeRecommendation('explore_nature', 'pattern',
          `Location diversity is low (${locationDiversity.diversityScore}/100) — a change of scenery can help`, 0.6),
      );
    }
    if (locationDiversity.isMonotonous && stress.stressScore >= THRESHOLDS.stressElevated) {
      candidates.push(
        makeRecommendation('explore_nature', 'pattern',
          'High stress + repetitive routine — nature exposure can break the cycle', 0.8),
      );
    }
  }

  // ---- Sunlight exposure-triggered recommendations ----
  if (sunlightExposure) {
    if (sunlightExposure.goalProgress < 0.5 && sunlightExposure.isVitaminDWindow) {
      candidates.push(
        makeRecommendation('outdoor_sunlight', 'pattern',
          `Only ${sunlightExposure.totalOutdoorMinutes}m of sunlight today — the vitamin D window is open now`, 0.8),
      );
    } else if (sunlightExposure.goalProgress < 0.5 && hour < 15) {
      candidates.push(
        makeRecommendation('outdoor_walk_sunshine', 'schedule',
          `You've had ${sunlightExposure.totalOutdoorMinutes}m of outdoor time — try to get more before 3pm`, 0.6),
      );
    }
    if (sunlightExposure.totalOutdoorMinutes < 10 && hour >= 14) {
      candidates.push(
        makeRecommendation('outdoor_sunlight', 'pattern',
          'Very low sunlight exposure today — even 15 minutes helps your mood', 0.75),
      );
    }
  }

  // ---- Filter out recently shown recommendations ----
  const filtered = candidates.filter(r => !recentRecommendations.includes(r.id));

  // ---- Deduplicate by intervention ID (keep highest priority) ----
  const deduped = new Map<string, Recommendation>();
  for (const rec of filtered) {
    const existing = deduped.get(rec.id);
    if (!existing || rec.priorityScore > existing.priorityScore) {
      deduped.set(rec.id, rec);
    }
  }

  // ---- Sort by priority and return top 3 ----
  return Array.from(deduped.values())
    .sort((a, b) => b.priorityScore - a.priorityScore)
    .slice(0, 3);
}

/**
 * Record recommendation outcome for effectiveness tracking.
 */
export function recordOutcome(
  recommendation: Recommendation,
  postStress: number,
  postHrv: number,
): Recommendation {
  const effectivenessScore = computeEffectiveness(
    recommendation.preStress ?? 0,
    postStress,
    recommendation.preHrv ?? 0,
    postHrv,
  );

  return {
    ...recommendation,
    status: 'completed',
    postStress,
    postHrv,
    effectivenessScore,
  };
}

// ============================================================
// Internal Helpers
// ============================================================

function makeRecommendation(
  interventionId: string,
  trigger: RecommendationTrigger,
  triggerReason: string,
  priorityScore: number,
): Recommendation {
  const template = INTERVENTIONS.find(i => i.id === interventionId);
  if (!template) {
    throw new Error(`Unknown intervention: ${interventionId}`);
  }

  return {
    id: template.id,
    category: template.category,
    title: template.title,
    description: template.description,
    durationMin: template.durationMin,
    instructions: template.instructions,
    trigger,
    triggerReason,
    priorityScore,
    evidenceLevel: template.evidenceLevel,
    citation: template.citation,
    status: 'pending',
  };
}

function computeEffectiveness(
  preStress: number,
  postStress: number,
  preHrv: number,
  postHrv: number,
): number {
  // Score 0-1 based on improvement in stress and HRV
  let score = 0.5; // Baseline

  // Stress reduction (max +0.3)
  if (preStress > 0) {
    const stressReduction = (preStress - postStress) / preStress;
    score += Math.min(0.3, stressReduction * 0.5);
  }

  // HRV improvement (max +0.2)
  if (preHrv > 0) {
    const hrvImprovement = (postHrv - preHrv) / preHrv;
    score += Math.min(0.2, hrvImprovement * 0.3);
  }

  return Math.max(0, Math.min(1, score));
}
