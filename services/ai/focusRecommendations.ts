/**
 * Seren — Focus Mood Recommendations via Groq
 * =============================================
 * Uses the elevated physiological features from the Focus model
 * to generate personalized, actionable tips via Groq (llama-3.1-8b-instant).
 * Falls back to rule-based tips if Groq is unavailable.
 */

import { ElevatedFeature, FocusLevel } from './types';
import { getLLMConfig, getLLMApiKey } from './llmService';

// ── Groq API Call ────────────────────────────────────────────

async function callGroq(prompt: string): Promise<string> {
  const apiKey = getLLMApiKey();

  if (!apiKey) {
    console.warn('[Seren] No Groq API key configured - using fallback tips');
    throw new Error('No API key');
  }

  console.log('[Seren] Calling Groq API (model: llama-3.1-8b-instant, max_tokens: 300)');
  const startTime = Date.now();

  // Create abort controller with 10 second timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);

  try {
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'llama-3.1-8b-instant',
        messages: [
          { role: 'system', content: FOCUS_SYSTEM_PROMPT },
          { role: 'user', content: prompt },
        ],
        max_tokens: 60,
        temperature: 0.5,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    const elapsed = Date.now() - startTime;

    if (!response.ok) {
      console.warn(`[Seren] Groq API error: HTTP ${response.status} (${elapsed}ms)`);
      const errorText = await response.text().catch(() => '');
      console.warn(`[Seren] Groq error response: ${errorText.substring(0, 200)}`);
      throw new Error(`Groq ${response.status}`);
    }

    const data = await response.json();
    const tips = data.choices?.[0]?.message?.content ?? '';

    if (!tips || tips.length === 0) {
      console.warn('[Seren] Groq returned empty response');
      throw new Error('empty response');
    }

    console.log(`[Seren] Groq tips generated successfully (${elapsed}ms): "${tips.substring(0, 80)}..."`);
    return tips;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      console.warn('[Seren] Groq API timeout after 10 seconds');
      throw new Error('Groq timeout');
    }
    throw error;
  }
}

const FOCUS_SYSTEM_PROMPT = `You are Seren, a calm physiological companion for students. Based on wearable biometric readings, give ONE gentle observation (under 20 words) that names what the body is doing and suggests a micro-action. Never use words like "distracted", "unfocused", or "scattered" — frame it in terms of the nervous system or physiology. Tone: warm, indirect, science-grounded. Start with "Your" or a soft verb. No intro, no punctuation at end.
Example: Your autonomic system is slightly activated — two slow exhales can bring heart rate back toward baseline`;

// ── Prompt Builder ───────────────────────────────────────────

function buildPrompt(
  focusScore: number,
  focusLevel: FocusLevel,
  elevatedFeatures: ElevatedFeature[],
  isStudent: boolean,
): string {
  const context = isStudent ? 'student heading into an exam' : 'professional before a demanding task';
  const readings = elevatedFeatures.length > 0
    ? elevatedFeatures.slice(0, 3)
        .map(f => `${f.label}: ${f.value.toFixed(1)} (${f.direction === 'high' ? 'above' : 'below'} resting baseline)`)
        .join('; ')
    : 'all signals within comfortable resting range';

  return `${context}. Readiness score: ${Math.round(focusScore)}/100. Live readings: ${readings}. One calm, science-grounded observation with a micro-action:`;
}

// ── Rule-Based Fallback ───────────────────────────────────────

const RULE_TIPS: Record<string, string[]> = {
  'Heart Rate':            ['Your heart rate is a touch elevated — a slow exhale can ease it back down'],
  'HRV (RMSSD)':          ['Your heart rhythm variability is low — a few box-breaths can invite more balance'],
  'HRV (SDNN)':           ['Your nervous system is carrying some load — a quiet minute to breathe can help it settle'],
  'Sympathovagal Balance': ['Your sympathetic system is slightly dominant — a long exhale shifts the balance gently'],
  'Mental Complexity':     ['Your signal entropy is lower than usual — stepping away briefly lets your brain reorganise'],
  'Skin Temperature':      ['Your peripheral temperature is elevated — cool wrists under water can signal safety to your body'],
  'Stress Pattern':        ['Your autonomic pattern shows sustained activation — even 90 seconds of stillness can reset it'],
  'Parasympathetic Tone':  ['Your recovery system is quiet right now — slow nasal breathing helps bring it online'],
};

const DEFAULT_TIP = 'Your body looks settled — a single slow breath before you begin can anchor your focus';

function ruleBasedTips(elevatedFeatures: ElevatedFeature[]): string[] {
  if (elevatedFeatures.length === 0) return [DEFAULT_TIP];
  const top = elevatedFeatures[0];
  const pool = RULE_TIPS[top.label] ?? [DEFAULT_TIP];
  return [pool[Math.floor(Math.random() * pool.length)]];
}

// ── Public API ───────────────────────────────────────────────

export async function getFocusTips(
  focusScore: number,
  focusLevel: FocusLevel,
  elevatedFeatures: ElevatedFeature[],
  isStudent: boolean,
  groqApiKey?: string,
): Promise<string[]> {
  if (!groqApiKey && !getLLMConfig()) {
    console.log('[Seren] No LLM configured - using rule-based tips');
    return ruleBasedTips(elevatedFeatures);
  }

  try {
    const prompt = buildPrompt(focusScore, focusLevel, elevatedFeatures, isStudent);
    const raw = await callGroq(prompt);
    console.log(`[Seren] Raw Groq response: "${raw.substring(0, 100)}..."`);

    const tip = raw.split('\n')[0].replace(/^[\|\-\*\d\.\s]+/, '').trim();

    if (tip.length < 8) {
      console.warn('[Seren] Groq response too short - falling back');
      throw new Error('empty response after parsing');
    }

    console.log(`[Seren] Groq tip: "${tip}"`);
    return [tip];
  } catch (err) {
    console.warn(`[Seren] Groq tips failed, falling back to rule-based: ${err instanceof Error ? err.message : String(err)}`);
    return ruleBasedTips(elevatedFeatures);
  }
}
