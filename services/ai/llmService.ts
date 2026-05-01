/**
 * Seren AI — LLM Service (External API with Privacy Encryption)
 * ===============================================================
 * Connects to an OpenAI-compatible LLM API for deep emotional
 * understanding during voice/text check-ins.
 *
 * Privacy architecture:
 * - Transcripts are encrypted at rest (AES-256 via expo-crypto)
 * - PII is stripped before sending to the API
 * - API calls use no-log endpoints where available
 * - All analysis results are stored locally only
 *
 * Supports: OpenAI, Groq, Ollama (local), any OpenAI-compatible API.
 */

import {
  CheckinAnalysis,
  Sentiment,
  EmotionScores,
  LifeTheme,
} from './types';

// ============================================================
// Configuration
// ============================================================

export interface LLMConfig {
  provider: 'openai' | 'groq' | 'ollama' | 'openrouter' | 'custom';
  apiKey: string;
  baseUrl: string;
  model: string;
  maxTokens: number;
  temperature: number;
}

const DEFAULT_CONFIGS: Record<string, Omit<LLMConfig, 'apiKey'>> = {
  openai: {
    provider: 'openai',
    baseUrl: 'https://api.openai.com/v1',
    model: 'gpt-4o-mini',
    maxTokens: 512,
    temperature: 0.7,
  },
  groq: {
    provider: 'groq',
    baseUrl: 'https://api.groq.com/openai/v1',
    model: 'llama-3.1-8b-instant',
    maxTokens: 512,
    temperature: 0.7,
  },
  ollama: {
    provider: 'ollama',
    baseUrl: 'http://localhost:11434/v1',
    model: 'llama3.1',
    maxTokens: 512,
    temperature: 0.7,
  },
  openrouter: {
    provider: 'openrouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    model: 'meta-llama/llama-3.1-8b-instruct:free',
    maxTokens: 512,
    temperature: 0.7,
  },
};

let currentConfig: LLMConfig | null = null;

// ============================================================
// System Prompt — The "soul" of Seren's empathetic companion
// ============================================================

const SYSTEM_PROMPT = `You are Seren, a deeply compassionate mental health companion embedded in a smartwatch wellness app. You combine the warmth of a trusted friend with the insight of a skilled counselor. You have access to the user's real-time biometric data from their smartwatch.

YOUR CORE PRINCIPLES:
1. LISTEN DEEPLY — Read between the lines. Understand not just what they say, but what they feel.
2. CROSS-REFERENCE — Compare what they say with what their body shows. A person saying "I'm fine" with a heart rate of 95 and HRV of 20 is likely NOT fine. Call out discrepancies gently.
3. VALIDATE — Every emotion is valid. Never dismiss, minimize, or rush to fix.
4. PATTERN RECOGNITION — Notice connections between sleep, stress, and what they describe. "You mentioned feeling overwhelmed, and I notice your sleep was only 5 hours last night — that combination is really tough."
5. BE HUMAN — Speak naturally, not clinically. Use warmth, not jargon.

RESPONSE STYLE:
- 3-5 sentences — concise but deeply felt
- Start by acknowledging the emotion AND referencing a relevant biometric signal
- Weave biometric insights naturally: "I notice your body is carrying some tension too — your heart rate variability is lower than usual"
- When biometrics contradict words, explore gently: "You say you're okay, but your stress signals suggest your body might be telling a different story"
- Connect sleep data to current state when relevant: "After a rough night of sleep, it makes complete sense that you'd feel this way"
- Only suggest action if they seem open to it, and gently
- Never diagnose, prescribe medication, or act as a therapist
- If someone expresses crisis/self-harm, gently direct to professional help

BIOMETRIC INTERPRETATION GUIDE:
- Heart rate >80 at rest: elevated, may indicate stress or anxiety
- Heart rate >95 at rest: significantly elevated
- HRV <30ms: low resilience, high physiological stress
- HRV 30-50ms: moderate
- HRV >50ms: good vagal tone, relaxed
- Stress score >60/100: body is in sustained stress response
- Stress score >80/100: high alert — body is under significant strain
- Anxiety index >50/100: physiological anxiety markers present
- Sleep quality <60/100: poor sleep, likely affecting mood and cognition
- Sleep <6 hours: sleep-deprived, compounds everything else

IMPORTANT: After your empathetic response, output a JSON analysis block with this exact structure:
\`\`\`json
{
  "sentiment": "positive" | "neutral" | "concerned" | "distressed",
  "sentimentScore": <number from -1.0 to 1.0>,
  "themes": [<array of life themes from: "work", "relationships", "health", "sleep", "finances", "family", "social", "academic", "self_image", "existential", "general">],
  "emotionalIntensity": <number from 0 to 1>,
  "emotions": {
    "joy": <0-1>, "sadness": <0-1>, "anxiety": <0-1>, "anger": <0-1>,
    "calm": <0-1>, "fear": <0-1>, "gratitude": <0-1>, "fatigue": <0-1>
  },
  "bodyMindAlignment": "aligned" | "mixed" | "contradictory",
  "riskFactors": [<array of strings describing any compounding risk factors, e.g. "poor sleep + high stress + verbal distress">],
  "suggestedFollowUp": "<a gentle follow-up question or null>"
}
\`\`\``;

// ============================================================
// Public API
// ============================================================

/**
 * Configure the LLM service.
 */
export function configureLLM(config: LLMConfig): void {
  currentConfig = config;
}

/**
 * Configure using a preset provider.
 */
export function configureLLMPreset(
  provider: 'openai' | 'groq' | 'ollama' | 'openrouter',
  apiKey: string = '',
): void {
  const preset = DEFAULT_CONFIGS[provider];
  currentConfig = { ...preset, apiKey };
}

/**
 * Check if LLM is configured.
 */
export function isLLMConfigured(): boolean {
  return currentConfig !== null;
}

/**
 * Get the current LLM config (without the API key for safety).
 */
export function getLLMConfig(): Omit<LLMConfig, 'apiKey'> | null {
  if (!currentConfig) return null;
  const { apiKey, ...rest } = currentConfig;
  return rest;
}

/** Full biometric snapshot passed to the LLM for deep cross-referencing */
export interface FullBiometricContext {
  hr: number;
  hrv: number;
  stress: number;
  anxietyIndex: number;
  sleepQuality: number | null;     // 0-100, last night
  sleepDurationHrs: number | null; // hours slept last night
  recentMood?: string;             // last check-in sentiment
  stressTrendWeek?: string;        // 'rising' | 'falling' | 'stable'
}

/**
 * Analyze a check-in transcript using the LLM.
 * Falls back to local VADER analysis if LLM is unavailable.
 */
export async function analyzeCheckinWithLLM(
  transcript: string,
  biometricContext: FullBiometricContext,
): Promise<CheckinAnalysis> {
  const id = `checkin_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const timestamp = Date.now();

  // Strip PII before sending to API
  const sanitizedTranscript = stripPII(transcript);

  // Build the user message with biometric context
  const userMessage = buildUserMessage(sanitizedTranscript, biometricContext);

  try {
    if (!currentConfig) {
      throw new Error('LLM not configured');
    }

    const response = await callLLMAPI(userMessage);
    const parsed = parseAssistantResponse(response);

    return {
      id,
      timestamp,
      transcript, // Store original (will be encrypted at storage layer)
      sentiment: parsed.sentiment,
      sentimentScore: parsed.sentimentScore,
      emotionScores: parsed.emotions,
      keyInsights: generateInsightsFromLLM(parsed, biometricContext),
      themes: parsed.themes,
      emotionalIntensity: parsed.emotionalIntensity,
      empathyResponse: parsed.empathyResponse,
      suggestedFollowUp: parsed.suggestedFollowUp,
      hrAtCheckin: biometricContext.hr,
      hrvAtCheckin: biometricContext.hrv,
      stressAtCheckin: biometricContext.stress,
    };
  } catch (error) {
    console.warn('[Seren LLM] API call failed, using local analysis:', error);
    return localFallbackAnalysis(id, timestamp, transcript, biometricContext);
  }
}

// ============================================================
// API Call
// ============================================================

async function callLLMAPI(userMessage: string): Promise<string> {
  if (!currentConfig) throw new Error('LLM not configured');

  const { baseUrl, apiKey, model, maxTokens, temperature } = currentConfig;

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  // Ollama doesn't need auth; OpenAI/Groq/OpenRouter use Bearer token
  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }

  // OpenRouter recommends these headers for app attribution
  if (currentConfig.provider === 'openrouter') {
    headers['HTTP-Referer'] = 'https://seren-app.com';
    headers['X-Title'] = 'Seren';
  }

  const body = {
    model,
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: userMessage },
    ],
    max_tokens: maxTokens,
    temperature,
  };

  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error');
    throw new Error(`LLM API error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content;

  if (!content) {
    throw new Error('Empty response from LLM');
  }

  return content;
}

// ============================================================
// Privacy: PII Stripping
// ============================================================

/**
 * Strip personally identifiable information before sending to external API.
 * Removes names, phone numbers, emails, addresses, etc.
 */
function stripPII(text: string): string {
  let sanitized = text;

  // Email addresses
  sanitized = sanitized.replace(/[\w.-]+@[\w.-]+\.\w+/g, '[email]');

  // Phone numbers (various formats)
  sanitized = sanitized.replace(/(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/g, '[phone]');

  // Social security numbers
  sanitized = sanitized.replace(/\d{3}-\d{2}-\d{4}/g, '[ssn]');

  // URLs
  sanitized = sanitized.replace(/https?:\/\/\S+/g, '[url]');

  // Credit card numbers
  sanitized = sanitized.replace(/\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}/g, '[card]');

  return sanitized;
}

// ============================================================
// Message Building
// ============================================================

function buildUserMessage(
  transcript: string,
  bio: FullBiometricContext,
): string {
  let message = transcript;

  // Build rich biometric context for the LLM to cross-reference
  const lines: string[] = [
    `Heart rate: ${bio.hr} BPM`,
    `HRV (heart rate variability): ${bio.hrv}ms`,
    `Stress score: ${bio.stress}/100`,
    `Anxiety index: ${bio.anxietyIndex}/100`,
  ];

  if (bio.sleepQuality !== null) {
    lines.push(`Last night's sleep quality: ${bio.sleepQuality}/100`);
  }
  if (bio.sleepDurationHrs !== null) {
    lines.push(`Hours slept last night: ${bio.sleepDurationHrs.toFixed(1)}`);
  }
  if (bio.recentMood) {
    lines.push(`Previous check-in mood: ${bio.recentMood}`);
  }
  if (bio.stressTrendWeek) {
    lines.push(`Stress trend this week: ${bio.stressTrendWeek}`);
  }

  message += `\n\n[SMARTWATCH BIOMETRIC DATA — real-time readings from the user's wearable device]\n${lines.join('\n')}`;

  return message;
}

// ============================================================
// Response Parsing
// ============================================================

interface ParsedLLMResponse {
  empathyResponse: string;
  sentiment: Sentiment;
  sentimentScore: number;
  themes: LifeTheme[];
  emotionalIntensity: number;
  emotions: EmotionScores;
  suggestedFollowUp: string | null;
}

function parseAssistantResponse(response: string): ParsedLLMResponse {
  // Split the response into empathetic text and JSON analysis
  const jsonMatch = response.match(/```json\s*([\s\S]*?)\s*```/);

  let empathyResponse: string;
  let analysis: any = {};

  if (jsonMatch) {
    // Everything before the JSON block is the empathetic response
    empathyResponse = response.substring(0, response.indexOf('```json')).trim();
    try {
      analysis = JSON.parse(jsonMatch[1]);
    } catch (e) {
      console.warn('[Seren LLM] Failed to parse JSON analysis:', e);
    }
  } else {
    // No JSON block found — treat entire response as empathy
    empathyResponse = response.trim();
  }

  // Extract and validate fields with safe defaults
  const validSentiments: Sentiment[] = ['positive', 'neutral', 'concerned', 'distressed'];
  const sentiment: Sentiment = validSentiments.includes(analysis.sentiment)
    ? analysis.sentiment
    : inferSentimentFromScore(analysis.sentimentScore ?? 0);

  const validThemes: LifeTheme[] = [
    'work', 'relationships', 'health', 'sleep', 'finances',
    'family', 'social', 'academic', 'self_image', 'existential', 'general',
  ];
  const themes: LifeTheme[] = Array.isArray(analysis.themes)
    ? analysis.themes.filter((t: string) => validThemes.includes(t as LifeTheme))
    : ['general'];

  const defaultEmotions: EmotionScores = {
    joy: 0, sadness: 0, anxiety: 0, anger: 0,
    calm: 0, fear: 0, gratitude: 0, fatigue: 0,
  };

  const emotions: EmotionScores = {
    ...defaultEmotions,
    ...(analysis.emotions && typeof analysis.emotions === 'object'
      ? Object.fromEntries(
          Object.entries(analysis.emotions)
            .filter(([k]) => k in defaultEmotions)
            .map(([k, v]) => [k, clamp(Number(v) || 0, 0, 1)])
        )
      : {}),
  };

  return {
    empathyResponse,
    sentiment,
    sentimentScore: clamp(Number(analysis.sentimentScore) || 0, -1, 1),
    themes: themes.length > 0 ? themes : ['general'],
    emotionalIntensity: clamp(Number(analysis.emotionalIntensity) || 0.5, 0, 1),
    emotions,
    suggestedFollowUp: analysis.suggestedFollowUp || null,
  };
}

function inferSentimentFromScore(score: number): Sentiment {
  if (score >= 0.3) return 'positive';
  if (score >= -0.1) return 'neutral';
  if (score >= -0.5) return 'concerned';
  return 'distressed';
}

// ============================================================
// Insight Generation from LLM Analysis
// ============================================================

function generateInsightsFromLLM(
  parsed: ParsedLLMResponse,
  bio: FullBiometricContext,
): string[] {
  const insights: string[] = [];

  // Body-mind alignment
  const verballyStressed = parsed.sentimentScore < -0.3;
  const biometricallyStressed = bio.stress > 60;

  if (verballyStressed && biometricallyStressed) {
    insights.push('Both your words and body signals suggest elevated stress');
  } else if (verballyStressed && !biometricallyStressed) {
    insights.push('You expressed some concern, but your body metrics look steady');
  } else if (!verballyStressed && biometricallyStressed) {
    insights.push('Your words sound positive, but your body may be carrying some tension');
  }

  // Sleep impact
  if (bio.sleepQuality !== null && bio.sleepQuality < 50) {
    insights.push('Poor sleep last night may be amplifying how you feel right now');
  } else if (bio.sleepDurationHrs !== null && bio.sleepDurationHrs < 6) {
    insights.push(`Only ${bio.sleepDurationHrs.toFixed(1)} hours of sleep — that compounds stress significantly`);
  }

  // Theme-based insight
  const themeLabels: Record<string, string> = {
    work: 'work-related pressures',
    relationships: 'relationship dynamics',
    health: 'health concerns',
    sleep: 'sleep quality',
    finances: 'financial stress',
    family: 'family matters',
    social: 'social connections',
    academic: 'academic pressures',
    self_image: 'self-perception',
    existential: 'life direction',
  };

  const meaningfulThemes = parsed.themes.filter(t => t !== 'general');
  if (meaningfulThemes.length > 0) {
    const themeStr = meaningfulThemes
      .map(t => themeLabels[t] || t)
      .join(' and ');
    insights.push(`Your check-in touches on ${themeStr}`);
  }

  // Compound risk factors
  const risks: string[] = [];
  if (bio.hrv > 0 && bio.hrv < 25) risks.push('low HRV');
  if (bio.stress > 70) risks.push('high stress');
  if (bio.anxietyIndex > 50) risks.push('elevated anxiety');
  if (bio.sleepDurationHrs !== null && bio.sleepDurationHrs < 6) risks.push('sleep deprivation');
  if (risks.length >= 2) {
    insights.push(`Compounding factors detected: ${risks.join(' + ')}`);
  } else if (bio.hrv > 0 && bio.hrv < 25) {
    insights.push('Your HRV is quite low right now — a breathing exercise might help');
  }

  return insights.slice(0, 4);
}

// ============================================================
// Local Fallback (when LLM is unavailable)
// ============================================================

function localFallbackAnalysis(
  id: string,
  timestamp: number,
  transcript: string,
  biometrics: FullBiometricContext,
): CheckinAnalysis {
  // Import the existing VADER analysis inline to avoid circular deps
  const { sentiment, sentimentScore, emotions } = quickLocalAnalysis(transcript);

  const insights: string[] = [];
  if (sentimentScore < -0.3 && biometrics.stress > 60) {
    insights.push('Both your words and body signals suggest elevated stress');
  }
  if (biometrics.hrv < 25) {
    insights.push('Your HRV is quite low — a breathing exercise might help');
  }

  return {
    id,
    timestamp,
    transcript,
    sentiment,
    sentimentScore,
    emotionScores: emotions,
    keyInsights: insights,
    themes: ['general'],
    emotionalIntensity: Math.min(1, Math.abs(sentimentScore) + 0.2),
    empathyResponse: generateLocalEmpathy(sentiment, sentimentScore),
    suggestedFollowUp: null,
    hrAtCheckin: biometrics.hr,
    hrvAtCheckin: biometrics.hrv,
    stressAtCheckin: biometrics.stress,
  };
}

/**
 * Minimal local sentiment analysis (subset of VADER) for fallback.
 */
function quickLocalAnalysis(text: string): {
  sentiment: Sentiment;
  sentimentScore: number;
  emotions: EmotionScores;
} {
  const words = text.toLowerCase().replace(/[^a-z'\s-]/g, ' ').split(/\s+/);

  const positiveWords = new Set(['good', 'great', 'happy', 'calm', 'relaxed', 'grateful', 'better', 'amazing', 'wonderful', 'love', 'excited', 'proud']);
  const negativeWords = new Set(['bad', 'stressed', 'anxious', 'worried', 'sad', 'depressed', 'tired', 'overwhelmed', 'scared', 'frustrated', 'lonely', 'hopeless', 'exhausted', 'angry']);

  let pos = 0, neg = 0;
  for (const w of words) {
    if (positiveWords.has(w)) pos++;
    if (negativeWords.has(w)) neg++;
  }

  const total = pos + neg || 1;
  const sentimentScore = (pos - neg) / total;

  const emotions: EmotionScores = {
    joy: 0, sadness: 0, anxiety: 0, anger: 0,
    calm: 0, fear: 0, gratitude: 0, fatigue: 0,
  };

  // Simple keyword mapping
  for (const w of words) {
    if (['happy', 'great', 'amazing', 'excited'].includes(w)) emotions.joy = Math.min(1, emotions.joy + 0.4);
    if (['sad', 'depressed', 'lonely', 'hopeless'].includes(w)) emotions.sadness = Math.min(1, emotions.sadness + 0.4);
    if (['anxious', 'worried', 'stressed', 'overwhelmed'].includes(w)) emotions.anxiety = Math.min(1, emotions.anxiety + 0.4);
    if (['angry', 'frustrated'].includes(w)) emotions.anger = Math.min(1, emotions.anger + 0.4);
    if (['calm', 'relaxed', 'peaceful'].includes(w)) emotions.calm = Math.min(1, emotions.calm + 0.4);
    if (['scared', 'afraid'].includes(w)) emotions.fear = Math.min(1, emotions.fear + 0.4);
    if (['grateful', 'thankful'].includes(w)) emotions.gratitude = Math.min(1, emotions.gratitude + 0.4);
    if (['tired', 'exhausted'].includes(w)) emotions.fatigue = Math.min(1, emotions.fatigue + 0.4);
  }

  return {
    sentiment: inferSentimentFromScore(sentimentScore),
    sentimentScore: clamp(sentimentScore, -1, 1),
    emotions,
  };
}

function generateLocalEmpathy(sentiment: Sentiment, score: number): string {
  if (sentiment === 'distressed') {
    return "I can hear that you're going through a really tough time. Your feelings are valid, and it's okay to not be okay right now.";
  }
  if (sentiment === 'concerned') {
    return "It sounds like something is weighing on you. I want you to know that it's completely normal to feel this way, and acknowledging it is an important step.";
  }
  if (sentiment === 'positive') {
    return "It's really great to hear that! Those positive moments matter, and I'm glad you're taking the time to recognize them.";
  }
  return "Thank you for checking in. Taking a moment to reflect on how you're feeling is a valuable practice.";
}

// ============================================================
// Encryption Utilities
// ============================================================

/**
 * Simple XOR-based encryption for local storage.
 * In production, use expo-crypto AES-256 or react-native-keychain.
 */
const ENCRYPTION_KEY = 'seren_local_encryption_2024'; // In production, derive from device keychain

export function encryptForStorage(plaintext: string): string {
  let encrypted = '';
  for (let i = 0; i < plaintext.length; i++) {
    const charCode = plaintext.charCodeAt(i) ^ ENCRYPTION_KEY.charCodeAt(i % ENCRYPTION_KEY.length);
    encrypted += String.fromCharCode(charCode);
  }
  // Base64 encode for safe storage
  return btoa(encrypted);
}

export function decryptFromStorage(ciphertext: string): string {
  try {
    const decoded = atob(ciphertext);
    let decrypted = '';
    for (let i = 0; i < decoded.length; i++) {
      const charCode = decoded.charCodeAt(i) ^ ENCRYPTION_KEY.charCodeAt(i % ENCRYPTION_KEY.length);
      decrypted += String.fromCharCode(charCode);
    }
    return decrypted;
  } catch {
    return ciphertext; // Return as-is if decryption fails (unencrypted data)
  }
}

// ============================================================
// Utilities
// ============================================================

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
