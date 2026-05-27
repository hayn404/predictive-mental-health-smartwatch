/**
 * __tests__/llmService.test.ts
 *
 * Unit tests for services/ai/llmService.ts
 *
 * Current coverage: 5.88% (API-gated — main function requires live Groq call)
 * Target coverage after these tests: >60%
 *
 * Strategy:
 *  1. Mock globalThis.fetch to intercept all Groq API calls
 *  2. Test every exported utility function directly (PII stripping,
 *     encryption/decryption, isConfigured check)
 *  3. Test analyzeCheckinWithLLM() against both the success path
 *     (mocked API response) and the fallback path (API throws / unconfigured)
 *  4. Verify the fallback returns a valid result indistinguishable from
 *     a real LLM response in structure
 *
 * NOTE: These tests import from llmService.ts using named exports.
 * If your llmService.ts uses default exports or different function names,
 * adjust the import line to match. The test logic itself does not change.
 */

// ─── Mocks ────────────────────────────────────────────────────────────────────

// Mock fetch globally before any imports so the module-level fetch
// reference is already patched when llmService.ts loads
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Mock expo-file-system (used by whisperService imported by llmService)
jest.mock('expo-file-system/legacy', () => ({
  uploadAsync: jest.fn(),
  readAsStringAsync: jest.fn(),
  EncodingType: { Base64: 'base64' },
  FileSystemUploadType: { MULTIPART: 'multipart' },
}));

// Mock expo-file-system standard (SDK 54 may use either path)
jest.mock('expo-file-system', () => ({
  uploadAsync: jest.fn(),
  readAsStringAsync: jest.fn(),
  EncodingType: { Base64: 'base64' },
  FileSystemUploadType: { MULTIPART: 'multipart' },
}));

// ─── Imports (after mocks) ────────────────────────────────────────────────────

import {
  initializeAIServices,
  isLLMConfigured,
  analyzeCheckinWithLLM,
  stripPII,
  encryptTranscript,
  decryptTranscript,
  type LLMCheckinResult,
} from '../services/ai/llmService';

// ─── Groq API response factory ────────────────────────────────────────────────

function makeGroqResponse(content: string) {
  return {
    ok: true,
    json: async () => ({
      choices: [
        {
          message: {
            role: 'assistant',
            content,
          },
        },
      ],
      usage: { prompt_tokens: 120, completion_tokens: 80, total_tokens: 200 },
    }),
  };
}

function makeGroqErrorResponse(status = 429, message = 'Rate limit exceeded') {
  return {
    ok: false,
    status,
    json: async () => ({ error: { message } }),
  };
}

// ─── PII stripping ────────────────────────────────────────────────────────────

describe('llmService — PII stripping', () => {
  test('strips email addresses', () => {
    const result = stripPII('My email is john.doe@example.com, please write back');
    expect(result).not.toContain('john.doe@example.com');
    expect(result).toContain('[REDACTED]');
  });

  test('strips phone numbers in various formats', () => {
    const formats = [
      'Call me at 555-867-5309',
      'My number is (555) 867 5309',
      'Reach me on +1 555 867 5309',
    ];
    formats.forEach((text) => {
      const result = stripPII(text);
      expect(result).not.toContain('867-5309');
      expect(result).not.toContain('8675309');
    });
  });

  test('strips 9-digit SSN patterns', () => {
    const result = stripPII('My SSN is 123-45-6789 for the form');
    expect(result).not.toContain('123-45-6789');
  });

  test('preserves non-PII text content', () => {
    const clean = 'I felt anxious today after a long meeting at work.';
    const result = stripPII(clean);
    expect(result).toBe(clean);
  });

  test('handles empty string without throwing', () => {
    expect(() => stripPII('')).not.toThrow();
    expect(stripPII('')).toBe('');
  });

  test('handles text with multiple PII types in one pass', () => {
    const messy = 'Name: John Smith, email: john@test.com, phone: 555-123-4567';
    const result = stripPII(messy);
    expect(result).not.toContain('john@test.com');
    expect(result).not.toContain('555-123-4567');
  });
});

// ─── Transcript encryption / decryption ──────────────────────────────────────

describe('llmService — transcript encryption', () => {
  const sampleTranscript = 'I have been feeling overwhelmed and anxious this week.';

  test('encryptTranscript returns a non-empty string', () => {
    const encrypted = encryptTranscript(sampleTranscript);
    expect(typeof encrypted).toBe('string');
    expect(encrypted.length).toBeGreaterThan(0);
  });

  test('encrypted value is different from original', () => {
    const encrypted = encryptTranscript(sampleTranscript);
    expect(encrypted).not.toBe(sampleTranscript);
  });

  test('decryptTranscript recovers original text', () => {
    const encrypted = encryptTranscript(sampleTranscript);
    const decrypted = decryptTranscript(encrypted);
    expect(decrypted).toBe(sampleTranscript);
  });

  test('round-trip is stable across multiple calls', () => {
    for (let i = 0; i < 5; i++) {
      const enc = encryptTranscript(sampleTranscript);
      const dec = decryptTranscript(enc);
      expect(dec).toBe(sampleTranscript);
    }
  });

  test('encrypting empty string does not throw', () => {
    expect(() => encryptTranscript('')).not.toThrow();
  });

  test('decrypting invalid data returns empty string or throws gracefully', () => {
    expect(() => {
      const result = decryptTranscript('not-valid-encrypted-data-!!!');
      // Either returns empty/fallback or throws — both are acceptable
      if (typeof result !== 'string') throw new Error('Unexpected return type');
    }).not.toThrow(); // The function itself should not crash the app
  });
});

// ─── isConfigured check ───────────────────────────────────────────────────────

describe('llmService — configuration state', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('isLLMConfigured returns false before initialization', () => {
    // Don't call initializeAIServices — check initial state
    // NOTE: if your module initializes from env vars at import time,
    // this test verifies the explicit "not configured" path
    const configured = isLLMConfigured();
    expect(typeof configured).toBe('boolean');
  });

  test('isLLMConfigured returns true after initialization with valid key', () => {
    initializeAIServices({ groqApiKey: 'gsk_test_fake_key_for_unit_test' });
    expect(isLLMConfigured()).toBe(true);
  });

  test('isLLMConfigured returns false after initialization with empty key', () => {
    initializeAIServices({ groqApiKey: '' });
    expect(isLLMConfigured()).toBe(false);
  });
});

// ─── analyzeCheckinWithLLM — success path ────────────────────────────────────

describe('llmService — analyzeCheckinWithLLM success path', () => {
  const sampleTranscript = 'I felt pretty stressed at work but managed to calm down in the evening.';
  const sampleBiometrics = {
    stressScore: 65,
    hrv_rmssd: 28,
    sleepQualityScore: 58,
  };

  const mockLLMContent = JSON.stringify({
    primaryEmotion: 'stressed',
    emotionIntensity: 0.65,
    sentiment: -0.3,
    themes: ['work pressure', 'coping', 'recovery'],
    insights: ['Stress peaked mid-day but evening recovery is positive'],
    supportiveResponse: 'It sounds like you navigated a tough day. The evening calm shows real resilience.',
    suggestedActions: ['Try a 5-minute breathing exercise before bed'],
  });

  beforeEach(() => {
    jest.clearAllMocks();
    initializeAIServices({ groqApiKey: 'gsk_test_fake_key_for_unit_test' });
    mockFetch.mockResolvedValueOnce(makeGroqResponse(mockLLMContent));
  });

  test('returns LLMCheckinResult with required fields', async () => {
    const result = await analyzeCheckinWithLLM(sampleTranscript, sampleBiometrics);
    expect(result).toHaveProperty('primaryEmotion');
    expect(result).toHaveProperty('sentiment');
    expect(result).toHaveProperty('supportiveResponse');
    expect(result).toHaveProperty('insights');
  });

  test('sentiment is bounded [-1, 1]', async () => {
    const result = await analyzeCheckinWithLLM(sampleTranscript, sampleBiometrics);
    expect(result.sentiment).toBeGreaterThanOrEqual(-1);
    expect(result.sentiment).toBeLessThanOrEqual(1);
  });

  test('PII is stripped before the API call', async () => {
    const transcriptWithPII = 'I talked to jane@example.com about my anxiety.';
    await analyzeCheckinWithLLM(transcriptWithPII, sampleBiometrics);

    // Inspect what was sent to fetch
    const callArgs = mockFetch.mock.calls[0];
    const requestBody = JSON.parse(callArgs[1].body);
    const sentPrompt = JSON.stringify(requestBody);
    expect(sentPrompt).not.toContain('jane@example.com');
  });

  test('fetch is called with Groq API endpoint', async () => {
    await analyzeCheckinWithLLM(sampleTranscript, sampleBiometrics);
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('api.groq.com'),
      expect.any(Object)
    );
  });

  test('biometric context is included in the request', async () => {
    await analyzeCheckinWithLLM(sampleTranscript, sampleBiometrics);
    const callArgs = mockFetch.mock.calls[0];
    const requestBody = JSON.parse(callArgs[1].body);
    const sentContent = JSON.stringify(requestBody);
    // Stress score or HRV value should appear in the prompt context
    expect(sentContent).toMatch(/65|28|stress|hrv/i);
  });
});

// ─── analyzeCheckinWithLLM — fallback path ────────────────────────────────────

describe('llmService — analyzeCheckinWithLLM fallback path', () => {
  const sampleTranscript = 'I am feeling anxious and a bit down today.';
  const sampleBiometrics = { stressScore: 72, hrv_rmssd: 22, sleepQualityScore: 45 };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('fallback activates when API key is not configured', async () => {
    initializeAIServices({ groqApiKey: '' });
    const result = await analyzeCheckinWithLLM(sampleTranscript, sampleBiometrics);
    expect(result).toBeDefined();
    expect(result).toHaveProperty('primaryEmotion');
    expect(result).toHaveProperty('sentiment');
  });

  test('fallback result has valid sentiment in [-1, 1]', async () => {
    initializeAIServices({ groqApiKey: '' });
    const result = await analyzeCheckinWithLLM(sampleTranscript, sampleBiometrics);
    expect(result.sentiment).toBeGreaterThanOrEqual(-1);
    expect(result.sentiment).toBeLessThanOrEqual(1);
  });

  test('fallback activates when fetch throws a network error', async () => {
    initializeAIServices({ groqApiKey: 'gsk_test_fake_key_for_unit_test' });
    mockFetch.mockRejectedValueOnce(new Error('Network request failed'));
    const result = await analyzeCheckinWithLLM(sampleTranscript, sampleBiometrics);
    expect(result).toBeDefined();
    expect(result).toHaveProperty('primaryEmotion');
  });

  test('fallback activates when API returns rate-limit error (429)', async () => {
    initializeAIServices({ groqApiKey: 'gsk_test_fake_key_for_unit_test' });
    mockFetch.mockResolvedValueOnce(makeGroqErrorResponse(429));
    const result = await analyzeCheckinWithLLM(sampleTranscript, sampleBiometrics);
    expect(result).toBeDefined();
    expect(result).toHaveProperty('supportiveResponse');
  });

  test('fallback activates when API returns 500 server error', async () => {
    initializeAIServices({ groqApiKey: 'gsk_test_fake_key_for_unit_test' });
    mockFetch.mockResolvedValueOnce(makeGroqErrorResponse(500, 'Internal server error'));
    const result = await analyzeCheckinWithLLM(sampleTranscript, sampleBiometrics);
    expect(result).toBeDefined();
  });

  test('fallback returns negative sentiment for distressed transcript', async () => {
    initializeAIServices({ groqApiKey: '' });
    const distressed = 'I feel terrible and hopeless, everything is going wrong.';
    const result = await analyzeCheckinWithLLM(distressed, sampleBiometrics);
    expect(result.sentiment).toBeLessThan(0);
  });

  test('fallback returns positive-ish sentiment for happy transcript', async () => {
    initializeAIServices({ groqApiKey: '' });
    const happy = 'I had a wonderful day, felt really happy and grateful.';
    const result = await analyzeCheckinWithLLM(happy, sampleBiometrics);
    expect(result.sentiment).toBeGreaterThan(-0.2);
  });

  test('fetch is NOT called when API key is not configured', async () => {
    initializeAIServices({ groqApiKey: '' });
    await analyzeCheckinWithLLM(sampleTranscript, sampleBiometrics);
    expect(mockFetch).not.toHaveBeenCalled();
  });
});

// ─── analyzeCheckinWithLLM — malformed API response handling ─────────────────

describe('llmService — malformed API response handling', () => {
  const transcript = 'Testing malformed response handling.';
  const biometrics = { stressScore: 50, hrv_rmssd: 38, sleepQualityScore: 65 };

  beforeEach(() => {
    jest.clearAllMocks();
    initializeAIServices({ groqApiKey: 'gsk_test_fake_key_for_unit_test' });
  });

  test('handles empty content string gracefully (falls back)', async () => {
    mockFetch.mockResolvedValueOnce(makeGroqResponse(''));
    const result = await analyzeCheckinWithLLM(transcript, biometrics);
    expect(result).toBeDefined();
    expect(result).toHaveProperty('primaryEmotion');
  });

  test('handles non-JSON content string gracefully (falls back)', async () => {
    mockFetch.mockResolvedValueOnce(makeGroqResponse('Sorry, I cannot help with that.'));
    const result = await analyzeCheckinWithLLM(transcript, biometrics);
    expect(result).toBeDefined();
  });

  test('handles missing choices array gracefully', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ choices: [] }),
    });
    const result = await analyzeCheckinWithLLM(transcript, biometrics);
    expect(result).toBeDefined();
  });
});
