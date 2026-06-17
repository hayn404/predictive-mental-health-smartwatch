/**
 * nemotron-relay — HTTP entrypoint
 * ====================================
 * Exposes the contract services/ai/nemotronAsrService.ts (in the Expo app)
 * expects:
 *   GET  /health      -> { status: 'ok'|'degraded', riva: 'reachable'|'unreachable', models?: string[] }
 *   POST /transcribe   -> multipart { file, language? } -> { text, confidence, durationMs, provider: 'nemotron' }
 *
 * Auth: if RELAY_SHARED_SECRET is set, /transcribe requires
 * `Authorization: Bearer <RELAY_SHARED_SECRET>` — a separate secret from
 * NVIDIA_API_KEY, so the phone never carries the real NVIDIA credential.
 */

require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const os = require('os');

const { recognize, checkConfig } = require('./rivaAsrClient');
const { transcodeToPCM16 } = require('./transcode');

const app = express();
app.use(cors());

const upload = multer({ dest: os.tmpdir(), limits: { fileSize: 25 * 1024 * 1024 } });

const RELAY_SHARED_SECRET = process.env.RELAY_SHARED_SECRET || '';

// Whisper-style short codes ('en') -> Riva-style BCP-47 ('en-US'), since
// nemotronAsrService.ts/aiConfig.ts may pass either depending on what's
// configured for the rest of the app.
const LANGUAGE_MAP = { en: 'en-US', es: 'es-US', fr: 'fr-FR', de: 'de-DE', hi: 'hi-IN', zh: 'zh-CN' };
function normalizeLanguage(lang) {
  if (!lang) return 'en-US';
  return LANGUAGE_MAP[lang] || lang;
}

function requireSharedSecret(req, res, next) {
  if (!RELAY_SHARED_SECRET) {
    // Dev-only fallback. Logged loudly on boot below — don't deploy this way.
    return next();
  }
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : '';
  if (token !== RELAY_SHARED_SECRET) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
}

app.get('/health', async (_req, res) => {
  try {
    const { models } = await checkConfig();
    res.json({ status: 'ok', riva: 'reachable', models });
  } catch (err) {
    res.status(503).json({ status: 'degraded', riva: 'unreachable', error: err.message });
  }
});

app.post('/transcribe', requireSharedSecret, upload.single('file'), async (req, res) => {
  const file = req.file;
  if (!file) {
    return res.status(400).json({ error: 'Missing "file" field in multipart body' });
  }

  const languageCode = normalizeLanguage(req.body?.language);
  const startedAt = Date.now();

  try {
    const pcm = await transcodeToPCM16(file.path, { sampleRate: 16000 });
    const { transcript, confidence } = await recognize(pcm, { sampleRate: 16000, languageCode });

    res.json({
      text: transcript,
      confidence,
      durationMs: Date.now() - startedAt,
      provider: 'nemotron',
    });
  } catch (err) {
    console.error('[nemotron-relay] /transcribe failed:', err);
    const isTimeout = /deadline|timeout/i.test(err.message || '');
    res.status(isTimeout ? 504 : 502).json({ error: err.message || 'ASR upstream failed' });
  } finally {
    fs.unlink(file.path, () => {}); // best-effort cleanup of the temp upload
  }
});

const PORT = process.env.RELAY_PORT || 8088;

app.listen(PORT, () => {
  console.log(`[nemotron-relay] listening on :${PORT}`);
  if (!RELAY_SHARED_SECRET) {
    console.warn('[nemotron-relay] RELAY_SHARED_SECRET is not set — this relay is open to anyone who finds its URL. Fine for local dev, NOT for deployment.');
  }
  if (!process.env.NVIDIA_API_KEY && !process.env.RIVA_GRPC_HOST?.includes('your-riva-host')) {
    console.warn('[nemotron-relay] NVIDIA_API_KEY is not set — requests to the default NVCF-hosted endpoint will fail auth. Set it, or point RIVA_GRPC_HOST at a self-hosted Riva server instead.');
  }
});
