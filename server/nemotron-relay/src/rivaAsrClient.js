/**
 * nemotron-relay — Riva ASR gRPC client
 * ========================================
 * Builds a gRPC client straight from NVIDIA's public riva_asr.proto (vendored
 * in proto/riva/proto/, fetched from github.com/nvidia-riva/common) using
 * @grpc/proto-loader. There's no official Node.js Riva/NIM SDK — the
 * official clients are Python/C++/Java — so this loads the .proto directly
 * rather than wrapping an SDK that doesn't exist for this runtime.
 *
 * Two deployment targets, both using the exact same proto:
 *  - NVIDIA NIM hosted via NVCF (default): grpc.nvcf.nvidia.com:443, with
 *    `function-id` + `authorization: Bearer <NVIDIA_API_KEY>` metadata.
 *    This is the same routing NVIDIA's own Riva Python client scripts use
 *    (see build.nvidia.com's "API Reference" tab on any ASR model).
 *  - Self-hosted Riva server: arbitrary host:port, no function-id/auth.
 *
 * Audio contract: Riva's `RecognitionConfig.encoding` field expects RAW
 * encoded bytes, not a WAV/container file — for LINEAR_PCM that means
 * headerless 16-bit signed little-endian PCM samples. transcode.js produces
 * exactly that from whatever audioRecorder.ts records (m4a/AAC).
 */

const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const path = require('path');

// Resolved relative to PROTO_INCLUDE_DIR below — matches how riva_asr.proto's
// own `import "riva/proto/riva_audio.proto"` statements resolve, and avoids
// a (harmless but noisy) proto-loader warning that shows up when the entry
// file is passed as an absolute path instead.
const PROTO_ENTRY = path.join('riva', 'proto', 'riva_asr.proto');
const PROTO_INCLUDE_DIR = path.join(__dirname, '..', 'proto');

let cachedClient = null;

function getClient() {
  if (cachedClient) return cachedClient;

  const packageDefinition = protoLoader.loadSync(PROTO_ENTRY, {
    keepCase: false, // sample_rate_hertz -> sampleRateHertz, etc.
    longs: String,
    enums: String, // encode/decode enums as their string names (e.g. 'LINEAR_PCM')
    defaults: true,
    oneofs: true,
    includeDirs: [PROTO_INCLUDE_DIR], // resolves riva_asr.proto's `import "riva/proto/riva_audio.proto"`
  });

  const proto = grpc.loadPackageDefinition(packageDefinition);
  const RivaSpeechRecognition = proto.nvidia.riva.asr.RivaSpeechRecognition;

  const host = process.env.RIVA_GRPC_HOST || 'grpc.nvcf.nvidia.com:443';
  const useSsl = (process.env.RIVA_USE_SSL ?? 'true') !== 'false';
  const credentials = useSsl ? grpc.credentials.createSsl() : grpc.credentials.createInsecure();

  cachedClient = new RivaSpeechRecognition(host, credentials, {
    'grpc.max_receive_message_length': 50 * 1024 * 1024,
  });

  return cachedClient;
}

function buildMetadata() {
  const metadata = new grpc.Metadata();
  if (process.env.RIVA_FUNCTION_ID) {
    metadata.add('function-id', process.env.RIVA_FUNCTION_ID);
  }
  if (process.env.NVIDIA_API_KEY) {
    metadata.add('authorization', `Bearer ${process.env.NVIDIA_API_KEY}`);
  }
  return metadata;
}

/**
 * Non-streaming recognition — mirrors what audioRecorder.ts/whisperService.ts
 * already do for a single recorded clip (record fully, then transcribe).
 *
 * @param {Buffer} pcmBuffer Raw 16-bit signed little-endian PCM, mono.
 * @param {{ sampleRate?: number, languageCode?: string, timeoutMs?: number }} opts
 * @returns {Promise<{ transcript: string, confidence: number }>}
 */
function recognize(pcmBuffer, opts = {}) {
  const { sampleRate = 16000, languageCode = 'en-US', timeoutMs = 15000 } = opts;

  return new Promise((resolve, reject) => {
    const client = getClient();
    const deadline = new Date(Date.now() + timeoutMs);

    const request = {
      config: {
        encoding: 'LINEAR_PCM',
        sampleRateHertz: sampleRate,
        languageCode,
        maxAlternatives: 1,
        enableAutomaticPunctuation: true,
      },
      audio: pcmBuffer,
    };

    client.Recognize(request, buildMetadata(), { deadline }, (err, response) => {
      if (err) return reject(err);

      const alternative = response?.results?.[0]?.alternatives?.[0];
      resolve({
        transcript: (alternative?.transcript || '').trim(),
        confidence: alternative?.confidence ?? 0,
      });
    });
  });
}

/**
 * Lightweight connectivity check — calls GetRivaSpeechRecognitionConfig
 * instead of sending real audio, so /health doesn't burn NVIDIA API usage.
 * @returns {Promise<{ models: string[] }>}
 */
function checkConfig(timeoutMs = 4000) {
  return new Promise((resolve, reject) => {
    const client = getClient();
    const deadline = new Date(Date.now() + timeoutMs);

    client.GetRivaSpeechRecognitionConfig({ modelName: '' }, buildMetadata(), { deadline }, (err, response) => {
      if (err) return reject(err);
      const models = (response?.modelConfig || []).map((m) => m.modelName).filter(Boolean);
      resolve({ models });
    });
  });
}

module.exports = { recognize, checkConfig };
