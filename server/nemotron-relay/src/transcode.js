/**
 * nemotron-relay — audio transcode
 * ====================================
 * Riva's ASR gRPC API expects raw, headerless PCM16 bytes for LINEAR_PCM —
 * not the m4a/AAC container audioRecorder.ts produces on the client. This
 * uses ffmpeg (via ffmpeg-static, so no system ffmpeg install is required)
 * to convert whatever comes in to 16kHz mono PCM16 little-endian.
 */

const ffmpeg = require('fluent-ffmpeg');
const ffmpegPath = require('ffmpeg-static');

ffmpeg.setFfmpegPath(ffmpegPath);

/**
 * @param {string} inputPath Path to the uploaded audio file (any container ffmpeg can read)
 * @param {{ sampleRate?: number }} opts
 * @returns {Promise<Buffer>} Raw PCM16 mono little-endian bytes
 */
function transcodeToPCM16(inputPath, opts = {}) {
  const { sampleRate = 16000 } = opts;

  return new Promise((resolve, reject) => {
    const chunks = [];
    let settled = false;

    const fail = (err) => {
      if (settled) return;
      settled = true;
      reject(err);
    };

    const command = ffmpeg(inputPath)
      .audioChannels(1)
      .audioFrequency(sampleRate)
      .format('s16le') // raw PCM16 little-endian, no container
      .on('error', fail);

    const stream = command.pipe();
    stream.on('data', (chunk) => chunks.push(chunk));
    stream.on('error', fail);
    stream.on('end', () => {
      if (settled) return;
      settled = true;
      resolve(Buffer.concat(chunks));
    });
  });
}

module.exports = { transcodeToPCM16 };
