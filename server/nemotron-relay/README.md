# nemotron-relay

A small Node/Express service that bridges Seren's Expo app to NVIDIA's
NIM/Riva ASR over gRPC. Expo/React Native JS can't speak gRPC directly, so
the app talks plain HTTP to this relay, and this relay does the actual gRPC
call using NVIDIA's public `riva_asr.proto` (vendored in `proto/`, fetched
from [nvidia-riva/common](https://github.com/nvidia-riva/common) — there's
no official Node SDK for Riva/NIM, only Python/C++/Java, so this loads the
proto directly via `@grpc/proto-loader` rather than wrapping an SDK that
doesn't exist for Node).

```
Expo app → POST /transcribe (HTTP, multipart audio) → this relay
                                                          │
                                          ffmpeg: m4a → raw PCM16/16kHz/mono
                                                          │
                                              gRPC Recognize() call
                                                          │
                                    grpc.nvcf.nvidia.com:443 (NVIDIA NIM)
```

## 1. Get NVIDIA API access

1. Create an account / sign in at [build.nvidia.com](https://build.nvidia.com).
2. Search for an ASR model — **nemotron-asr-streaming** is NVIDIA's own
   Nemotron ASR model; `parakeet-*` and `canary-*` are other Riva ASR
   options if you want to compare accuracy/latency.
3. Click "Get API Key" (top right) to get an `nvapi-...` key.
4. Open that model's **API Reference** tab and copy its **function-id** —
   this is a UUID specific to that model (NVIDIA's hosted gRPC gateway,
   `grpc.nvcf.nvidia.com:443`, is a single endpoint that routes to different
   hosted models by this header).

## 2. Configure

```bash
cp .env.example .env
```

Fill in `NVIDIA_API_KEY` and `RIVA_FUNCTION_ID` from step 1, and set
`RELAY_SHARED_SECRET` to a random string — this is what the Expo app sends
back to authenticate with *this relay*; it's deliberately separate from
`NVIDIA_API_KEY` so the phone never carries your real NVIDIA credential.

If you're running your own self-hosted Riva server instead of NVIDIA's
hosted NIM, see the alternative block in `.env.example` — same proto, just
point `RIVA_GRPC_HOST` at your server and drop the NVCF-specific fields.

## 3. Run locally

```bash
npm install
npm run dev
```

Check it's up: `curl http://localhost:8088/health` — this calls Riva's
lightweight `GetRivaSpeechRecognitionConfig` RPC (not real audio), so it
doesn't burn API usage just to check connectivity.

### Testing from a physical device / Expo Go

`localhost` on your laptop isn't reachable from a phone. Either:
- Run on the same Wi-Fi and use your laptop's LAN IP
  (`EXPO_PUBLIC_NEMOTRON_RELAY_URL=http://192.168.x.x:8088` in the **app's**
  `.env`), or
- Tunnel it (e.g. `ngrok http 8088`) and use the public URL.

## 4. Audio format note

Riva's `RecognitionConfig.encoding` expects **raw, headerless** audio bytes
matching the declared encoding — for `LINEAR_PCM` that's 16-bit signed
little-endian PCM samples, not a WAV file and definitely not the
m4a/AAC that `audioRecorder.ts` records on the client. `src/transcode.js`
handles this conversion with ffmpeg (via `ffmpeg-static`, so no system
ffmpeg install needed) on every request — that's the main extra latency
this relay adds versus calling Riva directly.

## 5. Deploy

```bash
docker build -t nemotron-relay .
docker run -p 8088:8088 --env-file .env nemotron-relay
```

Push that image to Fly.io, Render, a small VPS, or anywhere else that runs
containers and gives you a stable HTTPS URL. Then set
`EXPO_PUBLIC_NEMOTRON_RELAY_URL` (and `EXPO_PUBLIC_NEMOTRON_RELAY_KEY` to
match `RELAY_SHARED_SECRET`) in the **app's** `.env`.

If you don't have NVIDIA NIM/Riva access yet, or don't want to run this
relay at all: leave `EXPO_PUBLIC_NEMOTRON_RELAY_URL` unset in the app and
nothing breaks — `voiceAssistant.ts`'s ASR routing falls back to Whisper,
which is the only ASR path the app needs to function.

## Endpoints

### `GET /health`

```json
{ "status": "ok", "riva": "reachable", "models": ["nemotron-asr-streaming"] }
```

### `POST /transcribe`

Multipart form: `file` (audio blob, any container ffmpeg can decode),
optional `language` (e.g. `en` or `en-US`). Headers: `Authorization: Bearer
<RELAY_SHARED_SECRET>` if one is set.

```json
{ "text": "I've been feeling pretty good today", "confidence": 0.94, "durationMs": 612, "provider": "nemotron" }
```

Errors: `400` bad upload, `401` bad/missing shared secret, `502` Riva
upstream failure, `504` Riva timeout.
