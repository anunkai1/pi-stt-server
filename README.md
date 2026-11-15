# pi-stt-server

Minimal resident local speech-to-text HTTP server, mirroring the design and
lifecycle of `pi-voice-server` (Kokoro TTS): systemd socket activation, lazy
engine load, serialised transcription, clean idle exit.

It replaced agentchatbox's per-request `faster-whisper` shell-out
(`scripts/transcribe.py`), which reloaded the model from disk for every voice
note. Weights load once and stay warm; the process exits after an idle
timeout.

## HTTP surface

    GET  /health      → capability, residency, engine metadata
    POST /transcribe  → raw audio bytes (any container, any sample rate)
                        → {"text": ..., "duration": ..., "engine": ..., "model": ...}

No multipart parsing: the caller sends the audio file's bytes as the request
body (`Content-Type: application/octet-stream`). Audio is decoded in process
with PyAV to 16 kHz mono PCM.

Transcriptions are serialised — one model, one caller at a time. A hung
transcription is bounded by `STT_TIMEOUT_MS` (default 5 min) and releases the
queue.

## Engine

Runs on Server2's CPU only.

| Engine | Library | Notes |
|---|---|---|
| `fluid_speech` (default) | `sherpa-onnx`, `FluidVoice-Labs/fluid-speech-pro-onnx` (int8, CC BY 4.0) | 0.63B FastConformer-TDT model supporting 25 European languages (including English, Russian and Ukrainian) with automatic language detection, punctuation and casing. ~1 GB RSS, ~801 MB weights. `prepare_fluid_speech.py OUT_DIR` downloads and checksum-verifies the pinned Hugging Face revision (`fa588e9f87f1b9ad0ca2e2d8f7eaf84dc3803581`). |

Benchmark on Server2's CPU (4 threads, 66 Open ASR Leaderboard clips across AMI,
Earnings-22, LibriSpeech other, Common Voice, VoxPopuli; 5–25 s; ~10 min):

- Fluid Speech Pro (`sherpa-onnx` int8): 6.56 % WER / 0.155 RTF / 0.9 s cold load
- Bark 3.0 Plus (`transformers` fp32, retired — English-only): 6.14 % WER / 0.111 RTF
- qwen3_asr int8 (retired): 10.60 % WER / 0.241 RTF
- whisper large-v3-turbo (retired): 10.53 % WER / 1.617 RTF
- whisper small (retired): 17.22 % WER / 0.351 RTF

Fluid Speech Pro replaced the earlier two-engine (`bark` + `qwen3_asr`) split on
15/11/2026: it handles English and 24 other European languages automatically in
a single lightweight ONNX runtime without PyTorch or `transformers`.

Audio longer than five minutes is rejected with HTTP 413 before model loading;
raising the limit beyond 300 seconds is rejected at startup. Recordings longer
than 30 seconds are decoded in 30-second windows (the model's qualified window
length) and joined in order. `/health` advertises `maxAudioSeconds`. Clients
should allow the daemon's five-minute processing timeout and retain recordings
on failure.

## Configuration (env)

    STT_HOST               default 127.0.0.1
    STT_PORT               default 8182 (ignored under systemd socket activation)
    STT_PRIMARY_ENGINE     default fluid_speech
    STT_FALLBACK_ENGINE    default "" (disabled)
    STT_IDLE_TIMEOUT_MS    default 1800000 (30 min); min 60s, max 24h
    STT_TIMEOUT_MS         default 300000 (5 min per transcription)
    STT_MAX_AUDIO_BYTES    default 26214400 (25 MiB request cap)
    STT_MAX_AUDIO_SECONDS  default 300 (5 min); min 10, max 300
    FLUID_SPEECH_DIR       default ~/ai-models/fluid-speech-pro-onnx-fa588e9f
    FLUID_SPEECH_THREADS   default 4
    SHUTDOWN_DRAIN_TIMEOUT_MS default 600000

`FLUID_SPEECH_DIR` must be pre-populated by `prepare_fluid_speech.py` (handled
automatically by `deploy-pi-stt-server`).

## Tests

    venv/bin/python -m pytest test/ -q

Tests cover request validation, engine chain fallback, 30-second windowing,
serialisation, idle-timer semantics, provisioning checksum checks and the HTTP
routes with fake engines — no real weights required.
