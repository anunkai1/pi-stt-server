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
                        → {"text": ..., "language": ..., "duration": ...,
                           "engine": ..., "model": ..., "engineMs": ...}

No multipart parsing: the caller sends the audio file's bytes as the request
body (`Content-Type: application/octet-stream`). Audio is decoded in process
with PyAV to 16 kHz mono PCM.

Transcriptions are serialised — one model, one caller at a time. A hung
transcription is bounded by `STT_TIMEOUT_MS` (default 5 min) and releases the
queue.

## Engines

Runs on Server2's CPU only.

| Engine | Library | Notes |
|---|---|---|
| `bark` (optional, English only) | transformers `ParakeetForTDT`, Underdog Bark 3.0 Plus | **English only**. ~3 GB RSS. `prepare_bark.py OUT_DIR` downloads the pinned revision and unpacks it once (cold load 0.2 s instead of ~25 s) |
| `qwen3_asr` (default) | sherpa-onnx, `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` | multilingual (incl. Russian), silence-robust, ~1 GB |

Benchmark, 21/10/2026: 66 Open ASR Leaderboard clips (AMI, Earnings-22,
LibriSpeech other, Common Voice, VoxPopuli; 5–25 s; ~10 min), leaderboard
normaliser, word error rate / real-time factor / median latency per clip, on
Server2's CPU unless noted:

- Bark 3.0 Plus: 6.1 % / 0.11 / 0.9 s
- qwen3_asr int8: 10.6 % / 0.24 / 1.9 s
- whisper large-v3-turbo: 10.5 % / 1.62 (11.2 % / 0.37 / 3.2 s on a GTX 1650)
- whisper small: 17.2 % / 0.35

Bark ran no faster on the GTX 1650 than on this CPU (greedy TDT decoding is
per-step and host-bound), so the GPU copy and the whisper engine were retired.

Qwen is the default so Russian and other non-English speech is not silently
returned as English gibberish. The fallback is disabled by default: an
English-only engine is not a safe fallback for arbitrary languages. For an
explicitly English-only installation, Bark can still be selected with
`STT_PRIMARY_ENGINE=bark`.

Audio longer than five minutes is rejected with HTTP 413 before model loading;
raising the limit beyond 300 seconds is rejected at startup. This bounds Bark's
quadratic attention memory as well as queue time. Qwen decodes in 20-second
windows because its packaged ONNX decoder has a 512-token context; sending a
whole long clip silently truncates audio and substantially increases memory.
All windows are transcribed in order. `/health` advertises
`maxAudioSeconds`. Clients should allow the daemon's five-minute processing
timeout and retain recordings on failure.

Parakeet TDT v3 via sherpa-onnx was evaluated and rejected: the available
PyPI/GitHub packages ship a decoder whose NeMo metadata (`vocab_size`) is
stripped by int8 quantisation, and the fp32/sherpa export mismatches the
runtime input layout. Bark (a Parakeet fine-tune) avoids that by running in
plain transformers.

## Configuration (env)

    STT_HOST               default 127.0.0.1
    STT_PORT               default 8182 (ignored under systemd socket activation)
    STT_PRIMARY_ENGINE     default qwen3_asr
    STT_FALLBACK_ENGINE    default "" (disabled)
    STT_IDLE_TIMEOUT_MS    default 1800000 (30 min); min 60s, max 24h
    STT_TIMEOUT_MS         default 300000 (5 min per transcription)
    STT_MAX_AUDIO_BYTES    default 25165824 (25 MiB request cap)
    STT_MAX_AUDIO_SECONDS  default 300 (5 min); min 10, max 300
    BARK_DIR               default ~/ai-models/underdog-bark-3.0-plus (prepare_bark.py output)
    BARK_THREADS           default 4
    QWEN3_ASR_DIR          default ~/.pi/stt/models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
    QWEN3_THREADS          default 4
    SHUTDOWN_DRAIN_TIMEOUT_MS default 600000

`BARK_DIR` and `QWEN3_ASR_DIR` must be pre-populated (Bark by
`prepare_bark.py`); the deployer provisions both. Install PyTorch from
`requirements-torch-cpu.txt` before `requirements.txt`.

## Tests

    venv/bin/python -m pytest test/ -q

Tests cover request validation, engine chain fallback (including the
empty-primary case), serialisation, idle-timer semantics and the HTTP routes
with fake engines — no real weights required.
