# pi-stt-server

Minimal resident local speech-to-text HTTP server, mirroring the design and
lifecycle of `pi-voice-server` (Kokoro TTS): systemd socket activation, lazy
engine load, serialised transcription, clean idle exit.

It replaces agentchatbox's per-request `faster-whisper` shell-out
(`scripts/transcribe.py`), which reloaded the model from disk for every voice
note. Weights load once and stay warm; the process exits after an idle
timeout.

## HTTP surface

    GET  /health      → capability, residency, engine metadata
    POST /transcribe  → raw audio bytes (any container, any sample rate)
                        → {"text": ..., "language": ..., "duration": ...,
                           "engine": ..., "model": ..., "engineMs": ...}

No multipart parsing: the caller sends the audio file's bytes as the request
body (`Content-Type: application/octet-stream`). Decoding is in-process:

- the `whisper` engine (faster-whisper/CTranslate2) decodes via bundled PyAV;
- the `qwen3_asr` (sherpa-onnx) and `bark` (transformers) engines are fed
  16 kHz mono PCM resampled in process with PyAV.

Transcriptions are serialised — one model, one caller at a time. A hung
transcription is bounded by `STT_TIMEOUT_MS` (default 5 min) and releases the
queue.

## Engines

| Engine | Library | Notes |
|---|---|---|
| `qwen3_asr` | sherpa-onnx, `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` | multilingual (incl. Russian), silence-robust, RTF ≈ 0.24–0.30 on Server2's CPU |
| `whisper` | faster-whisper, `large-v3-turbo` | `WHISPER_DEVICE=cuda` + `WHISPER_COMPUTE=int8_float16` measured RTF ≈ 0.21 on Lappy's GTX 1650; near large-v3 accuracy, multilingual |
| `bark` | transformers `ParakeetForTDT`, Underdog Bark 3.0 Plus | **English only**. fp32 on CPU, fp16 on CUDA (fp32 does not fit beside Lappy's desktop in 4 GB). ~3 GB RSS on CPU, ~1.3 GB VRAM. `prepare_bark.py OUT_DIR` downloads the pinned revision and unpacks it once (cold load 0.2 s instead of ~25 s) |

Bark benchmark, 21/10/2026: 66 Open ASR Leaderboard clips (AMI, Earnings-22,
LibriSpeech other, Common Voice, VoxPopuli; 5–25 s; ~10 min), leaderboard
normaliser, word error rate / real-time factor / median latency per clip:

- Bark 3.0 Plus, Server2 CPU (4 threads): 6.1 % / 0.11 / 0.9 s
- Bark 3.0 Plus, Lappy GTX 1650 fp16: 6.1 % / 0.13 / 1.2 s (greedy TDT
  decoding is per-step and host-bound, so the GPU barely helps short clips)
- qwen3_asr int8, Server2 CPU: 10.6 % / 0.24 / 1.9 s
- whisper large-v3-turbo, Lappy GPU: 11.2 % / 0.37 / 3.2 s
- whisper large-v3-turbo, Server2 CPU: 10.5 % / 1.62
- whisper small, Server2 CPU: 17.2 % / 0.35

Measured on the 15.45 s deployment clip (`test/fixtures`, same text):
whisper `base` 0.072, `small` 0.215, `medium` 0.571, `large-v3-turbo` 0.773
(Server2 CPU, int8) — base is fast but the least accurate tier, and
medium/turbo load in 31–35 s, which is exactly why this server keeps them
resident. GPU fp16 whisper-turbo was measured at RTF 0.81 (autoregressive
decoding latency-bound on the 1650 Max-Q); `int8_float16` at 0.209.

Parakeet TDT v3 via sherpa-onnx was evaluated and rejected: the available
PyPI/GitHub packages ship a decoder whose NeMo metadata (`vocab_size`) is
stripped by int8 quantisation, and the fp32/sherpa export mismatches the
runtime input layout. Bark (a Parakeet fine-tune) avoids that by running in
plain transformers.

## Configuration (env)

    STT_HOST               default 127.0.0.1
    STT_PORT               default 8182 (ignored under systemd socket activation)
    STT_PRIMARY_ENGINE     default qwen3_asr
    STT_FALLBACK_ENGINE    default whisper  ("" disables)
    STT_IDLE_TIMEOUT_MS    default 1800000 (30 min); min 60s, max 24h
    STT_TIMEOUT_MS         default 300000 (5 min per transcription)
    STT_MAX_AUDIO_BYTES    default 25165824 (25 MiB request cap)
    STT_MAX_AUDIO_SECONDS  default 3600
    WHISPER_MODEL          default large-v3-turbo
    WHISPER_DEVICE         default cpu     (cuda on Lappy)
    WHISPER_COMPUTE        default int8    (int8_float16 for cuda)
    QWEN3_ASR_DIR          default ~/.pi/stt/models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
    QWEN3_THREADS          default 4
    BARK_DIR               default ~/ai-models/underdog-bark-3.0-plus (prepare_bark.py output)
    BARK_DEVICE            default cpu     (cuda on Lappy)
    BARK_THREADS           default 4
    SHUTDOWN_DRAIN_TIMEOUT_MS default 600000

The first request to the whisper engine downloads its weights into the
HuggingFace cache; `QWEN3_ASR_DIR` and `BARK_DIR` must be pre-populated (Bark
by `prepare_bark.py`). Production deployments provision all of them so the
first real voice note doesn't pay for a download.

Install PyTorch before `requirements.txt`: `requirements-torch-cpu.txt` on
Server2, `requirements-torch-cuda.txt` on Lappy.

## Tests

    venv/bin/python -m pytest test/ -q

Tests cover request validation, engine chain fallback (including the
empty-primary case), serialisation, idle-timer semantics and the HTTP routes
with fake engines — no real weights required.
