#!/usr/bin/env python3
"""
Resident local STT HTTP server (agentchatbox voice-note transcription).

Mirrors pi-voice-server's lifecycle: systemd socket activation, lazy engine
load, serialised work, clean idle exit. See README.md for the HTTP surface
and configuration.

Engines:
  qwen3_asr  — sherpa-onnx Qwen3-ASR 0.6B int8 (multilingual, silence-robust)
  whisper    — faster-whisper (PyAV decode; cpu or cuda)

Both engines decode in-process; the caller posts raw audio bytes.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

from aiohttp import web

import av
import numpy as np

MIN_IDLE_MS = 60_000
MAX_IDLE_MS = 24 * 60 * 60 * 1000
SYSTEMD_FD = 3
ENGINE_NAMES = ("qwen3_asr", "whisper")

STATE_DIR = Path.home() / ".pi" / "stt"


def log(*args: Any) -> None:
    print("[stt-server]", *args, flush=True)


def parse_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer from {minimum} to {maximum}")
    if value < minimum or value > maximum:
        raise SystemExit(f"{name} must be an integer from {minimum} to {maximum}")
    return value


class AudioTooLong(Exception):
    pass


class EngineError(Exception):
    pass


# ── Audio decoding (shared by both engines) ─────────────────────────


def decode_to_16k_mono(data: bytes, max_seconds: int) -> tuple[np.ndarray, float]:
    """Decode arbitrary container bytes to float32 mono samples at 16 kHz.

    PyAV handles webm/opus, mp4/aac, wav, ogg, flac and the rest of its
    built-in library. Returns (samples, original_duration_seconds).
    """
    container = av.open(io.BytesIO(data))
    try:
        audio_streams = container.streams.audio
        if not audio_streams:
            raise ValueError("no audio stream in uploaded audio")
        stream = audio_streams[0]
        if stream.duration and stream.time_base:
            original_duration = float(stream.duration * stream.time_base)
        else:
            original_duration = 0.0
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks: list[np.ndarray] = []
        total_samples = 0
        limit = max_seconds * 16000
        for packet in container.demux(stream):
            for frame in packet.decode():
                for rf in resampler.resample(frame):
                    arr = rf.to_ndarray()
                    if arr.ndim == 2:
                        arr = arr.reshape(-1)
                    arr = arr.astype(np.int16)
                    if total_samples + len(arr) > limit:
                        raise AudioTooLong(
                            f"decoded audio exceeds {max_seconds}s limit"
                        )
                    chunks.append(arr)
                    total_samples += len(arr)
        if total_samples == 0:
            raise ValueError("uploaded audio contains no decodable audio frames")
        pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
        return pcm, original_duration
    finally:
        container.close()


def save_state(engine: str, model: str) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        (STATE_DIR / "server-state.json").write_text(
            json.dumps({"engine": engine, "model": model, "resident": True,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


# ── Engines ─────────────────────────────────────────────────────────


class TranscriptionResult:
    __slots__ = ("text", "language", "duration", "engine", "model")

    def __init__(
        self, text: str, engine: str, model: str,
        language: Optional[str] = None, duration: Optional[float] = None,
    ) -> None:
        self.text = text
        self.engine = engine
        self.model = model
        self.language = language
        self.duration = duration


class Qwen3AsrEngine:
    """sherpa-onnx OfflineRecognizer over the packaged Qwen3-ASR 0.6B int8 model."""

    name = "qwen3_asr"
    default_model = "qwen3-asr-0.6B-int8-2026-03-25"

    def __init__(self, model_dir: str, threads: int) -> None:
        self._model_dir = model_dir
        self._threads = threads
        self._recognizer: Any = None

    @property
    def model(self) -> str:
        return Path(self._model_dir).name or self.default_model

    @property
    def resident(self) -> bool:
        return self._recognizer is not None

    def ensure(self) -> Any:
        if self._recognizer is None:
            import sherpa_onnx

            base = Path(self._model_dir).expanduser()
            if not base.is_dir():
                raise EngineError(
                    f"qwen3_asr model directory missing: {base} "
                    "(pre-populate it or set QWEN3_ASR_DIR)"
                )
            for required in ("conv_frontend.onnx", "encoder.int8.onnx",
                             "decoder.int8.onnx", "tokenizer"):
                if not (base / required).exists():
                    raise EngineError(
                        f"qwen3_asr model directory incomplete: "
                        f"missing {required} in {base}"
                    )
            try:
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                    conv_frontend=str(base / "conv_frontend.onnx"),
                    encoder=str(base / "encoder.int8.onnx"),
                    decoder=str(base / "decoder.int8.onnx"),
                    tokenizer=str(base / "tokenizer"),
                    num_threads=self._threads,
                )
            except Exception as e:
                raise EngineError(f"qwen3_asr engine failed to load: {e}")
            log(f"qwen3_asr engine ready ({self.model}, threads={self._threads})")
            save_state(self.name, self.model)
        return self._recognizer

    def transcribe(self, data: bytes) -> TranscriptionResult:
        recognizer = self.ensure()
        samples, original_duration = decode_to_16k_mono(data, MAX_AUDIO_SECONDS)
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        started = time.monotonic()
        recognizer.decode_stream(stream)
        decode_ms = (time.monotonic() - started) * 1000
        log(f"qwen3_asr: decoded {len(samples) / 16000:.1f}s in {decode_ms:.0f}ms")
        return TranscriptionResult(
            text=stream.result.text.strip(),
            engine=self.name,
            model=self.model,
            duration=original_duration or len(samples) / 16000.0,
        )


class WhisperEngine:
    """faster-whisper (CTranslate2), resident; decodes via bundled PyAV."""

    name = "whisper"
    default_model = "large-v3-turbo"

    def __init__(self, model: str, device: str, compute: str) -> None:
        self._model = model
        self._device = device
        self._compute = compute
        self._model_obj: Any = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def resident(self) -> bool:
        return self._model_obj is not None

    def ensure(self) -> Any:
        if self._model_obj is None:
            from faster_whisper import WhisperModel

            try:
                self._model_obj = WhisperModel(
                    self._model, device=self._device, compute_type=self._compute
                )
            except Exception as e:
                raise EngineError(
                    f"whisper engine failed to load {self._model}: {e}"
                )
            log(f"whisper engine ready ({self._model}, device={self._device}, "
                f"compute={self._compute})")
            save_state(self.name, self._model)
        return self._model_obj

    def transcribe(self, data: bytes) -> TranscriptionResult:
        model_obj = self.ensure()
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
        try:
            tmp.write(data)
            tmp.close()
            segments, info = model_obj.transcribe(tmp.name, beam_size=1, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            return TranscriptionResult(
                text=text, engine=self.name, model=self.model,
                language=info.language, duration=info.duration,
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


def build_engine(name: str, *, default: bool) -> Any:
    if name == "qwen3_asr":
        return Qwen3AsrEngine(
            os.environ.get("QWEN3_ASR_DIR")
            or str(Path.home() / ".pi" / "stt" / "models" / Qwen3AsrEngine.default_model),
            parse_int_env("QWEN3_THREADS", 4, 1, 64),
        )
    if name == "whisper":
        return WhisperEngine(
            os.environ.get("WHISPER_MODEL") or WhisperEngine.default_model,
            os.environ.get("WHISPER_DEVICE") or "cpu",
            os.environ.get("WHISPER_COMPUTE") or "int8",
        )
    if default:
        return None  # fallback disabled by empty env
    raise SystemExit(f"unknown engine '{name}' (expected one of {ENGINE_NAMES})")


# ── Lifecycle (semantics mirrored from pi-voice-server lifecycle.mjs) ──


class IdleShutdown:
    """Arm shutdown only when idle; transcription activity disarms it.

    Health probes deliberately do not reset the timer; work_started/work_finished
    wrap the complete serialised transcription.
    """

    def __init__(self, timeout_ms: int, on_idle: Any) -> None:
        self._timeout_s = timeout_ms / 1000.0
        self._on_idle = on_idle
        self._handle: Optional[asyncio.TimerHandle] = None
        self._stopped = False

    def _arm(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        if self._stopped:
            return
        self._handle = asyncio.get_running_loop().call_later(self._timeout_s, self._fire)

    def _fire(self) -> None:
        self._handle = None
        self._on_idle()

    def start(self) -> None:
        self._arm()

    def work_started(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def work_finished(self) -> None:
        self._arm()

    def stop(self) -> None:
        self._stopped = True
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None


# ── Server state ────────────────────────────────────────────────────

PRIMARY = build_engine(os.environ.get("STT_PRIMARY_ENGINE") or "qwen3_asr", default=False)
FALLBACK = build_engine(os.environ.get("STT_FALLBACK_ENGINE", "whisper"), default=True)
CHAIN = [PRIMARY] + ([FALLBACK] if FALLBACK else [])

IDLE_TIMEOUT_MS = parse_int_env("STT_IDLE_TIMEOUT_MS", 1_800_000, MIN_IDLE_MS, MAX_IDLE_MS)
TRANSCRIBE_TIMEOUT_MS = parse_int_env("STT_TIMEOUT_MS", 300_000, 10_000, 3_600_000)
MAX_AUDIO_BYTES = parse_int_env("STT_MAX_AUDIO_BYTES", 25 * 1024 * 1024, 1024 * 1024, 512 * 1024 * 1024)
MAX_AUDIO_SECONDS = parse_int_env("STT_MAX_AUDIO_SECONDS", 3600, 10, 86_400)
SHUTDOWN_DRAIN_TIMEOUT_MS = parse_int_env("SHUTDOWN_DRAIN_TIMEOUT_MS", 600_000, 10_000, 3_600_000)

listen_lock = asyncio.Lock()
active_transcriptions = 0
shutting_down = False
need_restart = False

idle: IdleShutdown  # bound below, after shutdown() is defined


def build_app() -> web.Application:
    application = web.Application(client_max_size=MAX_AUDIO_BYTES)
    application.router.add_get("/health", handle_health)
    application.router.add_post("/transcribe", handle_transcribe)
    return application


async def transcribe_chain(data: bytes) -> TranscriptionResult:
    """Run the engine chain: skip errored engines, escalate on empty results.

    Real weights make a transcription hang vanishingly unlikely, but a timeout
    still cannot be allowed to leave a wedge: it poisons the process, the
    request is answered 503, and the process exits so systemd replaces the
    model state (same fail-closed pattern as pi-voice-server).
    """
    global need_restart
    last_error: Optional[Exception] = None
    for engine in CHAIN:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(engine.transcribe, data),
                timeout=TRANSCRIBE_TIMEOUT_MS / 1000.0,
            )
        except asyncio.TimeoutError:
            need_restart = True
            last_error = EngineError(
                f"{engine.name} transcription timed out after {TRANSCRIBE_TIMEOUT_MS}ms"
            )
            log(str(last_error))
            continue
        except AudioTooLong as e:
            raise  # deterministic rejection: do not escalate
        except Exception as e:
            last_error = e
            log(f"{engine.name} engine error: {e}")
            continue
        if result.text or engine is CHAIN[-1]:
            return result
        log(f"{engine.name} produced empty transcript; escalating to fallback")
    raise last_error or EngineError("no engine produced a transcript")


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "modelAvailable": True,
            "modelResident": any(e.resident for e in CHAIN),
            "modelLoading": False,
            "engines": {e.name: {"resident": e.resident, "model": e.model}
                        for e in CHAIN},
            "idleTimeoutMs": IDLE_TIMEOUT_MS,
            "timeoutMs": TRANSCRIBE_TIMEOUT_MS,
            "maxAudioBytes": MAX_AUDIO_BYTES,
            "activeTranscriptions": active_transcriptions,
            "status": "draining" if shutting_down else "ok",
        }
    )


async def handle_transcribe(request: web.Request) -> web.Response:
    global active_transcriptions
    if request.content_length is None:
        return web.json_response(
            {"error": "Content-Length required (send raw audio bytes as body)"}, status=411
        )
    data = await request.read()
    if not data:
        return web.json_response(
            {"error": "no audio uploaded (send raw audio bytes)"}, status=400
        )
    async with listen_lock:
        active_transcriptions += 1
        idle.work_started()
        try:
            result = await transcribe_chain(data)
        except EngineError as e:
            return web.json_response({"error": str(e)}, status=503)
        except AudioTooLong as e:
            return web.json_response({"error": str(e)}, status=413)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log("transcription error:", e)
            return web.json_response({"error": f"transcription failed: {e}"}, status=500)
        finally:
            active_transcriptions -= 1
            idle.work_finished()
    response: dict[str, Any] = {"text": result.text, "engine": result.engine,
                                "model": result.model}
    if result.language:
        response["language"] = result.language
    if result.duration is not None:
        response["duration"] = result.duration
    if need_restart:
        # Answer first; a poisoned process must not linger with a wedged model.
        asyncio.get_running_loop().call_later(0.5, lambda: os._exit(1))
    return web.json_response(response)


app = build_app()
runner: Optional[web.AppRunner] = None


async def start_server() -> None:
    global runner
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    listen_pid = os.environ.get("LISTEN_PID")
    listen_fds = os.environ.get("LISTEN_FDS")
    if listen_pid == str(os.getpid()) and listen_fds:
        if int(listen_fds) != 1:
            log("FATAL: expected exactly one systemd socket, received", listen_fds)
            sys.exit(1)
        fdnames = os.environ.get("LISTEN_FDNAMES", "")
        if fdnames and fdnames != "stt":
            log("FATAL: unexpected systemd socket name:", fdnames)
            sys.exit(1)
        import socket

        sock = socket.socket(fileno=SYSTEMD_FD)
        await web.SockSite(runner, sock).start()
        log("listening on systemd socket fd 3")
    else:
        host = os.environ.get("STT_HOST") or "127.0.0.1"
        port = int(os.environ.get("STT_PORT") or "8182")
        await web.TCPSite(runner, host, port).start()
        log(f"listening on http://{host}:{port}")


async def stop_server() -> None:
    if runner is not None:
        await runner.cleanup()


async def shutdown(reason: str) -> None:
    global shutting_down
    if shutting_down:
        return
    shutting_down = True
    log(f"received {reason}; draining {active_transcriptions} transcription(s)")
    idle.stop()
    try:
        await asyncio.wait_for(_drain(), timeout=SHUTDOWN_DRAIN_TIMEOUT_MS / 1000.0)
        sys.exit(0)
    except asyncio.TimeoutError:
        log(f"shutdown drain exceeded {SHUTDOWN_DRAIN_TIMEOUT_MS}ms")
        sys.exit(1)


async def _drain() -> None:
    while active_transcriptions > 0:
        await asyncio.sleep(0.1)
    await stop_server()


def main() -> None:
    def _request_shutdown(*_: Any) -> None:
        asyncio.get_running_loop().call_soon(
            lambda: asyncio.ensure_future(shutdown("SIGTERM"))
        )

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    async def _run() -> None:
        await start_server()
        log(f"lazy engine load; idle shutdown={IDLE_TIMEOUT_MS}ms")
        idle.start()
        while True:
            await asyncio.sleep(3600)

    asyncio.run(_run())


idle = IdleShutdown(
    IDLE_TIMEOUT_MS,
    lambda: asyncio.get_running_loop().call_soon(
        lambda: asyncio.ensure_future(shutdown("idle timeout"))
    ),
)

if __name__ == "__main__":
    main()
