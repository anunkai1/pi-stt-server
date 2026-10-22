"""Tests for stt_server: routes, engine chain, validation, lifecycle.

Fake engines stand in for real weights, so nothing here needs network or GPU.
"""

import asyncio
import io
import wave

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import stt_server
from stt_server import (
    AudioTooLong,
    EngineError,
    IdleShutdown,
    TranscriptionResult,
    decode_to_16k_mono,
)


# ── Helpers ─────────────────────────────────────────────────────────


def make_wav(seconds: float, rate: int = 24000) -> bytes:
    """A valid single-channel WAV containing a simple tone."""
    n = int(seconds * rate)
    t = np.linspace(0, seconds, n, endpoint=False)
    samples = (3000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


class FakeEngine:
    def __init__(self, name, text="", error=None, delay=0.0):
        self.name = name
        self._text = text
        self._error = error
        self._delay = delay
        self.calls = []

    @property
    def model(self):
        return f"{self.name}-fake"

    @property
    def resident(self):
        return True

    def transcribe(self, data):
        self.calls.append(data)
        if self._delay:
            import time

            time.sleep(self._delay)
        if self._error:
            raise self._error
        return TranscriptionResult(
            text=self._text, engine=self.name, model=self.model,
            language="en", duration=1.5,
        )


@pytest.fixture
def chain(monkeypatch):
    """Replace the real engine chain with two controllable fakes."""
    primary = FakeEngine("primary", text="hello world")
    fallback = FakeEngine("fallback", text="fallback text")
    monkeypatch.setattr(stt_server, "CHAIN", [primary, fallback])
    return primary, fallback


@pytest.fixture
async def client(chain, aiohttp_client_factory=None):
    server = TestServer(stt_server.build_app())
    client = TestClient(server)
    await client.start_server()
    yield client
    await client.close()


# ── decode_to_16k_mono ──────────────────────────────────────────────


def test_decode_resamples_and_keeps_duration():
    data = make_wav(1.0, rate=24000)
    samples, duration = decode_to_16k_mono(data, max_seconds=10)
    assert samples.dtype == np.float32
    assert abs(len(samples) - 16000) <= 50  # 1s at 16 kHz (± resampler granularity)
    assert duration == pytest.approx(1.0, abs=0.01)


def test_decode_rejects_too_long():
    data = make_wav(2.0, rate=16000)
    with pytest.raises(AudioTooLong):
        decode_to_16k_mono(data, max_seconds=1)


def test_decode_rejects_non_audio():
    with pytest.raises(Exception):
        decode_to_16k_mono(b"not audio at all", max_seconds=10)


# ── HTTP routes ─────────────────────────────────────────────────────


async def test_health_contract(client):
    resp = await client.get("/health")
    assert resp.status == 200
    body = await resp.json()
    assert body["modelAvailable"] is True
    assert body["status"] == "ok"
    assert "engines" in body
    assert "idleTimeoutMs" in body
    assert body["maxAudioSeconds"] == stt_server.MAX_AUDIO_SECONDS


async def test_transcribe_roundtrip(client, chain):
    primary, _ = chain
    resp = await client.post(
        "/transcribe", data=b"fake-bytes", headers={"Content-Type": "application/octet-stream"}
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["text"] == "hello world"
    assert body["engine"] == "primary"
    assert primary.calls == [b"fake-bytes"]


async def test_transcribe_requires_content_length(client):
    resp = await client.post("/transcribe")
    assert resp.status in (400, 411)


async def test_transcribe_empty_body(client):
    resp = await client.post(
        "/transcribe", data=b"", headers={"Content-Length": "0"}
    )
    assert resp.status == 400


# ── Engine chain fallback ───────────────────────────────────────────


async def test_escalates_on_engine_error(client, chain):
    primary, fallback = chain
    primary._error = EngineError("boom")
    resp = await client.post(
        "/transcribe", data=b"x", headers={"Content-Type": "application/octet-stream"}
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["text"] == "fallback text"
    assert body["engine"] == "fallback"


async def test_escalates_on_empty_primary(client, chain):
    primary, fallback = chain
    primary._text = ""
    resp = await client.post(
        "/transcribe", data=b"x", headers={"Content-Type": "application/octet-stream"}
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["text"] == "fallback text"


async def test_empty_result_from_last_engine_is_returned(client, chain, monkeypatch):
    primary, fallback = chain
    primary._text = ""
    fallback._text = ""
    resp = await client.post(
        "/transcribe", data=b"x", headers={"Content-Type": "application/octet-stream"}
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["text"] == ""


async def test_all_engines_error_returns_503(client, chain):
    primary, fallback = chain
    primary._error = EngineError("boom")
    fallback._error = EngineError("also boom")
    resp = await client.post(
        "/transcribe", data=b"x", headers={"Content-Type": "application/octet-stream"}
    )
    assert resp.status == 503
    body = await resp.json()
    assert "boom" in body["error"]


def test_qwen_windows_cover_all_audio_without_exceeding_decoder_context(monkeypatch):
    from types import SimpleNamespace
    engine = stt_server.Qwen3AsrEngine("fake", 1)
    chunks = []
    def create_stream():
        return SimpleNamespace(
            accept_waveform=lambda rate, pcm: chunks.append(pcm.copy()),
            result=SimpleNamespace(text=f"part {len(chunks) + 1}"),
        )
    monkeypatch.setattr(engine, "ensure", lambda: SimpleNamespace(create_stream=create_stream, decode_stream=lambda stream: None))
    samples = np.arange(45 * 16000, dtype=np.float32)
    monkeypatch.setattr(stt_server, "decode_to_16k_mono", lambda *args: (samples, 45))
    result = engine.transcribe(b"audio")
    assert [len(chunk) for chunk in chunks] == [20 * 16000, 20 * 16000, 5 * 16000]
    np.testing.assert_array_equal(np.concatenate(chunks), samples)
    assert result.text == "part 1 part 2 part 3" and result.duration == 45


def test_default_chain_is_multilingual_and_limit_is_bounded():
    import os
    import subprocess
    import sys
    env = dict(os.environ)
    for key in ("STT_PRIMARY_ENGINE", "STT_FALLBACK_ENGINE", "STT_MAX_AUDIO_SECONDS"):
        env.pop(key, None)
    probe = "import stt_server as s; print([e.name for e in s.CHAIN], s.MAX_AUDIO_SECONDS)"
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "['qwen3_asr'] 300"
    result = subprocess.run([sys.executable, "-c", probe], env={**env, "STT_MAX_AUDIO_SECONDS": "3600"}, capture_output=True)
    assert result.returncode != 0


@pytest.mark.parametrize("engine_class", [stt_server.BarkEngine, stt_server.Qwen3AsrEngine])
def test_long_audio_rejected_before_model_loading(monkeypatch, engine_class):
    monkeypatch.setattr(stt_server, "MAX_AUDIO_SECONDS", 1)
    engine = engine_class("not-loaded", 1)
    def no_load():
        pytest.fail("must reject long audio before loading weights")
    monkeypatch.setattr(engine, "ensure", no_load)
    with pytest.raises(AudioTooLong, match="1s limit"):
        engine.transcribe(make_wav(2))


async def test_audio_limit_is_413_without_fallback(client, chain):
    primary, fallback = chain
    primary._error = AudioTooLong("decoded audio exceeds 300s limit")
    response = await client.post("/transcribe", data=b"too-long")
    assert response.status == 413
    assert "300s" in (await response.json())["error"]
    assert fallback.calls == []


# ── Lifecycle ───────────────────────────────────────────────────────


async def test_timeout_on_last_engine_restarts_the_process(client, monkeypatch):
    """A timed-out engine thread cannot be cancelled: answer 503, then exit."""
    import os

    monkeypatch.setattr(stt_server, "CHAIN", [FakeEngine("only", text="x", delay=0.6)])
    monkeypatch.setattr(stt_server, "TRANSCRIBE_TIMEOUT_MS", 100)
    monkeypatch.setattr(stt_server, "need_restart", False)
    monkeypatch.setattr(stt_server, "restart_scheduled", False)
    exits = []
    monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))
    r = await client.post("/transcribe", data=make_wav(1.0))
    assert r.status == 503
    assert "timed out" in (await r.json())["error"]
    await asyncio.sleep(0.8)
    assert exits == [1]


def test_unknown_fallback_engine_name_is_rejected():
    with pytest.raises(SystemExit):
        stt_server.build_engine("wisper", default=True)
    assert stt_server.build_engine("", default=True) is None


def test_bark_engine_config_and_unprepared_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("BARK_DIR", str(tmp_path / "bark"))
    engine = stt_server.build_engine("bark", default=False)
    assert (engine.name, engine.model, engine.resident) == ("bark", "bark", False)
    with pytest.raises(stt_server.EngineError, match="unprepared"):
        engine.ensure()


async def test_idle_shutdown_semantics():
    fired = []
    idle = IdleShutdown(50, lambda: fired.append(True))
    idle.start()
    await asyncio.sleep(0.15)
    assert fired == [True]


async def test_idle_shutdown_disarmed_by_work():
    fired = []
    idle = IdleShutdown(50, lambda: fired.append(True))
    idle.start()
    idle.work_started()
    await asyncio.sleep(0.15)
    assert fired == []
    idle.work_finished()
    await asyncio.sleep(0.15)
    assert fired == [True]


def test_transcribe_serialised():
    """Two slow calls must not overlap on one model."""
    order = []
    engines = [FakeEngine("a", text="A", delay=0.05), FakeEngine("b", text="B")]

    async def run():
        lock = asyncio.Lock()
        for engine in engines:
            async with lock:
                order.append(f"{engine.name}:start")
                await asyncio.to_thread(engine.transcribe, b"")
                order.append(f"{engine.name}:end")

    asyncio.run(run())
    assert order == ["a:start", "a:end", "b:start", "b:end"]
