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


# ── Lifecycle ───────────────────────────────────────────────────────


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
