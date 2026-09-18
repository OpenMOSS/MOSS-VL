"""Smoke-client retry policy against real local WebSocket connections."""

import asyncio
import importlib.util
import json
import time
from pathlib import Path

import pytest
from adapter import config
from adapter.server import AdapterServer
from websockets.asyncio.server import serve

spec = importlib.util.spec_from_file_location(
    "smoke_under_test", Path(__file__).with_name("smoke_client.py")
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_busy", [True, False])
async def test_busy_reconnects_and_does_not_taint_success(monkeypatch, retry_busy):
    monkeypatch.setattr(config, "MAX_INFLIGHT", 1)
    monkeypatch.setattr(smoke, "BUSY_RETRY_DELAYS", (0.01, 0.01, 0.01))
    adapter = AdapterServer()
    adapter._inflight = 1
    connections = []

    async def handler(ws):
        connections.append(ws)
        if len(connections) == 1:
            await adapter._handler(ws)
            return
        await ws.recv()
        await ws.send('{"type":"ready"}')
        await ws.recv()
        assert await ws.recv() == b"frame"
        await ws.send('{"type":"frame_ack"}')
        await ws.send('{"type":"output","text":"hello<|im_end|>"}')
        await ws.recv()

    async with serve(handler, "127.0.0.1", 0) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        result = await smoke.run_round(url, [b"frame"], retry_busy=retry_busy)
    assert len(connections) == (2 if retry_busy else 1)
    if retry_busy:
        assert result.marker_seen and result.acks == 1
        assert result.visible_text == "hello" and result.errors == []
        assert result.retry_errors == [smoke.BUSY_SUBSTRING]
    else:
        assert result.errors == [smoke.BUSY_SUBSTRING]
        assert not result.marker_seen and not result.retry_errors


@pytest.mark.asyncio
async def test_busy_retries_are_bounded(monkeypatch):
    monkeypatch.setattr(config, "MAX_INFLIGHT", 1)
    monkeypatch.setattr(smoke, "BUSY_RETRY_DELAYS", (0.01, 0.01, 0.01))
    adapter = AdapterServer()
    adapter._inflight = 1
    attempts = []

    async def handler(ws):
        attempts.append(ws)
        await adapter._handler(ws)

    async with serve(handler, "127.0.0.1", 0) as server:
        result = await smoke.run_round(
            f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}", []
        )
    assert len(attempts) == 4
    assert result.errors == [smoke.BUSY_SUBSTRING]
    assert result.retry_errors == [smoke.BUSY_SUBSTRING] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
async def test_ready_wait_and_backoff_share_one_deadline(monkeypatch, busy):
    monkeypatch.setattr(smoke, "ROUND_TIMEOUT_S", 0.05)
    monkeypatch.setattr(smoke, "BUSY_RETRY_DELAYS", (0.2, 0.2, 0.2))
    attempts = []

    async def handler(ws):
        attempts.append(ws)
        await ws.recv()
        if busy:
            await ws.send(
                json.dumps({"type": "error", "message": smoke.BUSY_SUBSTRING})
            )
        await ws.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        start = time.monotonic()
        result = await smoke.run_round(
            f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}", []
        )
        elapsed = time.monotonic() - start
    assert len(attempts) == 1 and result.errors
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_busy_after_ready_is_not_replayed(monkeypatch):
    calls = []

    async def once(*args, **kwargs):
        calls.append(kwargs)
        result = smoke.RoundResult()
        result.t_ready = time.monotonic()
        result.errors = [smoke.BUSY_SUBSTRING]
        return result

    monkeypatch.setattr(smoke, "_run_round_once", once)
    result = await smoke.run_round("ws://test", [])
    assert len(calls) == 1 and result.errors and not result.retry_errors


@pytest.mark.asyncio
async def test_vl02_keeps_strict_no_retry_contract(monkeypatch):
    calls = []

    async def run(*args, **kwargs):
        calls.append(kwargs)
        result = smoke.RoundResult()
        result.marker_seen = True
        return result

    monkeypatch.setattr(smoke, "run_round", run)
    assert await smoke.test_vl02("ws://test", [b"frame"] * 4)
    assert len(calls) == 2 and all(c["retry_busy"] is False for c in calls)


@pytest.mark.asyncio
async def test_non_busy_error_is_not_retried(monkeypatch):
    calls = []

    async def once(*args, **kwargs):
        calls.append(kwargs)
        result = smoke.RoundResult()
        result.errors = ["backend unavailable"]
        return result

    monkeypatch.setattr(smoke, "_run_round_once", once)
    result = await smoke.run_round("ws://test", [])
    assert len(calls) == 1 and result.errors == ["backend unavailable"]


@pytest.mark.asyncio
async def test_cancellation_is_not_retried_or_swallowed(monkeypatch):
    entered = asyncio.Event()

    async def once(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(smoke, "_run_round_once", once)
    task = asyncio.create_task(smoke.run_round("ws://test", []))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
