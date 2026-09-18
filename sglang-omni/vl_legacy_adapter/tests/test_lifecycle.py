"""CPU-only regression tests for frame reception, startup and admission."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import config, params, server, session, upstream


class Socket:
    def __init__(self):
        self.sent = []
        self.incoming = asyncio.Queue()
        self.sent_event = asyncio.Event()
        self.closed = False

    async def send(self, raw):
        self.sent.append(json.loads(raw))
        self.sent_event.set()

    async def recv(self):
        return await self.incoming.get()

    async def close(self, *args, **kwargs):
        self.closed = True


class FakeUpstream:
    instances = []
    failure = None

    def __init__(self, *args):
        self.frames = []
        self.terminal_seen = False
        self.aborted = False
        self.closed = asyncio.Event()
        self.last_error = None
        type(self).instances.append(self)

    async def connect(self):
        if self.failure == "hang":
            await asyncio.Future()
        if self.failure:
            raise self.failure

    async def configure(self, payload):
        pass

    async def send_frame(self, seq_no, timestamp, payload, **kwargs):
        self.frames.append(
            dict(seq_no=seq_no, timestamp=timestamp, payload=payload, **kwargs)
        )

    async def abort(self):
        self.aborted = True

    async def close(self):
        self.closed.set()

    async def wait_closed(self):
        await self.closed.wait()


@pytest.fixture
def fake_upstream(monkeypatch):
    monkeypatch.setattr(FakeUpstream, "instances", [])
    monkeypatch.setattr(FakeUpstream, "failure", None)
    monkeypatch.setattr(session, "OmniUpstream", FakeUpstream)
    return FakeUpstream


@asynccontextmanager
async def adapter_endpoint(monkeypatch):
    monkeypatch.setattr(config, "MAX_INFLIGHT", 1)
    adapter = server.AdapterServer()
    async with serve(adapter._handler, "127.0.0.1", 0) as listener:
        port = listener.sockets[0].getsockname()[1]
        yield adapter, f"ws://127.0.0.1:{port}/v1/realtime"


async def receive(ws):
    return json.loads(await asyncio.wait_for(ws.recv(), 1))


async def wait_released(adapter):
    async with asyncio.timeout(1):
        while adapter._inflight:
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_metadata_before_quiet_deadline_keeps_frame(monkeypatch):
    monkeypatch.setattr(config, "FINALIZE_QUIET_S", 0.03)
    bridge = session.BridgeSession(Socket())
    bridge.ready_sent = True
    bridge.upstream = FakeUpstream()
    await bridge._handle_frame_meta({"timestamp": 1})
    await bridge._handle_binary(b"first")
    pump = asyncio.create_task(bridge._pump_frames())
    try:
        await asyncio.sleep(0.005)
        await bridge._handle_frame_meta({"timestamp": 2})
        await asyncio.sleep(0.06)
        assert not pump.done(), "finalized while a binary frame was still expected"
        await bridge._handle_binary(b"second")
        await asyncio.wait_for(pump, 1)
        assert [f["payload"] for f in bridge.upstream.frames] == [b"first", b"second"]
        assert [f["final"] for f in bridge.upstream.frames] == [False, True]
        assert bridge.downstream.sent == [{"type": "frame_ack"}] * 2
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


@pytest.mark.asyncio
async def test_final_handshake_rejects_new_input(monkeypatch):
    monkeypatch.setattr(config, "FINALIZE_QUIET_S", 0.01)
    bridge = session.BridgeSession(Socket())
    bridge.ready_sent = True
    entered, unblock = asyncio.Event(), asyncio.Event()

    class SlowUpstream(FakeUpstream):
        async def send_frame(self, *args, **kwargs):
            entered.set()
            await unblock.wait()
            await super().send_frame(*args, **kwargs)

    bridge.upstream = SlowUpstream()
    await bridge._handle_frame_meta({"timestamp": 1})
    await bridge._handle_binary(b"first")
    pump = asyncio.create_task(bridge._pump_frames())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await bridge._handle_frame_meta({"timestamp": 2})
        await bridge._handle_binary(b"extra")
        assert bridge.pending_meta is None
        assert bridge.frames.empty()
        assert len([m for m in bridge.downstream.sent if m["type"] == "error"]) == 2
    finally:
        unblock.set()
        await asyncio.wait_for(pump, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_frame", [True, False])
async def test_missing_binary_times_out_and_releases(
    monkeypatch, fake_upstream, first_frame
):
    monkeypatch.setattr(config, "FRAME_RECEIVE_TIMEOUT_S", 0.05, raising=False)
    monkeypatch.setattr(config, "FINALIZE_QUIET_S", 0.01)
    async with adapter_endpoint(monkeypatch) as (adapter, url):
        async with connect(url, proxy=None) as ws:
            await ws.send('{"type":"start"}')
            assert (await receive(ws))["type"] == "ready"
            if not first_frame:
                await ws.send('{"type":"frame","timestamp":0}')
                await ws.send(b"first")
            await ws.send('{"type":"frame","timestamp":1}')
            reply = await receive(ws)
            assert reply["type"] == "error" and "binary" in reply["message"]
            await asyncio.wait_for(ws.wait_closed(), 1)
        await wait_released(adapter)
        assert fake_upstream.instances[-1].aborted
        assert fake_upstream.instances[-1].closed.is_set()
        async with connect(url, proxy=None) as ws:
            await ws.send('{"type":"start"}')
            assert (await receive(ws))["type"] == "ready"


@pytest.mark.asyncio
async def test_invalid_payload_clears_pending_deadline(monkeypatch, fake_upstream):
    monkeypatch.setattr(config, "FRAME_RECEIVE_TIMEOUT_S", 0.03, raising=False)
    monkeypatch.setattr(config, "MAX_FRAME_BYTES", 3)
    async with adapter_endpoint(monkeypatch) as (_, url):
        async with connect(url, proxy=None) as ws:
            await ws.send('{"type":"start"}')
            assert (await receive(ws))["type"] == "ready"
            await ws.send('{"type":"frame","timestamp":1}')
            await ws.send(b"oversize")
            assert (await receive(ws))["type"] == "error"
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(ws.recv(), 0.08)
            await ws.send('{"type":"stop"}')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["overflow", "handshake", "connect", "unexpected", "timeout"]
)
async def test_start_failure_closes_and_recovers(monkeypatch, fake_upstream, failure):
    monkeypatch.setattr(config, "SETUP_TIMEOUT_S", 0.03, raising=False)
    failures = {
        "handshake": InvalidStatus(Response(403, "Forbidden", Headers(), b"")),
        "connect": upstream.UpstreamError("connection refused"),
        "unexpected": RuntimeError("unexpected setup failure"),
        "timeout": "hang",
    }
    fake_upstream.failure = failures.get(failure)
    async with adapter_endpoint(monkeypatch) as (adapter, url):
        async with connect(url, proxy=None) as ws:
            await ws.send(
                '{"type":"start","max_new_tokens":1e309}'
                if failure == "overflow"
                else '{"type":"start"}'
            )
            assert (await receive(ws))["type"] == "error"
            await asyncio.wait_for(ws.wait_closed(), 1)
        await wait_released(adapter)
        assert all(u.closed.is_set() for u in fake_upstream.instances)
        fake_upstream.failure = None
        async with connect(url, proxy=None) as ws:
            await ws.send('{"type":"start"}')
            assert (await receive(ws))["type"] == "ready"


@pytest.mark.asyncio
async def test_stop_during_startup_cancels_and_cleans(monkeypatch, fake_upstream):
    fake_upstream.failure = "hang"
    async with adapter_endpoint(monkeypatch) as (adapter, url):
        async with connect(url, proxy=None) as ws:
            await ws.send('{"type":"start"}')
            async with asyncio.timeout(1):
                while not fake_upstream.instances:
                    await asyncio.sleep(0.005)
            await ws.send('{"type":"stop"}')
            await asyncio.wait_for(ws.wait_closed(), 1)
        await wait_released(adapter)
        assert fake_upstream.instances[0].closed.is_set()
        assert fake_upstream.instances[0].aborted


@pytest.mark.asyncio
async def test_busy_io_does_not_block_release_or_admission(monkeypatch):
    monkeypatch.setattr(config, "MAX_INFLIGHT", 1)
    done = asyncio.Event()

    class ActiveBridge:
        def __init__(self, socket, on_client_done):
            self.release = on_client_done

        async def run(self):
            await done.wait()
            await self.release()
            await self.release()

    monkeypatch.setattr(server, "BridgeSession", ActiveBridge)
    adapter = server.AdapterServer()
    active = asyncio.create_task(adapter._handler(Socket()))
    await asyncio.sleep(0)
    busy_socket = Socket()
    busy = asyncio.create_task(adapter._handler(busy_socket))
    try:
        await asyncio.wait_for(busy_socket.sent_event.wait(), 1)
        done.set()
        await asyncio.wait_for(asyncio.shield(active), 0.2)
        assert adapter._inflight == 0
        assert not busy.done()
        next_socket = Socket()
        await asyncio.wait_for(adapter._handler(next_socket), 0.2)
        assert not next_socket.sent
        assert adapter._inflight == 0
    finally:
        done.set()
        busy_socket.incoming.put_nowait('{"type":"start"}')
        await asyncio.gather(active, busy)


@pytest.mark.parametrize(
    "field",
    [
        "temperature",
        "top_p",
        "max_new_tokens",
        "max_tokens_per_second",
        "frame_queue_size",
    ],
)
@pytest.mark.parametrize("value", [float("inf"), float("nan"), float("-inf")])
def test_nonfinite_numeric_parameters_rejected(field, value):
    with pytest.raises((ValueError, TypeError)):
        params.map_start_to_configure({field: value})


def test_sampling_mapping_unchanged():
    assert params.map_start_to_configure(
        dict(
            do_sample=False,
            temperature=0.2,
            top_p=0.8,
            max_new_tokens=512,
            max_tokens_per_second=10,
            frame_queue_size=32,
            top_k=20,
        )
    ) == {
        "type": "session.configure",
        "temperature": 0.0,
        "top_p": 0.8,
        "max_new_tokens": 512,
        "max_tokens_per_turn": 10.0,
        "input_queue_capacity": 32,
    }


@pytest.mark.asyncio
async def test_upstream_wraps_handshake_error(monkeypatch):
    async def rejected(*args, **kwargs):
        raise InvalidStatus(Response(403, "Forbidden", Headers(), b""))

    monkeypatch.setattr(upstream, "connect", rejected)
    client = upstream.OmniUpstream("ws://localhost/wrong", None)
    with pytest.raises(upstream.UpstreamError, match="connection failed"):
        await client.connect()


@pytest.mark.asyncio
async def test_configure_send_failure_cancels_futures():
    class BrokenSocket(Socket):
        async def send(self, raw):
            raise OSError("closed during configure")

    client = upstream.OmniUpstream("ws://localhost", None)
    client.ws = BrokenSocket()
    with pytest.raises(upstream.UpstreamError):
        await client.configure({"type": "session.configure"})
    assert client._configured.done() and client._ready.done()


@pytest.mark.asyncio
async def test_frame_enqueued_at_quiet_timeout_is_consumed(monkeypatch):
    bridge = session.BridgeSession(Socket())
    bridge.ready_sent = True
    original = asyncio.wait_for

    async def deadline_race(awaitable, timeout):
        awaitable.close()
        await bridge._handle_frame_meta({"timestamp": 1})
        await bridge._handle_binary(b"at-deadline")
        raise TimeoutError

    monkeypatch.setattr(session.asyncio, "wait_for", deadline_race)
    item = await bridge._next_frame_or_none()
    assert item == (1.0, b"at-deadline")
    assert not bridge._input_closed
    monkeypatch.setattr(session.asyncio, "wait_for", original)


@pytest.mark.asyncio
async def test_timeout_does_not_submit_held_frame(monkeypatch, fake_upstream):
    monkeypatch.setattr(config, "FRAME_RECEIVE_TIMEOUT_S", 0.03)
    async with adapter_endpoint(monkeypatch) as (adapter, url):
        async with connect(url, proxy=None) as ws:
            await ws.send('{"type":"start"}')
            await receive(ws)
            await ws.send('{"type":"frame","timestamp":0}')
            await ws.send(b"first")
            await ws.send('{"type":"frame","timestamp":1}')
            assert (await receive(ws))["type"] == "error"
            await asyncio.wait_for(ws.wait_closed(), 1)
        await wait_released(adapter)
        assert fake_upstream.instances[0].frames == []


@pytest.mark.asyncio
async def test_unexpected_mapping_failure_is_supervised(monkeypatch, fake_upstream):
    def broken(start):
        raise RuntimeError("unexpected mapper exception")

    monkeypatch.setattr(session, "map_start_to_configure", broken)
    async with adapter_endpoint(monkeypatch) as (adapter, url):
        async with connect(url, proxy=None) as ws:
            await ws.send('{"type":"start"}')
            assert (await receive(ws))["type"] == "error"
            await asyncio.wait_for(ws.wait_closed(), 1)
        await wait_released(adapter)


@pytest.mark.asyncio
async def test_overall_setup_timeout_spans_stages(monkeypatch, fake_upstream):
    monkeypatch.setattr(config, "SETUP_TIMEOUT_S", 0.05)

    async def slow(*args):
        await asyncio.sleep(0.035)

    monkeypatch.setattr(fake_upstream, "connect", slow)
    monkeypatch.setattr(fake_upstream, "configure", slow)
    async with adapter_endpoint(monkeypatch) as (adapter, url):
        async with connect(url, proxy=None) as ws:
            await ws.send('{"type":"start"}')
            reply = await receive(ws)
            assert reply["type"] == "error" and "timed out" in reply["message"]
            await asyncio.wait_for(ws.wait_closed(), 1)
        await wait_released(adapter)


@pytest.mark.asyncio
async def test_busy_disconnect_does_not_change_capacity(monkeypatch):
    monkeypatch.setattr(config, "MAX_INFLIGHT", 1)
    adapter = server.AdapterServer()
    adapter._inflight = 1

    class BrokenSocket(Socket):
        async def send(self, raw):
            raise OSError("client disconnected")

    socket = BrokenSocket()
    await adapter._handler(socket)
    assert socket.closed
    assert adapter._inflight == 1


@pytest.mark.asyncio
async def test_unlimited_admission_does_not_count_slots(monkeypatch):
    monkeypatch.setattr(config, "MAX_INFLIGHT", 0)

    class ImmediateBridge:
        def __init__(self, socket, on_client_done):
            self.release = on_client_done

        async def run(self):
            await self.release()

    monkeypatch.setattr(server, "BridgeSession", ImmediateBridge)
    adapter = server.AdapterServer()
    await asyncio.gather(*(adapter._handler(Socket()) for _ in range(8)))
    assert adapter._inflight == 0
