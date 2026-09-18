"""Protocol-only latency collector regressions; no model server needed."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

spec = importlib.util.spec_from_file_location(
    "legacy_perf_probe", Path(__file__).parents[3] / "perf_probe.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Peer:
    def __init__(self, output):
        self.output = output
        self.queue = asyncio.Queue()
        self.closed = False
        self.stopped = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def recv(self):
        return json.dumps(await self.queue.get())

    async def send(self, raw):
        if isinstance(raw, bytes):
            for event in self.output:
                self.queue.put_nowait(event)
        else:
            kind = json.loads(raw)["type"]
            if kind == "start":
                self.queue.put_nowait({"type": "ready"})
            elif kind == "stop":
                self.stopped = True


@pytest.mark.asyncio
async def test_interleaved_text_before_ack_is_preserved(monkeypatch):
    peer = Peer(
        [
            {"type": "output", "text": "hello"},
            {"type": "frame_ack"},
            {"type": "output", "text": "<|im_end|>"},
        ]
    )
    monkeypatch.setattr(probe, "connect", lambda *a, **k: peer)
    result = await probe.one_round("ws://test", [b"jpeg"], "describe", {})
    assert result["text"] == "hello" and result["ttft"] is not None
    assert result["ack_count"] == 1 and peer.closed and peer.stopped


@pytest.mark.asyncio
async def test_split_marker_only_is_not_a_valid_answer(monkeypatch):
    peer = Peer(
        [
            {"type": "frame_ack"},
            {"type": "output", "text": "<|si"},
            {"type": "output", "text": "lence|>"},
        ]
    )
    monkeypatch.setattr(probe, "connect", lambda *a, **k: peer)
    with pytest.raises(RuntimeError, match="without visible text"):
        await probe.one_round("ws://test", [b"jpeg"], "describe", {})
    assert peer.closed and peer.stopped


@pytest.mark.asyncio
async def test_silent_peer_has_an_actual_deadline(monkeypatch):
    peer = Peer([])
    monkeypatch.setattr(probe, "connect", lambda *a, **k: peer)
    with pytest.raises(TimeoutError):
        await probe.one_round("ws://test", [b"jpeg"], "describe", {}, timeout_s=0.03)
    assert peer.closed and peer.stopped


@pytest.mark.parametrize("modern", [True, False])
@pytest.mark.parametrize(
    "relative", ["perf_probe.py", "vl_legacy_adapter/tests/smoke_client.py"]
)
def test_websocket_api_compatibility_keeps_direct_connections(
    monkeypatch, modern, relative
):
    module = ModuleType("websockets.asyncio.client")

    def old_connect(uri, *, max_size=None, **kwargs):
        assert "proxy" not in kwargs

    def new_connect(uri, *, proxy=True, **kwargs):
        assert proxy is None

    module.connect = new_connect if modern else old_connect
    monkeypatch.setitem(sys.modules, "websockets.asyncio.client", module)
    path = Path(__file__).parents[3] / relative
    spec = importlib.util.spec_from_file_location("compat_probe", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    assert loaded.CONNECT_OPTIONS == ({"proxy": None} if modern else {})
    loaded.connect("ws://test", **loaded.CONNECT_OPTIONS)


@pytest.mark.asyncio
async def test_environment_proxy_cannot_redirect_perf_probe(monkeypatch):
    from websockets.asyncio.server import serve

    monkeypatch.setenv("ws_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")

    async def handler(ws):
        await ws.recv()
        await ws.send('{"type":"ready"}')
        await ws.recv()
        await ws.recv()
        await ws.send('{"type":"frame_ack"}')
        await ws.send('{"type":"output","text":"hello<|im_end|>"}')
        await ws.recv()

    async with serve(handler, "127.0.0.1", 0) as server:
        result = await probe.one_round(
            f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}", [b"jpeg"], "", {}
        )
    assert result["text"] == "hello"
