"""Cancellation and admission deadlines must work while input credit is stalled."""

import asyncio
import json
from contextlib import asynccontextmanager
from io import BytesIO

import anyio
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketState

import sglang_omni.serve.video_realtime as video_realtime
from sglang_omni.client.types import GenerateChunk
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.video_realtime import VideoRealtimeSessionManager


@pytest.mark.asyncio
async def test_cleanup_survives_repeated_task_cancellation():
    from sglang_omni.serve._cleanup import await_cleanup

    gate, entered, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def cleanup():
        entered.set()
        await gate.wait()
        finished.set()

    owned = asyncio.create_task(cleanup())
    waiter = asyncio.create_task(await_cleanup(owned))
    await entered.wait()
    waiter.cancel()
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    assert not owned.cancelled() and not finished.is_set()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert finished.is_set()


@pytest.mark.asyncio
async def test_abort_cancellation_still_releases_local_resources():
    class CancelledClient(PendingClient):
        async def abort(self, rid):
            raise asyncio.CancelledError

    manager = VideoRealtimeSessionManager(client=CancelledClient(), model_name="test")
    session = manager.open(Socket())
    session.configured = True
    session.outstanding_seq_nos.add(0)
    session.frame_refs_by_seq[0] = manager.frame_store.put(
        session.request_id, b"unconsumed"
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await manager.close(session.session_id)
        assert not manager.sessions and not manager.frame_store._names_by_request
        assert not session.frame_refs_by_seq and not session.outstanding_seq_nos
    finally:
        manager.frame_store.close()


class Socket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []

    async def receive(self):
        message = await self.incoming.get()
        if message["type"] == "websocket.disconnect":
            self.client_state = WebSocketState.DISCONNECTED
        return message

    async def send_json(self, value):
        self.sent.append(value)

    async def close(self):
        self.application_state = WebSocketState.DISCONNECTED
        self.incoming.put_nowait(dict(type="websocket.disconnect"))

    def submit(self, value):
        if isinstance(value, bytes):
            self.incoming.put_nowait(dict(type="websocket.receive", bytes=value))
        else:
            self.incoming.put_nowait(
                dict(type="websocket.receive", text=json.dumps(value))
            )


class PendingClient:
    def __init__(self):
        self.updates = []
        self.aborted = []
        self.chunks = asyncio.Queue()
        self.ready_gate = asyncio.Event()
        self.ready_gate.set()

    async def generate(self, request, request_id=None):
        await self.ready_gate.wait()
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event="session.ready",
            control_data={},
        )
        while True:
            yield await self.chunks.get()

    async def update_request(self, request_id, data):
        self.updates.append(data)

    async def abort(self, request_id):
        self.aborted.append(request_id)


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), 2)


@asynccontextmanager
async def opened(*, configure_timeout_s=180, client=None):
    client = client or PendingClient()
    socket = Socket()
    manager = VideoRealtimeSessionManager(
        client=client,
        model_name="test",
        max_sessions=1,
        configure_timeout_s=configure_timeout_s,
    )
    session = manager.open(socket)

    async def serve():
        try:
            await session.run()
        finally:
            await manager.close(session.session_id)

    task = asyncio.create_task(serve())
    try:
        await until(lambda: bool(socket.sent))
        yield session, socket, client, manager, task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
        manager.frame_store.close()


async def configure(session, socket):
    socket.submit(dict(type="session.configure", input_queue_capacity=1))
    await until(lambda: session.ready)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["v1", "v2"])
@pytest.mark.parametrize("abort_mode", ["slow", "error", "timeout"])
async def test_cancelled_scope_completes_cleanup(protocol, abort_mode):
    from functools import partial

    from vl_api_adapter.adapter.session import V2Session
    from vl_api_adapter.adapter.usage import empty_usage

    class SlowClient(PendingClient):
        async def abort(self, rid):
            if abort_mode == "timeout":
                await asyncio.Event().wait()
            await asyncio.sleep(0.01)
            if abort_mode == "error":
                raise RuntimeError("synthetic abort failure")
            self.aborted.append(rid)

        async def admin(self, action, payload, **kwargs):
            await asyncio.sleep(0.01)
            return {
                "success": True,
                "results": [
                    {
                        "data": {
                            **payload,
                            "final": True,
                            "watermark": 0,
                            "usage": empty_usage(),
                        }
                    }
                ],
            }

    class FlexibleSocket(Socket):
        async def close(self, code=1000):
            await super().close()

    client, socket = SlowClient(), FlexibleSocket()
    manager = VideoRealtimeSessionManager(
        client=client, model_name="test", max_sessions=1
    )
    factory = (
        partial(V2Session, context_limit=131072, accounting_stage="test")
        if protocol == "v2"
        else None
    )
    session = manager.open(socket, session_factory=factory)
    session.cleanup_timeout_s = 0.03
    if protocol == "v2":
        # Exercise the same timeout branch without spending five seconds.
        original_abort = client.abort

        async def bounded_abort(rid):
            return await asyncio.wait_for(original_abort(rid), 0.03)

        client.abort = bounded_abort
    scopes, exited = [], asyncio.Event()

    async def serve():
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            try:
                await session.run()
            finally:
                await manager.close(session.session_id)
        exited.set()

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(serve)
            await until(lambda: bool(socket.sent))
            await configure(session, socket)
            socket.submit(
                dict(type="input.frame", seq_no=0, timestamp=0, mime_type="image/png")
            )
            await until(
                lambda: any(e["type"] == "input.frame.ready" for e in socket.sent)
            )
            payload = BytesIO()
            Image.new("RGB", (2, 2)).save(payload, format="PNG")
            socket.submit(payload.getvalue())
            await until(lambda: len(client.updates) == 1)
            assert manager.frame_store._names_by_request
            scopes[0].cancel()
            await asyncio.wait_for(exited.wait(), 2)
        assert not manager.sessions
        assert not manager.frame_store._names_by_request
        assert not session.outstanding_seq_nos and not session.frame_refs_by_seq
        assert session.response_task.done()
        before = list(socket.sent)
        await session.teardown()
        assert socket.sent == before
        replacement = manager.open(FlexibleSocket())
        await manager.close(replacement.session_id)
    finally:
        manager.frame_store.close()


@pytest.mark.parametrize("action", ["abort", "disconnect"])
@pytest.mark.parametrize("pending_kind", ["frame", "prompt"])
def test_control_releases_stalled_input_and_shared_frame(action, pending_kind):
    async def run():
        async with opened() as (session, socket, client, manager, task):
            await configure(session, socket)
            socket.submit(
                dict(type="input.frame", seq_no=0, timestamp=0, mime_type="image/png")
            )
            await until(
                lambda: any(e["type"] == "input.frame.ready" for e in socket.sent)
            )
            image = BytesIO()
            Image.new("RGB", (2, 2)).save(image, format="PNG")
            socket.submit(image.getvalue())
            await until(lambda: len(client.updates) == 1)
            assert manager.frame_store._names_by_request
            entered = asyncio.Event()
            reserve = session._reserve_input

            async def record_wait(seq):
                if seq == 1:
                    entered.set()
                await reserve(seq)

            session._reserve_input = record_wait
            if pending_kind == "prompt":
                socket.submit(dict(type="input.prompt", seq_no=1, prompt="next"))
            else:
                socket.submit(
                    dict(
                        type="input.frame", seq_no=1, timestamp=1, mime_type="image/png"
                    )
                )
            await asyncio.wait_for(entered.wait(), 1)
            if action == "abort":
                socket.submit(dict(type="session.abort"))
            else:
                socket.incoming.put_nowait(dict(type="websocket.disconnect"))
            await asyncio.wait_for(task, 1)
            assert client.aborted == [session.request_id]
            assert len(client.updates) == 1
            assert not session.outstanding_seq_nos and not session.accepted_by_seq
            assert not manager.frame_store._names_by_request
            assert not manager.sessions
            assert session.response_task.done()
            replacement = manager.open(Socket())
            await manager.close(replacement.session_id)

    asyncio.run(run())


def test_full_input_queue_is_bounded_and_closes_on_overflow():
    async def run():
        async with opened() as (session, socket, client, manager, task):
            await configure(session, socket)
            socket.submit(dict(type="input.prompt", seq_no=0, prompt="first"))
            await until(lambda: len(client.updates) == 1)
            for seq in range(1, 20):
                socket.submit(dict(type="input.prompt", seq_no=seq, prompt="queued"))
            await asyncio.wait_for(task, 1)
            assert any(e.get("code") == "input_queue_full" for e in socket.sent)
            assert len(client.updates) == 1 and client.aborted == [session.request_id]
            assert not manager.sessions and not session.outstanding_seq_nos

    asyncio.run(run())


def test_abort_seals_queued_configuration_before_dispatch():
    async def run():
        async with opened() as (session, socket, client, manager, task):
            entered = asyncio.Event()
            hold = asyncio.Event()
            handle = session.handle_json

            async def delayed_handle(payload):
                entered.set()
                await hold.wait()
                await handle(payload)

            session.handle_json = delayed_handle
            socket.submit(dict(type="session.configure"))
            await asyncio.wait_for(entered.wait(), 1)
            socket.submit(dict(type="session.abort"))
            await asyncio.wait_for(task, 1)
            assert not session.configured
            assert session.response_task is None
            assert not client.updates and not client.aborted and not manager.sessions

    asyncio.run(run())


def test_input_order_is_preserved_when_credit_returns():
    async def run():
        async with opened() as (session, socket, client, manager, task):
            await configure(session, socket)
            socket.submit(dict(type="input.prompt", seq_no=0, prompt="first"))
            await until(lambda: len(client.updates) == 1)
            socket.submit(dict(type="input.prompt", seq_no=1, prompt="second"))
            socket.submit(dict(type="input.prompt", seq_no=2, prompt="third"))
            for seq in range(3):
                await until(lambda: len(client.updates) == seq + 1)
                client.chunks.put_nowait(
                    GenerateChunk(
                        request_id=session.request_id,
                        modality="control",
                        control_event="input.prompt.processed",
                        control_data=dict(seq_no=seq, timestamp=0, final=False),
                    )
                )
                await until(
                    lambda: any(
                        e["type"] == "input.prompt.processed" and e["seq_no"] == seq
                        for e in socket.sent
                    )
                )
            assert [e["seq_no"] for e in client.updates] == [0, 1, 2]
            for seq in range(3):
                types = [e["type"] for e in socket.sent if e.get("seq_no") == seq]
                assert types.index("input.prompt.accepted") < types.index(
                    "input.prompt.processed"
                )
            socket.submit(dict(type="session.abort"))
            await asyncio.wait_for(task, 1)

    asyncio.run(run())


def test_oversized_queued_binary_ends_session(monkeypatch):
    monkeypatch.setattr(video_realtime, "MAX_FRAME_BYTES", 8)

    async def run():
        async with opened() as (session, socket, client, manager, task):
            await configure(session, socket)
            socket.submit(b"x" * 9)
            await asyncio.wait_for(task, 1)
            assert any(e.get("code") == "frame_too_large" for e in socket.sent)
            assert client.aborted == [session.request_id]
            assert not client.updates and not manager.sessions

    asyncio.run(run())


def test_configure_timeout_frees_slot_via_public_endpoint():
    client = PendingClient()
    app = create_app(
        client,
        model_name="test",
        enable_video_realtime=True,
        video_realtime_configure_timeout_s=0.05,
    )
    with TestClient(app) as http:
        with http.websocket_connect("/v1/video/realtime") as ws:
            assert ws.receive_json()["configure_timeout_s"] == 0.05
            assert ws.receive_json()["code"] == "configuration_timeout"
            assert ws.receive_json()["type"] == "session.done"
            assert ws.receive()["type"] == "websocket.close"
        with http.websocket_connect("/v1/video/realtime") as ws:
            assert ws.receive_json()["type"] == "session.created"
    assert not client.updates and not client.aborted


def test_invalid_traffic_does_not_extend_configuration_deadline():
    async def run():
        async with opened(configure_timeout_s=0.05) as (
            session,
            socket,
            client,
            manager,
            task,
        ):
            for _ in range(12):
                if task.done():
                    break
                socket.submit(dict(type="session.configure", input_queue_capacity=0))
                await asyncio.sleep(0.01)
            await asyncio.wait_for(task, 1)
            assert any(e.get("code") == "configuration_timeout" for e in socket.sent)
            assert not session.configured and not manager.sessions

    asyncio.run(run())


def test_configuration_deadline_does_not_limit_model_prefill():
    async def run():
        client = PendingClient()
        client.ready_gate.clear()
        async with opened(configure_timeout_s=0.05, client=client) as (
            session,
            socket,
            _,
            manager,
            task,
        ):
            socket.submit(dict(type="session.configure"))
            await until(lambda: session.configured)
            await asyncio.sleep(0.1)
            assert not task.done() and not session.ready
            assert not any(
                e.get("code") == "configuration_timeout" for e in socket.sent
            )
            socket.submit(dict(type="session.abort"))
            await asyncio.wait_for(task, 1)
            assert client.aborted == [session.request_id] and not manager.sessions

    asyncio.run(run())


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_configuration_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="configure_timeout_s"):
        create_app(
            PendingClient(),
            model_name="test",
            enable_video_realtime=True,
            video_realtime_configure_timeout_s=timeout,
        )
