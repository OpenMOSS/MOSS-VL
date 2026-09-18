from __future__ import annotations

import asyncio
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

import sglang_omni.serve.video_realtime as video_realtime_module
from sglang_omni.client.types import GenerateChunk
from sglang_omni.models.moss_vl_realtime.frame_store import (
    SharedMemoryFrameStore,
    resolve_shared_memory_frame,
)
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.video_realtime import (
    VideoFrameMetadata,
    VideoPromptInput,
    VideoRealtimeSession,
    VideoSessionConfigure,
    warmup_video_realtime,
)


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 3), color=(40, 50, 60)).save(output, format="PNG")
    return output.getvalue()


class _Client:
    def __init__(self) -> None:
        self.update_event = asyncio.Event()
        self.updates: list[tuple[str, dict]] = []
        self.aborted: list[str] = []

    async def generate(self, request, request_id=None):
        del request
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event="session.ready",
            control_data={},
        )
        await self.update_event.wait()
        update = self.updates[0][1]
        control_event = (
            "input.frame.processed"
            if update.get("frame_ref") is not None
            else "input.prompt.processed"
        )
        turn_id = 0
        turn_data = {}
        if update.get("prompt") is not None:
            turn_id = 1
            turn_data = {"interrupted_turn_id": 0, "turn_id": 1}
            yield GenerateChunk(
                request_id=request_id,
                modality="control",
                control_event="response.turn.interrupted",
                control_data={
                    "turn_id": 0,
                    "next_turn_id": 1,
                    "seq_no": update["seq_no"],
                },
            )
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event=control_event,
            control_data={
                "seq_no": update["seq_no"],
                "timestamp": update["timestamp"],
                "final": update["final"],
                **turn_data,
            },
        )
        yield GenerateChunk(request_id=request_id, text="car", turn_id=turn_id)
        yield GenerateChunk(
            request_id=request_id,
            text="car",
            finish_reason="stop",
            turn_id=turn_id,
        )

    async def update_request(self, request_id, data, *, stage_name=None):
        del stage_name
        self.updates.append((request_id, data))
        self.update_event.set()

    async def abort(self, request_id):
        self.aborted.append(request_id)
        return True


class _HoldingClient(_Client):
    async def generate(self, request, request_id=None):
        del request
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event="session.ready",
            control_data={},
        )
        await asyncio.Event().wait()
        yield GenerateChunk(request_id=request_id)

    async def update_request(self, request_id, data, *, stage_name=None):
        del stage_name
        self.updates.append((request_id, data))


class _WarmupClient(_Client):
    def __init__(self) -> None:
        super().__init__()
        self.request = None
        self.frame_sizes: list[tuple[int, int]] = []

    async def generate(self, request, request_id=None):
        self.request = request
        async for chunk in super().generate(request, request_id=request_id):
            yield chunk

    async def update_request(self, request_id, data, *, stage_name=None):
        frame = resolve_shared_memory_frame(data["frame_ref"])
        self.frame_sizes.append(frame.size)
        await super().update_request(request_id, data, stage_name=stage_name)


class _SilenceClient(_Client):
    async def generate(self, request, request_id=None):
        del request
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event="session.ready",
            control_data={"turn_id": 0},
        )
        await self.update_event.wait()
        update = self.updates[0][1]
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event="input.frame.processed",
            control_data={
                "seq_no": update["seq_no"],
                "timestamp": update["timestamp"],
                "final": update["final"],
            },
        )
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event="response.turn.silence",
            control_data={
                "turn_id": 0,
                "seq_no": update["seq_no"],
                "timestamp": update["timestamp"],
                "silence_seq": 0,
            },
        )
        yield GenerateChunk(request_id=request_id, finish_reason="stop", turn_id=0)


class _RejectingUpdateClient(_HoldingClient):
    async def update_request(self, request_id, data, *, stage_name=None):
        del request_id, data, stage_name
        raise RuntimeError("coordinator rejected active request update")


class _CapacityFailureClient(_Client):
    async def generate(self, request, request_id=None):
        del request
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event="session.ready",
            control_data={},
        )
        await self.update_event.wait()
        raise RuntimeError("KV layout does not fit in req_to_token row")


class _BrokenWebSocket:
    async def send_json(self, payload):
        del payload
        raise RuntimeError("WebSocket is already closed")


def test_video_realtime_defaults_to_four_outstanding_inputs() -> None:
    config = VideoSessionConfigure(type="session.configure")
    session = VideoRealtimeSession(
        _BrokenWebSocket(),  # type: ignore[arg-type]
        client=None,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        frame_store=SharedMemoryFrameStore(),
    )

    assert config.input_queue_capacity == 4
    assert config.max_tokens_per_turn == 86400.0
    assert session.input_queue_capacity == 4


def test_video_realtime_normalizes_or_rejects_blank_prompts() -> None:
    frame = VideoFrameMetadata(
        type="input.frame",
        seq_no=0,
        timestamp=0.0,
        mime_type="image/png",
        prompt="   ",
    )
    assert frame.prompt is None

    with pytest.raises(ValidationError, match="non-whitespace"):
        VideoPromptInput(
            type="input.prompt",
            seq_no=0,
            prompt="   ",
        )

    for value in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValidationError, match="max_tokens_per_turn"):
            VideoSessionConfigure(
                type="session.configure",
                max_tokens_per_turn=value,
            )


def test_frame_event_construction_failure_releases_reserved_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenFramePromptEvent:
        def __init__(self, **kwargs):
            del kwargs
            raise ValueError("synthetic event construction failure")

    client = _HoldingClient()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )
    manager = app.state.video_realtime_manager

    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        created = websocket.receive_json()
        websocket.send_json({"type": "session.configure"})
        assert websocket.receive_json()["type"] == "session.configured"
        assert websocket.receive_json()["type"] == "session.ready"
        monkeypatch.setattr(
            video_realtime_module,
            "FramePromptEvent",
            _BrokenFramePromptEvent,
        )
        websocket.send_json(
            {
                "type": "input.frame",
                "seq_no": 0,
                "timestamp": 0.0,
                "prompt": "What changed?",
                "mime_type": "image/png",
            }
        )
        assert websocket.receive_json()["type"] == "input.frame.ready"
        websocket.send_bytes(_png_bytes())
        error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["code"] == "invalid_request"
        assert "synthetic event construction failure" in error["message"]
        session = manager.sessions[created["session_id"]]
        assert session.outstanding_seq_nos == set()
        assert session.accepted_by_seq == {}
        assert session.frame_refs_by_seq == {}
        assert dict(manager.frame_store._names_by_request) == {}


def test_video_realtime_warmup_processes_one_final_frame() -> None:
    client = _WarmupClient()

    asyncio.run(
        warmup_video_realtime(
            client,  # type: ignore[arg-type]
            model_name="moss-vl-realtime",
            timeout_s=1.0,
        )
    )

    assert client.request is not None
    assert client.request.model == "moss-vl-realtime"
    assert client.request.extra_params == {"realtime_warmup": True}
    assert client.frame_sizes == [(640, 352)]
    assert len(client.updates) == 1
    request_id, event = client.updates[0]
    assert event["seq_no"] == 0
    assert event["timestamp"] == 0.0
    assert event["final"] is True
    assert client.aborted == [request_id]


def test_video_realtime_binary_frame_reaches_request_update() -> None:
    client = _Client()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        created = websocket.receive_json()
        websocket.send_json(
            {
                "type": "session.configure",
                "prompt": "Track the car.",
                "max_new_tokens": 8,
            }
        )
        assert websocket.receive_json()["type"] == "session.configured"
        assert websocket.receive_json()["type"] == "session.ready"
        websocket.send_json(
            {
                "type": "input.frame",
                "seq_no": 0,
                "timestamp": 1.5,
                "final": True,
                "mime_type": "image/png",
            }
        )
        assert websocket.receive_json() == {"type": "input.frame.ready", "seq_no": 0}
        websocket.send_bytes(_png_bytes())

        events = []
        event_types = set()
        while "response.done" not in event_types:
            event = websocket.receive_json()
            events.append(event)
            event_types.add(event["type"])

    assert {
        "input.frame.accepted",
        "input.frame.processed",
        "response.text.delta",
        "response.done",
    } <= event_types
    assert [
        event["delta"] for event in events if event["type"] == "response.text.delta"
    ] == ["car"]
    assert len(client.updates) == 1
    request_id, update = client.updates[0]
    assert request_id == created["request_id"]
    assert update["session_id"] == created["session_id"]
    assert update["timestamp"] == 1.5
    assert update["frame_ref"].startswith("shm://")


def test_video_realtime_forwards_silence_control_event() -> None:
    client = _SilenceClient()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "session.configure"})
        assert websocket.receive_json()["type"] == "session.configured"
        assert websocket.receive_json()["type"] == "session.ready"
        websocket.send_json(
            {
                "type": "input.frame",
                "seq_no": 0,
                "timestamp": 2.0,
                "final": True,
                "mime_type": "image/png",
            }
        )
        assert websocket.receive_json()["type"] == "input.frame.ready"
        websocket.send_bytes(_png_bytes())
        events = []
        while not any(event["type"] == "response.done" for event in events):
            events.append(websocket.receive_json())

    event_types = [event["type"] for event in events]
    assert (
        event_types.index("input.frame.accepted")
        < event_types.index("input.frame.processed")
        < event_types.index("response.turn.silence")
    )
    silence = next(
        event for event in events if event["type"] == "response.turn.silence"
    )
    assert silence == {
        "type": "response.turn.silence",
        "turn_id": 0,
        "seq_no": 0,
        "timestamp": 2.0,
        "silence_seq": 0,
    }


def test_video_realtime_allows_only_one_websocket() -> None:
    client = _Client()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )
    with (
        TestClient(app) as test_client,
        test_client.websocket_connect("/v1/video/realtime") as first,
    ):
        assert first.receive_json()["type"] == "session.created"
        with test_client.websocket_connect("/v1/video/realtime") as second:
            error = second.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "session_capacity_exceeded"


def test_video_realtime_admits_up_to_configured_session_capacity() -> None:
    client = _Client()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
        video_realtime_max_sessions=2,
    )
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/v1/video/realtime") as first:
            first_created = first.receive_json()
            assert first_created["type"] == "session.created"
            with test_client.websocket_connect("/v1/video/realtime") as second:
                second_created = second.receive_json()
                assert second_created["type"] == "session.created"
                assert second_created["session_id"] != first_created["session_id"]
                with test_client.websocket_connect("/v1/video/realtime") as third:
                    error = third.receive_json()
                    assert error["type"] == "error"
                    assert error["code"] == "session_capacity_exceeded"


def test_video_realtime_frees_session_slot_on_disconnect() -> None:
    client = _Client()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/v1/video/realtime") as first:
            assert first.receive_json()["type"] == "session.created"
        with test_client.websocket_connect("/v1/video/realtime") as second:
            assert second.receive_json()["type"] == "session.created"


def test_video_realtime_prompt_only_event_reaches_request_update() -> None:
    client = _Client()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        created = websocket.receive_json()
        websocket.send_json({"type": "session.configure", "prompt": "Watch."})
        assert websocket.receive_json()["type"] == "session.configured"
        assert websocket.receive_json()["type"] == "session.ready"
        websocket.send_json(
            {
                "type": "input.prompt",
                "seq_no": 0,
                "prompt": "How many?",
                "final": True,
            }
        )
        events = []
        event_types = set()
        while "response.done" not in event_types:
            event = websocket.receive_json()
            events.append(event)
            event_types.add(event["type"])

    assert {
        "input.prompt.accepted",
        "response.turn.interrupted",
        "input.prompt.processed",
    } <= event_types
    by_type = {event["type"]: event for event in events}
    assert [event["type"] for event in events].index("input.prompt.accepted") < [
        event["type"] for event in events
    ].index("response.turn.interrupted")
    assert by_type["response.turn.interrupted"] == {
        "type": "response.turn.interrupted",
        "turn_id": 0,
        "next_turn_id": 1,
        "seq_no": 0,
    }
    assert by_type["input.prompt.processed"]["interrupted_turn_id"] == 0
    assert by_type["input.prompt.processed"]["turn_id"] == 1
    assert by_type["response.text.delta"]["turn_id"] == 1
    assert by_type["response.done"]["turn_id"] == 1
    request_id, update = client.updates[0]
    assert request_id == created["request_id"]
    assert update["session_id"] == created["session_id"]
    assert update["prompt"] == "How many?"
    assert update["final"] is True
    assert "frame_ref" not in update


def test_video_realtime_accepts_multiple_inflight_frames() -> None:
    client = _HoldingClient()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json({"type": "session.configure"})
        assert websocket.receive_json()["type"] == "session.configured"
        assert websocket.receive_json()["type"] == "session.ready"
        websocket.send_json(
            {
                "type": "input.frame",
                "seq_no": 0,
                "timestamp": 0.0,
                "mime_type": "image/png",
            }
        )
        assert websocket.receive_json()["type"] == "input.frame.ready"
        websocket.send_bytes(_png_bytes())
        assert websocket.receive_json()["type"] == "input.frame.accepted"
        websocket.send_json(
            {
                "type": "input.frame",
                "seq_no": 1,
                "timestamp": 1.0,
                "mime_type": "image/png",
            }
        )
        assert websocket.receive_json()["type"] == "input.frame.ready"
        websocket.send_bytes(_png_bytes())
        accepted = websocket.receive_json()
        assert accepted["type"] == "input.frame.accepted"
        assert accepted["pending_events"] == 2

    assert [update[1]["seq_no"] for update in client.updates] == [0, 1]


def test_video_realtime_applies_bounded_input_backpressure() -> None:
    async def _run() -> None:
        session = VideoRealtimeSession(
            None,  # type: ignore[arg-type]
            client=None,  # type: ignore[arg-type]
            model_name="moss-vl-realtime",
            frame_store=SharedMemoryFrameStore(),
        )
        session.input_queue_capacity = 1
        await session._reserve_input(0)

        waiting = asyncio.create_task(session._reserve_input(1))
        await asyncio.sleep(0)
        assert waiting.done() is False

        await session._release_input(0)
        await asyncio.wait_for(waiting, timeout=1.0)
        assert session.outstanding_seq_nos == {1}
        await session._close_input_queue()

    asyncio.run(_run())


def test_video_realtime_rejects_input_before_session_ready() -> None:
    async def _run() -> None:
        session = VideoRealtimeSession(
            None,  # type: ignore[arg-type]
            client=None,  # type: ignore[arg-type]
            model_name="moss-vl-realtime",
            frame_store=SharedMemoryFrameStore(),
        )
        session.configured = True

        with pytest.raises(ValueError, match="session.ready"):
            await session.handle_prompt(
                VideoPromptInput(
                    type="input.prompt",
                    seq_no=0,
                    prompt="How many?",
                )
            )

    asyncio.run(_run())


def test_video_realtime_reports_input_submission_failure() -> None:
    client = _RejectingUpdateClient()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )

    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        created = websocket.receive_json()
        websocket.send_json({"type": "session.configure"})
        assert websocket.receive_json()["type"] == "session.configured"
        assert websocket.receive_json()["type"] == "session.ready"
        websocket.send_json(
            {
                "type": "input.prompt",
                "seq_no": 0,
                "prompt": "How many?",
            }
        )
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "input_submission_failed"
    assert "coordinator rejected" in error["message"]
    assert created["request_id"] in client.aborted
    assert app.state.video_realtime_manager.sessions == {}


def test_video_realtime_capacity_failure_cleans_resources_and_allows_reconnect() -> (
    None
):
    client = _CapacityFailureClient()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )
    manager = app.state.video_realtime_manager

    with TestClient(app) as test_client:
        with test_client.websocket_connect("/v1/video/realtime") as websocket:
            created = websocket.receive_json()
            websocket.send_json({"type": "session.configure"})
            assert websocket.receive_json()["type"] == "session.configured"
            assert websocket.receive_json()["type"] == "session.ready"
            websocket.send_json(
                {
                    "type": "input.frame",
                    "seq_no": 0,
                    "timestamp": 0.0,
                    "mime_type": "image/png",
                }
            )
            assert websocket.receive_json()["type"] == "input.frame.ready"
            websocket.send_bytes(_png_bytes())
            events = []
            while True:
                event = websocket.receive_json()
                events.append(event)
                if event["type"] == "error":
                    break

        assert events[-1]["code"] == "response_failed"
        assert "KV layout does not fit" in events[-1]["message"]
        assert created["request_id"] in client.aborted
        assert manager.sessions == {}
        assert dict(manager.frame_store._names_by_request) == {}

        with test_client.websocket_connect("/v1/video/realtime") as replacement:
            assert replacement.receive_json()["type"] == "session.created"


def test_video_realtime_error_delivery_is_safe_after_disconnect() -> None:
    async def _run() -> None:
        session = VideoRealtimeSession(
            _BrokenWebSocket(),  # type: ignore[arg-type]
            client=None,  # type: ignore[arg-type]
            model_name="moss-vl-realtime",
            frame_store=SharedMemoryFrameStore(),
        )

        delivered = await session.send_error_safely(
            "backend failed",
            code="response_failed",
        )

        assert delivered is False

    asyncio.run(_run())


class _RequestCaptureClient(_Client):
    captured_request = None

    async def generate(self, request, request_id=None):
        self.captured_request = request
        async for chunk in super().generate(request, request_id=request_id):
            yield chunk


class _ImmediateRequestCaptureClient(_Client):
    captured_request = None

    async def generate(self, request, request_id=None):
        self.captured_request = request
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event="session.ready",
            control_data={},
        )
        yield GenerateChunk(
            request_id=request_id,
            text="",
            finish_reason="stop",
            turn_id=0,
        )


def test_video_realtime_benchmark_ignore_eos_flows_to_extra_params() -> None:
    client = _RequestCaptureClient()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
        video_realtime_benchmark_mode=True,
    )
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "session.configure",
                "prompt": "Watch.",
                "benchmark_ignore_eos": True,
            }
        )
        assert websocket.receive_json()["type"] == "session.configured"
        assert websocket.receive_json()["type"] == "session.ready"
        websocket.send_json(
            {
                "type": "input.prompt",
                "seq_no": 0,
                "prompt": "How many?",
                "final": True,
            }
        )
        event_types = set()
        while "response.done" not in event_types:
            event_types.add(websocket.receive_json()["type"])

    assert client.captured_request is not None
    assert client.captured_request.extra_params == {
        "benchmark_ignore_eos": True,
        "max_tokens_per_turn": 86400.0,
    }


def test_video_realtime_token_rate_flows_to_request() -> None:
    class _CaptureWebSocket:
        def __init__(self) -> None:
            self.messages = []
            self.application_state = video_realtime_module.WebSocketState.DISCONNECTED
            self.client_state = video_realtime_module.WebSocketState.DISCONNECTED

        async def send_json(self, payload):
            self.messages.append(payload)

    async def _run():
        websocket = _CaptureWebSocket()
        client = _ImmediateRequestCaptureClient()
        session = VideoRealtimeSession(
            websocket,  # type: ignore[arg-type]
            client=client,  # type: ignore[arg-type]
            model_name="moss-vl-realtime",
            frame_store=SharedMemoryFrameStore(),
        )
        await session.configure(
            VideoSessionConfigure(
                type="session.configure",
                prompt="Watch.",
                max_tokens_per_turn=12.5,
            )
        )
        assert session.response_task is not None
        await session.response_task
        return websocket.messages, client

    messages, client = asyncio.run(_run())
    assert messages[0]["type"] == "session.configured"
    assert messages[0]["max_tokens_per_turn"] == 12.5
    assert [message["type"] for message in messages[1:]] == [
        "session.ready",
        "response.done",
        "session.done",
    ]
    assert client.captured_request is not None
    assert client.captured_request.extra_params["max_tokens_per_turn"] == 12.5


def test_video_realtime_rejects_benchmark_option_in_production_mode() -> None:
    client = _HoldingClient()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "session.configure",
                "prompt": "Watch.",
                "benchmark_ignore_eos": True,
            }
        )
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert "requires server benchmark mode" in error["message"]
    assert client.updates == []


def test_video_realtime_rejects_out_of_order_seq_without_killing_session() -> None:
    client = _Client()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )

    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        websocket.receive_json()  # session.created
        websocket.send_json({"type": "session.configure"})
        assert websocket.receive_json()["type"] == "session.configured"
        assert websocket.receive_json()["type"] == "session.ready"

        websocket.send_json({"type": "input.prompt", "seq_no": 5, "prompt": "skip?"})
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "invalid_request"
        assert "seq_no" in error["message"]

        # The session survives: a well-ordered event is still accepted.
        websocket.send_json(
            {"type": "input.prompt", "seq_no": 0, "prompt": "How many?"}
        )
        accepted = websocket.receive_json()
        assert accepted["type"] == "input.prompt.accepted"
        assert accepted["seq_no"] == 0


@pytest.mark.parametrize("submit_frame", [False, True])
def test_video_realtime_abort_stream_error_leaves_terminal_to_teardown(
    monkeypatch, submit_frame
) -> None:
    class AbortErrorClient(_HoldingClient):
        def __init__(self):
            super().__init__()
            self.abort_event = asyncio.Event()
            self.response_drained = asyncio.Event()

        async def generate(self, request, request_id=None):
            yield GenerateChunk(
                request_id=request_id,
                modality="control",
                control_event="session.ready",
                control_data={},
            )
            await self.abort_event.wait()
            raise RuntimeError("aborted")

        async def abort(self, request_id):
            await super().abort(request_id)
            self.abort_event.set()
            # Force the response error handler to finish before teardown runs.
            await asyncio.wait_for(self.response_drained.wait(), timeout=5)
            return True

    client = AbortErrorClient()
    original = VideoRealtimeSession.stream_response

    async def observed_response(self, request):
        try:
            await original(self, request)
        finally:
            client.response_drained.set()

    monkeypatch.setattr(VideoRealtimeSession, "stream_response", observed_response)
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )
    manager = app.state.video_realtime_manager
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        created = websocket.receive_json()
        websocket.send_json({"type": "session.configure"})
        assert websocket.receive_json()["type"] == "session.configured"
        assert websocket.receive_json()["type"] == "session.ready"
        if submit_frame:
            websocket.send_json(
                {
                    "type": "input.frame",
                    "seq_no": 0,
                    "timestamp": 0.0,
                    "mime_type": "image/png",
                }
            )
            assert websocket.receive_json()["type"] == "input.frame.ready"
            websocket.send_bytes(_png_bytes())
            assert websocket.receive_json()["type"] == "input.frame.accepted"
        websocket.send_json({"type": "session.abort"})
        assert websocket.receive_json() == {
            "type": "session.done",
            "session_id": created["session_id"],
            "aborted": True,
        }
        closed = websocket.receive()
        assert closed["type"] == "websocket.close"
        assert closed["code"] == 1000

    assert client.response_drained.is_set()
    assert client.aborted == [created["request_id"]]
    assert manager.sessions == {}
    assert dict(manager.frame_store._names_by_request) == {}


@pytest.mark.parametrize("message", ["aborted", "engine failed"])
def test_video_realtime_response_error_without_abort_is_reported(message) -> None:
    async def run():
        async def generate(*args, **kwargs):
            raise RuntimeError(message)
            yield  # pragma: no cover

        session = VideoRealtimeSession(
            _BrokenWebSocket(),  # type: ignore[arg-type]
            client=SimpleNamespace(generate=generate),
            model_name="moss-vl-realtime",
            frame_store=SharedMemoryFrameStore(),
        )
        events = []

        async def error(text, *, code):
            events.append((code, text))

        async def close():
            events.append("close")

        session.send_error_safely = error
        session.close_websocket = close
        await session.stream_response(None)
        assert events == [("response_failed", message), "close"]
        assert session.closed

    asyncio.run(run())


@pytest.mark.parametrize("abort_sent", [False, True])
def test_video_realtime_response_task_cancellation_propagates(abort_sent) -> None:
    async def run():
        async def generate(*args, **kwargs):
            raise asyncio.CancelledError()
            yield  # pragma: no cover

        session = VideoRealtimeSession(
            _BrokenWebSocket(),  # type: ignore[arg-type]
            client=SimpleNamespace(generate=generate),
            model_name="moss-vl-realtime",
            frame_store=SharedMemoryFrameStore(),
        )
        session.abort_sent = abort_sent
        with pytest.raises(asyncio.CancelledError):
            await session.stream_response(None)
        assert not session.closed

    asyncio.run(run())


def test_video_realtime_client_abort_emits_terminal_session_done() -> None:
    client = _Client()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        enable_video_realtime=True,
    )

    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        created = websocket.receive_json()
        websocket.send_json({"type": "session.configure"})
        assert websocket.receive_json()["type"] == "session.configured"
        assert websocket.receive_json()["type"] == "session.ready"

        websocket.send_json({"type": "session.abort"})
        done = websocket.receive_json()
        assert done["type"] == "session.done"
        assert done["aborted"] is True

    assert created["request_id"] in client.aborted
