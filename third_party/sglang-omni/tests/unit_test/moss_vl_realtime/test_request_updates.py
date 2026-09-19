from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from sglang_omni.models.moss_vl_realtime import (
    FramePromptEvent,
    MossVLRealtimeSessionController,
)
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.tp_control import TPWorkMessage
from sglang_omni.proto import RequestUpdateMessage
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeScheduler,
    RecordingCoordinatorControlPlane,
)
from tests.unit_test.pipeline.helpers import make_stage


def _event(
    seq_no: int,
    timestamp: float,
    *,
    request_id: str = "req-1",
    session_id: str = "session-1",
    final: bool = False,
) -> FramePromptEvent:
    return FramePromptEvent(
        request_id=request_id,
        session_id=session_id,
        seq_no=seq_no,
        timestamp=timestamp,
        frame_ref=f"relay://frame-{seq_no}",
        prompt="What changed?" if seq_no == 0 else None,
        final=final,
    )


def test_frame_update_round_trips_over_control_plane() -> None:
    event = _event(0, 1.25)
    message = RequestUpdateMessage(
        request_id=event.request_id,
        data=event.to_dict(),
    )

    restored = deserialize_message(serialize_message(message))

    assert restored == message
    assert FramePromptEvent.from_dict(restored.data) == event


def test_prompt_only_update_round_trips_without_frame_ref() -> None:
    event = FramePromptEvent(
        request_id="req-1",
        session_id="session-1",
        seq_no=1,
        timestamp=2.0,
        frame_ref=None,
        prompt="How many?",
        final=True,
    )

    restored = FramePromptEvent.from_dict(event.to_dict())

    assert restored == event
    assert restored.has_frame is False
    assert "frame_ref" not in event.to_dict()


def test_coordinator_routes_update_to_active_state_owner() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="decode",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("decode", "inproc://decode")

        await coordinator._submit_request("req-1", {"text": "initial context"})
        await coordinator.update_request("req-1", _event(0, 0.0).to_dict())

        target, endpoint, message = control_plane.submitted[-1]
        assert (target, endpoint) == ("decode", "inproc://decode")
        assert isinstance(message, RequestUpdateMessage)
        assert message.request_id == "req-1"

        assert await coordinator.abort("req-1") is True
        with pytest.raises(KeyError, match="does not exist"):
            await coordinator.update_request("req-1", _event(1, 1.0).to_dict())

    asyncio.run(_run())


def test_stage_delivers_request_update_to_scheduler_inbox() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()
        stage = make_stage(name="decode", scheduler=scheduler)
        stage._active_requests.add("req-1")
        event = _event(0, 0.0)

        await stage._handle_message(RequestUpdateMessage("req-1", event.to_dict()))

        incoming = scheduler.inbox.get_nowait()
        assert incoming.request_id == "req-1"
        assert incoming.type == "request_update"
        assert FramePromptEvent.from_dict(incoming.data) == event

        stage._on_abort("req-1")
        await stage._handle_message(
            RequestUpdateMessage("req-1", _event(1, 1.0).to_dict())
        )
        assert scheduler.inbox.empty()

        stage._aborted.clear()
        await stage._handle_message(
            RequestUpdateMessage("late-request", _event(0, 0.0).to_dict())
        )
        assert scheduler.inbox.empty()

    asyncio.run(_run())


def test_tp_work_marks_request_active_before_updates() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler()
        stage = make_stage(name="decode", role="follower", scheduler=scheduler)
        payload = type("Payload", (), {"request_id": "req-1"})()

        await stage._on_tp_work(TPWorkMessage("req-1", payload))
        await stage._handle_message(
            RequestUpdateMessage("req-1", _event(0, 0.0).to_dict())
        )

        first = scheduler.inbox.get_nowait()
        second = scheduler.inbox.get_nowait()
        assert first.type == "new_request"
        assert second.type == "request_update"

    asyncio.run(_run())


def test_omni_scheduler_buffers_early_update_and_replays_after_build() -> None:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.inbox = __import__("queue").Queue()
    scheduler.inbox.put(IncomingMessage("req-1", "request_update", {"seq_no": 0}))
    scheduler._aborted_request_ids = set()
    scheduler._completed_request_ids = {}
    scheduler._pending_request_updates = {}
    scheduler._request_update_handler = None
    scheduler._find_request_data = lambda request_id: None
    scheduler.tp_size = 1

    assert OmniScheduler.recv_requests(scheduler) == []
    assert list(scheduler._pending_request_updates["req-1"]) == [{"seq_no": 0}]

    req_data = type("RequestData", (), {})()
    payload = type(
        "Payload",
        (),
        {
            "request_id": "req-1",
            "prefetched_chunks": (),
            "prefetched_stream_done": False,
        },
    )()
    scheduler._stream_chunk_handler = None
    scheduler._stream_done_handler = None
    OmniScheduler._initialize_request_stream_state(scheduler, req_data, payload)

    assert list(req_data.request_updates) == [{"seq_no": 0}]
    assert "req-1" not in scheduler._pending_request_updates


def test_tp_scheduler_broadcasts_request_update_from_entry_rank(monkeypatch) -> None:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.inbox = __import__("queue").Queue()
    scheduler.tp_size = 2
    scheduler.is_entry_rank = True
    scheduler.tp_group = SimpleNamespace(rank=0, ranks=[0, 1])
    scheduler.tp_cpu_group = object()
    message = IncomingMessage("req-1", "request_update", {"seq_no": 0})
    scheduler.inbox.put(message)
    seen = {}

    def fake_broadcast(messages, rank, group, *, src):
        seen.update(messages=messages, rank=rank, group=group, src=src)
        return messages

    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler.broadcast_pyobj",
        fake_broadcast,
    )

    assert scheduler._recv_scheduler_messages() == [message]
    assert seen == {
        "messages": [message],
        "rank": 0,
        "group": scheduler.tp_cpu_group,
        "src": 0,
    }


def test_session_controller_enforces_order_time_and_final() -> None:
    controller = MossVLRealtimeSessionController()
    session = controller.open("req-1", "session-1")
    first = _event(0, 0.0)
    final = _event(1, 0.5, final=True)

    controller.ingest("req-1", first.to_dict())
    controller.ingest("req-1", final)
    assert session.drain(max_events=1) == [first]
    assert session.drain() == [final]

    with pytest.raises(RuntimeError, match="final event"):
        controller.ingest("req-1", _event(2, 1.0))
    assert controller.close("req-1") is True
    assert controller.close("req-1") is False


@pytest.mark.parametrize(
    ("event", "error"),
    [
        (_event(1, 0.0), "expected seq_no 0"),
        (_event(0, 0.0, request_id="other"), "request mismatch"),
        (_event(0, 0.0, session_id="other"), "session mismatch"),
    ],
)
def test_session_controller_rejects_mismatched_events(
    event: FramePromptEvent, error: str
) -> None:
    controller = MossVLRealtimeSessionController()
    controller.open("req-1", "session-1")
    with pytest.raises(ValueError, match=error):
        controller.ingest("req-1", event)


def test_session_controller_rejects_timestamp_regression() -> None:
    controller = MossVLRealtimeSessionController()
    controller.open("req-1", "session-1")
    controller.ingest("req-1", _event(0, 2.0))
    with pytest.raises(ValueError, match="timestamp moved backwards"):
        controller.ingest("req-1", _event(1, 1.0))
