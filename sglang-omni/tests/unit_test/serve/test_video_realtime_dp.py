# SPDX-License-Identifier: Apache-2.0
"""Native-DP realtime serving: replica admission aggregation and session pinning."""
from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.types import GenerateChunk
from sglang_omni.config import (
    ParallelismConfig,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.serve.launcher import (
    _entry_stage_dp_size,
    _video_realtime_max_sessions,
)
from sglang_omni.serve.video_realtime import (
    VideoRealtimeSessionManager,
    VideoSessionConfigure,
    warmup_video_realtime,
)

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


class _Socket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def receive(self) -> dict[str, Any]:
        await asyncio.Event().wait()  # pragma: no cover - sessions never run here
        raise AssertionError("unreachable")

    async def send_json(self, value: dict[str, Any]) -> None:
        self.sent.append(value)

    async def close(self, code: int = 1000) -> None:
        del code
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED


class _RecordingClient:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.aborted: list[str] = []
        self.chunks: asyncio.Queue[GenerateChunk] = asyncio.Queue()

    async def generate(
        self,
        request: Any,
        request_id: str | None = None,
        **kwargs: Any,
    ):
        self.generate_calls.append(kwargs)
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event="session.ready",
            control_data={},
        )
        while True:
            yield await self.chunks.get()

    async def update_request(self, request_id: str, data: dict[str, Any]) -> None:
        del request_id, data

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


async def _until(predicate) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), 2)


def _manager(**overrides):
    kwargs = {"client": _RecordingClient(), "model_name": "test"}
    kwargs.update(overrides)
    return VideoRealtimeSessionManager(**kwargs)


def test_dp1_sessions_are_not_pinned() -> None:
    manager = _manager(max_sessions=2)

    sessions = [manager.open(_Socket()) for _ in range(2)]
    assert [s.dp_rank for s in sessions] == [None, None]
    with pytest.raises(RuntimeError, match="no free session slot"):
        manager.open(_Socket())


def test_replicated_admission_balances_and_caps_per_replica() -> None:
    manager = _manager(max_sessions=4, replica_count=2)

    sessions = [manager.open(_Socket()) for _ in range(4)]
    assert [s.dp_rank for s in sessions] == [0, 1, 0, 1]
    with pytest.raises(RuntimeError, match="no free session slot"):
        manager.open(_Socket())


def test_closed_session_frees_its_replica_slot() -> None:
    async def _run() -> None:
        manager = _manager(max_sessions=2, replica_count=2)

        first = manager.open(_Socket())
        second = manager.open(_Socket())
        assert (first.dp_rank, second.dp_rank) == (0, 1)
        with pytest.raises(RuntimeError, match="no free session slot"):
            manager.open(_Socket())

        await manager.close(first.session_id)
        third = manager.open(_Socket())
        assert third.dp_rank == 0
        await manager.close(second.session_id)
        assert manager.sessions.keys() == {third.session_id}

    asyncio.run(_run())


def test_session_configure_pins_generate_to_the_assigned_replica() -> None:
    async def _run() -> None:
        client = _RecordingClient()
        manager = VideoRealtimeSessionManager(
            client=client, model_name="test", max_sessions=2, replica_count=2
        )
        first = manager.open(_Socket())
        second = manager.open(_Socket())
        assert second.dp_rank == 1

        await second.configure(VideoSessionConfigure(type="session.configure"))
        await _until(lambda: bool(client.generate_calls))
        assert client.generate_calls[-1]["dp_rank"] == 1

        await manager.close(second.session_id)
        await manager.close(first.session_id)
        assert client.aborted

    asyncio.run(_run())


class _WarmupClient:
    """Session-ready, then a processed frame once the frame update arrives."""

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    async def generate(self, request: Any, request_id: str | None = None, **kwargs: Any):
        self.generate_calls.append(kwargs)
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event="session.ready",
            control_data={},
        )
        while not self.updates:
            await asyncio.sleep(0.001)
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event="input.frame.processed",
            control_data={},
        )

    async def update_request(self, request_id: str, data: dict[str, Any]) -> None:
        self.updates.append(data)

    async def abort(self, request_id: str) -> None:
        del request_id


def test_warmup_runs_per_replica_with_explicit_pins() -> None:
    async def _run() -> None:
        client = _WarmupClient()

        await warmup_video_realtime(client, model_name="test", dp_rank=1)
        assert client.generate_calls[-1]["dp_rank"] == 1

        await warmup_video_realtime(client, model_name="test")
        assert "dp_rank" not in client.generate_calls[-1]

    asyncio.run(_run())


class _RealtimePipelineConfig(PipelineConfig):
    supports_video_realtime: ClassVar[bool] = True


def _realtime_config(*, dp: int, max_running_requests: int) -> PipelineConfig:
    gpus = list(range(dp)) if dp > 1 else 0
    return _RealtimePipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="ar",
                process="pipeline",
                factory=_FACTORY,
                factory_args={"max_running_requests": max_running_requests},
                gpu=gpus,
                parallelism=ParallelismConfig(dp=dp),
                terminal=True,
            )
        ],
    )


def test_video_realtime_max_sessions_scales_with_replicas() -> None:
    config = _realtime_config(dp=2, max_running_requests=3)

    assert _entry_stage_dp_size(config) == 2
    assert _video_realtime_max_sessions(config) == 6


def test_video_realtime_max_sessions_unchanged_without_dp() -> None:
    config = _realtime_config(dp=1, max_running_requests=3)

    assert _entry_stage_dp_size(config) == 1
    assert _video_realtime_max_sessions(config) == 3


def test_create_app_threads_replica_count_to_manager() -> None:
    from sglang_omni.serve.openai_api import create_app

    client = _RecordingClient()
    app = create_app(
        client,
        model_name="test",
        enable_video_realtime=True,
        video_realtime_max_sessions=4,
        video_realtime_replica_count=2,
    )

    manager = app.state.video_realtime_manager
    assert manager.max_sessions == 4
    assert manager.replica_count == 2


def test_replica_count_must_divide_session_capacity() -> None:
    with pytest.raises(ValueError, match="multiple of replica_count"):
        _manager(max_sessions=3, replica_count=2)
