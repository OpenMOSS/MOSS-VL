"""Unit tests for the MOSS-VL realtime async-decode (lookahead) machinery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_vl_realtime import (
    FramePromptEvent,
    MossVLRealtimePhase,
    MossVLRealtimeRuntimeState,
)
from sglang_omni.models.moss_vl_realtime.model_runner import MossVLRealtimeModelRunner
from sglang_omni.models.moss_vl_realtime.scheduler import MossVLRealtimeScheduler
from sglang_omni.models.moss_vl_realtime.session_state import (
    MossVLRealtimeSessionController,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def _state(
    request_id: str,
    phase: MossVLRealtimePhase = MossVLRealtimePhase.DECODING,
    decoder_length: int = 3,
    next_mrope_position: int = 9,
) -> MossVLRealtimeRuntimeState:
    return MossVLRealtimeRuntimeState(
        request_id=request_id,
        session_id=f"session-{request_id}",
        decoder_length=decoder_length,
        next_mrope_position=next_mrope_position,
        phase=phase,
    )


def _moss_req(
    request_id: str,
    phase: MossVLRealtimePhase = MossVLRealtimePhase.DECODING,
    *,
    finished: bool = False,
    is_retracted: bool = False,
) -> SimpleNamespace:
    state = _state(request_id, phase)
    req = SimpleNamespace(
        rid=request_id,
        sampling_params=SimpleNamespace(
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            min_new_tokens=0,
        ),
        finished=lambda: finished,
        is_retracted=is_retracted,
    )
    req._moss_vl_realtime_state = state
    return req


def _runner() -> MossVLRealtimeModelRunner:
    return MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)


def _scheduler() -> MossVLRealtimeScheduler:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler._async_pending = None
    scheduler._realtime_extend_batch = None
    scheduler.parked_reqs = {}
    scheduler.parked_since = {}
    scheduler.realtime_sessions = MossVLRealtimeSessionController()
    return scheduler


def test_lookahead_eligible_gates_non_moss_batches() -> None:
    runner = _runner()
    plain_req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            min_new_tokens=0,
        )
    )

    assert runner.lookahead_eligible(SimpleNamespace(reqs=[plain_req])) is False
    assert runner.lookahead_eligible(SimpleNamespace(reqs=[])) is False
    assert runner.lookahead_eligible(SimpleNamespace(reqs=[_moss_req("req-1")])) is True


def test_lookahead_eligible_rejects_history_dependent_sampling() -> None:
    runner = _runner()
    req = _moss_req("req-1")
    req.sampling_params.repetition_penalty = 1.1

    assert runner.lookahead_eligible(SimpleNamespace(reqs=[req])) is False


def test_finalize_skip_rids_covers_parked_rows_only() -> None:
    runner = _runner()
    scheduler_output = SimpleNamespace(
        requests=[
            SimpleNamespace(
                request_id="req-live",
                data=SimpleNamespace(req=_moss_req("req-live")),
            ),
            SimpleNamespace(
                request_id="req-parked",
                data=SimpleNamespace(
                    req=_moss_req("req-parked", MossVLRealtimePhase.WAITING_FOR_EVENT)
                ),
            ),
        ]
    )

    assert runner.finalize_skip_rids(scheduler_output) == {"req-parked"}


def test_post_decode_resolve_skips_parked_and_finished_rows() -> None:
    runner = _runner()
    decoding = _moss_req("req-decoding")
    parked = _moss_req("req-parked", MossVLRealtimePhase.WAITING_FOR_EVENT)
    finished = _moss_req("req-finished", MossVLRealtimePhase.FINISHED)
    schedule_batch = SimpleNamespace(reqs=[decoding, parked, finished])
    result = SimpleNamespace(next_token_ids=None)

    # launch_buf=None short-circuits the base token wiring, isolating the
    # runtime-state advance under test.
    runner.post_decode_resolve(None, result, None, schedule_batch, [])

    decoding_state = decoding._moss_vl_realtime_state
    assert decoding_state.decoder_length == 4
    assert decoding_state.next_mrope_position == 10
    assert decoding_state.phase is MossVLRealtimePhase.DECODING
    for req in (parked, finished):
        state = req._moss_vl_realtime_state
        assert state.decoder_length == 3
        assert state.next_mrope_position == 9


def _event(request_id: str, seq_no: int = 0) -> FramePromptEvent:
    return FramePromptEvent(
        request_id=request_id,
        session_id=f"session-{request_id}",
        seq_no=seq_no,
        timestamp=0.0,
        frame_ref=f"relay://frame-{seq_no}",
    )


def test_update_barrier_resolves_pending_step_before_materialize(monkeypatch) -> None:
    scheduler = _scheduler()
    req = _moss_req("req-1")
    scheduler.running_batch = SimpleNamespace(reqs=[req])
    session = scheduler.realtime_sessions.open("req-1", "session-req-1")
    session.accept(_event("req-1"))
    scheduler._async_pending = (SimpleNamespace(reqs=[req]), object(), object())
    calls: list[str] = []

    def _resolve() -> None:
        calls.append("resolve")
        scheduler._async_pending = None

    scheduler._resolve_pending_async = _resolve
    extension = object()
    scheduler._materialize_realtime_extensions = (
        lambda: calls.append("materialize") or extension
    )
    scheduler._expire_parked_requests = lambda: None

    def plan_after_handoff(self):
        calls.append("super")
        plan = self.get_new_batch_prefill(self.running_batch)
        assert plan.batch_to_run is extension
        return "plan"

    monkeypatch.setattr(
        OmniScheduler,
        "get_next_batch_to_run",
        plan_after_handoff,
    )

    assert scheduler.get_next_batch_to_run() == "plan"
    assert calls == ["resolve", "super", "materialize"]


def test_update_barrier_skips_resolve_without_pending_events(monkeypatch) -> None:
    scheduler = _scheduler()
    req = _moss_req("req-1")
    scheduler.running_batch = SimpleNamespace(reqs=[req])
    scheduler.realtime_sessions.open("req-1", "session-req-1")
    scheduler._async_pending = (SimpleNamespace(reqs=[req]), object(), object())
    scheduler._resolve_pending_async = lambda: pytest.fail("unexpected resolve")
    scheduler._materialize_realtime_extensions = lambda: None
    scheduler._expire_parked_requests = lambda: None
    monkeypatch.setattr(
        OmniScheduler,
        "get_next_batch_to_run",
        lambda self: "plan",
    )

    assert scheduler.get_next_batch_to_run() == "plan"
    assert scheduler._async_pending is not None


def test_resolve_and_process_drops_parked_overrun_row() -> None:
    scheduler = _scheduler()
    live = _moss_req("req-live")
    parked = _moss_req("req-parked", MossVLRealtimePhase.WAITING_FOR_EVENT)
    scheduler.parked_reqs = {"req-parked": parked}
    batch = SimpleNamespace(
        reqs=[live, parked],
        out_cache_loc=torch.tensor([10, 11]),
    )
    captured: dict[str, object] = {}

    def _run_batch_resolve(batch_arg, sched_output, pending_step, *, skip_rids):
        captured["skip_rids"] = skip_rids
        return SimpleNamespace(next_token_ids=torch.tensor([100, 101]))

    scheduler._run_batch_resolve = _run_batch_resolve
    freed: list[tuple[object, list[int]]] = []
    scheduler._free_parked_overrun_step_slots = lambda batch_arg, indices: (
        freed.append((batch_arg, indices))
    )
    processed: list[tuple[object, object]] = []
    scheduler.process_batch_result = lambda b, r: processed.append((b, r))

    scheduler._resolve_and_process(batch, object(), object())

    assert captured["skip_rids"] == {"req-parked"}
    assert freed == [(batch, [1])]
    assert batch.reqs == [live]
    result = processed[0][1]
    assert result.next_token_ids.tolist() == [100]


def test_resolve_and_process_delegates_when_no_rows_drop(monkeypatch) -> None:
    scheduler = _scheduler()
    batch = SimpleNamespace(reqs=[_moss_req("req-live")])
    calls: list[tuple[object, object, object]] = []
    monkeypatch.setattr(
        OmniScheduler,
        "_resolve_and_process",
        lambda self, b, s, p: calls.append((b, s, p)),
    )

    scheduler._resolve_and_process(batch, "sched", "pending")

    assert calls == [(batch, "sched", "pending")]


def test_abort_flushes_pending_step_covering_the_request(monkeypatch) -> None:
    scheduler = _scheduler()
    req = _moss_req("req-1")
    scheduler._async_pending = (SimpleNamespace(reqs=[req]), object(), object())
    scheduler.running_batch = SimpleNamespace(reqs=[req])
    scheduler._find_request_data = lambda rid: None
    calls: list[str] = []

    def _resolve() -> None:
        calls.append("resolve")
        scheduler._async_pending = None

    scheduler._resolve_pending_async = _resolve
    monkeypatch.setattr(
        OmniScheduler,
        "abort",
        lambda self, rid, *, defer_running_cleanup=True: calls.append("abort"),
    )

    scheduler.abort("req-1")

    assert calls == ["resolve", "abort"]
    assert scheduler._async_pending is None


def test_abort_without_pending_row_skips_resolve(monkeypatch) -> None:
    scheduler = _scheduler()
    other = _moss_req("req-other")
    scheduler._async_pending = (SimpleNamespace(reqs=[other]), object(), object())
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler._find_request_data = lambda rid: None
    scheduler._resolve_pending_async = lambda: pytest.fail("unexpected resolve")
    monkeypatch.setattr(
        OmniScheduler,
        "abort",
        lambda self, rid, *, defer_running_cleanup=True: None,
    )

    scheduler.abort("req-1")

    assert scheduler._async_pending is not None
