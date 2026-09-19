"""Capacity and allocation ownership across reused decode batches."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers import schedule_batch as upstream
from sglang.srt.mem_cache import allocation

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_vl_realtime.batch_adapter import (
    MossVLRealtimeScheduleBatch,
    commit_moss_vl_realtime_batch,
    prepare_moss_vl_realtime_encoder_info_extend,
    realtime_decode_capacity_error,
    rollback_moss_vl_realtime_batch,
)
from sglang_omni.models.moss_vl_realtime.model_runner import MossVLRealtimeModelRunner
from sglang_omni.models.moss_vl_realtime.runtime_state import MossVLRealtimeRuntimeState
from sglang_omni.models.moss_vl_realtime.scheduler import (
    MossVLRealtimeScheduler,
    _append_segment_to_request,
    _guard_realtime_context_capacity,
    _undo_appended_segment,
    bind_realtime_page_row,
)
from tests.unit_test.moss_vl_realtime.test_batch_adapter import _batch
from tests.unit_test.moss_vl_realtime.test_realtime_scheduler import _Req, _segment


def prepared_batch():
    fixture, req, state = _batch()
    batch = MossVLRealtimeScheduleBatch(reqs=[req])
    batch.__dict__.update(vars(fixture))
    prepare_moss_vl_realtime_encoder_info_extend(
        batch, [[999, -101, -101, 101, 102]], [9]
    )
    commit_moss_vl_realtime_batch(batch)
    table = batch.req_to_token_pool.req_to_token
    batch.req_to_token_pool.write = lambda index, value: table.__setitem__(
        index, value.to(table.dtype)
    )
    batch.maybe_evict_swa = lambda: None
    batch.tree_cache = None
    batch.model_config = SimpleNamespace(is_encoder_decoder=True)
    batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    batch.req_pool_indices = torch.tensor([req.req_pool_idx])
    batch.orig_seq_lens = batch.seq_lens.clone()
    batch.hisparse_coordinator = None
    req.decode_batch_idx = 0
    req.finished = lambda: False
    req.is_retracted = False
    return batch, req, state


def decode(monkeypatch, batch, slot):
    monkeypatch.setattr(
        upstream,
        "get_server_args",
        lambda: SimpleNamespace(enable_mamba_extra_buffer=lambda: False),
    )
    monkeypatch.setattr(allocation, "_alloc_page_size", lambda batch: 1)
    monkeypatch.setattr(
        allocation, "alloc_token_slots", lambda *a: torch.tensor([slot])
    )
    batch.prepare_for_decode()


def released(batch):
    return [
        int(slot)
        for slots in batch.token_to_kv_pool_allocator.released
        for slot in slots
    ]


def test_reused_prefill_batch_rolls_back_only_failed_decode(monkeypatch):
    batch, req, state = prepared_batch()
    decode(monkeypatch, batch, 26)
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    def fail(*args):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(ModelRunner, "execute", fail)
    with pytest.raises(RuntimeError, match="forward failed"):
        runner.execute(SimpleNamespace(batch_data=batch))
    assert released(batch) == [26]
    assert req.kv.kv_allocated_len == req.kv_committed_len == 9
    assert batch.req_to_token_pool.req_to_token[1, 9] == 0
    assert state.encoder_length + state.decoder_length == 9
    rollback_moss_vl_realtime_batch(batch)
    assert released(batch) == [26]


def test_committed_prefill_snapshot_cannot_release_owned_kv():
    batch, req, _ = prepared_batch()
    snapshot = batch.copy()
    rollback_moss_vl_realtime_batch(snapshot)
    assert released(batch) == []
    assert req.kv.kv_allocated_len == 9


def test_error_after_decode_commit_does_not_free_committed_slot(monkeypatch):
    batch, req, state = prepared_batch()
    decode(monkeypatch, batch, 26)
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    def fail_after_commit(*args):
        runner.post_decode(None, None, batch, [])
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(ModelRunner, "execute", fail_after_commit)
    with pytest.raises(RuntimeError, match="sampling failed"):
        runner.execute(SimpleNamespace(batch_data=batch))
    assert released(batch) == []
    assert req.kv.kv_allocated_len == 10
    assert batch.req_to_token_pool.req_to_token[1, 9] == 26
    assert state.encoder_length + state.decoder_length == 10


def test_decode_preparation_failure_before_allocation_keeps_old_kv(monkeypatch):
    batch, req, _ = prepared_batch()

    def fail(*args):
        raise RuntimeError("before allocation")

    monkeypatch.setattr(upstream.ScheduleBatch, "prepare_for_decode", fail)
    with pytest.raises(RuntimeError, match="before allocation"):
        batch.prepare_for_decode()
    assert batch.out_cache_loc is None
    assert released(batch) == []
    assert req.kv.kv_allocated_len == 9


def test_decode_preparation_failure_after_allocation_releases_current_slot(monkeypatch):
    batch, req, _ = prepared_batch()
    original = upstream.ScheduleBatch.prepare_for_decode

    def fail(self):
        original(self)
        raise RuntimeError("after allocation")

    monkeypatch.setattr(upstream.ScheduleBatch, "prepare_for_decode", fail)
    with pytest.raises(RuntimeError, match="after allocation"):
        decode(monkeypatch, batch, 26)
    assert released(batch) == [26]
    assert req.kv.kv_allocated_len == 9


def test_async_snapshot_keeps_its_own_decode_allocation(monkeypatch):
    batch, req, state = prepared_batch()
    decode(monkeypatch, batch, 26)
    snapshot = batch.copy()
    decode(monkeypatch, batch, 27)
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)
    runner.post_decode_resolve(
        None, SimpleNamespace(next_token_ids=None), None, snapshot, []
    )
    rollback_moss_vl_realtime_batch(batch)
    rollback_moss_vl_realtime_batch(snapshot)
    assert released(batch) == [27]
    assert req.kv.kv_allocated_len == 10
    assert state.encoder_length + state.decoder_length == 10
    assert batch.req_to_token_pool.req_to_token[1, 9] == 26


def test_older_attempt_cannot_rollback_over_newer_allocation(monkeypatch):
    batch, req, _ = prepared_batch()
    decode(monkeypatch, batch, 26)
    snapshot = batch.copy()
    decode(monkeypatch, batch, 27)
    with pytest.raises(RuntimeError, match="newer"):
        rollback_moss_vl_realtime_batch(snapshot)
    assert released(batch) == []
    assert req.kv.kv_allocated_len == 11
    rollback_moss_vl_realtime_batch(batch)
    rollback_moss_vl_realtime_batch(snapshot)
    assert released(batch) == [27, 26]
    assert req.kv.kv_allocated_len == 9


def test_extend_generation_budget_is_capped_by_context():
    req = _Req()
    state = MossVLRealtimeRuntimeState(
        request_id="req-1", session_id="session-1", req_pool_index=0, decoder_length=2
    )
    req._moss_vl_realtime_state = state
    table = torch.zeros((1, 8), dtype=torch.int32)
    table[0, :2] = torch.tensor([11, 12])
    bind_realtime_page_row(req, table)
    _append_segment_to_request(req, state, _segment())
    assert len(req.origin_input_ids) + req.sampling_params.max_new_tokens <= 8
    assert state.decode_allowance == 9


def test_extend_must_leave_room_for_its_sampled_output():
    state = MossVLRealtimeRuntimeState(
        request_id="req", session_id="req", decoder_length=2
    )
    pool = SimpleNamespace(req_to_token=torch.zeros((1, 8)))
    with pytest.raises(RuntimeError, match="context"):
        _guard_realtime_context_capacity(
            None, state, SimpleNamespace(raw_append_ids=[1] * 5), pool
        )


def test_configured_context_does_not_include_page_table_padding():
    state = MossVLRealtimeRuntimeState(
        request_id="req", session_id="req", decoder_length=2, context_limit=8
    )
    pool = SimpleNamespace(req_to_token=torch.zeros((1, 64)))
    with pytest.raises(RuntimeError, match="context"):
        _guard_realtime_context_capacity(
            None, state, SimpleNamespace(raw_append_ids=[1] * 5), pool
        )


@pytest.mark.parametrize(
    "decoder,allocated,limit,width,history,allowed",
    [
        (4, 6, 8, 16, 2, True),
        (5, 7, 8, 16, 2, False),
        (6, 8, 8, 16, 2, False),
        (4, 7, 8, 16, 2, False),  # Unresolved lookahead already owns a slot.
        (3, 5, 104, 16, 100, False),  # Eviction cannot reset token history.
        (3, 5, 105, 16, 100, True),
        (6, 8, 64, 8, 2, False),  # Physical row can be smaller than context.
    ],
)
def test_decode_checks_logical_and_physical_capacity(
    decoder, allocated, limit, width, history, allowed
):
    state = MossVLRealtimeRuntimeState(
        request_id="req",
        session_id="req",
        encoder_length=2,
        decoder_length=decoder,
        appended_encoder_length=history,
        context_limit=limit,
    )
    req = SimpleNamespace(
        _moss_vl_realtime_state=state, kv=SimpleNamespace(kv_allocated_len=allocated)
    )
    pool = SimpleNamespace(req_to_token=torch.zeros((1, width)))
    assert (realtime_decode_capacity_error(req, pool) is None) is allowed


def capacity_scheduler():
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    reqs = []
    for i in range(4):
        state = MossVLRealtimeRuntimeState(
            request_id=str(i),
            session_id=str(i),
            encoder_length=2,
            decoder_length=5 if i == 0 else 4,
            context_limit=8,
        )
        req = SimpleNamespace(rid=str(i), _moss_vl_realtime_state=state, done=False)
        req.finished = lambda req=req: req.done
        reqs.append(req)
    scheduler.running_batch = SimpleNamespace(reqs=reqs)
    scheduler.req_to_token_pool = SimpleNamespace(req_to_token=torch.zeros((4, 16)))
    scheduler._async_pending = None
    scheduler.errors = []
    scheduler.aborted = []
    scheduler._emit_request_error = lambda rid, error: scheduler.errors.append(rid)

    def abort(rid, **kwargs):
        scheduler.aborted.append(rid)
        scheduler.running_batch.reqs = [
            r for r in scheduler.running_batch.reqs if r.rid != rid
        ]

    scheduler.abort = abort
    return scheduler


def test_context_exhaustion_only_ends_the_affected_session():
    scheduler = capacity_scheduler()
    scheduler._guard_decode_capacity()
    assert scheduler.aborted == scheduler.errors == ["0"]
    assert [r.rid for r in scheduler.running_batch.reqs] == ["1", "2", "3"]


def test_near_limit_resolves_lookahead_before_deciding_to_abort():
    scheduler = capacity_scheduler()
    scheduler._async_pending = object()
    calls = []

    def resolve():
        calls.append("resolve")
        scheduler._async_pending = None
        scheduler.running_batch.reqs[0].done = True

    scheduler._resolve_pending_async = resolve
    scheduler._guard_decode_capacity()
    assert calls == ["resolve"]
    assert scheduler.aborted == []  # Normal length completion won the race.


def test_context_guard_uses_the_tp_decision(monkeypatch):
    from sglang_omni.models.moss_vl_realtime import scheduler as module

    scheduler = capacity_scheduler()
    scheduler.tp_size = 2
    scheduler.tp_group = SimpleNamespace(rank=1, ranks=[0, 1])
    scheduler.tp_cpu_group = object()
    monkeypatch.setattr(
        module, "broadcast_pyobj", lambda *a, **k: [{"0": "context exhausted"}]
    )
    scheduler._guard_decode_capacity()
    assert scheduler.aborted == ["0"]


def test_prepare_refuses_overflow_before_any_allocation(monkeypatch):
    batch, req, state = prepared_batch()
    state.context_limit = 10
    with pytest.raises(RuntimeError, match="context"):
        decode(monkeypatch, batch, 26)
    assert req.kv.kv_allocated_len == 9
    assert released(batch) == []


def test_prefill_handoff_preserves_fields_and_installs_decode_hooks():
    source = upstream.ScheduleBatch(reqs=[])
    source.custom_marker = object()
    source.seq_lens = torch.tensor([5])
    converted = MossVLRealtimeScheduleBatch.from_batch(source)
    assert isinstance(converted, MossVLRealtimeScheduleBatch)
    assert converted.custom_marker is source.custom_marker
    assert converted.seq_lens is source.seq_lens
    assert MossVLRealtimeScheduleBatch.from_batch(converted) is converted


def test_undo_restores_the_original_clipped_allowance():
    req = _Req()
    req.sampling_params.max_new_tokens = 4
    state = MossVLRealtimeRuntimeState(
        request_id="req-1",
        session_id="session-1",
        req_pool_index=0,
        decoder_length=2,
        decode_allowance=6,
    )
    req._moss_vl_realtime_state = state
    table = torch.zeros((1, 8), dtype=torch.int32)
    table[0, :2] = torch.tensor([11, 12])
    bind_realtime_page_row(req, table)
    segment = _segment()
    _append_segment_to_request(req, state, segment)
    _undo_appended_segment(req, segment)
    assert req.sampling_params.max_new_tokens == 4
    assert state.decode_allowance == 6


def test_async_launch_failure_waits_before_rollback(monkeypatch):
    batch, _, _ = prepared_batch()
    decode(monkeypatch, batch, 26)
    calls = []
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)
    runner._execution_bridge = SimpleNamespace(
        record_completion=lambda: SimpleNamespace(
            synchronize=lambda: calls.append("wait")
        )
    )
    free = batch.token_to_kv_pool_allocator.free

    def checked_free(slots):
        assert calls == ["wait"]
        free(slots)

    def fail(*args):
        raise RuntimeError("launch failure")

    monkeypatch.setattr(batch.token_to_kv_pool_allocator, "free", checked_free)
    monkeypatch.setattr(ModelRunner, "execute_launch", fail)
    with pytest.raises(RuntimeError, match="launch failure"):
        runner.execute_launch(SimpleNamespace(batch_data=batch))
    assert released(batch) == [26]


def test_resolve_failure_keeps_newer_step_kv_for_coordinated_abort(monkeypatch):
    batch, req, _ = prepared_batch()
    decode(monkeypatch, batch, 26)
    snapshot = batch.copy()
    decode(monkeypatch, batch, 27)
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)
    calls = []
    runner._execution_bridge = SimpleNamespace(
        record_completion=lambda: SimpleNamespace(
            synchronize=lambda: calls.append("wait")
        )
    )

    def fail(*args):
        raise RuntimeError("resolve failure")

    monkeypatch.setattr(ModelRunner, "execute_resolve", fail)
    with pytest.raises(RuntimeError, match="resolve failure"):
        runner.execute_resolve(SimpleNamespace(schedule_batch=snapshot))
    assert calls == ["wait"]
    assert released(batch) == []
    assert req.kv.kv_allocated_len == 11
    assert batch.req_to_token_pool.req_to_token[1, 9:11].tolist() == [26, 27]
