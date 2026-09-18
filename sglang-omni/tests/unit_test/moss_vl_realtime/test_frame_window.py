from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_vl_realtime import (
    FramePromptEvent,
    MossVLRealtimeRuntimeState,
    RealtimeFrameRecord,
    RealtimeFrameWindowConfig,
    apply_frame_window_plan,
    plan_frame_window,
    stage_segment_frame_records,
)
from sglang_omni.models.moss_vl_realtime.frame_window import (
    DEFAULT_POOL_RATIO,
    DEFAULT_POOL_WINDOW_S,
    DEFAULT_RAW_WINDOW_S,
    FRAME_RECORDS_STAGED_ATTR,
    covered_spans,
    visible_count_remap,
)


def _record(ts: float, *, slots: int = 2, pooled: bool = False, sources: int = 1):
    return RealtimeFrameRecord(
        timestamp=ts,
        grid_h=2,
        grid_w=2,
        slots=slots,
        pooled=pooled,
        pooled_sources=sources,
    )


def _config(**overrides) -> RealtimeFrameWindowConfig:
    base: dict = {
        "enabled": True,
        "raw_window_s": 35.0,
        "pool_window_s": 120.0,
        "pool_ratio": 2,
        # Tests exercise the pooling path explicitly; the production default
        # is pooling off.
        "pooling_enabled": True,
    }
    base.update(overrides)
    return RealtimeFrameWindowConfig(**base)


class _FakeAllocator:
    def __init__(self, free_ids: list[int], *, fail_alloc: bool = False) -> None:
        self._free = list(free_ids)
        self._fail_alloc = fail_alloc
        self.freed: list[list[int]] = []

    def alloc(self, n: int):
        if self._fail_alloc or n > len(self._free):
            return None
        out = torch.tensor(self._free[:n], dtype=torch.int64)
        self._free = self._free[n:]
        return out

    def free(self, slots) -> None:
        self.freed.extend(int(v) for v in slots.tolist())

    def available_size(self) -> int:
        return len(self._free)


class _FakeKVPool:
    def __init__(self, size: int, layers: int = 2, heads: int = 2, dim: int = 4):
        self.k_buffer = [torch.zeros(size, heads, dim) for _ in range(layers)]
        self.v_buffer = [torch.zeros(size, heads, dim) for _ in range(layers)]


def _enabled_state(records, *, grid_rows=None):
    state = MossVLRealtimeRuntimeState(
        request_id="req-fw",
        session_id="session-fw",
        req_pool_index=0,
        encoder_length=sum(r.slots for r in records),
        decoder_length=3,
        surviving_frame_count=len(records),
    )
    state.frame_records = list(records)
    rows = grid_rows if grid_rows is not None else [r.grid_row for r in records]
    state.full_grid_thw = torch.tensor(rows, dtype=torch.long)
    return state


def _harness(records, *, allocator_ids=(900, 901, 902, 903, 904, 905)):
    """One request with 2-slot frames and a 3-token decoder tail."""
    state = _enabled_state(records)
    req_to_token = torch.zeros((1, 64), dtype=torch.int64)
    encoder_slots = list(range(100, 100 + state.encoder_length))
    decoder_slots = [50, 60, 70]
    req_to_token[0, : len(encoder_slots) + 3] = torch.tensor(
        encoder_slots + decoder_slots
    )
    allocator = _FakeAllocator(list(allocator_ids))
    pool = _FakeKVPool(size=1024)
    req = SimpleNamespace(
        rid="req-fw",
        multimodal_inputs=SimpleNamespace(
            visible_frame_counts=None,
            num_image_tokens=state.encoder_length,
            media_nums_per_sample=[state.full_grid_thw.shape[0]],
            mm_items=[
                SimpleNamespace(model_specific_data={"realtime_full_grid_thw": None})
            ],
        ),
        kv=SimpleNamespace(kv_allocated_len=state.encoder_length + 3),
        kv_committed_len=state.encoder_length + 3,
    )
    return state, req, req_to_token, allocator, pool


# --- config resolution -----------------------------------------------------


@pytest.mark.parametrize("field", ["raw_window_s", "pool_window_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_window_thresholds_must_be_finite(field, value):
    with pytest.raises(ValueError, match="finite"):
        RealtimeFrameWindowConfig(**{field: value})


@pytest.mark.parametrize(
    "key", ["REALTIME_FRAME_WINDOW_RAW_S", "REALTIME_FRAME_POOL_WINDOW_S"]
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_environment_window_thresholds_must_be_finite(key, value):
    with pytest.raises(ValueError, match="finite"):
        RealtimeFrameWindowConfig.resolve(env={key: value})


def test_config_defaults_disabled() -> None:
    config = RealtimeFrameWindowConfig.resolve(env={})
    assert config.enabled is False
    assert config.raw_window_s == DEFAULT_RAW_WINDOW_S == 60.0
    assert config.pool_window_s == DEFAULT_POOL_WINDOW_S == 240.0
    assert config.pool_ratio == DEFAULT_POOL_RATIO == 4


def test_config_env_overrides_explicit_values() -> None:
    config = RealtimeFrameWindowConfig.resolve(
        enabled=False,
        raw_window_s=5.0,
        pool_window_s=50.0,
        pool_ratio=2,
        env={
            "REALTIME_FRAME_WINDOW_ENABLED": "1",
            "REALTIME_FRAME_WINDOW_RAW_S": "45",
            "REALTIME_FRAME_POOL_WINDOW_S": "300",
            "REALTIME_FRAME_POOL_RATIO": "3",
        },
    )
    assert config.enabled is True
    assert config.raw_window_s == 45.0
    assert config.pool_window_s == 300.0
    assert config.pool_ratio == 3


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="raw_window_s"):
        RealtimeFrameWindowConfig(enabled=True, raw_window_s=0)
    with pytest.raises(ValueError, match="pool_ratio"):
        RealtimeFrameWindowConfig(enabled=True, pool_ratio=1)
    with pytest.raises(ValueError, match="boolean"):
        RealtimeFrameWindowConfig.resolve(
            env={"REALTIME_FRAME_WINDOW_ENABLED": "maybe"}
        )


# --- planning ---------------------------------------------------------------


def test_plan_noop_when_disabled_or_within_window() -> None:
    records = [_record(0.0), _record(10.0), _record(20.0)]
    assert plan_frame_window(records, RealtimeFrameWindowConfig()) is None
    # Newest=20, oldest=0: span 20 < window 35 -> nothing ages out.
    assert plan_frame_window(records, _config()) is None


def test_plan_raw_window_boundary_is_strictly_greater() -> None:
    records = [_record(0.0), _record(35.0)]
    # Span exactly equals the window: the oldest frame is kept.
    assert plan_frame_window(records, _config()) is None
    records = [_record(0.0), _record(1.0), _record(36.0 + 1e-6)]
    plan = plan_frame_window(records, _config())
    assert plan is not None
    assert plan.pooled_raw_count == 2
    assert plan.dropped_raw_count == 0
    # One full ratio group folds into a virtual copy ahead of the raw region.
    assert [r.timestamp for r in plan.new_records] == [1.0, 36.0 + 1e-6]
    assert plan.new_records[0].pooled is True
    assert plan.new_records[0].pooled_sources == 2
    assert plan.new_records[1].pooled is False


def test_plan_holds_aged_remainder_until_ratio_group_fills() -> None:
    """1fps aging produces one aged frame per evaluation; pooling partial
    tail chunks would compress 1:1 forever. The remainder stays raw (a soft
    overshoot of at most ratio-1 frames) until a full group is available."""
    config = _config(raw_window_s=10.0, pool_ratio=4)
    # 33 frames 1s apart; newest=32: frames with 32-ts>10 (ts<22) are aged.
    records = [_record(float(i)) for i in range(33)]
    plan = plan_frame_window(records, config)
    assert plan is not None
    # 22 aged, floor to 4 full groups; ts 20 and 21 stay raw beyond window.
    assert plan.raw_chunks == (
        (0, 4, True),
        (4, 4, True),
        (8, 4, True),
        (12, 4, True),
        (16, 4, True),
    )
    assert plan.pooled_raw_count == 20
    assert plan.produced_virtual_count == 5
    kept_raw = [r.timestamp for r in plan.new_records if not r.pooled]
    assert kept_raw == [float(i) for i in range(20, 33)]
    assert len(plan.new_records) == 5 + 13
    produced = [r for r in plan.new_records if r.pooled]
    assert [r.pooled_sources for r in produced] == [4] * 5
    assert [r.timestamp for r in produced] == [3.0, 7.0, 11.0, 15.0, 19.0]

    # With only 3 aged frames (< ratio), nothing pools yet.
    records = [_record(float(i)) for i in range(14)]
    assert plan_frame_window(records, config) is None


def test_plan_pools_aged_frames_into_virtual_prefix() -> None:
    records = [_record(t) for t in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)]
    plan = plan_frame_window(records, _config())
    assert plan is not None
    # Newest raw = 50; frames trailing by >35s are 0 and 10.
    assert plan.raw_chunks == ((0, 2, True),)
    assert plan.pooled_raw_count == 2
    assert plan.produced_virtual_count == 1
    assert plan.evicted_virtual_count == 0
    timestamps = [r.timestamp for r in plan.new_records]
    assert timestamps == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert plan.new_records[0].pooled is True
    assert plan.new_records[0].pooled_sources == 2
    assert all(not r.pooled for r in plan.new_records[1:])


def test_plan_dropped_chunks_for_mixed_grids() -> None:
    mixed = [
        _record(0.0, slots=2),
        RealtimeFrameRecord(timestamp=10.0, grid_h=4, grid_w=4, slots=5),
        _record(20.0),
        _record(50.0),
    ]
    config = _config(raw_window_s=35.0)
    plan = plan_frame_window(mixed, config)
    assert plan is not None
    # Chunk (0,10) has mixed grids: dropped, not pooled.
    assert plan.raw_chunks == ((0, 2, False),)
    assert plan.dropped_raw_count == 2
    assert plan.produced_virtual_count == 0
    assert [r.timestamp for r in plan.new_records] == [20.0, 50.0]


def test_plan_pooling_disabled_drops_aged_raws() -> None:
    """Raw-only comparison mode: aged frames are dropped, no virtual tier."""
    records = [_record(t) for t in (0.0, 10.0, 20.0, 30.0, 80.0)]
    plan = plan_frame_window(records, _config(pooling_enabled=False))
    assert plan is not None
    # 80-0/10/20/30 all exceed the 35s raw window: one whole dropped chunk.
    assert plan.raw_chunks == ((0, 4, False),)
    assert plan.pooled_raw_count == 0
    assert plan.dropped_raw_count == 4
    assert plan.produced_virtual_count == 0
    assert [r.timestamp for r in plan.new_records] == [80.0]
    assert all(not r.pooled for r in plan.new_records)


def test_pooling_enabled_flag_resolution_and_validation() -> None:
    # Pooling defaults to off (train-infer consistency); opt in explicitly.
    default = RealtimeFrameWindowConfig(enabled=True)
    assert default.pooling_enabled is False
    pooled = RealtimeFrameWindowConfig(enabled=True, pooling_enabled=True)
    assert pooled.pooling_enabled is True
    raw_only = RealtimeFrameWindowConfig.resolve(
        enabled=True, pooling_enabled=True, env={"REALTIME_FRAME_POOLING_ENABLED": "0"}
    )
    assert raw_only.pooling_enabled is False
    with pytest.raises(TypeError, match="pooling_enabled"):
        RealtimeFrameWindowConfig(enabled=True, pooling_enabled="yes")


def test_plan_virtual_window_boundary() -> None:
    records = [
        _record(0.0, pooled=True, sources=2),
        _record(100.0, pooled=True, sources=2),
        _record(200.0, pooled=True, sources=2),
        _record(300.0),
    ]
    # 200 - 0 > 120 -> oldest virtual evicted.
    plan = plan_frame_window(records, _config())
    assert plan is not None
    assert plan.evicted_virtual_count == 1
    assert [r.timestamp for r in plan.new_records] == [100.0, 200.0, 300.0]

    boundary = [
        _record(80.0, pooled=True, sources=2),
        _record(200.0, pooled=True, sources=2),
        _record(300.0),
    ]
    # Span exactly 120 == window: kept (strictly-greater eviction).
    assert plan_frame_window(boundary, _config()) is None


def test_plan_rejects_pooled_record_after_raw() -> None:
    records = [_record(0.0), _record(10.0, pooled=True, sources=2)]
    with pytest.raises(RuntimeError, match="leading encoder region"):
        plan_frame_window(records, _config())


# --- record staging ---------------------------------------------------------


def test_stage_segment_frame_records_orders_and_spans() -> None:
    events = [
        FramePromptEvent(
            request_id="r",
            session_id="s",
            seq_no=1,
            timestamp=20.0,
            frame_ref="x://b",
        ),
        FramePromptEvent(
            request_id="r",
            session_id="s",
            seq_no=0,
            timestamp=5.0,
            frame_ref="x://a",
        ),
    ]
    segment = SimpleNamespace(
        events=events,
        full_grid_thw=torch.tensor([[1, 2, 2], [1, 4, 4]], dtype=torch.long),
    )
    records = stage_segment_frame_records(segment, merge_size=2)
    assert records is not None
    assert [r.timestamp for r in records] == [5.0, 20.0]
    # 2x2 grid with merge 2 -> 1 vision token + separator = 2 slots.
    assert records[0].slots == 2
    # 4x4 grid -> 4 vision tokens + separator = 5 slots.
    assert records[1].slots == 5
    assert all(not r.pooled for r in records)


def test_stage_segment_frame_records_none_for_text_only() -> None:
    event = FramePromptEvent(
        request_id="r",
        session_id="s",
        seq_no=0,
        timestamp=0.0,
        frame_ref=None,
        prompt="hello",
    )
    segment = SimpleNamespace(
        events=[event], full_grid_thw=torch.empty((0, 3), dtype=torch.long)
    )
    assert stage_segment_frame_records(segment, merge_size=2) is None


# --- metadata remap ---------------------------------------------------------


def test_visible_count_remap_pooled_group() -> None:
    spans = ((0, 2), (2, 3), (3, 4))  # rows 0-1 pooled to one row
    assert visible_count_remap(4, spans) == [0, 1, 1, 2, 3]


def test_visible_count_remap_identity_gaps_and_partial_visibility() -> None:
    # Identity spans leave counts untouched.
    assert visible_count_remap(3, ((0, 1), (1, 2), (2, 3))) == [0, 1, 2, 3]
    # Leading rows dropped (e.g. evicted virtuals / non-poolable chunks).
    assert visible_count_remap(4, ((2, 4),)) == [0, 0, 0, 1, 1]
    # A token sees the pooled virtual once it saw any member of the chunk.
    assert visible_count_remap(4, ((0, 2), (2, 4))) == [0, 1, 1, 2, 2]


def test_covered_spans_mixed_layout() -> None:
    records = [
        _record(0.0, pooled=True, sources=2),
        _record(10.0, pooled=True, sources=2),
        _record(20.0),
        _record(23.0),
        _record(27.0),
        _record(40.0),
    ]
    plan = plan_frame_window(records, _config(raw_window_s=15.0, pool_window_s=10.0))
    assert plan is not None
    # Raw window 15: aged raw = 20 and 23 (40-23=17>15; 40-27=13 stays).
    # One full chunk (2 members) pooled -> produced virtual ts=23. Virtuals
    # 0,10,23 with pool window 10: 23-0=23>10 evict 0, 23-10=13>10 evict 1.
    assert plan.evicted_virtual_count == 2
    spans = covered_spans(records, plan)
    # new_records: produced virtual covering old raw rows 2-3, kept raws 4,5.
    assert spans == ((2, 4), (4, 5), (5, 6))
    assert [r.timestamp for r in plan.new_records] == [23.0, 27.0, 40.0]


# --- apply: page-table compaction + pooling ---------------------------------


def test_apply_pools_two_frames_and_compacts_row() -> None:
    records = [_record(t) for t in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)]
    state, req, req_to_token, allocator, pool = _harness(records)
    state.visible_frame_counts = torch.tensor([0, 3, 5], dtype=torch.int32)
    req.multimodal_inputs.visible_frame_counts = torch.tensor([3, 5], dtype=torch.int32)
    # Fill the members' K/V with recognizable per-slot values.
    for layer in range(2):
        for slot in (100, 101):  # frame ts=0
            pool.k_buffer[layer][slot] = 1.0
            pool.v_buffer[layer][slot] = 2.0
        for slot in (102, 103):  # frame ts=10
            pool.k_buffer[layer][slot] = 3.0
            pool.v_buffer[layer][slot] = 4.0

    plan = plan_frame_window(records, _config())
    assert plan is not None
    event = apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: pool,
    )

    assert (event.pooled_raw_frames, event.produced_virtual_frames) == (2, 1)
    assert (event.encoder_length_before, event.encoder_length_after) == (12, 10)
    # Pooled into fresh slots 900-901; rows: virtual, raws 20..50, decoder.
    expected = [900, 901, 104, 105, 106, 107, 108, 109, 110, 111, 50, 60, 70]
    assert req_to_token[0, :13].tolist() == expected
    assert req_to_token[0, 13:].count_nonzero().item() == 0
    assert sorted(allocator.freed) == [100, 101, 102, 103]
    # Pooling math: mean of the two members, per layer, at the dst slots.
    assert torch.allclose(pool.k_buffer[0][900], torch.full((2, 4), 2.0))
    assert torch.allclose(pool.v_buffer[1][901], torch.full((2, 4), 3.0))
    # Metadata resync.
    assert state.encoder_length == 10
    assert state.surviving_frame_count == 5
    assert state.evicted_frame_count == 0
    assert state.full_grid_thw.tolist() == [[1, 2, 2]] * 5
    # Kept-count remap: K = [0,1,1,2,3,4,5]; [0,3,5] -> [0,2,4].
    assert state.visible_frame_counts.tolist() == [0, 2, 4]
    assert req.multimodal_inputs.visible_frame_counts.tolist() == [2, 4]
    assert req.multimodal_inputs.num_image_tokens == 10
    assert req.multimodal_inputs.media_nums_per_sample == [5]
    assert (
        req.multimodal_inputs.mm_items[0]
        .model_specific_data["realtime_full_grid_thw"]
        .tolist()
        == [[1, 2, 2]] * 5
    )
    # The state and request copies of the counts stay distinct objects.
    assert state.visible_frame_counts is not req.multimodal_inputs.visible_frame_counts
    assert req.kv.kv_allocated_len == 13
    assert req.kv_committed_len == 13
    assert [r.pooled for r in state.frame_records] == [True, False, False, False, False]
    assert event.pool_free_slots == allocator.available_size()


@pytest.mark.parametrize("widths", [(2, 5), (5, 2)])
def test_pooling_regression_each_chunk_uses_its_own_width(widths) -> None:
    records = [
        RealtimeFrameRecord(
            float(i), 2 if width == 2 else 4, 2 if width == 2 else 4, width
        )
        for i, width in enumerate([widths[0]] * 2 + [widths[1]] * 2)
    ] + [_record(100.0)]
    state, req, table, allocator, pool = _harness(
        records, allocator_ids=tuple(range(900, 907))
    )
    state.visible_frame_counts = torch.arange(6, dtype=torch.int32)
    for layer, (k, v) in enumerate(zip(pool.k_buffer, pool.v_buffer)):
        values = torch.arange(k.shape[0]).reshape(-1, 1, 1).float()
        k.copy_(values + layer)
        v.copy_(values * 2 + layer)
    before_k = [buf.clone() for buf in pool.k_buffer]
    before_v = [buf.clone() for buf in pool.v_buffer]
    plan = plan_frame_window(records, _config())
    event = apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=table,
        allocator=allocator,
        kv_pool_provider=lambda: pool,
    )
    assert event.produced_virtual_frames == 2
    assert state.encoder_length == 9
    raw_start = 100 + 2 * sum(widths)
    assert table[0, :12].tolist() == list(range(900, 907)) + [
        raw_start,
        raw_start + 1,
        50,
        60,
        70,
    ]
    assert not table[0, 12:].count_nonzero()
    src, dst = 100, 900
    for width in widths:
        for layer in range(2):
            expected_k = (
                before_k[layer][src : src + width]
                + before_k[layer][src + width : src + 2 * width]
            ) / 2
            expected_v = (
                before_v[layer][src : src + width]
                + before_v[layer][src + width : src + 2 * width]
            ) / 2
            torch.testing.assert_close(
                pool.k_buffer[layer][dst : dst + width], expected_k
            )
            torch.testing.assert_close(
                pool.v_buffer[layer][dst : dst + width], expected_v
            )
        src += 2 * width
        dst += width
    assert sorted(allocator.freed) == list(range(100, raw_start))
    assert allocator._free == []
    assert state.visible_frame_counts.tolist() == [0, 1, 1, 2, 2, 3]
    assert state.full_grid_thw.tolist() == [
        list(record.grid_row) for record in plan.new_records
    ]
    assert req.kv_committed_len == req.kv.kv_allocated_len == 12


@pytest.mark.parametrize("provider_raises", [False, True])
def test_pooling_regression_unavailable_provider_does_not_lose_slots(
    provider_raises,
) -> None:
    records = [_record(ts) for ts in (0.0, 1.0, 100.0)]
    state, req, table, allocator, _ = _harness(records)
    before = table.clone()
    initial_free = allocator._free[:]

    def provider():
        if provider_raises:
            raise RuntimeError("pool provider failed")
        return None

    def apply():
        return apply_frame_window_plan(
            req,
            state,
            plan_frame_window(records, _config()),
            records=records,
            req_to_token=table,
            allocator=allocator,
            kv_pool_provider=provider,
        )

    if provider_raises:
        with pytest.raises(RuntimeError, match="pool provider failed"):
            apply()
        assert torch.equal(table, before)
        assert state.frame_records == records
        assert state.encoder_length == 6
        assert req.kv_committed_len == 9
    else:
        event = apply()
        assert event.dropped_raw_frames == 2
        assert event.produced_virtual_frames == 0
        assert table[0, :5].tolist() == [104, 105, 50, 60, 70]
    allocated = set(initial_free) - set(allocator._free)
    assert allocated <= set(allocator.freed)
    assert len(allocator.freed) == len(set(allocator.freed))


@pytest.mark.parametrize("old_virtuals", [0, 2])
@pytest.mark.parametrize("pool_mode", ["available", "missing", "pressure"])
def test_pooling_regression_immediately_expired_virtuals_are_dropped(
    old_virtuals, pool_mode
) -> None:
    records = [
        _record(-20.0 + i * 10, pooled=True, sources=4) for i in range(old_virtuals)
    ]
    records += [
        _record(0.0),
        _record(1.0),
        _record(10.0),
        RealtimeFrameRecord(11.0, 4, 4, 5),
        RealtimeFrameRecord(100.0, 4, 4, 5),
        RealtimeFrameRecord(101.0, 4, 4, 5),
        _record(200.0),
    ]
    state, req, table, allocator, pool = _harness(records)
    before_encoder = state.encoder_length
    state.visible_frame_counts = torch.arange(len(records) + 1, dtype=torch.int32)
    allocator._fail_alloc = pool_mode == "pressure"
    plan = plan_frame_window(records, _config(raw_window_s=10, pool_window_s=10))
    # The first poolable chunk expires immediately; the next is mixed-grid.
    assert plan.raw_chunks == ((0, 2, False), (2, 2, False), (4, 2, True))
    assert plan.evicted_virtual_count == old_virtuals
    assert covered_spans(tuple(records), plan) == (
        (old_virtuals + 4, old_virtuals + 6),
        (old_virtuals + 6, old_virtuals + 7),
    )
    event = apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=table,
        allocator=allocator,
        kv_pool_provider=lambda: None if pool_mode == "missing" else pool,
    )
    pooled = pool_mode == "available"
    raw_start = 100 + before_encoder - 2
    expected_encoder = (list(range(900, 905)) if pooled else []) + [
        raw_start,
        raw_start + 1,
    ]
    assert table[0, : len(expected_encoder) + 3].tolist() == expected_encoder + [
        50,
        60,
        70,
    ]
    assert sorted(allocator.freed) == list(range(100, raw_start))
    assert event.evicted_virtual_frames == old_virtuals
    assert event.produced_virtual_frames == int(pooled)
    assert event.dropped_raw_frames == (4 if pooled else 6)
    assert state.evicted_frame_count == old_virtuals * 4 + (4 if pooled else 6)
    assert state.encoder_length == len(expected_encoder)
    assert req.kv_committed_len == req.kv.kv_allocated_len == len(expected_encoder) + 3
    expected_counts = [0] * (old_virtuals + 5) + ([1, 1, 2] if pooled else [0, 0, 1])
    assert state.visible_frame_counts.tolist() == expected_counts
    assert [record.timestamp for record in state.frame_records] == (
        [101.0, 200.0] if pooled else [200.0]
    )


def test_pooling_regression_partial_write_failure_preserves_committed_layout(
    monkeypatch,
) -> None:
    import sglang_omni.models.moss_vl_realtime.frame_window as window

    records = [
        _record(0.0),
        _record(1.0),
        RealtimeFrameRecord(2.0, 4, 4, 5),
        RealtimeFrameRecord(3.0, 4, 4, 5),
        _record(100.0),
    ]
    state, req, table, allocator, pool = _harness(
        records, allocator_ids=tuple(range(900, 907))
    )
    before = table.clone()
    original = window._pool_group_into_slots
    calls = []

    def fail_second(kv_pool, runs, dst):
        calls.append(dst.numel())
        if len(calls) == 2:
            raise RuntimeError("second group failed")
        original(kv_pool, runs, dst)

    monkeypatch.setattr(window, "_pool_group_into_slots", fail_second)
    with pytest.raises(RuntimeError, match="second group failed"):
        apply_frame_window_plan(
            req,
            state,
            plan_frame_window(records, _config()),
            records=records,
            req_to_token=table,
            allocator=allocator,
            kv_pool_provider=lambda: pool,
        )
    assert calls == [2, 5]
    assert sorted(allocator.freed) == list(range(900, 907))
    assert torch.equal(table, before)
    assert state.frame_records == records
    assert state.encoder_length == 16
    assert req.kv_committed_len == req.kv.kv_allocated_len == 19


def test_pooling_regression_raw_only_never_requests_pool_or_new_slots() -> None:
    records = [_record(0.0), RealtimeFrameRecord(1.0, 4, 4, 5), _record(100.0)]
    state, req, table, allocator, _ = _harness(records)
    allocator.alloc = lambda n: pytest.fail("raw-only must not allocate pooled slots")
    event = apply_frame_window_plan(
        req,
        state,
        plan_frame_window(records, _config(pooling_enabled=False)),
        records=records,
        req_to_token=table,
        allocator=allocator,
        kv_pool_provider=lambda: pytest.fail("raw-only must not request a KV pool"),
    )
    assert event.produced_virtual_frames == 0
    assert table[0, :5].tolist() == [107, 108, 50, 60, 70]
    assert sorted(allocator.freed) == list(range(100, 107))


def test_apply_degrades_to_eviction_when_alloc_fails() -> None:
    records = [_record(t) for t in (0.0, 10.0, 50.0)]
    state, req, req_to_token, allocator, pool = _harness(records)
    allocator._fail_alloc = True

    plan = plan_frame_window(records, _config())
    assert plan is not None
    event = apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: pool,
    )
    assert event.produced_virtual_frames == 0
    assert event.dropped_raw_frames == 2
    assert event.evicted_virtual_frames == 0
    assert event.encoder_length_after == 2
    assert req_to_token[0, :5].tolist() == [104, 105, 50, 60, 70]
    assert sorted(allocator.freed) == [100, 101, 102, 103]
    assert state.evicted_frame_count == 2
    assert state.full_grid_thw.tolist() == [[1, 2, 2]]


def test_apply_evicts_virtuals_and_counts_pooled_sources() -> None:
    records = [
        _record(0.0, pooled=True, sources=4),
        _record(200.0, pooled=True, sources=4),
        _record(300.0),
        _record(310.0),
    ]
    state, req, req_to_token, allocator, pool = _harness(records)
    plan = plan_frame_window(records, _config())  # 200-0 > 120 -> evict oldest
    assert plan is not None
    event = apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: pool,
    )
    assert event.evicted_virtual_frames == 1
    # Evicting the oldest virtual frame counts its 4 original frames.
    assert state.evicted_frame_count == 4
    assert req_to_token[0, :9].tolist() == [102, 103, 104, 105, 106, 107, 50, 60, 70]
    assert sorted(allocator.freed) == [100, 101]
    assert state.frame_records[0].pooled is True


def test_apply_updates_running_batch_encoder_lens() -> None:
    records = [_record(t) for t in (0.0, 10.0, 50.0)]
    state, req, req_to_token, allocator, pool = _harness(records)
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="other"), req],
        encoder_lens_cpu=[7, 6],
        encoder_lens=torch.tensor([7, 6]),
    )
    plan = plan_frame_window(records, _config())
    apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: pool,
        running_batch=batch,
    )
    assert batch.encoder_lens_cpu == [7, 4]
    assert batch.encoder_lens.tolist() == [7, 4]


def test_apply_rejects_mismatched_record_spans() -> None:
    records = [_record(0.0, slots=3), _record(1.0, slots=3), _record(50.0)]
    state, req, req_to_token, allocator, pool = _harness(
        [_record(t) for t in (0.0, 1.0, 50.0)]
    )
    plan = plan_frame_window(records, _config())
    assert plan is not None
    with pytest.raises(RuntimeError, match="disagree with the encoder"):
        apply_frame_window_plan(
            req,
            state,
            plan,
            records=records,
            req_to_token=req_to_token,
            allocator=allocator,
            kv_pool_provider=lambda: pool,
        )


def test_apply_validates_plan_coverage_before_rewriting_row() -> None:
    """A plan/record coverage mismatch must abort before any page-row mutation.

    The applied records (two 3-slot frames) and the planned records (three
    2-slot frames) agree on the slot sum, so only the span-coverage check can
    catch the mismatch. Past the fix, that check runs before the rewrite: the
    committed row and the bookkeeping must be left exactly as they were, with
    only the freshly allocated pooled slots returned to the allocator.
    """
    records_apply = [_record(0.0, slots=3), _record(1.0, slots=3)]
    state, req, req_to_token, allocator, pool = _harness(records_apply)
    plan = plan_frame_window([_record(0.0), _record(10.0), _record(50.0)], _config())
    assert plan is not None
    row_before = req_to_token.clone()

    with pytest.raises(RuntimeError, match="does not cover"):
        apply_frame_window_plan(
            req,
            state,
            plan,
            records=records_apply,
            req_to_token=req_to_token,
            allocator=allocator,
            kv_pool_provider=lambda: pool,
        )

    assert torch.equal(req_to_token, row_before)
    # Only the freshly allocated dst slots (3-wide pooled frame) come back.
    assert sorted(allocator.freed) == [900, 901, 902]
    assert state.encoder_length == 6
    assert [r.timestamp for r in state.frame_records] == [0.0, 1.0]
    assert state.surviving_frame_count == 2
    assert req.kv_committed_len == 9


def test_apply_supports_consecutive_eviction_rounds() -> None:
    """A second plan/apply round must round-trip on the compacted row."""
    from sglang_omni.models.moss_vl_realtime.kv_layout import insert_encoder_slots

    records = [_record(t) for t in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)]
    state, req, req_to_token, allocator, pool = _harness(records)
    cfg = _config(raw_window_s=25.0)

    # Round 1: (0,10) pool into V1; frame 20 ages but waits for its ratio peer.
    plan1 = plan_frame_window(records, cfg)
    assert plan1 is not None
    event1 = apply_frame_window_plan(
        req,
        state,
        plan1,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: pool,
    )
    assert (event1.pooled_raw_frames, event1.produced_virtual_frames) == (2, 1)
    assert state.encoder_length == 10  # [V1][20][30][40][50]

    # Emulate the next segment commit: one new 2-slot frame lands in the
    # encoder region (inserted ahead of the decoder tail).
    insert_encoder_slots(
        req_to_token,
        req_pool_index=0,
        encoder_length=state.encoder_length,
        decoder_length=state.decoder_length,
        new_slots=[950, 951],
    )
    state.encoder_length += 2
    state.frame_records.append(_record(60.0))
    req.kv_committed_len += 2
    req.kv.kv_allocated_len += 2

    # Round 2 on the compacted row: (20,30) pool into V2.
    plan2 = plan_frame_window(tuple(state.frame_records), cfg)
    assert plan2 is not None
    event2 = apply_frame_window_plan(
        req,
        state,
        plan2,
        records=tuple(state.frame_records),
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: pool,
    )

    assert (event2.pooled_raw_frames, event2.produced_virtual_frames) == (2, 1)
    assert event2.evicted_virtual_frames == 0
    assert req_to_token[0, :13].tolist() == [
        900,
        901,
        902,
        903,
        108,
        109,
        110,
        111,
        950,
        951,
        50,
        60,
        70,
    ]
    assert req_to_token[0, 13:].count_nonzero().item() == 0
    assert sorted(allocator.freed) == list(range(100, 108))
    assert state.encoder_length == 10
    assert state.surviving_frame_count == 5
    assert state.evicted_frame_count == 0
    assert [r.pooled for r in state.frame_records] == [True, True, False, False, False]
    assert state.full_grid_thw.tolist() == [[1, 2, 2]] * 5
    assert req.kv_committed_len == 13
    assert req.kv.kv_allocated_len == 13
    assert req.multimodal_inputs.num_image_tokens == 10


# --- scheduler hook ---------------------------------------------------------


def _hook_scheduler(records, config):
    from sglang_omni.models.moss_vl_realtime.scheduler import MossVLRealtimeScheduler

    state, req, req_to_token, allocator, pool = _harness(records)
    req._moss_vl_realtime_state = state
    req.finished = lambda: False
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.frame_window_config = config
    scheduler.running_batch = SimpleNamespace(reqs=[req])
    scheduler.parked_reqs = {}
    scheduler._realtime_extend_batch = None
    scheduler._async_pending = None
    scheduler.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler._token_to_kv_pool = lambda: pool
    return scheduler, state, req, allocator


def test_scheduler_hook_noop_when_window_disabled() -> None:
    records = [_record(t) for t in (0.0, 10.0, 50.0)]
    scheduler, state, _req, allocator = _hook_scheduler(records, None)
    scheduler._evaluate_frame_window()
    assert allocator.freed == []
    assert state.encoder_length == 6


def test_scheduler_hook_evicts_during_decode_gap() -> None:
    records = [_record(t) for t in (0.0, 10.0, 50.0)]
    scheduler, state, _req, allocator = _hook_scheduler(records, _config())
    scheduler._evaluate_frame_window()
    assert state.encoder_length == 4  # 0 and 10 pooled into one virtual frame
    assert sorted(allocator.freed) == [100, 101, 102, 103]


def test_scheduler_hook_skips_mid_transaction() -> None:
    records = [_record(t) for t in (0.0, 10.0, 50.0)]
    scheduler, state, _req, allocator = _hook_scheduler(records, _config())
    state._append_inflight = True
    scheduler._evaluate_frame_window()
    assert allocator.freed == []
    assert state.encoder_length == 6


def test_undo_appended_segment_clears_staged_frame_records() -> None:
    """Rollback of a staged segment must not leak staged frame records."""
    from sglang_omni.models.moss_vl_realtime.scheduler import _undo_appended_segment

    state = MossVLRealtimeRuntimeState(request_id="req-fw", session_id="session-fw")
    req = SimpleNamespace(
        output_ids=[11, 12, 13],
        sampling_params=SimpleNamespace(max_new_tokens=8),
        multimodal_inputs="staged-mm",
        extend_range=(2, 3),
        prefix_indices=[7],
        skip_radix_cache_insert=True,
        _refresh_fill_ids=lambda: None,
        _moss_vl_realtime_previous_mm_inputs="prev-mm",
        _moss_vl_realtime_previous_extend_range=(0, 2),
        _moss_vl_realtime_previous_prefix_indices=[5],
        _moss_vl_realtime_previous_skip_radix_cache_insert=False,
        _moss_vl_realtime_previous_max_new_tokens=7,
        _moss_vl_realtime_staged_mrope_positions=torch.zeros(3, 3),
        _moss_vl_realtime_staged_visible_frame_counts=torch.zeros(3),
        _moss_vl_realtime_staged_full_grid_thw=torch.zeros(1, 3),
        _moss_vl_realtime_staged_events=[],
    )
    req._moss_vl_realtime_state = state
    setattr(req, FRAME_RECORDS_STAGED_ATTR, [_record(0.0)])
    segment = SimpleNamespace(raw_append_ids=[13])

    _undo_appended_segment(req, segment)

    assert not hasattr(req, FRAME_RECORDS_STAGED_ATTR)
    assert req.output_ids == [11, 12]
    assert req.sampling_params.max_new_tokens == 7
    assert req.multimodal_inputs == "prev-mm"
    assert req.extend_range == (0, 2)
    assert req.prefix_indices == [5]
    assert req.skip_radix_cache_insert is False
    assert state.pending_token_id is None


# --- model-runner record consumption --------------------------------------


def test_post_prefill_consumes_staged_frame_records() -> None:
    from sglang_omni.models.moss_vl_realtime.model_runner import (
        MossVLRealtimeModelRunner,
    )

    state = MossVLRealtimeRuntimeState(
        request_id="req-fw",
        session_id="session-fw",
        req_pool_index=0,
    )
    req = SimpleNamespace()
    req._moss_vl_realtime_state = state
    records = [_record(0.0)]
    setattr(req, FRAME_RECORDS_STAGED_ATTR, records)
    batch = SimpleNamespace(reqs=[req])
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_prefill(None, None, batch, [])

    assert state.frame_records == records
    assert not hasattr(req, FRAME_RECORDS_STAGED_ATTR)


def test_post_prefill_without_records_is_unchanged() -> None:
    from sglang_omni.models.moss_vl_realtime.model_runner import (
        MossVLRealtimeModelRunner,
    )

    state = MossVLRealtimeRuntimeState(
        request_id="req-fw",
        session_id="session-fw",
        req_pool_index=0,
    )
    req = SimpleNamespace()
    req._moss_vl_realtime_state = state
    batch = SimpleNamespace(reqs=[req])
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_prefill(None, None, batch, [])

    assert state.frame_records is None


# --- regression: extend materialization after a virtual-frame eviction ------


def _post_eviction_harness():
    """State as if 3 frames were appended and one virtual frame evicted.

    Token space keeps the evicted frame's pad placeholders
    (appended_encoder_length=6) while the KV encoder region shrank to 4 slots.
    """
    from array import array
    from collections import namedtuple

    Range = namedtuple("Range", ["start", "end"])
    Range.length = property(lambda self: self.end - self.start)

    records = [
        _record(0.0, pooled=True, sources=2),
        _record(3.0, pooled=True, sources=2),
        _record(10.0),
    ]
    state = _enabled_state(records)
    state.decoder_length = 2
    state.appended_encoder_length = 6
    req_to_token = torch.zeros((1, 32), dtype=torch.int64)
    req_to_token[0, :8] = torch.tensor([11, 12, 13, 14, 15, 16, 21, 22])
    allocator = _FakeAllocator([])
    req = SimpleNamespace(
        rid="req-fw",
        kv=SimpleNamespace(kv_allocated_len=8),
        kv_committed_len=8,
        multimodal_inputs=SimpleNamespace(
            visible_frame_counts=None,
            num_image_tokens=6,
            media_nums_per_sample=[3],
            mm_items=[
                SimpleNamespace(model_specific_data={"realtime_full_grid_thw": None})
            ],
        ),
    )
    req._moss_vl_realtime_state = state
    req._moss_vl_realtime_page_row = req_to_token[0]

    # Fill ids: origin tokens + token-space history + one pending sampled id.
    req.origin_input_ids = array("q", [101])
    token_history = array(
        "q", [-101] * 6  # two slots per historical frame (incl. evicted)
    )
    req.output_ids = token_history + array("q", [5, 999])  # 1 text + pending
    req.sampling_params = SimpleNamespace(max_new_tokens=16)
    req.skip_radix_cache_insert = False
    req.prefix_indices = torch.tensor([11, 12, 13, 14, 15, 16, 21, 22])
    req.extend_range = Range(0, 9)

    def _refresh():
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids

    req._refresh_fill_ids = _refresh
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", Range(start, end)
    )
    _refresh()
    return state, req, req_to_token, allocator, records, Range


def test_extend_materialization_after_virtual_eviction() -> None:
    """Dropping a virtual frame must not break the token-space invariants the
    next realtime extend relies on (previously raised "must coalesce with
    exactly one pending sampled token")."""
    from sglang_omni.models.moss_vl_realtime.scheduler import _append_segment_to_request

    state, req, req_to_token, allocator, records, Range = _post_eviction_harness()
    assert len(req.full_untruncated_fill_ids) == 9  # 1 origin + 6 pads + 2 + pending

    # pool window 2s: newest virtual ts=3, oldest ts=0 trails by >2 -> evicted.
    plan = plan_frame_window(records, _config(raw_window_s=100.0, pool_window_s=2.0))
    assert plan is not None and plan.evicted_virtual_count == 1
    event = apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: None,
    )
    assert event.encoder_length_after == 4
    assert allocator.freed == [11, 12]
    assert state.encoder_length == 4
    assert state.effective_appended_encoder_length == 6

    # The next drain materializes on top of the compacted row without raising.
    event_frame = FramePromptEvent(
        request_id="req-fw",
        session_id="session-fw",
        seq_no=3,
        timestamp=11.0,
        frame_ref="x://c",
    )
    segment = SimpleNamespace(
        events=(event_frame,),
        raw_append_ids=(-101, -101, 301),  # 2 encoder pads + 1 text token
        full_grid_thw=torch.tensor([[1, 2, 2]] * 3),
        multimodal_inputs=SimpleNamespace(
            mrope_positions=torch.tensor([[0, 1, 7], [0, 1, 7], [0, 1, 7]]),
            visible_frame_counts=torch.tensor([0, 1, 2], dtype=torch.int32),
        ),
    )
    _append_segment_to_request(req, state, segment)

    # Token space: 6 pads + 2 text committed; extend covers pending + appends.
    assert req.extend_range == Range(8, 12)
    assert len(req.prefix_indices) == 8
    assert req.prefix_indices[:6].tolist() == [13, 14, 15, 16, 21, 22]
    assert state.pending_token_id == 999
    # Committed KV region is untouched by the token-space bookkeeping.
    assert req_to_token[0, :6].tolist() == [13, 14, 15, 16, 21, 22]


def test_build_segment_pending_check_uses_token_space() -> None:
    """_build_segment must count the pending token against token history, not
    the compacted KV encoder length."""
    from sglang_omni.models.moss_vl_realtime.scheduler import (
        MossVLRealtimeScheduler,
        _guard_realtime_context_capacity,
    )

    state, req, req_to_token, allocator, records, _ = _post_eviction_harness()
    plan = plan_frame_window(records, _config(raw_window_s=100.0, pool_window_s=2.0))
    apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: None,
    )

    captured: dict = {}

    class _FakeBuilder:
        def build(self, events, images, **common):
            captured.update(common)
            return "segment"

    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.segment_builder = _FakeBuilder()
    scheduler._resolve_frame_events_tp = lambda events: [object()] * len(events)

    events = [
        FramePromptEvent(
            request_id="req-fw",
            session_id="session-fw",
            seq_no=3,
            timestamp=11.0,
            frame_ref="x://c",
        )
    ]
    assert scheduler._build_segment(req, state, events) == "segment"
    # KV-space values still flow to the segment builder for mask sizing.
    assert captured["committed_encoder_length"] == 4
    assert captured["committed_decoder_length"] == 2

    # Eight historical positions + three appends + pending input + sampled output.
    pool_ns = SimpleNamespace(req_to_token=torch.zeros((1, 12)))
    with pytest.raises(RuntimeError, match="context length"):
        _guard_realtime_context_capacity(
            req,
            state,
            SimpleNamespace(raw_append_ids=(-101, -101, 301)),
            pool_ns,
        )
    _guard_realtime_context_capacity(
        req,
        state,
        SimpleNamespace(raw_append_ids=(-101, -101, 301)),
        SimpleNamespace(req_to_token=torch.zeros((1, 13))),
    )


def test_guard_without_eviction_uses_kv_space_as_before() -> None:
    state = _enabled_state([_record(0.0), _record(1.0)])
    state.decoder_length = 2
    from sglang_omni.models.moss_vl_realtime.scheduler import (
        _guard_realtime_context_capacity,
    )

    # token == KV: 4+2+3+1 pending input+1 sampled output needs 11 positions.
    segment = SimpleNamespace(raw_append_ids=(-101, -101, 301))
    _guard_realtime_context_capacity(
        SimpleNamespace(),
        state,
        segment,
        SimpleNamespace(req_to_token=torch.zeros((1, 11))),
    )
    with pytest.raises(RuntimeError, match="context length"):
        _guard_realtime_context_capacity(
            SimpleNamespace(),
            state,
            segment,
            SimpleNamespace(req_to_token=torch.zeros((1, 10))),
        )
