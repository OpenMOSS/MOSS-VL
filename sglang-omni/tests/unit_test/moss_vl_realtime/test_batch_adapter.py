from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace

import torch

from sglang_omni.models.moss_vl_realtime import (
    KV_TRANSACTION_ATTR,
    RUNTIME_STATE_ATTR,
    MossVLRealtimePhase,
    MossVLRealtimeRuntimeState,
    commit_moss_vl_realtime_batch,
    prepare_moss_vl_realtime_encoder_info_extend,
    rollback_moss_vl_realtime_batch,
)

Range = namedtuple("Range", ["start", "end"])
Range.length = property(lambda self: self.end - self.start)


class _Allocator:
    def __init__(self) -> None:
        self.released: list[torch.Tensor] = []

    def free(self, slots: torch.Tensor) -> None:
        self.released.append(slots.clone())


def _batch():
    state = MossVLRealtimeRuntimeState(
        request_id="req-1",
        session_id="session-1",
        req_pool_index=1,
        encoder_length=2,
        decoder_length=2,
        surviving_frame_count=1,
        next_mrope_position=7,
        pending_token_id=999,
    )
    req = SimpleNamespace(
        rid="req-1",
        req_pool_idx=1,
        prefix_indices=torch.tensor([11, 12, 21, 22]),
        extend_range=Range(4, 9),
        logprob_start_len=0,
        multimodal_inputs=SimpleNamespace(
            num_image_tokens=4,
            mrope_positions=torch.tensor(
                [[0, 1, 7, 8, 9], [0, 1, 7, 8, 9], [0, 1, 7, 8, 9]]
            ),
            mm_items=[
                SimpleNamespace(
                    model_specific_data={"realtime_added_frames": 1},
                ),
            ],
        ),
        kv_committed_len=4,
        kv=SimpleNamespace(kv_allocated_len=4),
    )
    setattr(req, RUNTIME_STATE_ATTR, state)
    req_to_token = torch.zeros((2, 16), dtype=torch.int64)
    req_to_token[1, :9] = torch.tensor([11, 12, 21, 22, 23, 13, 14, 24, 25])
    batch = SimpleNamespace(
        reqs=[req],
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        out_cache_loc=torch.tensor([23, 13, 14, 24, 25]),
        return_logprob=False,
        device=torch.device("cpu"),
        token_to_kv_pool_allocator=_Allocator(),
    )
    return batch, req, state


def test_realtime_batch_adapter_splits_delta_encoder_and_relayouts_row() -> None:
    batch, req, state = _batch()
    raw_input_ids = [[999, -101, -101, 101, 102]]

    prepare_moss_vl_realtime_encoder_info_extend(batch, raw_input_ids, [9])

    assert batch.encoder_lens_cpu == [4]
    assert batch.encoder_cached == [False]
    assert batch.encoder_out_cache_loc.tolist() == [13, 14]
    assert batch.out_cache_loc.tolist() == [23, 24, 25]
    assert batch.prefix_lens == [2]
    assert batch.extend_lens == [3]
    assert batch.seq_lens_cpu.tolist() == [5]
    assert batch.prefill_input_ids_cpu.tolist() == [999, 101, 102]
    assert batch.req_to_token_pool.req_to_token[1, :9].tolist() == [
        11,
        12,
        13,
        14,
        21,
        22,
        23,
        24,
        25,
    ]
    assert req.extend_range == Range(6, 9)
    assert state.phase is MossVLRealtimePhase.EXTENDING
    assert hasattr(req, KV_TRANSACTION_ATTR)

    commit_moss_vl_realtime_batch(batch)
    assert state.encoder_length == 4
    assert state.decoder_length == 5
    assert state.surviving_frame_count == 2
    assert state.next_mrope_position == 10
    assert state.phase is MossVLRealtimePhase.DECODING
    assert not hasattr(req, KV_TRANSACTION_ATTR)
    assert batch.token_to_kv_pool_allocator.released == []


def test_realtime_batch_adapter_rollback_restores_committed_layout() -> None:
    batch, req, state = _batch()

    prepare_moss_vl_realtime_encoder_info_extend(
        batch,
        [[999, -101, -101, 101, 102]],
        [9],
    )
    rollback_moss_vl_realtime_batch(batch)

    assert batch.req_to_token_pool.req_to_token[1, :4].tolist() == [11, 12, 21, 22]
    assert batch.req_to_token_pool.req_to_token[1, 4:].count_nonzero().item() == 0
    assert state.encoder_length == 2
    assert state.decoder_length == 2
    assert state.phase is MossVLRealtimePhase.WAITING_FOR_EVENT
    assert not hasattr(req, KV_TRANSACTION_ATTR)
    assert req.kv_committed_len == 4
    assert req.kv.kv_allocated_len == 4
    assert len(batch.token_to_kv_pool_allocator.released) == 1
    assert batch.token_to_kv_pool_allocator.released[0].tolist() == [
        23,
        13,
        14,
        24,
        25,
    ]

    rollback_moss_vl_realtime_batch(batch)
    assert len(batch.token_to_kv_pool_allocator.released) == 1


def test_initial_text_prefill_advances_mrope_without_multimodal_input() -> None:
    state = MossVLRealtimeRuntimeState(
        request_id="req-text",
        session_id="session-text",
        req_pool_index=0,
    )
    req = SimpleNamespace(
        rid="req-text",
        req_pool_idx=0,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        extend_range=Range(0, 3),
        logprob_start_len=0,
        multimodal_inputs=None,
        kv_committed_len=3,
        kv=SimpleNamespace(kv_allocated_len=3),
    )
    setattr(req, RUNTIME_STATE_ATTR, state)
    table = torch.zeros((1, 8), dtype=torch.int64)
    table[0, :3] = torch.tensor([31, 32, 33])
    batch = SimpleNamespace(
        reqs=[req],
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        out_cache_loc=torch.tensor([31, 32, 33]),
        return_logprob=False,
        device=torch.device("cpu"),
        token_to_kv_pool_allocator=_Allocator(),
    )

    prepare_moss_vl_realtime_encoder_info_extend(batch, [[101, 102, 103]], [3])
    commit_moss_vl_realtime_batch(batch)

    assert state.encoder_length == 0
    assert state.decoder_length == 3
    assert state.next_mrope_position == 3


def test_initial_text_prefill_adopts_radix_cache_hit() -> None:
    state = MossVLRealtimeRuntimeState(
        request_id="req-cached",
        session_id="session-cached",
        req_pool_index=0,
    )
    req = SimpleNamespace(
        rid="req-cached",
        req_pool_idx=0,
        prefix_indices=torch.tensor([41, 42], dtype=torch.int64),
        extend_range=Range(2, 3),
        logprob_start_len=0,
        multimodal_inputs=None,
        kv_committed_len=3,
        kv=SimpleNamespace(kv_allocated_len=3),
    )
    setattr(req, RUNTIME_STATE_ATTR, state)
    table = torch.zeros((1, 8), dtype=torch.int64)
    table[0, :3] = torch.tensor([41, 42, 43])
    batch = SimpleNamespace(
        reqs=[req],
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        out_cache_loc=torch.tensor([43]),
        return_logprob=False,
        device=torch.device("cpu"),
        token_to_kv_pool_allocator=_Allocator(),
    )

    prepare_moss_vl_realtime_encoder_info_extend(batch, [[103]], [3])

    assert batch.prefix_lens == [2]
    assert batch.extend_lens == [1]
    assert state.decoder_length == 2
    assert state.next_mrope_position == 2
    commit_moss_vl_realtime_batch(batch)
    assert state.decoder_length == 3
    assert state.next_mrope_position == 3


def test_rollback_noop_after_successful_commit() -> None:
    """A failure after commit must not free slots the request now owns."""
    batch, req, state = _batch()
    prepare_moss_vl_realtime_encoder_info_extend(
        batch,
        [[999, -101, -101, 101, 102]],
        [9],
    )
    commit_moss_vl_realtime_batch(batch)
    assert not hasattr(req, KV_TRANSACTION_ATTR)

    rollback_moss_vl_realtime_batch(batch)

    allocator = batch.token_to_kv_pool_allocator
    assert allocator.released == []
    # Bookkeeping restore converges on the committed lengths; not a truncation.
    assert req.kv_committed_len == 9
    assert req.kv.kv_allocated_len == 9


def test_rollback_frees_current_step_slots_without_any_transaction() -> None:
    """Decode-step failure: no extend transaction exists, so the batch's fresh
    step slots are still owned by this attempt and must be freed here."""
    batch, req, state = _batch()
    assert not hasattr(req, KV_TRANSACTION_ATTR)

    rollback_moss_vl_realtime_batch(batch)

    allocator = batch.token_to_kv_pool_allocator
    assert len(allocator.released) == 1
    assert allocator.released[0].tolist() == [23, 13, 14, 24, 25]
    assert req.kv_committed_len == 4
    assert req.kv.kv_allocated_len == 4
