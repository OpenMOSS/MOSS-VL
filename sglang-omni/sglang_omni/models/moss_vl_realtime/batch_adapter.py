"""Adapt an allocated SGLang extend batch for incremental vision KV."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.utils.common import is_pin_memory_available

from sglang_omni.models.moss_vl_realtime.runtime_state import (
    MossVLRealtimeKVAppendTransaction,
    MossVLRealtimeRuntimeState,
)
from sglang_omni.models.moss_vl_realtime.segment import REALTIME_ADDED_FRAMES_KEY

RUNTIME_STATE_ATTR = "_moss_vl_realtime_state"
KV_TRANSACTION_ATTR = "_moss_vl_realtime_kv_transaction"
_ALLOCATED_SLOTS_ATTR = "_moss_vl_realtime_allocated_slots"
_ALLOCATION_RELEASED_ATTR = "_moss_vl_realtime_allocation_released"
_ALLOCATION_COMMITTED_ATTR = "_moss_vl_realtime_allocation_committed"
_DECODE_ALLOCATION_ATTR = "_moss_vl_realtime_decode_allocation"


@dataclass(slots=True)
class _DecodeAllocation:
    reqs: tuple[Any, ...]
    positions: tuple[int, ...]
    committed_lengths: tuple[int, ...]
    pool_indices: tuple[int, ...]
    slots: torch.Tensor | None = None
    committed: bool = False
    released: bool = False


class MossVLRealtimeScheduleBatch(ScheduleBatch):
    """ScheduleBatch with request-local incremental vision preparation."""

    @classmethod
    def from_batch(cls, batch: ScheduleBatch) -> MossVLRealtimeScheduleBatch:
        if isinstance(batch, cls):
            return batch
        # Initial text prefill comes from the upstream planner. Preserve all
        # its tensors and dynamic metadata when installing realtime hooks.
        converted = cls(reqs=batch.reqs)
        converted.__dict__.update(vars(batch))
        return converted

    def copy(self) -> ScheduleBatch:
        snapshot = super().copy()
        snapshot.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
        record = getattr(self, _DECODE_ALLOCATION_ATTR, None)
        if record is not None:
            setattr(snapshot, _DECODE_ALLOCATION_ATTR, record)
        for flag in (_ALLOCATION_COMMITTED_ATTR, _ALLOCATION_RELEASED_ATTR):
            if getattr(self, flag, False):
                setattr(snapshot, flag, True)
        return snapshot

    def prepare_encoder_info_extend(
        self,
        input_ids: list[Sequence[int]],
        seq_lens: list[int],
    ) -> None:
        prepare_moss_vl_realtime_encoder_info_extend(self, input_ids, seq_lens)

    def prepare_for_extend(self) -> None:
        _begin_allocation(self)
        try:
            super().prepare_for_extend()
        except Exception:
            rollback_moss_vl_realtime_batch(self)
            raise

    def prepare_for_decode(self) -> None:
        for req in self.reqs:
            error = realtime_decode_capacity_error(req, self.req_to_token_pool)
            if error is not None:
                from sglang_omni.models.moss_vl_realtime.accounting import (
                    ContextExhaustedError,
                )

                raise ContextExhaustedError(error)
        record = _DecodeAllocation(
            reqs=tuple(self.reqs),
            positions=tuple(int(req.kv.kv_allocated_len) for req in self.reqs),
            committed_lengths=tuple(int(req.kv_committed_len) for req in self.reqs),
            pool_indices=tuple(int(req.req_pool_idx) for req in self.reqs),
        )
        _begin_allocation(self)
        setattr(self, _DECODE_ALLOCATION_ATTR, record)
        try:
            super().prepare_for_decode()
        except Exception:
            record.slots = self.out_cache_loc
            rollback_moss_vl_realtime_batch(self)
            raise
        record.slots = self.out_cache_loc


def _begin_allocation(batch: Any) -> None:
    batch.out_cache_loc = None
    setattr(batch, _ALLOCATED_SLOTS_ATTR, None)
    setattr(batch, _ALLOCATION_COMMITTED_ATTR, False)
    setattr(batch, _ALLOCATION_RELEASED_ATTR, False)
    if hasattr(batch, _DECODE_ALLOCATION_ATTR):
        delattr(batch, _DECODE_ALLOCATION_ATTR)


def realtime_decode_capacity_error(req: Any, pool: Any) -> str | None:
    state = getattr(req, RUNTIME_STATE_ATTR, None)
    table = getattr(pool, "req_to_token", None)
    if not isinstance(state, MossVLRealtimeRuntimeState) or table is None:
        return None
    width = int(table.shape[1])
    limit = state.context_limit or width
    committed = state.encoder_length + state.decoder_length
    allocated = max(
        committed, int(getattr(getattr(req, "kv", None), "kv_allocated_len", committed))
    )
    # Allocated but unresolved lookahead tokens count too. The next forward
    # consumes one pending token and samples one more logical position.
    logical = state.effective_appended_encoder_length + allocated - state.encoder_length
    if allocated + 1 > width or logical + 2 > limit:
        return (
            "realtime request exhausted the context length before decode: "
            f"projected token positions {logical + 2}/{limit}, "
            f"KV positions {allocated + 1}/{width}"
        )
    return None


def commit_moss_vl_realtime_decode(batch: Any) -> None:
    record = getattr(batch, _DECODE_ALLOCATION_ATTR, None)
    if record is not None:
        record.committed = True


def _rollback_decode(batch: Any, record: _DecodeAllocation) -> None:
    if record.committed or record.released:
        return
    if record.slots is None:
        record.released = True
        return
    active = {id(req) for req in batch.reqs}
    indices = []
    for i, req in enumerate(record.reqs):
        if (
            id(req) not in active
            or req.req_pool_idx != record.pool_indices[i]
            or req.kv is None
        ):
            continue
        state = getattr(req, RUNTIME_STATE_ATTR)
        if state.encoder_length + state.decoder_length > record.positions[i]:
            continue
        if int(req.kv.kv_allocated_len) > record.positions[i] + 1:
            raise RuntimeError(
                "resolve newer decode allocations before rolling back this step"
            )
        indices.append(i)
    if indices:
        batch.token_to_kv_pool_allocator.free(record.slots[indices])
        for i in indices:
            req = record.reqs[i]
            req.kv.kv_allocated_len = record.positions[i]
            req.kv_committed_len = record.committed_lengths[i]
            batch.req_to_token_pool.req_to_token[
                record.pool_indices[i], record.positions[i]
            ] = 0
    record.released = True


@dataclass(slots=True)
class _RealtimeExtendPlan:
    req: Any
    state: MossVLRealtimeRuntimeState
    raw_input_ids: Sequence[int]
    new_encoder_slots: torch.Tensor
    new_decoder_slots: torch.Tensor
    total_encoder_length: int
    total_decoder_length: int
    decoder_extend_length: int
    leading_decoder_length: int


def is_moss_vl_realtime_batch(batch: Any) -> bool:
    reqs = getattr(batch, "reqs", ())
    return bool(reqs) and all(
        isinstance(getattr(req, RUNTIME_STATE_ATTR, None), MossVLRealtimeRuntimeState)
        for req in reqs
    )


def prepare_moss_vl_realtime_encoder_info_extend(
    batch: Any,
    input_ids: list[Sequence[int]],
    seq_lens: list[int],
) -> None:
    """Replace SGLang's all-or-nothing encoder handling for delta frames.

    ``alloc_for_extend`` has already allocated and appended fresh physical KV
    slots. This function splits those slots into delta-encoder and text parts,
    stages the request page-table transaction, and exposes decoder-only query
    lengths to the language model while retaining total encoder lengths for
    cross-attention.
    """
    if not is_moss_vl_realtime_batch(batch):
        raise TypeError("batch does not contain only MOSS-VL realtime requests")
    if len(input_ids) != len(batch.reqs) or len(seq_lens) != len(batch.reqs):
        raise ValueError("input_ids and seq_lens must be request aligned")
    if getattr(batch, "return_logprob", False):
        raise ValueError("MOSS-VL realtime does not support prompt logprobs")

    setattr(batch, _ALLOCATED_SLOTS_ATTR, batch.out_cache_loc.clone())
    setattr(batch, _ALLOCATION_RELEASED_ATTR, False)
    setattr(batch, _ALLOCATION_COMMITTED_ATTR, False)
    if hasattr(batch, _DECODE_ALLOCATION_ATTR):
        delattr(batch, _DECODE_ALLOCATION_ATTR)
    transactions: list[MossVLRealtimeKVAppendTransaction] = []
    try:
        plans = _build_plans(batch, input_ids, seq_lens)
        for plan in plans:
            transaction = plan.state.begin_kv_append(
                batch.req_to_token_pool.req_to_token,
                new_encoder_slots=plan.new_encoder_slots,
                new_decoder_slots=plan.new_decoder_slots,
                added_frames=_added_frames(plan),
                next_mrope_position=_next_mrope_position(plan),
                # Slot release on rollback is handled by the batch-level
                # _release_allocated_slots; per-transaction release_slots is
                # reserved for callers that allocate outside the batch path.
                release_slots=None,
            )
            setattr(plan.req, KV_TRANSACTION_ATTR, transaction)
            transactions.append(transaction)
    except Exception:
        rollback_moss_vl_realtime_batch(batch)
        raise

    _install_batch_metadata(batch, plans)


def commit_moss_vl_realtime_batch(batch: Any) -> None:
    for req in batch.reqs:
        transaction = getattr(req, KV_TRANSACTION_ATTR, None)
        if transaction is None:
            continue
        transaction.commit()
        delattr(req, KV_TRANSACTION_ATTR)
        _reconcile_committed_kv_bookkeeping(batch, req)
    setattr(batch, _ALLOCATION_COMMITTED_ATTR, True)
    _clear_allocation_rollback_metadata(batch)


def _reconcile_committed_kv_bookkeeping(batch: Any, req: Any) -> None:
    """Restore KV-space accounting after upstream's token-space bookkeeping.

    ``prepare_for_extend`` records ``kv_committed_len``/``kv.kv_allocated_len``
    in token space (extend_range.end). Once the frame window has evicted
    frames, token space includes pad placeholders of dropped frames and leads
    the live KV layout; release would read duplicate slot ids out of the row's
    dead zone. Resync to the committed KV layout and zero the dead zone so no
    path can observe it. Without eviction the two spaces coincide and this is
    a value-preserving no-op.
    """
    state = getattr(req, RUNTIME_STATE_ATTR, None)
    if not isinstance(state, MossVLRealtimeRuntimeState):
        return
    kv_total = state.encoder_length + state.decoder_length
    token_total = state.effective_appended_encoder_length + state.decoder_length
    kv = getattr(req, "kv", None)
    if kv is not None:
        kv.kv_allocated_len = kv_total
    if hasattr(req, "kv_committed_len"):
        req.kv_committed_len = kv_total
    if token_total > kv_total and state.req_pool_index is not None:
        batch.req_to_token_pool.req_to_token[
            state.req_pool_index, kv_total:token_total
        ] = 0


def rollback_moss_vl_realtime_batch(batch: Any) -> None:
    record = getattr(batch, _DECODE_ALLOCATION_ATTR, None)
    if record is not None:
        _rollback_decode(batch, record)
        return
    if getattr(batch, _ALLOCATION_RELEASED_ATTR, False):
        return
    if getattr(batch, _ALLOCATION_COMMITTED_ATTR, False):
        # A late error after commit belongs to request teardown, not rollback.
        return
    has_active_transaction = False
    for req in reversed(batch.reqs):
        transaction = getattr(req, KV_TRANSACTION_ATTR, None)
        if transaction is None:
            continue
        has_active_transaction = True
        transaction.rollback()
        delattr(req, KV_TRANSACTION_ATTR)
    _restore_request_kv_bookkeeping(batch)
    if has_active_transaction:
        _release_allocated_slots(batch)
        return
    # No transaction ever staged: a decode-step failure. Restore bookkeeping
    # (done above) and free this step's freshly allocated slots.
    _release_allocated_slots(batch)


def _restore_request_kv_bookkeeping(batch: Any) -> None:
    table = batch.req_to_token_pool.req_to_token
    for req in batch.reqs:
        state = getattr(req, RUNTIME_STATE_ATTR, None)
        if not isinstance(state, MossVLRealtimeRuntimeState):
            continue
        committed_length = state.encoder_length + state.decoder_length
        allocated_length = committed_length
        if getattr(req, "kv", None) is not None:
            allocated_length = int(req.kv.kv_allocated_len)
        req.kv_committed_len = committed_length
        if getattr(req, "kv", None) is not None:
            req.kv.kv_allocated_len = committed_length
        req_pool_index = getattr(req, "req_pool_idx", None)
        if req_pool_index is not None and allocated_length > committed_length:
            table[int(req_pool_index), committed_length:allocated_length] = 0


def _release_allocated_slots(batch: Any) -> None:
    if getattr(batch, _ALLOCATION_RELEASED_ATTR, False):
        return
    allocator = batch.token_to_kv_pool_allocator
    if int(getattr(allocator, "page_size", 1)) != 1:
        raise RuntimeError("MOSS-VL realtime rollback requires page_size == 1")
    slots = getattr(
        batch,
        _ALLOCATED_SLOTS_ATTR,
        getattr(batch, "out_cache_loc", None),
    )
    if isinstance(slots, torch.Tensor) and slots.numel():
        allocator.free(slots)
    setattr(batch, _ALLOCATION_RELEASED_ATTR, True)


def _clear_allocation_rollback_metadata(batch: Any) -> None:
    for name in (_ALLOCATED_SLOTS_ATTR, _ALLOCATION_RELEASED_ATTR):
        if hasattr(batch, name):
            delattr(batch, name)


def _build_plans(
    batch: Any,
    input_ids: list[Sequence[int]],
    seq_lens: list[int],
) -> list[_RealtimeExtendPlan]:
    plans: list[_RealtimeExtendPlan] = []
    out_offset = 0
    for index, req in enumerate(batch.reqs):
        state = getattr(req, RUNTIME_STATE_ATTR)
        req_pool_index = _scalar_int(req.req_pool_idx, "req_pool_idx")
        state.bind_req_pool_index(req_pool_index)
        mm_inputs = req.multimodal_inputs
        total_encoder_length = int(
            getattr(mm_inputs, "num_image_tokens", state.encoder_length) or 0
        )
        # Initial text-only prefill is the one path with no realtime segment:
        # bootstrap the decoder length/mrope baseline from the committed prefix
        # so the first frame extend can splice on top of it.
        if (
            state.encoder_length == 0
            and state.decoder_length == 0
            and mm_inputs is None
            and state.pending_token_id is None
            and len(req.prefix_indices) > 0
        ):
            state.decoder_length = len(req.prefix_indices)
            state.next_mrope_position = state.decoder_length
        encoder_delta_length = total_encoder_length - state.encoder_length
        if encoder_delta_length < 0:
            raise RuntimeError("total encoder length moved backwards")

        raw_extend_length = int(req.extend_range.length)
        decoder_extend_length = raw_extend_length - encoder_delta_length
        if decoder_extend_length <= 0:
            raise RuntimeError("realtime extend must contain at least one text token")
        if len(input_ids[index]) != raw_extend_length:
            raise RuntimeError("raw extend input length does not match request range")
        # prefix_indices is token-space by upstream contract
        # (seq_len - len(prefix_indices) == extend_len); the committed row
        # prefix itself is KV-space (surviving slots only after frame window
        # evictions compacted the encoder region).
        expected_prefix_length = (
            state.effective_appended_encoder_length + state.decoder_length
        )
        if len(req.prefix_indices) != expected_prefix_length:
            raise RuntimeError(
                "cached prefix does not match committed realtime KV lengths"
            )
        committed_row_length = state.encoder_length + state.decoder_length
        committed_row = batch.req_to_token_pool.req_to_token[
            req_pool_index, :committed_row_length
        ]
        invalid = torch.nonzero(committed_row <= 0, as_tuple=False).flatten()
        if invalid.numel():
            prefix_invalid = torch.nonzero(
                req.prefix_indices <= 0, as_tuple=False
            ).flatten()
            raise RuntimeError(
                "SGLang extend allocation corrupted the committed KV prefix "
                f"(row_indices={invalid[:8].tolist()}, "
                f"prefix_indices={prefix_invalid[:8].tolist()}, "
                f"encoder_length={state.encoder_length}, "
                f"decoder_length={state.decoder_length}, "
                f"raw_extend_length={req.extend_range.length})"
            )
        expected_raw_seq_len = (
            state.effective_appended_encoder_length
            + encoder_delta_length
            + state.decoder_length
            + decoder_extend_length
        )
        if seq_lens[index] != expected_raw_seq_len:
            raise RuntimeError("raw sequence length is inconsistent with runtime state")

        new_slots = batch.out_cache_loc[out_offset : out_offset + raw_extend_length]
        if new_slots.numel() != raw_extend_length:
            raise RuntimeError("allocated KV slots are not request aligned")
        leading_decoder_length = 1 if state.pending_token_id is not None else 0
        encoder_start = leading_decoder_length
        encoder_end = encoder_start + encoder_delta_length
        new_encoder_slots = new_slots[encoder_start:encoder_end]
        new_decoder_slots = torch.cat(
            [new_slots[:encoder_start], new_slots[encoder_end:]]
        )
        out_offset += raw_extend_length
        plans.append(
            _RealtimeExtendPlan(
                req=req,
                state=state,
                raw_input_ids=input_ids[index],
                new_encoder_slots=new_encoder_slots,
                new_decoder_slots=new_decoder_slots,
                total_encoder_length=total_encoder_length,
                total_decoder_length=state.decoder_length + decoder_extend_length,
                decoder_extend_length=decoder_extend_length,
                leading_decoder_length=leading_decoder_length,
            )
        )
    if out_offset != int(batch.out_cache_loc.numel()):
        raise RuntimeError("allocated KV slots contain an unclaimed tail")
    return plans


def _install_batch_metadata(batch: Any, plans: list[_RealtimeExtendPlan]) -> None:
    pin_memory = is_pin_memory_available(batch.device)
    device = batch.device
    batch.encoder_lens_cpu = [plan.total_encoder_length for plan in plans]
    batch.encoder_cached = [not plan.new_encoder_slots.numel() for plan in plans]
    batch.encoder_lens = torch.tensor(
        batch.encoder_lens_cpu,
        dtype=torch.int64,
        pin_memory=pin_memory,
    ).to(device, non_blocking=True)
    batch.encoder_out_cache_loc = torch.cat([plan.new_encoder_slots for plan in plans])
    batch.out_cache_loc = torch.cat([plan.new_decoder_slots for plan in plans])
    batch.extend_lens = [plan.decoder_extend_length for plan in plans]
    batch.prefix_lens = [plan.state.decoder_length for plan in plans]
    batch.extend_num_tokens = sum(batch.extend_lens)
    decoder_seq_lens = [plan.total_decoder_length for plan in plans]
    batch.seq_lens = torch.tensor(
        decoder_seq_lens,
        dtype=torch.int64,
        pin_memory=pin_memory,
    ).to(device, non_blocking=True)
    batch.seq_lens_cpu = torch.tensor(decoder_seq_lens, dtype=torch.int64)
    batch.seq_lens_sum = sum(decoder_seq_lens)

    decoder_inputs: list[int] = []
    for plan in plans:
        encoder_delta_length = int(plan.new_encoder_slots.numel())
        encoder_start = plan.leading_decoder_length
        encoder_end = encoder_start + encoder_delta_length
        decoder_inputs.extend(plan.raw_input_ids[:encoder_start])
        decoder_inputs.extend(plan.raw_input_ids[encoder_end:])
        plan.req.extend_range = plan.req.extend_range._replace(
            start=plan.req.extend_range.start + encoder_delta_length
        )
        plan.req.logprob_start_len = max(
            plan.req.logprob_start_len,
            plan.total_encoder_length,
        )
    batch.prefill_input_ids_cpu = torch.tensor(
        decoder_inputs,
        dtype=torch.int64,
        pin_memory=pin_memory,
    )


def _added_frames(plan: _RealtimeExtendPlan) -> int:
    if not plan.new_encoder_slots.numel():
        return 0
    mm_inputs = plan.req.multimodal_inputs
    items = getattr(mm_inputs, "mm_items", None) or ()
    if not items:
        raise RuntimeError("realtime extend with encoder slots has no media item")
    added = items[0].model_specific_data.get(REALTIME_ADDED_FRAMES_KEY)
    if not isinstance(added, int) or added <= 0:
        raise RuntimeError("realtime media item is missing the added-frame count")
    return added


def _next_mrope_position(plan: _RealtimeExtendPlan) -> int:
    mm_inputs = plan.req.multimodal_inputs
    positions = getattr(mm_inputs, "mrope_positions", None)
    if not isinstance(positions, torch.Tensor) or positions.numel() == 0:
        return max(plan.state.next_mrope_position, plan.total_decoder_length)
    return max(plan.state.next_mrope_position, int(positions.max().item()) + 1)


def _scalar_int(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(f"{name} must be scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)
