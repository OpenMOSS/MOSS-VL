"""Scheduler-owned runtime state and transactional KV layout updates."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self

import torch

from sglang_omni.models.moss_vl_realtime.kv_layout import (
    MossVLRealtimeKVLayout,
    read_req_to_token_layout,
    write_req_to_token_layout,
)


class MossVLRealtimePhase(str, Enum):
    """Lifecycle phases for one persistent realtime decode request."""

    WAITING_FOR_EVENT = "waiting_for_event"
    EXTENDING = "extending"
    DECODING = "decoding"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass(slots=True)
class MossVLRealtimeRuntimeState:
    """Lengths and position metadata that cannot be inferred from token IDs."""

    request_id: str
    session_id: str
    req_pool_index: int | None = None
    encoder_length: int = 0
    decoder_length: int = 0
    # Frames currently represented in the encoder region. Before any frame
    # window eviction this equals the cumulative committed frame count;
    # apply_frame_window_plan resets it to the surviving window rows.
    surviving_frame_count: int = 0
    next_mrope_position: int = 0
    turn_id: int = 0
    max_tokens_per_turn: float = 86400.0
    # Per-extend decode allowance: the constant gap kept between
    # len(req.output_ids) and sampling_params.max_new_tokens. Decode tokens
    # (silence or text) drain the gap; every extend re-anchors it, so a
    # input refreshes its local allowance; context_limit still bounds the
    # whole session. None is derived once from the first extend.
    decode_allowance: int | None = None
    context_limit: int | None = field(default=None, kw_only=True)
    next_decode_not_before: float = 0.0
    pending_token_id: int | None = None
    mrope_positions: torch.Tensor | None = field(default=None, repr=False)
    visible_frame_counts: torch.Tensor | None = field(default=None, repr=False)
    full_grid_thw: torch.Tensor | None = field(default=None, repr=False)
    # Frame-window bookkeeping (sglang_omni .../frame_window.py). Both stay at
    # their defaults while the window is disabled, so the off path is
    # byte-identical to the un-windowed server.
    frame_records: list | None = field(default=None, repr=False)
    evicted_frame_count: int = 0
    # Token-space encoder history: total encoder slots ever appended, counting
    # frames the window already evicted. ``full_untruncated_fill_ids`` keeps
    # one pad placeholder per historical encoder slot, so all token-space
    # accounting (pending-token check, extend_range, prefix_indices length)
    # must use this instead of encoder_length once frames are dropped. None
    # means "not tracked yet" and behaves as encoder_length: no eviction can
    # have happened, so the two spaces coincide.
    appended_encoder_length: int | None = None
    phase: MossVLRealtimePhase = MossVLRealtimePhase.WAITING_FOR_EVENT
    _append_inflight: bool = field(default=False, init=False, repr=False)
    accounting: Any = field(default=None, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        if not self.request_id or not self.session_id:
            raise ValueError("request_id and session_id must be non-empty")
        for name in (
            "encoder_length",
            "decoder_length",
            "surviving_frame_count",
            "next_mrope_position",
            "turn_id",
            "evicted_frame_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.context_limit is not None and (
            isinstance(self.context_limit, bool)
            or not isinstance(self.context_limit, int)
            or self.context_limit <= 0
        ):
            raise ValueError("context_limit must be a positive integer or None")
        if self.decode_allowance is not None and (
            isinstance(self.decode_allowance, bool)
            or not isinstance(self.decode_allowance, int)
            or self.decode_allowance <= 0
        ):
            raise ValueError("decode_allowance must be a positive integer or None")
        if self.req_pool_index is not None:
            self.bind_req_pool_index(self.req_pool_index)
        if (
            isinstance(self.max_tokens_per_turn, bool)
            or not isinstance(self.max_tokens_per_turn, (int, float))
            or not math.isfinite(self.max_tokens_per_turn)
            or self.max_tokens_per_turn <= 0
        ):
            raise ValueError("max_tokens_per_turn must be finite and positive")
        self.max_tokens_per_turn = float(self.max_tokens_per_turn)
        if (
            isinstance(self.next_decode_not_before, bool)
            or not isinstance(self.next_decode_not_before, (int, float))
            or not math.isfinite(self.next_decode_not_before)
            or self.next_decode_not_before < 0
        ):
            raise ValueError("next_decode_not_before must be finite and non-negative")
        self.next_decode_not_before = float(self.next_decode_not_before)
        if self.appended_encoder_length is not None and (
            isinstance(self.appended_encoder_length, bool)
            or not isinstance(self.appended_encoder_length, int)
            or self.appended_encoder_length < 0
        ):
            raise ValueError(
                "appended_encoder_length must be a non-negative integer or None"
            )

    @property
    def effective_appended_encoder_length(self) -> int:
        """Token-space encoder-slot history, including evicted frames."""
        if self.appended_encoder_length is None:
            return self.encoder_length
        return self.appended_encoder_length

    def bind_req_pool_index(self, req_pool_index: int) -> None:
        if isinstance(req_pool_index, bool) or not isinstance(req_pool_index, int):
            raise TypeError("req_pool_index must be an integer")
        if req_pool_index < 0:
            raise ValueError("req_pool_index must be non-negative")
        if self.req_pool_index is not None and self.req_pool_index != req_pool_index:
            raise RuntimeError("request KV pool ownership cannot change while live")
        self.req_pool_index = req_pool_index

    def mark_waiting(self) -> None:
        self._require_open()
        if self._append_inflight:
            raise RuntimeError("cannot park a request during a KV append")
        self.phase = MossVLRealtimePhase.WAITING_FOR_EVENT

    def mark_decoding(self) -> None:
        self._require_open()
        if self._append_inflight:
            raise RuntimeError("commit or roll back the KV append first")
        self.phase = MossVLRealtimePhase.DECODING

    def finish(self, *, aborted: bool = False) -> None:
        if self._append_inflight:
            raise RuntimeError("cannot finish with an in-flight KV append")
        self.phase = (
            MossVLRealtimePhase.ABORTED if aborted else MossVLRealtimePhase.FINISHED
        )

    def begin_kv_append(
        self,
        req_to_token: torch.Tensor,
        *,
        new_encoder_slots: Iterable[int] = (),
        new_decoder_slots: Iterable[int] = (),
        added_frames: int = 0,
        next_mrope_position: int | None = None,
        release_slots: Callable[[tuple[int, ...]], None] | None = None,
    ) -> MossVLRealtimeKVAppendTransaction:
        self._require_open()
        if self.req_pool_index is None:
            raise RuntimeError("bind req_pool_index before appending KV")
        if self._append_inflight:
            raise RuntimeError("another KV append is already in flight")
        transaction = MossVLRealtimeKVAppendTransaction.create(
            state=self,
            req_to_token=req_to_token,
            new_encoder_slots=new_encoder_slots,
            new_decoder_slots=new_decoder_slots,
            added_frames=added_frames,
            next_mrope_position=next_mrope_position,
            release_slots=release_slots,
        )
        transaction.stage()
        return transaction

    def _require_open(self) -> None:
        if self.phase in (
            MossVLRealtimePhase.FINISHED,
            MossVLRealtimePhase.ABORTED,
        ):
            raise RuntimeError(f"request is already {self.phase.value}")


@dataclass(slots=True)
class MossVLRealtimeKVAppendTransaction:
    """One page-table rewrite whose state changes commit after model forward."""

    state: MossVLRealtimeRuntimeState
    req_to_token: torch.Tensor
    before: MossVLRealtimeKVLayout
    after: MossVLRealtimeKVLayout
    added_frames: int
    staged_next_mrope_position: int
    new_slots: tuple[int, ...]
    release_slots: Callable[[tuple[int, ...]], None] | None = None
    _staged: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(
        cls,
        *,
        state: MossVLRealtimeRuntimeState,
        req_to_token: torch.Tensor,
        new_encoder_slots: Iterable[int],
        new_decoder_slots: Iterable[int],
        added_frames: int,
        next_mrope_position: int | None,
        release_slots: Callable[[tuple[int, ...]], None] | None,
    ) -> MossVLRealtimeKVAppendTransaction:
        if isinstance(added_frames, bool) or not isinstance(added_frames, int):
            raise TypeError("added_frames must be an integer")
        if added_frames < 0:
            raise ValueError("added_frames must be non-negative")
        staged_position = (
            state.next_mrope_position
            if next_mrope_position is None
            else next_mrope_position
        )
        if (
            isinstance(staged_position, bool)
            or not isinstance(staged_position, int)
            or staged_position < state.next_mrope_position
        ):
            raise ValueError("next_mrope_position must be monotonic")
        assert state.req_pool_index is not None
        before = read_req_to_token_layout(
            req_to_token,
            req_pool_index=state.req_pool_index,
            encoder_length=state.encoder_length,
            decoder_length=state.decoder_length,
        )
        encoder_layout = before.append_encoder(new_encoder_slots)
        encoder_slots = encoder_layout.encoder_slots[before.encoder_length :]
        after = encoder_layout.append_decoder(new_decoder_slots)
        decoder_slots = after.decoder_slots[before.decoder_length :]
        return cls(
            state=state,
            req_to_token=req_to_token,
            before=before,
            after=after,
            added_frames=added_frames,
            staged_next_mrope_position=staged_position,
            new_slots=encoder_slots + decoder_slots,
            release_slots=release_slots,
        )

    @property
    def encoder_length(self) -> int:
        return self.after.encoder_length

    @property
    def decoder_length(self) -> int:
        return self.after.decoder_length

    def stage(self) -> None:
        if self._closed or self._staged:
            raise RuntimeError("KV append transaction cannot be staged again")
        assert self.state.req_pool_index is not None
        write_req_to_token_layout(
            self.req_to_token,
            req_pool_index=self.state.req_pool_index,
            layout=self.after,
        )
        self.state._append_inflight = True
        self.state.phase = MossVLRealtimePhase.EXTENDING
        self._staged = True

    def commit(self) -> None:
        self._require_active()
        self.state.encoder_length = self.after.encoder_length
        self.state.decoder_length = self.after.decoder_length
        self.state.surviving_frame_count += self.added_frames
        appended_baseline = (
            self.before.encoder_length
            if self.state.appended_encoder_length is None
            else self.state.appended_encoder_length
        )
        self.state.appended_encoder_length = appended_baseline + (
            self.after.encoder_length - self.before.encoder_length
        )
        self.state.next_mrope_position = self.staged_next_mrope_position
        self.state._append_inflight = False
        self.state.phase = MossVLRealtimePhase.DECODING
        self._closed = True

    def rollback(self) -> None:
        self._require_active()
        assert self.state.req_pool_index is not None
        write_req_to_token_layout(
            self.req_to_token,
            req_pool_index=self.state.req_pool_index,
            layout=self.before,
            clear_tail=True,
        )
        try:
            if self.release_slots is not None and self.new_slots:
                self.release_slots(self.new_slots)
        finally:
            self.state._append_inflight = False
            self.state.phase = MossVLRealtimePhase.WAITING_FOR_EVENT
            self._closed = True

    def _require_active(self) -> None:
        if not self._staged or self._closed:
            raise RuntimeError("KV append transaction is not active")

    def __enter__(self) -> Self:
        self._require_active()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc, traceback
        if self._closed:
            return False
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False
