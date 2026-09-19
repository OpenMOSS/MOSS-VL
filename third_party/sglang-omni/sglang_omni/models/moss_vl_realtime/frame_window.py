"""Two-level sliding window over the vision (cross-attention) KV region.

Layer one (raw window, ``raw_window_s``): frames whose timestamp trails the
newest raw frame by more than ``raw_window_s`` leave the raw region of the
request's page-table row.

Layer two (compressed window, ``pool_window_s`` + ``pool_ratio``): aged-out raw
frames are not dropped outright — full chunks of exactly ``pool_ratio``
consecutive frames with identical spatial grids fold into one *virtual frame*
whose K/V is the per-position mean of the members across every KV layer (T
axis ratio→1, H×W unchanged). Aged remainders below one full chunk stay raw
until the group fills (streaming at 1fps would otherwise emit 1:1 virtual
frames and never realize the compression). The virtual frame's slots are
freshly allocated and inserted at the end of the compressed region (the
page-row prefix ahead of the raw frames).
Virtual frames trailing the newest virtual frame by more than ``pool_window_s``
are freed for good — the frame pixels long ago entered the embedding store, so
dropping the KV is "rolling the frame into memory".
Layer two is **off by default** (``pooling_enabled=False``): the mean-pooled
virtual K/V is never seen in training, and the A/B experiment on the 420s
jumping-jack case showed no quality gain over the raw-only window while
doubling steady-state vision KV. Enable it explicitly via config field
``realtime_frame_pooling_enabled`` or env ``REALTIME_FRAME_POOLING_ENABLED=1``.

Everything here is inert unless ``RealtimeFrameWindowConfig.enabled`` is set
(config field ``realtime_frame_window_enabled`` or env
``REALTIME_FRAME_WINDOW_ENABLED=1``). With the feature off none of these
functions run and serving behavior is unchanged.

Correctness anchors:
- The row layout is [compressed virtuals]{oldest→newest} ++ [raw frames] ++
  [decoder text]; ``full_grid_thw`` rows and visible-frame counts stay aligned
  with that exact frame order, so the delta-mask builder
  (``sglang_model.use_realtime_full_grid_thw``) needs no changes.
- Vision K was rotated with absolute MRoPE positions at write time, so evicting
  a prefix needs no position rewrite on the survivors.
- Compaction deliberately does **not** go through the KV-append transaction:
  ``apply_frame_window_plan`` validates everything that can fail (record/span
  coverage, visible-count remap) and builds every derived tensor *before* the
  page-row rewrite, so nothing past the ``allocator.free`` call can raise. A
  pre-rewrite failure frees only the freshly allocated pooled slots and leaves
  the committed layout untouched.
- Pooled-virtual visibility is deliberately relaxed: a text token that saw at
  least one member frame of a chunk is remapped to see the whole pooled frame,
  including members that had not arrived when the token was generated. Combined
  with the mean-pooled K representation (not a legal RoPE K at any single
  position), this is a training-side-unseen distribution — which is why the
  feature stays opt-in.
- The window reclaims KV memory only. Token space keeps one pad placeholder per
  historical encoder slot in ``full_untruncated_fill_ids``, so the context
  budget (``_guard_realtime_context_capacity`` bills ``max(token, KV)``) still
  bounds session lifetime; the window does not extend it.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.models.moss_vl_realtime.kv_layout import (
    read_req_to_token_layout,
    write_req_to_token_layout,
)
from sglang_omni.models.moss_vl_realtime.segment import (
    REALTIME_FULL_GRID_THW_KEY,
    _encoder_length,
)

logger = logging.getLogger(__name__)

REALTIME_FRAME_WINDOW_ENABLED_ENV = "REALTIME_FRAME_WINDOW_ENABLED"
REALTIME_FRAME_WINDOW_RAW_S_ENV = "REALTIME_FRAME_WINDOW_RAW_S"
REALTIME_FRAME_POOL_WINDOW_S_ENV = "REALTIME_FRAME_POOL_WINDOW_S"
REALTIME_FRAME_POOL_RATIO_ENV = "REALTIME_FRAME_POOL_RATIO"
REALTIME_FRAME_POOLING_ENABLED_ENV = "REALTIME_FRAME_POOLING_ENABLED"

# Request attribute holding per-frame records between segment staging and the
# KV append commit (mirrors the other ``_moss_vl_realtime_staged_*`` attrs).
FRAME_RECORDS_STAGED_ATTR = "_moss_vl_realtime_staged_frame_records"

DEFAULT_RAW_WINDOW_S = 60.0
DEFAULT_POOL_WINDOW_S = 240.0
DEFAULT_POOL_RATIO = 4


def _env_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"invalid boolean environment value: {value!r}")


@dataclass(frozen=True, slots=True)
class RealtimeFrameWindowConfig:
    """Resolved thresholds for the vision-KV two-level sliding window."""

    enabled: bool = False
    raw_window_s: float = DEFAULT_RAW_WINDOW_S
    pool_window_s: float = DEFAULT_POOL_WINDOW_S
    pool_ratio: int = DEFAULT_POOL_RATIO
    # When False, aged raw frames are dropped outright and the pooled virtual
    # tier never exists (plain raw sliding window). Defaults to False: the
    # mean-pooled virtual K/V is never seen in training, and the A/B experiment
    # on the 420s case showed no quality gain over the raw-only window.
    pooling_enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("enabled", "pooling_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in ("raw_window_s", "pool_window_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "raw_window_s", float(self.raw_window_s))
        object.__setattr__(self, "pool_window_s", float(self.pool_window_s))
        if isinstance(self.pool_ratio, bool) or not isinstance(self.pool_ratio, int):
            raise TypeError("pool_ratio must be an integer")
        if self.pool_ratio < 2:
            raise ValueError("pool_ratio must be at least 2")

    @classmethod
    def resolve(
        cls,
        *,
        enabled: bool | None = None,
        raw_window_s: float | None = None,
        pool_window_s: float | None = None,
        pool_ratio: int | None = None,
        pooling_enabled: bool | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RealtimeFrameWindowConfig:
        """Resolve explicit settings against environment overrides.

        Precedence is env > explicit > default, so a deployment can always
        retune or disable the window from the process environment without
        rebuilding the pipeline config.
        """
        if env is None:
            env = os.environ
        resolved_enabled = cls().enabled if enabled is None else bool(enabled)
        raw = cls().raw_window_s if raw_window_s is None else float(raw_window_s)
        pool = cls().pool_window_s if pool_window_s is None else float(pool_window_s)
        ratio = cls().pool_ratio if pool_ratio is None else int(pool_ratio)
        pooling = (
            cls().pooling_enabled if pooling_enabled is None else bool(pooling_enabled)
        )
        if REALTIME_FRAME_WINDOW_ENABLED_ENV in env:
            resolved_enabled = _env_flag(env[REALTIME_FRAME_WINDOW_ENABLED_ENV])
        if REALTIME_FRAME_WINDOW_RAW_S_ENV in env:
            raw = float(env[REALTIME_FRAME_WINDOW_RAW_S_ENV])
        if REALTIME_FRAME_POOL_WINDOW_S_ENV in env:
            pool = float(env[REALTIME_FRAME_POOL_WINDOW_S_ENV])
        if REALTIME_FRAME_POOL_RATIO_ENV in env:
            ratio = int(env[REALTIME_FRAME_POOL_RATIO_ENV])
        if REALTIME_FRAME_POOLING_ENABLED_ENV in env:
            pooling = _env_flag(env[REALTIME_FRAME_POOLING_ENABLED_ENV])
        return cls(
            enabled=resolved_enabled,
            raw_window_s=raw,
            pool_window_s=pool,
            pool_ratio=ratio,
            pooling_enabled=pooling,
        )


@dataclass(frozen=True, slots=True)
class RealtimeFrameRecord:
    """One committed encoder frame (real or pooled-virtual) in row order."""

    timestamp: float
    grid_h: int
    grid_w: int
    slots: int
    pooled: bool = False
    pooled_sources: int = 1

    @property
    def grid_row(self) -> tuple[int, int, int]:
        # Realtime frames always have T == 1 (enforced upstream); a pooled
        # virtual frame keeps the members' HxW while the T axis folded to 1.
        return (1, self.grid_h, self.grid_w)


@dataclass(frozen=True, slots=True)
class FrameWindowPlan:
    """One eviction round: what leaves, what gets pooled, what stays."""

    # Only existing virtual frames; immediately expired new groups are raw drops.
    evicted_virtual_count: int
    # (start index into the raw region, member count, poolable) chunks covering
    # the aged raw prefix, in chronological order; non-poolable chunks are
    # dropped outright.
    raw_chunks: tuple[tuple[int, int, bool], ...]
    # The full surviving frame list after the plan applies: kept old virtuals,
    # then the virtual records produced from poolable chunks, then kept raws.
    new_records: tuple[RealtimeFrameRecord, ...]

    @property
    def pooled_raw_count(self) -> int:
        return sum(count for _, count, poolable in self.raw_chunks if poolable)

    @property
    def dropped_raw_count(self) -> int:
        return sum(count for _, count, poolable in self.raw_chunks if not poolable)

    @property
    def produced_virtual_count(self) -> int:
        return sum(1 for _, _, poolable in self.raw_chunks if poolable)

    @property
    def removes_frames(self) -> bool:
        return self.evicted_virtual_count > 0 or bool(self.raw_chunks)


@dataclass(frozen=True, slots=True)
class FrameWindowEvent:
    """Applied-plan summary surfaced to logs and the probe harness."""

    request_id: str
    evicted_virtual_frames: int
    pooled_raw_frames: int
    produced_virtual_frames: int
    dropped_raw_frames: int
    encoder_length_before: int
    encoder_length_after: int
    surviving_frame_count: int
    pool_free_slots: int | None


def plan_frame_window(
    records: tuple[RealtimeFrameRecord, ...] | list[RealtimeFrameRecord],
    config: RealtimeFrameWindowConfig,
) -> FrameWindowPlan | None:
    """Compute one eviction round from committed frame records only.

    Timestamps come from the frame events (not wall clocks), so every TP rank
    derives the identical plan. Returns None when nothing changes.
    """
    if not config.enabled or not records:
        return None
    records = tuple(records)
    virtual_end = 0
    for record in records:
        if not record.pooled:
            break
        virtual_end += 1
    if any(record.pooled for record in records[virtual_end:]):
        raise RuntimeError("pooled frames must form the leading encoder region")
    virtuals = records[:virtual_end]
    raws = records[virtual_end:]

    # Layer one: raw frames trailing the newest raw frame by strictly more
    # than raw_window_s leave the raw region; exactly-at-window frames stay.
    aged_raw = 0
    if raws:
        newest_raw = raws[-1].timestamp
        while (
            aged_raw < len(raws)
            and (newest_raw - raws[aged_raw].timestamp) > config.raw_window_s
        ):
            aged_raw += 1

    # Layer two, step one: fold the aged raw prefix into full chunks of
    # pool_ratio. Frames age out of the raw window one at a time while the
    # scheduler evaluates every decode gap; pooling partial chunks would fold
    # each aged frame 1:1 into an uncompressed virtual frame and never realize
    # the R:1 compression. Aged remainders (< ratio frames) therefore stay
    # raw — a soft overshoot of at most ratio-1 frames until the group fills.
    # A full chunk with mixed spatial grids cannot be position-wise
    # mean-pooled, so it is dropped instead.
    raw_chunks: list[tuple[int, int, bool]] = []
    produced: list[RealtimeFrameRecord] = []
    if config.pooling_enabled:
        poolable_count = aged_raw // config.pool_ratio * config.pool_ratio
        cursor = 0
        while cursor < poolable_count:
            end = cursor + config.pool_ratio
            members = raws[cursor:end]
            poolable = all(
                (member.grid_h, member.grid_w, member.slots)
                == (members[0].grid_h, members[0].grid_w, members[0].slots)
                for member in members
            )
            raw_chunks.append((cursor, len(members), poolable))
            if poolable:
                produced.append(
                    RealtimeFrameRecord(
                        timestamp=members[-1].timestamp,
                        grid_h=members[0].grid_h,
                        grid_w=members[0].grid_w,
                        slots=members[0].slots,
                        pooled=True,
                        pooled_sources=len(members),
                    )
                )
            cursor += len(members)
    else:
        # Raw-only comparison window (experiments): the whole aged prefix is
        # dropped outright and the pooled virtual tier never exists.
        poolable_count = aged_raw
        if aged_raw:
            raw_chunks.append((0, aged_raw, False))

    # Layer two, step two: evict leading virtual frames (existing plus newly
    # produced) whose span trails the newest virtual frame beyond the pool
    # window. Also strictly-greater; at least one virtual frame survives.
    virtuals_all = virtuals + tuple(produced)
    evicted_virtual = 0
    if len(virtuals_all) >= 2:
        newest_virtual = virtuals_all[-1].timestamp
        while (
            evicted_virtual < len(virtuals_all) - 1
            and (newest_virtual - virtuals_all[evicted_virtual].timestamp)
            > config.pool_window_s
        ):
            evicted_virtual += 1

    if evicted_virtual == 0 and not raw_chunks:
        return None
    new_records = virtuals_all[evicted_virtual:] + raws[poolable_count:]
    # Do not materialize virtuals that this same round would immediately evict.
    # Express them as raw drops so spans and slot ownership refer to old rows.
    expired_produced = max(0, evicted_virtual - len(virtuals))
    if expired_produced:
        surviving_chunks = []
        for start, count, poolable in raw_chunks:
            if poolable and expired_produced:
                poolable = False
                expired_produced -= 1
            surviving_chunks.append((start, count, poolable))
        raw_chunks = surviving_chunks
        evicted_virtual = len(virtuals)
    return FrameWindowPlan(
        evicted_virtual_count=evicted_virtual,
        raw_chunks=tuple(raw_chunks),
        new_records=tuple(new_records),
    )


def stage_segment_frame_records(
    segment: Any,
    *,
    merge_size: int,
) -> list[RealtimeFrameRecord] | None:
    """Derive per-frame records for one staged segment.

    Returns None for text-only segments. Frame ordering matches the segment
    builder: events sorted by timestamp, grid rows from the delta tail of
    ``segment.full_grid_thw``.
    """
    frame_events = sorted(
        (event for event in segment.events if event.has_frame),
        key=lambda event: event.timestamp,
    )
    if not frame_events:
        return None
    grid = segment.full_grid_thw
    if grid.shape[0] < len(frame_events):
        raise RuntimeError("segment grid history is shorter than its frame count")
    delta_rows = grid[grid.shape[0] - len(frame_events) :]
    merge_square = int(merge_size) ** 2
    if merge_square <= 0:
        raise ValueError("merge_size must be positive")
    records: list[RealtimeFrameRecord] = []
    for event, row in zip(frame_events, delta_rows.tolist()):
        temporal, grid_h, grid_w = (int(value) for value in row)
        if temporal != 1:
            raise RuntimeError("realtime frame records require T == 1")
        if (grid_h * grid_w) % merge_square:
            raise RuntimeError("frame grid must be divisible by merge_size")
        records.append(
            RealtimeFrameRecord(
                timestamp=float(event.timestamp),
                grid_h=grid_h,
                grid_w=grid_w,
                # Slot accounting is single-sourced from the segment builder;
                # with T == 1 enforced above this is grid_h*grid_w/merge**2 + 1.
                slots=_encoder_length(
                    torch.tensor([[temporal, grid_h, grid_w]], dtype=torch.long),
                    int(merge_size),
                ),
            )
        )
    return records


def covered_spans(
    records: tuple[RealtimeFrameRecord, ...],
    plan: FrameWindowPlan,
) -> tuple[tuple[int, int], ...]:
    """Old-row span covered by each record of ``plan.new_records``, in order."""
    old_virtual_count = sum(1 for record in records if record.pooled)
    spans: list[tuple[int, int]] = []
    cursor = plan.evicted_virtual_count
    for _ in range(old_virtual_count - plan.evicted_virtual_count):
        spans.append((cursor, cursor + 1))
        cursor += 1
    for start, count, poolable in plan.raw_chunks:
        if poolable:
            spans.append((cursor, cursor + count))
        cursor += count
    kept_raw = sum(1 for record in plan.new_records if not record.pooled)
    for _ in range(kept_raw):
        spans.append((cursor, cursor + 1))
        cursor += 1
    if cursor != len(records):
        raise RuntimeError("frame window plan does not cover every old record")
    expected = len(spans)
    if expected != len(plan.new_records):
        raise RuntimeError("frame window plan spans disagree with new records")
    return tuple(spans)


def visible_count_remap(
    old_row_count: int,
    spans: tuple[tuple[int, int], ...],
) -> list[int]:
    """Map old visible-frame counts to surviving-frame counts ("kept counts").

    ``spans[i]`` is the half-open old-row span covered by new row ``i``; old
    rows covered by no span were dropped outright. The table ``K`` maps an old
    cumulative count ``c`` to the number of new rows intersecting the first
    ``c`` old rows — a pooled virtual frame becomes visible to a text token as
    soon as that token saw at least one member of its chunk.
    """
    remap = [0] * (old_row_count + 1)
    cursor = 0
    produced = 0
    for start, end in spans:
        if start < cursor or end <= start:
            raise ValueError("plan spans must be ordered and non-empty")
        for c in range(cursor, start + 1):
            remap[c] = produced
        for c in range(start + 1, end + 1):
            remap[c] = produced + 1
        produced += 1
        cursor = end
    for c in range(cursor, old_row_count + 1):
        remap[c] = produced
    return remap


def apply_frame_window_plan(
    req: Any,
    state: Any,
    plan: FrameWindowPlan,
    *,
    records: tuple[RealtimeFrameRecord, ...] | list[RealtimeFrameRecord],
    req_to_token: torch.Tensor,
    allocator: Any,
    kv_pool_provider: Callable[[], Any | None],
    running_batch: Any | None = None,
) -> FrameWindowEvent:
    """Apply one plan: pool, rewrite the page row, free, then resync metadata.

    The page-row rewrite is the point of no return: pooling writes go to
    freshly allocated slots beforehand, so a pre-rewrite failure frees those
    new slots and leaves the committed layout untouched.
    """
    if not plan.removes_frames:
        raise ValueError("refusing to apply an empty frame window plan")
    records = tuple(records)
    if state.req_pool_index is None:
        raise RuntimeError("frame window requires a bound page-table row")
    layout = read_req_to_token_layout(
        req_to_token,
        req_pool_index=state.req_pool_index,
        encoder_length=state.encoder_length,
        decoder_length=state.decoder_length,
    )
    runs = _record_slot_runs(records, layout.encoder_slots)
    old_virtual_count = sum(1 for record in records if record.pooled)
    encoder_length_before = state.encoder_length

    # Allocate pooled slots up front. On allocator pressure (or a missing KV
    # pool) degrade the round to pure eviction: retaining aged raw frames
    # without pooled copies would break the raw-window bound.
    poolable_chunks = [chunk for chunk in plan.raw_chunks if chunk[2]]
    dst_slot_ids: torch.Tensor | None = None
    kv_pool = None
    if poolable_chunks:
        widths = [
            len(runs[old_virtual_count + start]) for start, _, _ in poolable_chunks
        ]
        # A missing or failing provider must not leave newly allocated slots orphaned.
        kv_pool = kv_pool_provider()
        if kv_pool is not None:
            dst_slot_ids = allocator.alloc(sum(widths))
        if dst_slot_ids is None or kv_pool is None:
            logger.warning(
                "frame window pooling degraded to eviction for %s "
                "(allocator pressure or missing KV pool)",
                state.request_id,
            )
            dst_slot_ids = None
            plan = FrameWindowPlan(
                evicted_virtual_count=plan.evicted_virtual_count,
                raw_chunks=tuple(
                    (start, count, False) for start, count, _ in plan.raw_chunks
                ),
                new_records=_degraded_new_records(records, plan),
            )
            poolable_chunks = []

    produced_runs: list[tuple[int, ...]] = []
    try:
        # Validate the metadata side first: nothing past the row rewrite may
        # raise, so coverage/remap checks and every derived tensor are built
        # here, while the committed layout is still untouched.
        prepared = _prepare_metadata_update(req, state, records=records, plan=plan)
        if poolable_chunks:
            if dst_slot_ids is None or kv_pool is None:
                raise RuntimeError("pooling slots were not allocated")
            dst_slot_ids = dst_slot_ids.to(dtype=torch.long)
            dst_offset = 0
            for (start, count, _), width in zip(poolable_chunks, widths, strict=True):
                member_runs = [
                    runs[old_virtual_count + start + offset] for offset in range(count)
                ]
                dst = dst_slot_ids[dst_offset : dst_offset + width]
                _pool_group_into_slots(kv_pool, member_runs, dst)
                produced_runs.append(tuple(int(v) for v in dst.tolist()))
                dst_offset += width

        aged_raw_total = plan.pooled_raw_count + plan.dropped_raw_count
        kept_virtual_runs = list(runs[plan.evicted_virtual_count : old_virtual_count])
        kept_raw_runs = list(runs[old_virtual_count + aged_raw_total :])
        new_encoder_slots = tuple(
            slot
            for chunk_run in kept_virtual_runs + produced_runs + kept_raw_runs
            for slot in chunk_run
        )
        new_layout = layout.__class__(
            encoder_slots=new_encoder_slots,
            decoder_slots=layout.decoder_slots,
        )
        # Point of no return: the page row now exposes the compacted layout.
        write_req_to_token_layout(
            req_to_token,
            req_pool_index=state.req_pool_index,
            layout=new_layout,
            clear_tail=True,
        )
    except Exception:
        if dst_slot_ids is not None and dst_slot_ids.numel():
            allocator.free(dst_slot_ids)
        raise

    consumed_raw_slots = [
        slot
        for run in runs[old_virtual_count : old_virtual_count + aged_raw_total]
        for slot in run
    ]
    freed = [
        slot for run in runs[: plan.evicted_virtual_count] for slot in run
    ] + consumed_raw_slots
    if freed:
        allocator.free(
            torch.tensor(freed, dtype=torch.long, device=req_to_token.device)
        )

    _commit_metadata_update(
        req,
        state,
        prepared=prepared,
        plan=plan,
        new_encoder_slots=new_encoder_slots,
    )
    if running_batch is not None:
        _update_running_batch_encoder_lens(running_batch, req, len(new_encoder_slots))
    pool_free = None
    available = getattr(allocator, "available_size", None)
    if callable(available):
        try:
            pool_free = int(available())
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning("frame window: allocator available_size() failed: %s", exc)
    event = FrameWindowEvent(
        request_id=state.request_id,
        evicted_virtual_frames=plan.evicted_virtual_count,
        pooled_raw_frames=plan.pooled_raw_count,
        produced_virtual_frames=len(produced_runs),
        dropped_raw_frames=plan.dropped_raw_count,
        encoder_length_before=encoder_length_before,
        encoder_length_after=len(new_encoder_slots),
        surviving_frame_count=len(plan.new_records),
        pool_free_slots=pool_free,
    )
    state.encoder_length = len(new_encoder_slots)
    return event


def _degraded_new_records(
    records: tuple[RealtimeFrameRecord, ...],
    plan: FrameWindowPlan,
) -> tuple[RealtimeFrameRecord, ...]:
    """Survivors when the round degrades from pooling to pure eviction."""
    old_virtual_count = sum(1 for record in records if record.pooled)
    aged_raw = plan.pooled_raw_count + plan.dropped_raw_count
    return (
        records[plan.evicted_virtual_count : old_virtual_count]
        + records[old_virtual_count + aged_raw :]
    )


def _record_slot_runs(
    records: tuple[RealtimeFrameRecord, ...],
    encoder_slots: tuple[int, ...],
) -> list[tuple[int, ...]]:
    total = sum(record.slots for record in records)
    if total != len(encoder_slots):
        raise RuntimeError(
            "frame records disagree with the encoder page-table region "
            f"(records={total} slots, row={len(encoder_slots)} slots)"
        )
    runs: list[tuple[int, ...]] = []
    cursor = 0
    for record in records:
        runs.append(tuple(encoder_slots[cursor : cursor + record.slots]))
        cursor += record.slots
    return runs


def _pool_group_into_slots(
    kv_pool: Any,
    member_runs: list[tuple[int, ...]],
    dst_slot_ids: torch.Tensor,
) -> None:
    """Mean-pool one chunk's K/V across every KV layer into the new slots."""
    width = len(member_runs[0])
    if any(len(run) != width for run in member_runs):
        raise RuntimeError("pool chunks require uniform member slot widths")
    device = dst_slot_ids.device
    src = torch.tensor(
        [slot for run in member_runs for slot in run],
        dtype=torch.long,
        device=device,
    )
    group = len(member_runs)
    for k_buf, v_buf in zip(
        list(kv_pool.k_buffer), list(kv_pool.v_buffer), strict=True
    ):
        pooled_k = k_buf.index_select(0, src).view(group, width, -1)
        pooled_k = pooled_k.float().mean(dim=0).to(k_buf.dtype)
        pooled_v = v_buf.index_select(0, src).view(group, width, -1)
        pooled_v = pooled_v.float().mean(dim=0).to(v_buf.dtype)
        k_buf.index_copy_(0, dst_slot_ids, pooled_k.view(width, *k_buf.shape[1:]))
        v_buf.index_copy_(0, dst_slot_ids, pooled_v.view(width, *v_buf.shape[1:]))


@dataclass(frozen=True, slots=True)
class _PreparedMetadataUpdate:
    """Pre-validated metadata side of one apply; see ``_prepare_metadata_update``."""

    dropped_units: int
    full_grid_thw: torch.Tensor | None
    # (owner tensor, replacement) pairs in ``_visible_counts_owners`` order.
    visible_counts: tuple[tuple[Any, torch.Tensor], ...]


def _prepare_metadata_update(
    req: Any,
    state: Any,
    *,
    records: tuple[RealtimeFrameRecord, ...],
    plan: FrameWindowPlan,
) -> _PreparedMetadataUpdate:
    """Compute every fallible part of the metadata update up front.

    Runs strictly before the page-row rewrite: span coverage and the
    visible-count remap are validated and every derived tensor is built while
    the committed layout is still untouched, so the post-rewrite commit step
    degrades to assignments that cannot raise.
    """
    spans = covered_spans(records, plan)
    remap = visible_count_remap(len(records), spans)
    full_grid_thw = None
    if state.full_grid_thw is not None:
        rows = [record.grid_row for record in plan.new_records]
        full_grid_thw = torch.tensor(rows, dtype=torch.long)
    visible_counts = []
    for owner in _visible_counts_owners(req, state):
        if owner.numel():
            mapped = [remap[min(int(v), len(records))] for v in owner.tolist()]
            new_counts = torch.tensor(mapped, dtype=owner.dtype, device=owner.device)
        else:
            new_counts = owner
        visible_counts.append((owner, new_counts))
    dropped_units = plan.dropped_raw_count + sum(
        record.pooled_sources for record in records[: plan.evicted_virtual_count]
    )
    return _PreparedMetadataUpdate(
        dropped_units=dropped_units,
        full_grid_thw=full_grid_thw,
        visible_counts=tuple(visible_counts),
    )


def _commit_metadata_update(
    req: Any,
    state: Any,
    *,
    prepared: _PreparedMetadataUpdate,
    plan: FrameWindowPlan,
    new_encoder_slots: tuple[int, ...],
) -> None:
    """Assign the prepared metadata; runs after the old slots were freed.

    Nothing below may raise: a failure past the free would leave
    ``kv_committed_len`` spanning cleared row space (slot id 0), which the
    release path would then free into the shared allocator.
    """
    state.frame_records = list(plan.new_records)
    state.evicted_frame_count += prepared.dropped_units
    state.surviving_frame_count = len(plan.new_records)

    if state.full_grid_thw is not None:
        state.full_grid_thw = prepared.full_grid_thw

    for owner, new_counts in prepared.visible_counts:
        _assign_visible_counts(req, state, owner, new_counts)

    mm_inputs = getattr(req, "multimodal_inputs", None)
    if mm_inputs is not None:
        mm_inputs.num_image_tokens = len(new_encoder_slots)
        if (
            getattr(mm_inputs, "media_nums_per_sample", None) is not None
            and state.full_grid_thw is not None
        ):
            mm_inputs.media_nums_per_sample = [int(state.full_grid_thw.shape[0])]
        items = getattr(mm_inputs, "mm_items", None) or ()
        if items and state.full_grid_thw is not None:
            items[0].model_specific_data[
                REALTIME_FULL_GRID_THW_KEY
            ] = state.full_grid_thw

    # The request frees KV via row[0:kv_committed_len] on release; shrink the
    # committed lengths so the evicted slots (already freed above) are never
    # freed a second time.
    committed_total = len(new_encoder_slots) + state.decoder_length
    if getattr(req, "kv", None) is not None:
        req.kv.kv_allocated_len = committed_total
    if hasattr(req, "kv_committed_len"):
        req.kv_committed_len = committed_total


def _visible_counts_owners(req: Any, state: Any) -> list[torch.Tensor]:
    owners: list[torch.Tensor] = []
    if isinstance(state.visible_frame_counts, torch.Tensor):
        owners.append(state.visible_frame_counts)
    mm_inputs = getattr(req, "multimodal_inputs", None)
    counts = getattr(mm_inputs, "visible_frame_counts", None) if mm_inputs else None
    if isinstance(counts, torch.Tensor) and all(
        counts is not existing for existing in owners
    ):
        owners.append(counts)
    return owners


def _assign_visible_counts(
    req: Any,
    state: Any,
    old: torch.Tensor,
    new: torch.Tensor,
) -> None:
    if state.visible_frame_counts is old:
        state.visible_frame_counts = new
    mm_inputs = getattr(req, "multimodal_inputs", None)
    if (
        mm_inputs is not None
        and getattr(mm_inputs, "visible_frame_counts", None) is old
    ):
        mm_inputs.visible_frame_counts = new


def _update_running_batch_encoder_lens(
    running_batch: Any, req: Any, new_encoder_length: int
) -> None:
    """Point the live decode batch at the shrunken encoder region.

    FlashInfer decode re-plans cross-attention kv_indices from
    ``batch.encoder_lens`` every step (graph replay included), so an in-place
    tensor write plus the CPU list is the whole propagation path.
    """
    reqs = getattr(running_batch, "reqs", ()) or ()
    index = next((i for i, candidate in enumerate(reqs) if candidate is req), None)
    if index is None:
        return
    lens_cpu = getattr(running_batch, "encoder_lens_cpu", None)
    if lens_cpu is not None:
        lens_cpu[index] = new_encoder_length
    lens = getattr(running_batch, "encoder_lens", None)
    if isinstance(lens, torch.Tensor):
        lens[index] = new_encoder_length
