"""Persistent-request scheduler for MOSS-VL realtime frame updates."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import torch
from PIL import Image

import sglang_omni.compat as _compat
from sglang_omni.scheduling.messages import IncomingMessage

_compat.apply_all()
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    NextBatchPlan,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (  # noqa: E402
    PrefillStats,
)
from sglang.srt.observability.metrics_collector import QueueCount  # noqa: E402
from sglang.srt.utils import broadcast_pyobj  # noqa: E402

from sglang_omni.models.moss_vl_realtime.accounting import (
    FINALIZE_ACTION,
    ContextExhaustedError,
    RealtimeAccounting,
    error_code,
)
from sglang_omni.models.moss_vl_realtime.batch_adapter import (
    RUNTIME_STATE_ATTR,
    MossVLRealtimeScheduleBatch,
    realtime_decode_capacity_error,
)
from sglang_omni.models.moss_vl_realtime.frame_store import resolve_shared_memory_frame
from sglang_omni.models.moss_vl_realtime.frame_window import (
    FRAME_RECORDS_STAGED_ATTR,
    RealtimeFrameWindowConfig,
    apply_frame_window_plan,
    plan_frame_window,
    stage_segment_frame_records,
)
from sglang_omni.models.moss_vl_realtime.payload_types import FramePromptEvent
from sglang_omni.models.moss_vl_realtime.runtime_state import (
    MossVLRealtimePhase,
    MossVLRealtimeRuntimeState,
)
from sglang_omni.models.moss_vl_realtime.segment import (
    MossVLRealtimeSegment,
    MossVLRealtimeSegmentBuilder,
)
from sglang_omni.models.moss_vl_realtime.session_state import (
    MossVLRealtimeSessionController,
)
from sglang_omni.scheduling.omni_scheduler import _FAILED_BATCH_RESULT, OmniScheduler

logger = logging.getLogger(__name__)


class _RealtimeDecodeDeferred(Exception):
    """Stop planning after prefill handoff, before decode allocates step KV."""


class MossVLRealtimeScheduler(OmniScheduler):
    """Interrupt decode between steps to extend one or more frame events."""

    def __init__(
        self,
        *args: Any,
        segment_builder: MossVLRealtimeSegmentBuilder,
        frame_resolver: Callable[[FramePromptEvent], Any] | None = None,
        silence_token_ids: tuple[int, ...],
        parked_request_timeout_s: float = 300.0,
        frame_window_config: RealtimeFrameWindowConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("enable_overlap", False):
            raise ValueError("MOSS-VL realtime requires overlap scheduling disabled")
        server_args = kwargs.get("server_args")
        if server_args is not None and int(server_args.page_size) != 1:
            raise ValueError("MOSS-VL realtime requires page_size == 1")
        if server_args is not None:
            self._validate_kv_pool_capacity(
                kwargs.get("token_to_kv_pool_allocator"), server_args
            )
        self.segment_builder = segment_builder
        self.frame_resolver = frame_resolver or resolve_local_frame
        self.realtime_sessions = MossVLRealtimeSessionController()
        self._realtime_extend_batch: ScheduleBatch | None = None
        self.frame_window_config = (
            frame_window_config
            if frame_window_config is not None and frame_window_config.enabled
            else None
        )
        if self.frame_window_config is not None:
            logger.info(
                "Realtime frame window enabled: raw_window=%ss pooling=%s "
                "pool_window=%ss pool_ratio=%d",
                self.frame_window_config.raw_window_s,
                self.frame_window_config.pooling_enabled,
                self.frame_window_config.pool_window_s,
                self.frame_window_config.pool_ratio,
            )
        self.silence_token_ids = tuple(int(token_id) for token_id in silence_token_ids)
        if not self.silence_token_ids:
            raise ValueError("silence_token_ids must not be empty")
        self.parked_request_timeout_s = float(parked_request_timeout_s)
        if (
            not math.isfinite(self.parked_request_timeout_s)
            or self.parked_request_timeout_s <= 0
        ):
            raise ValueError("parked_request_timeout_s must be finite and positive")
        self.parked_reqs: dict[str, Any] = {}
        self.parked_since: dict[str, float] = {}
        self._accounting_records: dict[str, RealtimeAccounting] = {}
        kwargs["request_update_handler"] = self._ingest_request_update
        super().__init__(*args, **kwargs)

    @staticmethod
    def _validate_kv_pool_capacity(allocator: Any, server_args: Any) -> None:
        """Fail fast when the KV pool can never serve the configured shape.

        A pool smaller than one session's context guarantees a mid-stream OOM
        abort for every long session; refuse to boot with an actionable
        message instead. Overcommit (pool < N × context) stays legal — the
        memory-pressure guards degrade heaviest-first — but earns a warning.
        """
        pool_tokens = int(getattr(allocator, "size", 0) or 0)
        context_length = int(getattr(server_args, "context_length", 0) or 0)
        max_running = int(getattr(server_args, "max_running_requests", 1) or 1)
        if not pool_tokens or not context_length:
            return
        if pool_tokens < context_length:
            raise ValueError(
                f"KV pool holds {pool_tokens} tokens, less than one realtime "
                f"session context ({context_length}); raise "
                "--mem-fraction-static or lower --context-length"
            )
        if pool_tokens < max_running * context_length:
            logger.warning(
                "KV pool (%d tokens) cannot hold %d full %d-token sessions; "
                "under memory pressure the heaviest session is aborted first",
                pool_tokens,
                max_running,
                context_length,
            )

    def _enqueue_built_request(
        self,
        payload: Any,
        pending_stream_done: bool,
        req_data: Any,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
        state = _runtime_state(req_data)
        if state.accounting is not None:
            self._prune_accounting_records()
            self._accounting_records[state.request_id] = state.accounting
        context_limit = getattr(
            getattr(self, "server_args", None), "context_length", None
        )
        if context_limit is not None:
            state.context_limit = int(context_limit)
        if (
            state.accounting is not None
            and state.context_limit is not None
            and len(req_data.req.origin_input_ids) + 1 > state.context_limit
        ):
            state.accounting.failure_code = "context_exhausted"
            state.accounting.freeze()
            self._emit_request_error(
                state.request_id,
                ContextExhaustedError(
                    "initial prompt exceeds the realtime context limit"
                ),
            )
            self.abort(state.request_id, defer_running_cleanup=False)
            return
        self.realtime_sessions.open(state.request_id, state.session_id)
        try:
            super()._enqueue_built_request(
                payload,
                pending_stream_done,
                req_data,
                request_admission_lock_held=request_admission_lock_held,
            )
            if (
                state.accounting is not None
                and state.request_id in self._aborted_request_ids
            ):
                state.accounting.freeze()
                self.realtime_sessions.close(state.request_id)
        except Exception:
            self.realtime_sessions.close(state.request_id)
            record = self._accounting_records.get(state.request_id)
            if record is not None:
                record.freeze()
            raise

    def _prune_accounting_records(self) -> None:
        records = self._accounting_records
        now = time.monotonic()
        retired = sorted(
            (record.retired_at, rid)
            for rid, record in records.items()
            if record.retired_at is not None
        )
        for retired_at, rid in retired:
            if now - retired_at > 300 or len(records) >= 4096:
                records.pop(rid, None)

    def _emit_request_error(self, request_id: str, error: Exception) -> None:
        record = getattr(self, "_accounting_records", {}).get(request_id)
        if record is not None and record.failure_code is None:
            record.failure_code = error_code(error)
            record.failure_seq = getattr(error, "seq_no", record.failure_seq)
        super()._emit_request_error(request_id, error)

    def _run_admin_action(self, action, payload=None):
        if action == "realtime_v2_info":
            return {
                "success": True,
                "message": "ok",
                "data": {
                    "realtime_v2": True,
                    "context_limit": int(self.server_args.context_length),
                },
            }
        if action != FINALIZE_ACTION:
            return super()._run_admin_action(action, payload)
        payload = dict(payload or {})
        rid, sid = payload.get("request_id"), payload.get("session_id")
        if not isinstance(rid, str) or not isinstance(sid, str):
            raise ValueError("request_id and session_id are required")
        # Admin operations execute on the scheduler owner, on all TP ranks.
        # Resolve any launched step before abort/freezing its logical counters.
        pending = getattr(self, "_async_pending", None)
        if pending is not None and any(req.rid == rid for req in pending[0].reqs):
            self._resolve_pending_async()
        record = self._accounting_records.get(rid)
        data = self._find_request_data(rid)
        if record is None and data is not None:
            raise ValueError("request does not have v2 accounting")
        if record is not None and record.session_id != sid:
            raise ValueError("accounting session ownership mismatch")
        if record is None:
            # Cancelling a not-yet-admitted build cannot have model positions.
            self.abort(rid, defer_running_cleanup=False)
            record = RealtimeAccounting(sid)
        elif not record.frozen:
            self.abort(rid, defer_running_cleanup=False)
        snapshot = record.freeze()
        return {
            "success": True,
            "message": "ok",
            "data": {
                "request_id": rid,
                "session_id": sid,
                "usage": snapshot,
                "watermark": record.step,
                "final": True,
                "failure_code": record.failure_code,
                "seq_no": record.failure_seq,
            },
        }

    def _ingest_request_update(self, req_data: Any, data: Any) -> None:
        state = _runtime_state(req_data)
        try:
            self.realtime_sessions.ingest(state.request_id, data)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            # A malformed/out-of-order client event must not crash the stage:
            # reject the update, surface the error, and abort only the
            # offending request.
            logger.exception("Rejected realtime update for %s", state.request_id)
            self._emit_request_error(state.request_id, exc)
            self.abort(state.request_id, defer_running_cleanup=False)

    def get_next_batch_to_run(self) -> Any | None:
        self._expire_parked_requests()
        self._abort_retracted_realtime_requests()
        if (
            self._realtime_extend_batch is None
            and self._async_pending is not None
            and self._has_pending_realtime_events()
        ):
            self._resolve_pending_async()
        try:
            return super().get_next_batch_to_run()
        except _RealtimeDecodeDeferred:
            # get_new_batch_prefill retained the reconciled running batch.
            # The event loop can now clear last_batch without losing requests.
            return None

    def _evaluate_frame_window(self) -> None:
        """Evict/pool aged vision frames during a decode gap.

        Runs between scheduling rounds (never mid-transaction). The plan is
        derived purely from committed frame timestamps, so every TP rank
        reaches the identical decision without a broadcast. An in-flight
        async lookahead step is resolved first: rewriting the page row while
        its forward reads the encoder region would serve stale indices.
        """
        config = getattr(self, "frame_window_config", None)
        if config is None:
            return
        candidates: list[tuple[Any, MossVLRealtimeRuntimeState]] = []
        seen: set[str] = set()
        running_reqs = tuple(getattr(self.running_batch, "reqs", ()) or ())
        for req in running_reqs + tuple(self.parked_reqs.values()):
            if req.rid in seen or req.finished():
                continue
            seen.add(req.rid)
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            if not isinstance(state, MossVLRealtimeRuntimeState):
                continue
            if state._append_inflight or state.phase not in (
                MossVLRealtimePhase.DECODING,
                MossVLRealtimePhase.WAITING_FOR_EVENT,
            ):
                continue
            if not state.frame_records or state.req_pool_index is None:
                continue
            candidates.append((req, state))
        if not candidates or self._realtime_extend_batch is not None:
            return
        if self._async_pending is not None:
            self._resolve_pending_async()
        for req, state in candidates:
            try:
                plan = plan_frame_window(tuple(state.frame_records), config)
                if plan is None:
                    continue
                event = apply_frame_window_plan(
                    req,
                    state,
                    plan,
                    records=tuple(state.frame_records),
                    req_to_token=self.req_to_token_pool.req_to_token,
                    allocator=self.token_to_kv_pool_allocator,
                    kv_pool_provider=self._token_to_kv_pool,
                    running_batch=self.running_batch,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to apply realtime frame window for %s", req.rid
                )
                self._emit_request_error(req.rid, exc)
                self.abort(req.rid, defer_running_cleanup=False)
                continue
            logger.info(
                "Realtime frame window %s: evicted_virtual=%d pooled_raw=%d "
                "produced_virtual=%d dropped_raw=%d encoder_length=%d->%d "
                "surviving_frames=%d pool_free_slots=%s",
                event.request_id,
                event.evicted_virtual_frames,
                event.pooled_raw_frames,
                event.produced_virtual_frames,
                event.dropped_raw_frames,
                event.encoder_length_before,
                event.encoder_length_after,
                event.surviving_frame_count,
                event.pool_free_slots,
            )

    def _token_to_kv_pool(self) -> Any | None:
        worker = getattr(self, "tp_worker", None) or getattr(self, "model_worker", None)
        runner = getattr(worker, "model_runner", None)
        return getattr(runner, "token_to_kv_pool", None)

    def _decode_rate_limited(self) -> bool:
        """Defer ordinary decode without delaying frame or prompt ingestion.

        Keeping requests in ``running_batch`` lets the scheduler continue
        draining control messages while preserving their KV ownership. With
        multiple live sessions the decode step is shared, so decode is
        deferred only while *every* request is still inside its token-rate
        interval; a request whose interval has elapsed releases the batch
        (requests not yet due ride along, which makes their pacing a soft
        target rather than an exact throttle).
        """
        reqs = tuple(getattr(self.running_batch, "reqs", ()))
        limited = False
        found = False
        for req in reqs:
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            if not isinstance(state, MossVLRealtimeRuntimeState):
                continue
            found = True
            if time.monotonic() >= state.next_decode_not_before:
                limited = False
                break
            limited = True
        if not found:
            return False
        return self._tp_consistent_decision(limited)

    def _tp_consistent_decision(self, decision: Any) -> Any:
        """Keep batch-production decisions identical across TP ranks.

        Rate limits and parked-expiry deadlines read each rank's local clock;
        without this, ranks could disagree at boundary instants and diverge
        the forward lockstep (NCCL hang). Rank 0 decides and the value is
        broadcast, mirroring the OmniScheduler idiom that disables
        clock-based prefill coalescing for TP>1.
        """
        if getattr(self, "tp_size", 1) == 1:
            return decision
        # broadcast_pyobj serializes on the source rank (len() required), so
        # scalar decisions travel in a one-element list.
        (result,) = broadcast_pyobj(
            [decision],
            self.tp_group.rank,
            self.tp_cpu_group,
            src=self.tp_group.ranks[0],
        )
        return result

    def _abort_retracted_realtime_requests(self) -> None:
        """Fail fast if a realtime request got retracted anyway.

        A retracted realtime request cannot be re-prefilled: its
        encoder-region KV was written incrementally by the vision encoder
        and the pixel inputs of earlier frames no longer exist, so upstream's
        re-prefill from token ids would corrupt the session. Remove it from
        the waiting queue and abort the session cleanly instead. This covers
        forced retraction (SGLANG_TEST_RETRACT) and any retract entry point
        the preemption guard in ``_preempt_decode_memory_pressure`` misses.
        """
        waiting = getattr(self, "waiting_queue", None)
        if not waiting:
            return
        victims = [
            req.rid
            for req in waiting
            if getattr(req, "is_retracted", False)
            and getattr(req, RUNTIME_STATE_ATTR, None) is not None
        ]
        for rid in self._tp_consistent_decision(victims):
            logger.error(
                "Realtime request %s was retracted; aborting the session "
                "because incremental encoder state cannot be re-prefilled",
                rid,
            )
            self._emit_request_error(
                rid,
                RuntimeError(
                    "realtime session was retracted under memory pressure and "
                    "cannot be re-prefilled; the session is aborted"
                ),
            )
            self.abort(rid, defer_running_cleanup=False)

    @staticmethod
    def _realtime_kv_cost(req: Any) -> int:
        state = getattr(req, RUNTIME_STATE_ATTR, None)
        if isinstance(state, MossVLRealtimeRuntimeState):
            return state.encoder_length + state.decoder_length
        return len(getattr(req, "output_ids", ()))

    def _preempt_decode_memory_pressure(self) -> None:
        """Abort the heaviest session when the next decode round cannot fit.

        Upstream ``retract_decode`` resets the least-preferred request for
        re-prefill, which corrupts realtime sessions (see
        ``_abort_retracted_realtime_requests``). When the KV pool cannot fit
        the next decode round, abort the heaviest session through the
        realtime-aware abort path instead: its rows are released immediately
        and the client gets an explicit error. Parked sessions retain KV and
        participate in victim selection. The last remaining live request is
        left to upstream's graceful OOM abort.
        """
        batch = getattr(self, "running_batch", None)
        reqs = getattr(batch, "reqs", None)
        if not reqs:
            return
        while reqs and not batch.check_decode_mem():
            live = {req.rid: req for req in reqs}
            for req in getattr(self, "parked_reqs", {}).values():
                live.setdefault(req.rid, req)
            if len(live) <= 1:
                break
            victim = max(live.values(), key=self._realtime_kv_cost)
            victim_rid = self._tp_consistent_decision(victim.rid)
            logger.warning(
                "KV pool cannot fit the next decode round; aborting realtime "
                "session %s (the heaviest request) instead of retracting it",
                victim_rid,
            )
            self._emit_request_error(
                victim_rid,
                RuntimeError(
                    "KV pool exhausted by concurrent sessions; this session "
                    "was aborted so remaining sessions can continue"
                ),
            )
            self.abort(victim_rid, defer_running_cleanup=False)
            batch = self.running_batch
            reqs = getattr(batch, "reqs", None) or []

    def _stamp_batch_launch(self, batch: Any) -> None:
        """Start the token-rate interval at forward launch, matching HF."""
        super()._stamp_batch_launch(batch)
        launch_time = float(batch.launch_ts)
        for req in batch.reqs:
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            if isinstance(state, MossVLRealtimeRuntimeState):
                state.next_decode_not_before = (
                    launch_time + 1.0 / state.max_tokens_per_turn
                )

    def _has_pending_realtime_events(self) -> bool:
        running_reqs = tuple(getattr(self.running_batch, "reqs", ()))
        for req in running_reqs + tuple(self.parked_reqs.values()):
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            if not isinstance(state, MossVLRealtimeRuntimeState) or req.finished():
                continue
            session = self.realtime_sessions.get(req.rid)
            if session is not None and session.pending_events:
                return True
        return False

    def get_new_batch_prefill(self, running_batch: Any) -> NextBatchPlan:
        # SGLang 0.5.16 calls this hook AFTER filtering/merging last_batch.
        # Use that authoritative batch for updates, eviction and rate checks.
        self.running_batch = running_batch
        self.last_batch = None
        self._evaluate_frame_window()
        if self._realtime_extend_batch is None:
            self._realtime_extend_batch = self._materialize_realtime_extensions()
        realtime_batch = self._realtime_extend_batch
        if realtime_batch is not None:
            self._realtime_extend_batch = None
            return NextBatchPlan(
                batch_to_run=realtime_batch,
                running_batch=self.running_batch,
            )
        plan = super().get_new_batch_prefill(self.running_batch)
        self.running_batch = plan.running_batch
        if plan.batch_to_run is not None:
            return plan
        if isinstance(self.running_batch, ScheduleBatch):
            self.running_batch = MossVLRealtimeScheduleBatch.from_batch(
                self.running_batch
            )
        self._guard_decode_capacity()
        if self._decode_rate_limited():
            # NextBatchPlan has no "idle with a nonempty running batch" flag;
            # stop here, before the upstream planner prepares decode KV.
            raise _RealtimeDecodeDeferred()
        self._preempt_decode_memory_pressure()
        return NextBatchPlan(batch_to_run=None, running_batch=self.running_batch)

    def _guard_decode_capacity(self) -> None:
        pool = getattr(self, "req_to_token_pool", None)
        if getattr(pool, "req_to_token", None) is None:
            return

        def violations():
            return {
                req.rid: error
                for req in tuple(getattr(self.running_batch, "reqs", ()))
                if not req.finished()
                and (error := realtime_decode_capacity_error(req, pool)) is not None
            }

        errors = self._tp_consistent_decision(violations())
        if errors and self._async_pending is not None:
            # Near the limit, resolve the outstanding step first: it may
            # finish the request, and its KV must not be freed while in flight.
            self._resolve_pending_async()
            errors = self._tp_consistent_decision(violations())
        for rid, message in errors.items():
            self._emit_request_error(rid, ContextExhaustedError(message))
            self.abort(rid, defer_running_cleanup=False)

    def _materialize_realtime_extensions(self) -> ScheduleBatch | None:
        running_batch = self.running_batch
        running_reqs = tuple(getattr(running_batch, "reqs", ()))
        candidate_reqs = running_reqs + tuple(self.parked_reqs.values())
        if not candidate_reqs:
            return None

        selected: list[tuple[Any, list[FramePromptEvent], MossVLRealtimeSegment]] = []
        seen_request_ids: set[str] = set()
        for req in candidate_reqs:
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            if not isinstance(state, MossVLRealtimeRuntimeState):
                continue
            if req.rid in seen_request_ids:
                # A request can appear in both running_batch and parked_reqs
                # transiently; never materialize it twice in one round.
                continue
            seen_request_ids.add(req.rid)
            session = self.realtime_sessions.get(req.rid)
            if (
                session is None
                or not session.pending_events
                or state.phase
                not in (
                    MossVLRealtimePhase.DECODING,
                    MossVLRealtimePhase.WAITING_FOR_EVENT,
                )
                or req.finished()
            ):
                continue
            # Drain every queued event in one round so the appended segment
            # matches the Transformers reference drain: prompts first in
            # arrival order, then frames sorted by timestamp.
            events = session.drain()
            try:
                segment = self._build_segment(req, state, events)
                _guard_realtime_context_capacity(
                    req, state, segment, self.req_to_token_pool
                )
                bind_realtime_page_row(
                    req,
                    self.req_to_token_pool.req_to_token,
                )
                _append_segment_to_request(req, state, segment)
                if self.frame_window_config is not None:
                    frame_records = stage_segment_frame_records(
                        segment,
                        merge_size=self.segment_builder.merge_size,
                    )
                    if frame_records is not None:
                        setattr(req, FRAME_RECORDS_STAGED_ATTR, frame_records)
            except Exception as exc:
                logger.exception("Failed to materialize realtime event for %s", req.rid)
                if state.accounting is not None and events:
                    state.accounting.failure_seq = events[-1].seq_no
                self._emit_request_error(req.rid, exc)
                self.abort(req.rid, defer_running_cleanup=False)
                continue
            selected.append((req, events, segment))

        if not selected:
            return None

        selected = self._enforce_extend_memory_budget(selected)
        if not selected:
            return None
        # Budget-enforcement aborts may have shrunk the live batch; refresh
        # the views the tail of this function filters on.
        running_batch = self.running_batch
        running_reqs = tuple(getattr(running_batch, "reqs", ()))

        reqs = [req for req, _, _ in selected]
        extend_batch = MossVLRealtimeScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )
        try:
            extend_batch.prepare_for_extend()
        # This boundary must contain allocator, processor, and model-specific errors.
        except Exception as exc:  # noqa: BLE001
            for req, _, segment in selected:
                _undo_appended_segment(req, segment)
                self._emit_request_error(req.rid, exc)
                self.abort(req.rid, defer_running_cleanup=False)
            return None
        extend_batch.prefill_stats = PrefillStats(
            log_input_tokens=sum(extend_batch.extend_lens),
            log_hit_tokens=sum(extend_batch.prefix_lens),
            new_token_ratio=float(getattr(self, "new_token_ratio", 0.0)),
            num_running_reqs=QueueCount.from_reqs(
                list(running_reqs),
                self.enable_priority_scheduling,
            ),
            num_new_seqs=len(reqs),
        )

        selected_ids = {req.rid for req in reqs}
        for request_id in selected_ids:
            if self.parked_reqs.pop(request_id, None) is not None:
                logger.info("Realtime request %s woke for a frame event", request_id)
            self.parked_since.pop(request_id, None)
        keep_indices = [
            index
            for index, req in enumerate(running_reqs)
            if req.rid not in selected_ids
        ]
        if running_batch is not None:
            running_batch.filter_batch(keep_indices=keep_indices)
            running_batch.batch_is_full = False
        return extend_batch

    def _enforce_extend_memory_budget(
        self,
        selected: list[tuple[Any, list[FramePromptEvent], MossVLRealtimeSegment]],
    ) -> list[tuple[Any, list[FramePromptEvent], MossVLRealtimeSegment]]:
        """Guarantee the extend batch fits the KV pool before allocation.

        ``prepare_for_extend`` allocates after the request mutations are
        staged; a mid-batch allocator failure strands partially allocated
        slots and has been observed to corrupt later sessions ("encoder and
        decoder slots must not overlap"). Measure first; when the pool cannot
        fit the batch, abort the heaviest live session (including parked
        ones, which still hold KV) through the realtime-aware path — explicit
        client error, immediate KV release — until it fits. Victimized
        extending requests are un-appended first so no staged mutation
        survives.
        """
        allocator = self.token_to_kv_pool_allocator
        if allocator is None or not hasattr(allocator, "available_size"):
            return selected
        staged_by_rid = {req.rid: (req, segment) for req, _, segment in selected}
        remaining = list(selected)

        def _needed() -> int:
            # Segment tokens +1 for the re-extended pending token and +1
            # decode-step headroom per extending request.
            return sum(len(segment.raw_append_ids) + 2 for _, _, segment in remaining)

        virtual_available = int(allocator.available_size())
        victims: list[str] = []
        live = {req.rid: req for req in getattr(self.running_batch, "reqs", ())}
        for req in self.parked_reqs.values():
            live.setdefault(req.rid, req)
        for req, _, _ in selected:
            live.setdefault(req.rid, req)
        while remaining and virtual_available < _needed():
            if not live:
                break
            victim = max(live.values(), key=self._realtime_kv_cost)
            victims.append(victim.rid)
            # Aborts run after planning; count each request's slots only once.
            del live[victim.rid]
            # Aborts free the victim's committed KV synchronously; kv_cost
            # slightly underestimates, which is the safe direction here.
            virtual_available += self._realtime_kv_cost(victim)
            remaining = [entry for entry in remaining if entry[0].rid != victim.rid]
        victims = self._tp_consistent_decision(victims)
        for victim_rid in victims:
            staged = staged_by_rid.get(victim_rid)
            if staged is not None:
                _undo_appended_segment(*staged)
            logger.warning(
                "KV pool cannot fit the pending frame extend; aborting "
                "realtime session %s (the heaviest request)",
                victim_rid,
            )
            self._emit_request_error(
                victim_rid,
                RuntimeError(
                    "KV pool exhausted by concurrent sessions; this session "
                    "was aborted so remaining sessions can continue"
                ),
            )
            self.abort(victim_rid, defer_running_cleanup=False)
        victim_ids = set(victims)
        return [entry for entry in selected if entry[0].rid not in victim_ids]

    def _resolve_and_process(
        self, batch: Any, sched_output: Any, pending_step: Any
    ) -> None:
        """Extend the lookahead overrun drop with silence-parked requests.

        A request parked after step N is still present in step N+1's
        already-launched batch: its overrun token must not be emitted,
        appended, or counted, and its decode KV slot must be freed. The runner
        side skips the state advance via the WAITING_FOR_EVENT phase check in
        ``post_decode_resolve``. Mirrors the upstream finished/retracted
        pre-drop, adding ``rid in parked_reqs`` to the predicate.
        """
        pre_drop = [
            r.finished() or r.is_retracted or r.rid in self.parked_reqs
            for r in batch.reqs
        ]
        if not any(pre_drop):
            return super()._resolve_and_process(batch, sched_output, pending_step)
        skip_rids = {batch.reqs[i].rid for i, was in enumerate(pre_drop) if was}
        result = self._run_batch_resolve(
            batch, sched_output, pending_step, skip_rids=skip_rids
        )
        if result is _FAILED_BATCH_RESULT:
            return
        keep = [i for i, was in enumerate(pre_drop) if not was]
        # Free only the parked rows' overrun slots: finished/retracted rows
        # follow the upstream cleanup contract (released with the request).
        parked_drop_indices = [
            i
            for i, req in enumerate(batch.reqs)
            if pre_drop[i] and not req.finished() and not req.is_retracted
        ]
        self._free_parked_overrun_step_slots(batch, parked_drop_indices)
        if result.next_token_ids is not None and keep:
            idx = torch.tensor(keep, device=result.next_token_ids.device)
            result.next_token_ids = result.next_token_ids[idx]
        batch.reqs = [batch.reqs[i] for i in keep]
        if batch.reqs:
            self.process_batch_result(batch, result)

    def _free_parked_overrun_step_slots(
        self, batch: Any, drop_indices: list[int]
    ) -> None:
        """Free parked rows' overrun decode slots and reconcile their accounting.

        Unlike finished/retracted rows (whose slots are released with the
        request), a parked request stays live: its overrun slot was counted in
        ``kv_allocated_len`` and written into the page row, so freeing it here
        *without* adjusting the request bookkeeping would double-free it when
        the parked request is later aborted/expired and releases its KV cache.
        """
        if not drop_indices:
            return
        if self.page_size != 1 or self.server_args.disable_radix_cache:
            return
        out_cache_loc = batch.out_cache_loc
        if out_cache_loc is None:
            logger.warning(
                "parked overrun step-slot free skipped: out_cache_loc is None"
            )
            return
        assert max(drop_indices) < out_cache_loc.numel(), (
            f"overrun drop index {max(drop_indices)} out of range "
            f"({out_cache_loc.numel()} step slots)"
        )
        idx = torch.tensor(drop_indices, dtype=torch.long, device=out_cache_loc.device)
        self.token_to_kv_pool_allocator.free(out_cache_loc[idx])
        req_to_token = self.req_to_token_pool.req_to_token
        for index in drop_indices:
            req = batch.reqs[index]
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            if state is None or state.req_pool_index is None:
                continue
            committed_total = state.encoder_length + state.decoder_length
            kv = getattr(req, "kv", None)
            if kv is not None and int(kv.kv_allocated_len) > committed_total:
                kv.kv_allocated_len = committed_total
            if int(getattr(req, "kv_committed_len", committed_total)) > committed_total:
                req.kv_committed_len = committed_total
            req_to_token[state.req_pool_index, committed_total] = 0

    def process_batch_result(self, batch: Any, result: Any) -> None:
        for req in batch.reqs:
            if getattr(req, "_moss_vl_realtime_final_extend", False):
                if not getattr(req, "_moss_vl_realtime_keep_ignore_eos", False):
                    req.sampling_params.ignore_eos = False
                del req._moss_vl_realtime_final_extend
        super().process_batch_result(batch, result)
        for req in tuple(batch.reqs):
            if req.finished():
                record = getattr(self, "_accounting_records", {}).get(req.rid)
                if record is not None:
                    state = getattr(req, RUNTIME_STATE_ATTR)
                    finish = req.finished_reason.to_json()
                    if (
                        finish.get("type") == "length"
                        and state.context_limit is not None
                        and record.snapshot()["total_tokens"] >= state.context_limit
                        and record.failure_code is None
                    ):
                        record.failure_code = "context_exhausted"
                    record.freeze()
                self.realtime_sessions.close(req.rid)
                self.parked_reqs.pop(req.rid, None)
                self.parked_since.pop(req.rid, None)
        self._park_silent_requests(batch)

    def _park_silent_requests(self, batch: Any) -> None:
        keep_indices: list[int] = []
        newly_parked_rids: list[str] = []
        for index, req in enumerate(tuple(batch.reqs)):
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            session = self.realtime_sessions.get(req.rid)
            is_silence = bool(req.output_ids) and _ends_with_token_ids(
                req.output_ids,
                self.silence_token_ids,
            )
            if is_silence or req.finished():
                reason = getattr(req, "finished_reason", None)
                logger.info(
                    "Realtime request %s result token=%s silence=%s "
                    "ignore_eos=%s finished=%s",
                    req.rid,
                    int(req.output_ids[-1]) if req.output_ids else None,
                    is_silence,
                    getattr(getattr(req, "sampling_params", None), "ignore_eos", None),
                    reason.to_json() if reason is not None else None,
                )
            should_park = (
                isinstance(state, MossVLRealtimeRuntimeState)
                and session is not None
                and not session.pending_events
                and not session.final_received
                and not req.finished()
                and is_silence
            )
            if not should_park:
                keep_indices.append(index)
                continue
            state.mark_waiting()
            self.parked_reqs[req.rid] = req
            self.parked_since[req.rid] = time.monotonic()
            newly_parked_rids.append(req.rid)
            logger.info("Realtime request %s parked after silence", req.rid)
        if len(keep_indices) != len(batch.reqs):
            if batch.seq_lens is None:
                # Async result snapshots omit the forward-only batch tensors.
                batch.reqs = [batch.reqs[i] for i in keep_indices]
            else:
                batch.filter_batch(keep_indices=keep_indices)
            batch.batch_is_full = False
        if newly_parked_rids:
            self._drop_parked_from_live_running_batch(batch, newly_parked_rids)

    def _drop_parked_from_live_running_batch(
        self, processed_batch: Any, parked_rids: list[str]
    ) -> None:
        """Remove newly parked requests from the live running batch.

        On the async (one-step lookahead) resolve path ``processed_batch`` is
        a snapshot copy; without this the parked request would keep ghost
        decoding from the live batch every step.
        """
        running_batch = getattr(self, "running_batch", None)
        if running_batch is None or running_batch is processed_batch:
            return
        running_reqs = tuple(getattr(running_batch, "reqs", ()))
        parked = set(parked_rids)
        keep = [i for i, req in enumerate(running_reqs) if req.rid not in parked]
        if len(keep) != len(running_reqs):
            running_batch.filter_batch(keep_indices=keep)
            running_batch.batch_is_full = False

    def _expire_parked_requests(self) -> None:
        now = time.monotonic()
        expired = [
            request_id
            for request_id, parked_at in self.parked_since.items()
            if now - parked_at >= self.parked_request_timeout_s
        ]
        expired = self._tp_consistent_decision(expired)
        for request_id in expired:
            self._emit_request_error(
                request_id,
                TimeoutError("realtime request exceeded parked idle timeout"),
            )
            self.abort(request_id, defer_running_cleanup=False)

    def _find_request_data(self, request_id: str) -> Any | None:
        data = super()._find_request_data(request_id)
        if data is not None:
            return data
        req = self.parked_reqs.get(request_id)
        return None if req is None else req._omni_data

    def _build_segment(
        self,
        req: Any,
        state: MossVLRealtimeRuntimeState,
        events: list[FramePromptEvent],
    ) -> MossVLRealtimeSegment:
        # Token space: fill ids retain one pad placeholder per historical
        # encoder slot, including frames the sliding window already evicted.
        committed_total = state.effective_appended_encoder_length + state.decoder_length
        req._refresh_fill_ids()
        pending_count = len(req.full_untruncated_fill_ids) - committed_total
        if pending_count != 1:
            raise RuntimeError(
                "realtime event must coalesce with exactly one pending sampled token"
            )
        common = {
            "previous_mrope_positions": state.mrope_positions,
            "previous_visible_counts": state.visible_frame_counts,
            "previous_grid_thw": state.full_grid_thw,
            "next_mrope_position": state.next_mrope_position,
            "committed_encoder_length": state.encoder_length,
            "committed_decoder_length": state.decoder_length,
            "pending_text_tokens": 1,
        }
        frame_events = [event for event in events if event.has_frame]
        images = self._resolve_frame_events_tp(frame_events)
        return self.segment_builder.build(events, images, **common)

    def _resolve_frame_events_tp(
        self, frame_events: list[FramePromptEvent]
    ) -> list[Any]:
        """Resolve frame pixels; under TP>1 only rank 0 touches the reference.

        Shared-memory frames are single-consumer (read-and-unlink): letting
        every rank resolve locally races (one rank unlinks before another
        opens). Rank 0 resolves and broadcasts (size, mode, raw bytes); other
        ranks decode from the broadcast, so every rank encodes identical
        pixels. A rank-0 failure is broadcast too, so all ranks raise the same
        error and abort in lockstep.
        """
        if getattr(self, "tp_size", 1) == 1:
            return [self.frame_resolver(event) for event in frame_events]
        marker: Any = None
        if self.tp_rank == 0:
            try:
                images = [self.frame_resolver(event) for event in frame_events]
                marker = (
                    True,
                    [(_as_raw_pixel_payload(img)) for img in images],
                )
            except Exception as exc:
                marker = (False, f"{type(exc).__name__}: {exc}")
        ok, result = broadcast_pyobj(
            marker,
            self.tp_group.rank,
            self.tp_cpu_group,
            src=self.tp_group.ranks[0],
        )
        if not ok:
            raise RuntimeError(f"rank-0 frame resolution failed: {result}")
        return [
            Image.frombytes(mode, size, raw).convert("RGB")
            for size, mode, raw in result
        ]

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        owner = getattr(self, "_scheduler_thread_id", None)
        if (
            getattr(self, "_running", False)
            and owner is not None
            and owner != threading.get_ident()
        ):
            # The control-plane listener is a different thread. Page rewrites,
            # in-flight forwards and KV release must stay on the scheduler owner.
            # Only the entry rank enqueues; recv_requests broadcasts to TP peers.
            if getattr(self, "is_entry_rank", getattr(self, "tp_rank", 0) == 0):
                self.inbox.put(
                    IncomingMessage(
                        request_id=request_id,
                        type="abort",
                        data={"defer_running_cleanup": defer_running_cleanup},
                    )
                )
            return
        pending = getattr(self, "_async_pending", None)
        if pending is not None and any(r.rid == request_id for r in pending[0].reqs):
            # Flush the in-flight lookahead step before touching this
            # request's state/KV; its row resolves (and is dropped when
            # already parked) through the normal path first.
            self._resolve_pending_async()
        parked_req = self.parked_reqs.get(request_id)
        data = self._find_request_data(request_id)
        if data is not None:
            state = getattr(data, "runtime_state", None)
            if (
                isinstance(state, MossVLRealtimeRuntimeState)
                and not state._append_inflight
            ):
                state.finish(aborted=True)
        self.realtime_sessions.close(request_id)
        super().abort(
            request_id,
            defer_running_cleanup=defer_running_cleanup,
        )
        self.parked_since.pop(request_id, None)
        parked_req = self.parked_reqs.pop(request_id, parked_req)
        if parked_req is not None:
            self._release_request_kv_cache(parked_req)
            parked_req._omni_data = None
        record = getattr(self, "_accounting_records", {}).get(request_id)
        if record is not None:
            record.freeze()

    def stop(self) -> None:
        for request_id in tuple(self.parked_reqs):
            self.abort(request_id, defer_running_cleanup=False)
        super().stop()


def _append_segment_to_request(
    req: Any,
    state: MossVLRealtimeRuntimeState,
    segment: MossVLRealtimeSegment,
) -> None:
    # Two coordinate spaces once the frame window has evicted frames:
    # - KV space (encoder_length): surviving page-table entries only.
    # - Token space (appended_encoder_length): every pad placeholder ever
    #   appended; fill ids, extend ranges and prefix_indices lengths live here
    #   because upstream asserts seq_len - len(prefix_indices) == extend_len.
    committed_total = state.encoder_length + state.decoder_length
    token_total = state.effective_appended_encoder_length + state.decoder_length
    req._refresh_fill_ids()
    if len(req.full_untruncated_fill_ids) != token_total + 1:
        raise RuntimeError("request does not have exactly one pending sampled token")
    state.pending_token_id = int(req.full_untruncated_fill_ids[-1])
    page_prefix = req._moss_vl_realtime_page_row[:committed_total]
    invalid = torch.nonzero(page_prefix <= 0, as_tuple=False).flatten()
    if invalid.numel():
        first_invalid = invalid[:8].tolist()
        kv = getattr(req, "kv", None)
        raise RuntimeError(
            "realtime committed KV prefix contains empty slots "
            f"(indices={first_invalid}, encoder_length={state.encoder_length}, "
            f"decoder_length={state.decoder_length}, "
            f"req_kv_committed_len={getattr(req, 'kv_committed_len', None)}, "
            f"req_kv_allocated_len={getattr(kv, 'kv_allocated_len', None)})"
        )
    req._moss_vl_realtime_previous_mm_inputs = req.multimodal_inputs
    req._moss_vl_realtime_previous_extend_range = req.extend_range
    req._moss_vl_realtime_previous_prefix_indices = req.prefix_indices
    req._moss_vl_realtime_previous_skip_radix_cache_insert = req.skip_radix_cache_insert
    req._moss_vl_realtime_previous_max_new_tokens = req.sampling_params.max_new_tokens
    req.skip_radix_cache_insert = True
    req._moss_vl_realtime_staged_mrope_positions = (
        segment.multimodal_inputs.mrope_positions.clone()
    )
    req._moss_vl_realtime_staged_visible_frame_counts = (
        segment.multimodal_inputs.visible_frame_counts.clone()
    )
    req._moss_vl_realtime_staged_full_grid_thw = segment.full_grid_thw.clone()
    req._moss_vl_realtime_staged_events = [event.to_dict() for event in segment.events]
    prompt_seq_nos = [
        event.seq_no for event in segment.events if event.prompt is not None
    ]
    if prompt_seq_nos:
        req._moss_vl_realtime_staged_turn_transition = {
            "interrupted_turn_id": state.turn_id,
            "turn_id": state.turn_id + len(prompt_seq_nos),
            "prompt_seq_nos": prompt_seq_nos,
        }
    if any(event.final for event in segment.events):
        req._moss_vl_realtime_final_extend = True
    allowance = _decode_allowance(req)
    req.output_ids.extend(segment.raw_append_ids)
    # Decode tokens (silence or text) grow output_ids without replenishing
    # max_new_tokens, so a purely additive update lets the upstream length
    # check kill any session whose cumulative decode count reaches the
    # initial allowance (~128 silences with the default client budget).
    # Re-anchor instead: the remaining allowance stays constant, i.e. the
    # session may decode up to <allowance> fresh tokens after every extend,
    # while the whole-session context remains a separate upper bound.
    context_limit = state.context_limit or int(req._moss_vl_realtime_page_row.numel())
    req.sampling_params.max_new_tokens = min(
        len(req.output_ids) + allowance,
        max(0, context_limit - len(req.origin_input_ids)),
    )
    req.multimodal_inputs = segment.multimodal_inputs
    req._refresh_fill_ids()
    # Token-length prefix for the upstream extend invariants; its content is
    # the current row itself, so the extend kernel's prefix rewrite is
    # idempotent (the token-only dead zone past the compacted KV prefix
    # round-trips unchanged).
    req.prefix_indices = req._moss_vl_realtime_page_row[:token_total].to(
        dtype=torch.int64
    )
    req.set_extend_range(token_total, len(req.full_untruncated_fill_ids))


def _decode_allowance(req: Any) -> int:
    """The constant output_ids-to-max_new_tokens gap for a realtime session.

    Established at request build time (state.decode_allowance); lazily
    derived once from the first extend for states that predate the field.
    """
    state = getattr(req, RUNTIME_STATE_ATTR)
    allowance = state.decode_allowance
    if allowance is None:
        allowance = req.sampling_params.max_new_tokens - len(req.output_ids)
        if allowance <= 0:
            raise ValueError(
                f"realtime request {state.request_id} has no decode allowance "
                f"left (max_new_tokens={req.sampling_params.max_new_tokens}, "
                f"output_ids={len(req.output_ids)})"
            )
        state.decode_allowance = allowance
    return allowance


def _undo_appended_segment(req: Any, segment: MossVLRealtimeSegment) -> None:
    count = len(segment.raw_append_ids)
    if count:
        del req.output_ids[-count:]
    req.sampling_params.max_new_tokens = req._moss_vl_realtime_previous_max_new_tokens
    del req._moss_vl_realtime_previous_max_new_tokens
    req.multimodal_inputs = req._moss_vl_realtime_previous_mm_inputs
    req.extend_range = req._moss_vl_realtime_previous_extend_range
    req.prefix_indices = req._moss_vl_realtime_previous_prefix_indices
    req.skip_radix_cache_insert = req._moss_vl_realtime_previous_skip_radix_cache_insert
    del req._moss_vl_realtime_previous_mm_inputs
    del req._moss_vl_realtime_previous_extend_range
    del req._moss_vl_realtime_previous_prefix_indices
    del req._moss_vl_realtime_previous_skip_radix_cache_insert
    del req._moss_vl_realtime_staged_mrope_positions
    del req._moss_vl_realtime_staged_visible_frame_counts
    del req._moss_vl_realtime_staged_full_grid_thw
    del req._moss_vl_realtime_staged_events
    if hasattr(req, FRAME_RECORDS_STAGED_ATTR):
        delattr(req, FRAME_RECORDS_STAGED_ATTR)
    if hasattr(req, "_moss_vl_realtime_staged_turn_transition"):
        del req._moss_vl_realtime_staged_turn_transition
    if hasattr(req, "_moss_vl_realtime_final_extend"):
        del req._moss_vl_realtime_final_extend
    req._refresh_fill_ids()
    state = getattr(req, RUNTIME_STATE_ATTR)
    state.pending_token_id = None


def _guard_realtime_context_capacity(
    req: Any,
    state: MossVLRealtimeRuntimeState,
    segment: MossVLRealtimeSegment,
    req_to_token_pool: Any,
) -> None:
    """Reject appends that would write past the page-table row width.

    Realtime requests bypass admission's capacity check because every extend
    also grows ``max_new_tokens``; without this guard, an over-length session
    would reach the upstream triton kernel's unbounded req_to_token write.
    Checked before any request mutation, so the request stays coherent and is
    finished through the normal error path.
    """
    row_width = int(req_to_token_pool.req_to_token.shape[1])
    context_length = state.context_limit or row_width
    # Historical token positions and retained physical KV have separate
    # bounds once window eviction has compacted the page row.
    kv_total = state.encoder_length + state.decoder_length
    token_total = state.effective_appended_encoder_length + state.decoder_length
    # The pending sampled token is part of this forward's input. Reserve
    # another logical position for the token produced by that forward.
    projected_kv = kv_total + len(segment.raw_append_ids) + 1
    projected_total = token_total + len(segment.raw_append_ids) + 2
    if projected_total > context_length or projected_kv > row_width:
        raise ContextExhaustedError(
            "realtime request exhausted the context length: "
            f"projected token positions {projected_total}/{context_length}, "
            f"KV positions {projected_kv}/{row_width}; "
            "the session cannot accept more input and must end"
        )


def bind_realtime_page_row(req: Any, req_to_token: torch.Tensor) -> None:
    """Bind the live page-table row used when converting decode to extend."""
    req_pool_index = int(req.req_pool_idx)
    req._moss_vl_realtime_page_row = req_to_token[req_pool_index]


def _as_raw_pixel_payload(image: Any) -> tuple[tuple[int, int], str, bytes]:
    """Pack a resolved frame for the TP pixel broadcast as (size, mode, raw)."""
    size = getattr(image, "size", None)
    mode = getattr(image, "mode", None)
    if size is None or mode is None or not hasattr(image, "tobytes"):
        raise TypeError("frame resolvers must return a PIL image")
    return tuple(size), str(mode), image.tobytes()


def resolve_local_frame(event: FramePromptEvent) -> Image.Image:
    """Resolve local/file refs and one-consumer shared-memory frames."""
    if event.frame_ref is None:
        raise ValueError("text-only event has no frame to resolve")
    parsed = urlparse(event.frame_ref)
    if parsed.scheme == "shm":
        return resolve_shared_memory_frame(event.frame_ref)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme == "":
        path = Path(event.frame_ref)
    else:
        raise ValueError(
            f"frame_ref scheme {parsed.scheme!r} needs a media-relay resolver"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB")


def _runtime_state(req_data: Any) -> MossVLRealtimeRuntimeState:
    state = getattr(req_data, "runtime_state", None)
    if not isinstance(state, MossVLRealtimeRuntimeState):
        raise TypeError("MOSS-VL realtime request data is missing runtime_state")
    return state


def _ends_with_token_ids(values: Any, suffix: tuple[int, ...]) -> bool:
    if len(values) < len(suffix):
        return False
    return tuple(int(value) for value in values[-len(suffix) :]) == suffix
