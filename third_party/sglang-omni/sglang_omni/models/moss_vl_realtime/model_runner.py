"""Model-runner lifecycle hooks for MOSS-VL realtime KV state."""

from __future__ import annotations

from typing import Any

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.moss_vl_realtime.batch_adapter import (
    RUNTIME_STATE_ATTR,
    commit_moss_vl_realtime_batch,
    commit_moss_vl_realtime_decode,
    is_moss_vl_realtime_batch,
    rollback_moss_vl_realtime_batch,
)
from sglang_omni.models.moss_vl_realtime.frame_window import FRAME_RECORDS_STAGED_ATTR
from sglang_omni.models.moss_vl_realtime.runtime_state import (
    MossVLRealtimePhase,
    MossVLRealtimeRuntimeState,
)


class MossVLRealtimeModelRunner(ModelRunner):
    """Commit staged encoder appends and advance decoder position state."""

    def execute(self, scheduler_output: Any):
        batch = scheduler_output.batch_data
        try:
            return super().execute(scheduler_output)
        except Exception:
            if batch is not None and is_moss_vl_realtime_batch(batch):
                self._wait_failed_forward()
                rollback_moss_vl_realtime_batch(batch)
            raise

    def _wait_failed_forward(self) -> None:
        bridge = getattr(self, "_execution_bridge", None)
        if bridge is not None:
            bridge.record_completion().synchronize()

    def execute_launch(self, scheduler_output):
        try:
            return super().execute_launch(scheduler_output)
        except Exception:
            self._wait_failed_forward()
            batch = scheduler_output.batch_data
            if batch is not None and is_moss_vl_realtime_batch(batch):
                rollback_moss_vl_realtime_batch(batch)
            raise

    def execute_resolve(self, pending):
        try:
            return super().execute_resolve(pending)
        except Exception:
            # A newer lookahead step may already reference this step's KV.
            # Keep all request-owned slots intact for scheduler abort, which
            # drains the newer pending step and then releases the whole row.
            self._wait_failed_forward()
            raise

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del result, forward_batch, requests
        if is_moss_vl_realtime_batch(schedule_batch):
            accounting_before = {}
            for req in schedule_batch.reqs:
                state = getattr(req, RUNTIME_STATE_ATTR)
                if state.accounting is not None:
                    accounting_before[req.rid] = (
                        state.effective_appended_encoder_length,
                        state.decoder_length,
                        int(state.pending_token_id is not None),
                    )
            commit_moss_vl_realtime_batch(schedule_batch)
            for index, req in enumerate(schedule_batch.reqs):
                state = getattr(req, RUNTIME_STATE_ATTR)
                if state.req_pool_index is None:
                    state.bind_req_pool_index(int(req.req_pool_idx))
                    decoder_length = int(schedule_batch.seq_lens_cpu[index].item())
                    state.decoder_length = decoder_length
                    state.next_mrope_position = decoder_length
                    state.phase = MossVLRealtimePhase.DECODING
                state.pending_token_id = None
                staged_events = getattr(
                    req,
                    "_moss_vl_realtime_staged_events",
                    None,
                )
                if staged_events is not None:
                    transition = getattr(
                        req,
                        "_moss_vl_realtime_staged_turn_transition",
                        None,
                    )
                    # Assign one turn boundary per prompt, in prompt order.
                    turn_transitions: dict[int, dict[str, int]] = {}
                    if transition is not None:
                        turn_cursor = int(transition["interrupted_turn_id"])
                        for seq_no in transition["prompt_seq_nos"]:
                            turn_transitions[int(seq_no)] = {
                                "interrupted_turn_id": turn_cursor,
                                "turn_id": turn_cursor + 1,
                            }
                            turn_cursor += 1
                        state.turn_id = int(transition["turn_id"])
                        del req._moss_vl_realtime_staged_turn_transition
                    processed_events: list[dict[str, Any]] = []
                    for staged_event in staged_events:
                        processed_event = dict(staged_event)
                        turn_transition = turn_transitions.get(
                            int(processed_event["seq_no"])
                        )
                        if turn_transition is not None:
                            processed_event.update(turn_transition)
                        processed_events.append(processed_event)
                    req._moss_vl_realtime_processed_events = processed_events
                    del req._moss_vl_realtime_staged_events
                for attr_name, state_name in (
                    ("_moss_vl_realtime_staged_mrope_positions", "mrope_positions"),
                    (
                        "_moss_vl_realtime_staged_visible_frame_counts",
                        "visible_frame_counts",
                    ),
                    ("_moss_vl_realtime_staged_full_grid_thw", "full_grid_thw"),
                ):
                    if hasattr(req, attr_name):
                        setattr(state, state_name, getattr(req, attr_name))
                        delattr(req, attr_name)
                # Frame-window bookkeeping: only ever staged when the window
                # is enabled, so this is a no-op on the default path.
                staged_records = getattr(req, FRAME_RECORDS_STAGED_ATTR, None)
                if staged_records is not None:
                    if state.frame_records is None:
                        state.frame_records = []
                    state.frame_records.extend(staged_records)
                    delattr(req, FRAME_RECORDS_STAGED_ATTR)
                for name in (
                    "_moss_vl_realtime_previous_mm_inputs",
                    "_moss_vl_realtime_previous_extend_range",
                    "_moss_vl_realtime_previous_prefix_indices",
                    "_moss_vl_realtime_previous_skip_radix_cache_insert",
                    "_moss_vl_realtime_previous_max_new_tokens",
                ):
                    if hasattr(req, name):
                        delattr(req, name)
                if state.accounting is not None:
                    vision, decoder, pending = accounting_before[req.rid]
                    state.accounting.commit(
                        vision=state.effective_appended_encoder_length - vision,
                        text_input=state.decoder_length - decoder - pending,
                    )

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del result, forward_batch, requests
        for req in schedule_batch.reqs:
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            if not isinstance(state, MossVLRealtimeRuntimeState):
                continue
            if state.phase in (
                MossVLRealtimePhase.FINISHED,
                MossVLRealtimePhase.ABORTED,
            ):
                continue
            state.decoder_length += 1
            state.next_mrope_position += 1
            state.phase = MossVLRealtimePhase.DECODING
            if state.accounting is not None and not state.accounting.frozen:
                state.accounting.commit()

        commit_moss_vl_realtime_decode(schedule_batch)

    def finalize_skip_rids(self, scheduler_output: Any) -> set[str]:
        """Keep the silence-park overrun row from counting as a generation step.

        A parked (WAITING_FOR_EVENT) request can only appear in a batch as the
        one-step lookahead overrun; sync batches never contain parked requests
        (they are filtered out at park time), so this is a no-op off the async
        path.
        """
        skip: set[str] = set()
        for sched_req in scheduler_output.requests:
            state = getattr(sched_req.data.req, RUNTIME_STATE_ATTR, None)
            if (
                isinstance(state, MossVLRealtimeRuntimeState)
                and state.phase is MossVLRealtimePhase.WAITING_FOR_EVENT
            ):
                skip.add(sched_req.request_id)
        return skip

    def lookahead_eligible(self, batch: Any) -> bool:
        """Allow one-step lookahead only for MOSS realtime decode batches.

        The base default only gates history-dependent sampling; MOSS batches
        additionally rely on the runner/scheduler async machinery shipped
        together with this override: state advancement here in
        ``post_decode_resolve``, the scheduler-side update barrier, and the
        parked-overrun row drop.
        """
        if not is_moss_vl_realtime_batch(batch):
            return False
        return super().lookahead_eligible(batch)

    def post_decode_resolve(
        self,
        launch_buf: Any,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        super().post_decode_resolve(
            launch_buf, result, forward_batch, schedule_batch, requests
        )
        for req in schedule_batch.reqs:
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            if not isinstance(state, MossVLRealtimeRuntimeState):
                continue
            if state.phase in (
                MossVLRealtimePhase.FINISHED,
                MossVLRealtimePhase.ABORTED,
                # WAITING_FOR_EVENT rows are the silence-park overrun step:
                # launched one iteration before the park decision was known.
                # The scheduler drops their tokens; the runtime state must not
                # advance for a step that never commits.
                MossVLRealtimePhase.WAITING_FOR_EVENT,
            ):
                continue
            state.decoder_length += 1
            state.next_mrope_position += 1
            state.phase = MossVLRealtimePhase.DECODING
            if state.accounting is not None and not state.accounting.frozen:
                state.accounting.commit()

        commit_moss_vl_realtime_decode(schedule_batch)
