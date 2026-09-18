# SPDX-License-Identifier: Apache-2.0
"""Coordinator for managing the multi-stage pipeline."""

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from sglang_omni.pipeline.control_plane import CoordinatorControlPlane
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import (
    AbortMessage,
    AdminMessage,
    AdminOperation,
    AdminResult,
    AdminResultMessage,
    CompleteMessage,
    OmniRequest,
    RequestInfo,
    RequestState,
    RequestUpdateMessage,
    StagePayload,
    StreamMessage,
    SubmitMessage,
    is_update_action,
)

logger = logging.getLogger(__name__)


@dataclass
class _AdminPendingOperation:
    expected_stages: set[str]
    action: str
    # Per stage, every replica's result in reply order. Aggregation collapses
    # to one entry per stage (replica detail attached under ``data`` when the
    # stage is replicated).
    results: dict[str, list[AdminResult]] = field(default_factory=dict)
    # Remaining replica results per stage; reaches zero when every replica of
    # every target stage reported.
    pending_replica_counts: dict[str, int] = field(default_factory=dict)
    future: asyncio.Future | None = None


@dataclass
class ReplicaInfo:
    """One data-parallel replica of a registered stage.

    ``capacity``/``inflight`` are the occupancy hook for admission-aware
    replica selection: ``capacity=None`` means "no admission limit known to
    the coordinator", in which case selection is plain round-robin.
    """

    dp_rank: int
    endpoint: str
    inflight: int = 0
    capacity: int | None = None

    @property
    def is_full(self) -> bool:
        return self.capacity is not None and self.inflight >= self.capacity


class Coordinator:
    """Central coordinator for the multi-stage pipeline.

    Responsibilities:
    - Register stages
    - Submit requests to entry stage
    - Track request state
    - Handle completions
    - Broadcast abort signals
    """

    def __init__(
        self,
        completion_endpoint: str,
        abort_endpoint: str,
        entry_stage: str,
        terminal_stages: list[str] | None = None,
        terminal_stages_resolver: (
            Callable[[OmniRequest], list[str] | None] | None
        ) = None,
    ):
        """Initialize coordinator.

        Args:
            completion_endpoint: ZMQ endpoint to receive completions
            abort_endpoint: ZMQ endpoint for abort broadcasts
            entry_stage: Name of the entry stage for new requests
            terminal_stages: Terminal stage names. When multiple are given,
                the coordinator waits for all to complete before resolving.
        """
        self.entry_stage = entry_stage
        self._terminal_stages: set[str] = (
            set(terminal_stages) if terminal_stages else set()
        )
        self._terminal_stages_resolver = terminal_stages_resolver
        self._partial_results: dict[str, dict[str, Any]] = {}

        # Control plane
        self.control_plane = CoordinatorControlPlane(
            completion_endpoint=completion_endpoint,
            abort_endpoint=abort_endpoint,
        )

        # Stage registry: stage name -> per-dp-replica endpoints (one entry
        # per replica; len == 1 for stages without native data parallelism).
        self._stages: dict[str, list[ReplicaInfo]] = {}
        self._rr_cursors: dict[str, int] = {}

        # Request tracking
        self._requests: dict[str, RequestInfo] = {}
        self._completion_futures: dict[str, asyncio.Future] = {}
        self._stream_queues: dict[
            str, asyncio.Queue[CompleteMessage | StreamMessage]
        ] = {}
        # Abort messages carry only the request ID. A strongly held task keeps
        # local admission closed and lets the broadcast survive caller cancellation.
        self._abort_tasks: dict[str, asyncio.Task[bool]] = {}
        self._admin_ops: dict[str, _AdminPendingOperation] = {}
        self._admin_lock = asyncio.Lock()

        # State
        self._running = False
        self._fatal_error: str | None = None

    def register_stage(
        self,
        name: str,
        endpoint: str,
        *,
        dp_rank: int | None = None,
        capacity: int | None = None,
    ) -> None:
        """Register a stage endpoint, appending a replica slot per DP rank.

        Args:
            name: Stage name
            endpoint: ZMQ endpoint for the stage replica's external I/O
            dp_rank: Data-parallel replica index; ``None`` assigns the next
                rank in registration order (single-replica stages get 0).
            capacity: Optional admission limit used by replica selection
                (e.g. per-replica max concurrent requests).
        """
        replicas = self._stages.setdefault(name, [])
        resolved_dp_rank = len(replicas) if dp_rank is None else dp_rank
        replicas.append(
            ReplicaInfo(dp_rank=resolved_dp_rank, endpoint=endpoint, capacity=capacity)
        )
        logger.info("Coordinator registered stage: %s at %s", name, endpoint)
        if len(replicas) > 1:
            logger.info(
                "Stage %s now has %d data-parallel replicas", name, len(replicas)
            )

    def replicas(self, stage_name: str) -> list[ReplicaInfo]:
        """Live replica handles for a stage (Phase-3 admission reads/mutates
        occupancy counters through these shared objects)."""
        return list(self._stages.get(stage_name, ()))

    def _replica_for(self, stage_name: str, dp_rank: int) -> ReplicaInfo | None:
        for replica in self._stages.get(stage_name, ()):
            if replica.dp_rank == dp_rank:
                return replica
        return None

    def _select_replica(self, stage_name: str) -> ReplicaInfo:
        """Round-robin over replicas, preferring ones below their capacity."""
        replicas = self._stages[stage_name]
        count = len(replicas)
        cursor = self._rr_cursors.get(stage_name, 0) % count
        for offset in range(count):
            replica = replicas[(cursor + offset) % count]
            if not replica.is_full:
                self._rr_cursors[stage_name] = (cursor + offset + 1) % count
                return replica
        # Every replica is at capacity: keep plain round-robin admission and
        # let the stage scheduler make the final accept/reject decision.
        self._rr_cursors[stage_name] = (cursor + 1) % count
        return replicas[cursor]

    def _release_replica(self, info: RequestInfo) -> None:
        if info.owner_dp_rank is None:
            return
        replica = self._replica_for(
            info.current_stage or self.entry_stage, info.owner_dp_rank
        )
        if replica is not None:
            replica.inflight = max(0, replica.inflight - 1)
        info.owner_dp_rank = None
        info.owner_endpoint = None

    async def start(self) -> None:
        """Start the coordinator."""
        await self.control_plane.start()
        self._running = True
        logger.info("Coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False
        self.control_plane.close()
        logger.info("Coordinator stopped")

    async def fail_pending_requests(self, error: BaseException | str) -> None:
        """Fail all requests currently owned by the coordinator."""
        self._running = False
        message = str(error)
        self._fatal_error = message
        for request_id, info in list(self._requests.items()):
            info.state = RequestState.FAILED
            info.error = message
            self._reject_completion_future(request_id, RuntimeError(message))
            queue = self._stream_queues.get(request_id)
            if queue is not None:
                await queue.put(
                    CompleteMessage(
                        request_id=request_id,
                        from_stage="coordinator",
                        success=False,
                        error=message,
                    )
                )
            self._release_replica(info)
        self._requests.clear()
        self._partial_results.clear()

    async def shutdown_stages(self) -> None:
        """Send shutdown signal to all registered stages."""
        for name, replicas in self._stages.items():
            for replica in replicas:
                try:
                    await self.control_plane.send_shutdown(name, replica.endpoint)
                    logger.info("Sent shutdown to stage: %s", name)
                except Exception as e:
                    logger.warning("Failed to send shutdown to stage %s: %s", name, e)

    async def admin(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Run an administrative operation against one or more stages."""
        if not self._running:
            raise RuntimeError("Coordinator is not running")

        target_stages = self._resolve_admin_stages(stages)
        if not target_stages:
            raise ValueError("No stages registered for admin operation")

        op_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        pending = _AdminPendingOperation(
            expected_stages=set(target_stages),
            action=action,
            pending_replica_counts={
                stage_name: len(self._stages[stage_name])
                for stage_name in target_stages
            },
            future=loop.create_future(),
        )
        operation = AdminOperation(
            op_id=op_id,
            action=action,
            payload=dict(payload or {}),
            target_stages=list(target_stages),
            timeout_s=timeout_s,
        )

        async with self._admin_lock:
            self._admin_ops[op_id] = pending
            try:
                for stage_name in target_stages:
                    for replica in self._stages[stage_name]:
                        await self.control_plane.send_admin(
                            stage_name,
                            replica.endpoint,
                            AdminMessage(operation=operation),
                        )

                assert pending.future is not None
                results_by_stage = await asyncio.wait_for(
                    pending.future, timeout=timeout_s
                )
            finally:
                self._admin_ops.pop(op_id, None)

        return self._aggregate_admin_results(
            op_id=op_id,
            action=action,
            results_by_stage=results_by_stage,
        )

    async def model_info(
        self,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "model_info",
            stages=stages,
            timeout_s=timeout_s,
        )

    async def pause_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "pause_generation",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def continue_generation(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "continue_generation",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_disk(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "update_weights_from_disk",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def init_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "init_weights_update_group",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def destroy_weights_update_group(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "destroy_weights_update_group",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def update_weights_from_distributed(
        self,
        payload: dict[str, Any],
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "update_weights_from_distributed",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def weights_checker(
        self,
        payload: dict[str, Any] | None = None,
        *,
        stages: Sequence[str] | None = None,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        return await self.admin(
            "weights_checker",
            payload,
            stages=stages,
            timeout_s=timeout_s,
        )

    async def submit(self, request_id: str, request: OmniRequest | Any) -> Any:
        """Submit a request to the pipeline and wait for completion."""
        await self._submit_request(request_id, request)

        future = self._completion_futures[request_id]
        try:
            result = await future
            return result
        finally:
            self._completion_futures.pop(request_id, None)

    async def stream(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        dp_rank: int | None = None,
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        """Submit a request and yield stream events until completion.

        ``dp_rank`` pins admission to a specific entry-stage replica (used by
        replica-aware admission that must pre-assign ownership); ``None``
        leaves replica selection to :meth:`_select_replica`.
        """
        queue: asyncio.Queue[CompleteMessage | StreamMessage] = asyncio.Queue()

        try:
            if dp_rank is None:
                # Keep the default path on the pre-DP _submit_request signature.
                await self._submit_request(request_id, request, stream_queue=queue)
            else:
                await self._submit_request(
                    request_id, request, stream_queue=queue, dp_rank=dp_rank
                )
            expected_terminal_stages = self._expected_terminal_stages(request_id)

            completed_stages: set[str] = set()
            while True:
                msg = await queue.get()
                if isinstance(msg, CompleteMessage):
                    if not msg.success:
                        raise RuntimeError(msg.error or "Unknown error")
                    yield msg
                    completed_stages.add(msg.from_stage)
                    if (
                        not expected_terminal_stages
                        or completed_stages >= expected_terminal_stages
                    ):
                        return
                else:
                    yield msg
        finally:
            if self._stream_queues.get(request_id) is queue:
                try:
                    if request_id in self._requests:
                        try:
                            await self.abort(request_id)
                        except Exception:
                            # The coordinator-owned abort task logs its own failure.
                            # Do not replace the exception already leaving the stream.
                            pass
                finally:
                    if self._stream_queues.get(request_id) is queue:
                        self._stream_queues.pop(request_id, None)
                        self._completion_futures.pop(request_id, None)

    async def submit_to_replica(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        dp_rank: int,
    ) -> Any:
        """Submit to a specific entry-stage replica and wait for completion.

        Internal API for replica-scoped traffic (e.g. per-replica warmup)
        that must bypass load-balancing replica selection.
        """
        await self._submit_request(request_id, request, dp_rank=dp_rank)

        future = self._completion_futures[request_id]
        try:
            result = await future
            return result
        finally:
            self._completion_futures.pop(request_id, None)

    async def _submit_request(
        self,
        request_id: str,
        request: OmniRequest | Any,
        *,
        stream_queue: asyncio.Queue[CompleteMessage | StreamMessage] | None = None,
        dp_rank: int | None = None,
    ) -> None:
        """Submit a request without waiting for completion."""
        if self._fatal_error is not None:
            raise RuntimeError(self._fatal_error)
        if self._request_id_is_reserved(request_id):
            raise ValueError(f"Request {request_id} already exists")

        replicas = self._stages.get(self.entry_stage)
        if not replicas:
            raise ValueError(f"Entry stage {self.entry_stage} not registered")
        if dp_rank is None:
            entry_replica = self._select_replica(self.entry_stage)
        else:
            entry_replica = self._replica_for(self.entry_stage, dp_rank)
            if entry_replica is None:
                raise ValueError(
                    f"Entry stage {self.entry_stage} has no replica "
                    f"dp_rank={dp_rank}"
                )

        if not isinstance(request, OmniRequest):
            request = OmniRequest(inputs=request)

        # Track request, bound to the selected entry-stage replica.
        entry_replica.inflight += 1
        self._requests[request_id] = RequestInfo(
            request_id=request_id,
            state=RequestState.PENDING,
            current_stage=self.entry_stage,
            terminal_stages=self._resolve_terminal_stages(request),
            owner_endpoint=entry_replica.endpoint,
            owner_dp_rank=entry_replica.dp_rank,
        )

        # Create future for completion
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._completion_futures[request_id] = future
        if stream_queue is not None:
            self._stream_queues[request_id] = stream_queue

        payload = StagePayload(
            request_id=request_id,
            request=request,
            data={"raw_inputs": request.inputs},
        )

        admission_metadata: dict[str, Any] = {"entry_stage": self.entry_stage}
        if len(replicas) > 1:
            admission_metadata["dp_rank"] = entry_replica.dp_rank
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="request_admission",
            metadata=admission_metadata,
        )

        # Submit to the selected entry-stage replica
        try:
            await self.control_plane.submit_to_stage(
                self.entry_stage,
                entry_replica.endpoint,
                SubmitMessage(request_id=request_id, data=payload),
            )
        except Exception:
            self._release_replica(self._requests[request_id])
            raise

        # Update state
        info = self._requests.get(request_id)
        if info is not None:
            info.state = RequestState.RUNNING

        logger.info(
            "Coordinator submitted req=%s to %s at %s",
            request_id,
            self.entry_stage,
            entry_replica.endpoint,
        )

    def _request_id_is_reserved(self, request_id: str) -> bool:
        """Return whether any coordinator owner still holds this request ID."""
        return (
            request_id in self._requests
            or request_id in self._completion_futures
            or request_id in self._stream_queues
            or request_id in self._abort_tasks
        )

    def _reject_completion_future(
        self,
        request_id: str,
        exc: BaseException,
    ) -> None:
        # Note: (Akazaakane) Non-streaming callers await the completion future,
        # so errors must be propagated with set_exception(). Streaming callers
        # receive errors through the stream queue and never await that future;
        # cancel it instead to avoid "Future exception was never retrieved".
        future = self._completion_futures.get(request_id)
        if future is None or future.done():
            return
        if request_id in self._stream_queues:
            future.cancel()
        else:
            future.set_exception(exc)

    async def abort(self, request_id: str) -> bool:
        """Abort a request.

        Args:
            request_id: Request to abort

        Returns:
            True if aborted, False if not found
        """
        abort_task = self._abort_tasks.get(request_id)
        if abort_task is not None:
            return await asyncio.shield(abort_task)

        info = self._requests.get(request_id)
        if info is None:
            return False

        if info.state in (
            RequestState.COMPLETED,
            RequestState.FAILED,
            RequestState.ABORTED,
        ):
            return False

        abort_task = asyncio.create_task(
            self._run_abort(request_id),
            name=f"coordinator-abort-{request_id}",
        )
        self._abort_tasks[request_id] = abort_task
        abort_task.add_done_callback(
            lambda done, rid=request_id: self._on_abort_task_done(rid, done)
        )
        return await asyncio.shield(abort_task)

    async def update_request(
        self,
        request_id: str,
        data: dict[str, Any],
        *,
        stage_name: str | None = None,
    ) -> None:
        """Append an event to an active request, by default to the entry stage.

        NOTE(routing): ``current_stage`` is only assigned at submit time and
        never advanced, so without an explicit ``stage_name`` the update goes
        to the entry stage — not "the stage currently holding the request",
        which multi-stage pipelines do not track here. Callers targeting a
        downstream stage must also tolerate the control-plane vs data-plane
        ordering gap (an update sent this way can arrive before the submit).
        """
        info = self._requests.get(request_id)
        if info is None:
            raise KeyError(f"Request {request_id} does not exist")
        if info.state is not RequestState.RUNNING:
            raise RuntimeError(
                f"Request {request_id} is not running: {info.state.value}"
            )
        target = stage_name or info.current_stage or self.entry_stage
        if stage_name is None and info.owner_endpoint is not None:
            # Sticky routing: updates stay on the entry-stage replica that
            # owns the request (its shm/KV state lives there).
            target_endpoint = info.owner_endpoint
        else:
            replicas = self._stages.get(target)
            if not replicas:
                raise ValueError(f"Stage {target} is not registered")
            target_endpoint = replicas[0].endpoint
        if not isinstance(data, dict):
            raise TypeError("request update data must be a dict")
        await self.control_plane.submit_to_stage(
            target,
            target_endpoint,
            RequestUpdateMessage(request_id=request_id, data=data),
        )

    async def _run_abort(
        self,
        request_id: str,
    ) -> bool:
        await self.control_plane.broadcast_abort(AbortMessage(request_id=request_id))

        info = self._requests.get(request_id)
        if info is None:
            return False

        info.state = RequestState.ABORTED
        self._reject_completion_future(
            request_id, asyncio.CancelledError(f"Request {request_id} aborted")
        )
        stream_queue = self._stream_queues.get(request_id)
        if stream_queue is not None:
            await stream_queue.put(
                CompleteMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    success=False,
                    error="aborted",
                )
            )

        self._release_replica(info)
        self._requests.pop(request_id, None)
        self._partial_results.pop(request_id, None)

        logger.info("Coordinator aborted req=%s", request_id)
        return True

    def _on_abort_task_done(
        self,
        request_id: str,
        task: asyncio.Task[bool],
    ) -> None:
        if self._abort_tasks.get(request_id) is task:
            self._abort_tasks.pop(request_id, None)
        if task.cancelled():
            logger.warning("Coordinator abort task cancelled for req=%s", request_id)
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "Failed to abort request %s",
                request_id,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def run_completion_loop(self) -> None:
        """Run the completion receiving loop.

        This should be run as a background task.
        """
        try:
            while self._running:
                msg = await self.control_plane.recv_event()
                if isinstance(msg, StreamMessage):
                    await self._handle_stream(msg)
                elif isinstance(msg, AdminResultMessage):
                    self._handle_admin_result(msg.result)
                else:
                    await self._handle_completion(msg)
        except asyncio.CancelledError:
            logger.info("Coordinator completion loop cancelled")
        except Exception as e:
            logger.error("Coordinator completion loop error: %s", e)
            raise

    async def _handle_completion(self, msg: CompleteMessage) -> None:
        """Handle a completion message from a stage."""
        request_id = msg.request_id
        logger.debug(
            "Coordinator received completion: req=%s from %s success=%s",
            request_id,
            msg.from_stage,
            msg.success,
        )
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="terminal_response",
            metadata={
                "from_stage": msg.from_stage,
                "success": msg.success,
            },
        )

        if request_id not in self._requests:
            logger.warning(
                "Coordinator received completion for unknown req=%s", request_id
            )
            return

        info = self._requests[request_id]

        # Fail-fast: any terminal failure -> fail entire request
        if not msg.success:
            info.state = RequestState.FAILED
            info.error = msg.error
            await self.control_plane.broadcast_abort(
                AbortMessage(request_id=request_id)
            )
            self._partial_results.pop(request_id, None)
            self._reject_completion_future(
                request_id, RuntimeError(msg.error or "Unknown error")
            )
            if request_id in self._stream_queues:
                await self._stream_queues[request_id].put(msg)
            self._release_replica(info)
            self._requests.pop(request_id, None)
            return

        expected_terminal_stages = self._expected_terminal_stages(request_id)
        if expected_terminal_stages and msg.from_stage not in expected_terminal_stages:
            logger.debug(
                "Coordinator ignoring completion from inactive terminal: "
                "req=%s stage=%s expected=%s",
                request_id,
                msg.from_stage,
                sorted(expected_terminal_stages),
            )
            return

        # Single active terminal (original behavior) or no terminal_stages configured
        if len(expected_terminal_stages) <= 1:
            info.state = RequestState.COMPLETED
            info.result = msg.result
            if request_id in self._completion_futures:
                future = self._completion_futures[request_id]
                if not future.done():
                    future.set_result(msg.result)
            if request_id in self._stream_queues:
                await self._stream_queues[request_id].put(msg)
            self._release_replica(info)
            self._requests.pop(request_id, None)
            return

        # Multi-terminal: collect partial results
        partials = self._partial_results.setdefault(request_id, {})
        partials[msg.from_stage] = msg.result

        # Forward stream completion per-stage
        if request_id in self._stream_queues:
            await self._stream_queues[request_id].put(msg)

        if set(partials) < expected_terminal_stages:
            return  # still waiting

        # All terminal stages done -> merge and resolve
        merged = dict(partials)
        self._partial_results.pop(request_id)
        info.state = RequestState.COMPLETED
        info.result = merged

        if request_id in self._completion_futures:
            future = self._completion_futures[request_id]
            if not future.done():
                future.set_result(merged)
        self._release_replica(info)
        self._requests.pop(request_id, None)

    async def _handle_stream(self, msg: StreamMessage) -> None:
        """Handle a stream chunk from a stage."""
        request_id = msg.request_id
        if request_id not in self._stream_queues:
            return
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="coordinator_stream_received",
            metadata={
                "from_stage": msg.from_stage,
                "chunk_id": msg.chunk_id,
                "modality": msg.modality,
            },
        )
        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="stage_stream_chunk_received",
            metadata={
                "from_stage": msg.from_stage,
                "chunk_id": msg.chunk_id,
                "modality": msg.modality,
            },
        )
        await self._stream_queues[request_id].put(msg)

    def _handle_admin_result(self, result: AdminResult) -> None:
        pending = self._admin_ops.get(result.op_id)
        if pending is None:
            logger.warning(
                "Coordinator received admin result for unknown op=%s stage=%s",
                result.op_id,
                result.stage,
            )
            return
        if (
            result.dp_rank is None
            and len(self._stages.get(result.stage, ())) > 1
        ):
            # The replica could not be attributed (stage scheduler does not
            # report dp_rank); keep it unattributed instead of guessing, so
            # single-replica result payloads stay exactly as before.
            logger.warning(
                "Coordinator could not attribute admin result to a replica: "
                "op=%s stage=%s",
                result.op_id,
                result.stage,
            )
        pending.results.setdefault(result.stage, []).append(result)
        remaining = pending.pending_replica_counts.get(result.stage)
        if remaining is not None:
            pending.pending_replica_counts[result.stage] = max(0, remaining - 1)
        if (
            pending.future is not None
            and pending.results.keys() >= pending.expected_stages
            and all(count <= 0 for count in pending.pending_replica_counts.values())
        ):
            if not pending.future.done():
                pending.future.set_result(dict(pending.results))

    def _resolve_admin_stages(self, stages: Sequence[str] | None) -> list[str]:
        if stages is None:
            return sorted(self._stages)
        resolved = list(stages)
        unknown = sorted(set(resolved) - set(self._stages))
        if unknown:
            raise ValueError(f"Unknown admin target stage(s): {unknown}")
        return resolved

    @staticmethod
    def _merge_stage_replica_results(
        op_id: str,
        action: str,
        stage_name: str,
        items: list[AdminResult],
    ) -> AdminResult:
        """Collapse a stage's replica replies into one stage-level result.

        Single-replica stages return their one result untouched, keeping the
        pre-DP response shape. A replicated stage succeeds only when every
        replica succeeded; per-replica detail rides along under
        ``data["replica_results"]``.
        """
        if len(items) == 1:
            return items[0]
        ordered = sorted(
            items, key=lambda item: (item.dp_rank is None, item.dp_rank or 0)
        )
        errors = [item.error for item in ordered if item.error]
        success = all(item.success for item in ordered)
        data = dict(ordered[0].data)
        data["replica_results"] = [item.to_dict() for item in ordered]
        return AdminResult(
            op_id=op_id,
            stage=stage_name,
            action=action,
            success=success,
            message=ordered[0].message if success else "; ".join(errors),
            data=data,
            error=None if success else ("; ".join(errors) or ordered[0].error),
            rank=ordered[0].rank,
            role=ordered[0].role,
        )

    def _aggregate_admin_results(
        self,
        *,
        op_id: str,
        action: str,
        results_by_stage: dict[str, list[AdminResult]],
    ) -> dict[str, Any]:
        results = [
            self._merge_stage_replica_results(op_id, action, stage, items)
            for stage, items in results_by_stage.items()
        ]
        updated_results = [
            item
            for item in results
            if not item.data.get("skipped") and not item.data.get("unsupported")
        ]
        if is_update_action(action):
            success = bool(updated_results) and all(
                item.success for item in updated_results
            )
        else:
            success = all(item.success for item in results)

        errors = [item.error for item in results if item.error]
        if success:
            message = "ok"
        elif errors:
            message = "; ".join(errors)
        else:
            message = "admin operation did not complete successfully"

        return {
            "op_id": op_id,
            "action": action,
            "success": success,
            "message": message,
            "results": [item.to_dict() for item in results],
        }

    def get_request_info(self, request_id: str) -> RequestInfo | None:
        """Get info about a request."""
        return self._requests.get(request_id)

    def _resolve_terminal_stages(self, request: OmniRequest) -> set[str]:
        if self._terminal_stages_resolver is None:
            return set(self._terminal_stages)
        resolved = self._terminal_stages_resolver(request)
        if resolved is None:
            return set(self._terminal_stages)
        if isinstance(resolved, str) or not isinstance(resolved, Sequence):
            raise ValueError(
                "terminal_stages_resolver must return a sequence of terminal "
                "stage names or None"
            )
        if not all(isinstance(stage, str) for stage in resolved):
            raise ValueError(
                "terminal_stages_resolver must return terminal stage names"
            )
        resolved_stages = set(resolved)
        if not resolved_stages:
            raise ValueError("terminal_stages_resolver returned no terminal stages")
        unknown = resolved_stages - self._terminal_stages
        if unknown:
            raise ValueError(
                "terminal_stages_resolver returned stages outside the static "
                f"terminal stages: {sorted(unknown)}. Allowed terminal stages: "
                f"{sorted(self._terminal_stages)}"
            )
        return resolved_stages

    def _expected_terminal_stages(self, request_id: str) -> set[str]:
        info = self._requests.get(request_id)
        if info is None or info.terminal_stages is None:
            return set(self._terminal_stages)
        return info.terminal_stages

    def health(self) -> dict[str, Any]:
        """Return health status."""
        state_counts = {}
        for info in self._requests.values():
            state = info.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "running": self._running,
            "stages": list(self._stages.keys()),
            "entry_stage": self.entry_stage,
            "total_requests": len(self._requests),
            "pending_completions": len(self._completion_futures),
            "request_states": state_counts,
        }


async def run_coordinator(
    completion_endpoint: str,
    abort_endpoint: str,
    entry_stage: str,
    stages: dict[str, str],  # name -> endpoint
    terminal_stages: list[str] | None = None,
    terminal_stages_resolver: Callable[[OmniRequest], list[str] | None] | None = None,
) -> Coordinator:
    """Create and start a coordinator.

    Args:
        completion_endpoint: ZMQ endpoint to receive completions
        abort_endpoint: ZMQ endpoint for abort broadcasts
        entry_stage: Name of the entry stage
        stages: Dict of stage_name -> stage_endpoint
        terminal_stages: Optional list of terminal stage names for multi-terminal merge

    Returns:
        Started Coordinator instance
    """
    coordinator = Coordinator(
        completion_endpoint=completion_endpoint,
        abort_endpoint=abort_endpoint,
        entry_stage=entry_stage,
        terminal_stages=terminal_stages,
        terminal_stages_resolver=terminal_stages_resolver,
    )

    # Register stages
    for name, endpoint in stages.items():
        coordinator.register_stage(name, endpoint)

    # Start
    await coordinator.start()

    return coordinator
