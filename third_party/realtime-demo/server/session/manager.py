"""Session lifecycle: create / attach / detach / grace-GC (backend_overhaul.md §B2).

One `SessionManager` lives in the Runtime. Capacity is owned by the VLM replica
pool (one live session per GPU worker; `NoFreeReplica` → 409 in the router) —
the manager only enforces a MAX_SESSIONS backstop against runaway session
counts (scripted samp sessions bypass the pool). It owns the reconnect grace
timers and is the only component that tears an orchestrator down.

WS attachment model: at most one socket per session. A new attach *supersedes* a
lingering old socket (flaky-gateway reconnects must not lock the client out): the
old pump is released via its close-event and its half-read items are recovered by
the replay ring, which is the authoritative catch-up source (`?last_seq=N`).
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ..config import Settings
from ..logging_conf import get_logger
from ..schemas import SessionConfig
from .orchestrator import EngineSet, Orchestrator
from .state import (
    PHASE_ACTIVE,
    PHASE_CLOSED,
    PHASE_DETACHED,
    OutboundItem,
    SessionState,
    new_session_id,
)

log = get_logger(__name__)


class SessionConflict(RuntimeError):
    """Session capacity exhausted (MAX_SESSIONS backstop)."""


class SessionManager:
    def __init__(self, settings: Settings, history: Any = None,
                 capacity: Optional[Callable[[], int]] = None,
                 memory: Any = None, memory_facts: Any = None,
                 memory_plane: Any = None):
        self.settings = settings
        self._history = history  # persistence.HistoryRecorder (None = recording off)
        # (MemoryStore, MemoryWriter) or (None, None) — shared process-wide; each
        # session gets its own MemorySession view scoped to its conversation id
        self._memory_store, self._memory_writer = memory or (None, None)
        # background fact extractor (memory/facts.py), shared process-wide too
        self._memory_facts = memory_facts
        # offline sglang plane for the rollover summarizer (rt.vlm_offline);
        # None on 1-GPU boxes → rollover degrades to verbatim-tail-only
        self._memory_plane = memory_plane
        # VLM replica count provider (Runtime wires the pool's capacity in);
        # feeds the auto MAX_SESSIONS backstop
        self._capacity = capacity
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._grace_timers: Dict[str, asyncio.TimerHandle] = {}
        self._close_events: Dict[str, asyncio.Event] = {}
        self._closed = False

    def _max_sessions(self) -> int:
        """Explicit MAX_SESSIONS wins; auto = replica capacity + headroom for
        pool-bypassing sessions (samp scripted demos don't hold a GPU)."""
        if self.settings.max_sessions > 0:
            return self.settings.max_sessions
        capacity = 0
        if self._capacity is not None:
            try:
                capacity = int(self._capacity() or 0)
            except Exception:  # noqa: BLE001
                capacity = 0
        return max(1, capacity) + 8

    # ------------------------------------------------------------------ lifecycle

    async def create(
        self,
        config: SessionConfig,
        engines_factory: Callable[[], Awaitable[EngineSet]],
        reseat_factory: Optional[Callable[..., Any]] = None,
    ) -> SessionState:
        """Mint a session and start its orchestrator.

        The engines factory runs under the manager lock so the capacity check and
        the (exclusive) VLM realtime-session start are atomic.

        `reseat_factory` (router-supplied closure over the same engine params)
        starts a REPLACEMENT VLM session for a rollover re-seat; None leaves
        rollover inert even when memory is on.
        """
        async with self._lock:
            if self._closed:
                raise RuntimeError("session manager is shut down")
            limit = self._max_sessions()
            if len(self._sessions) >= limit:
                raise SessionConflict(
                    f"session limit reached ({len(self._sessions)}/{limit}); "
                    "delete a session or wait for a grace expiry")
            engines = await engines_factory()
            try:
                state = SessionState(
                    session_id=new_session_id(),
                    config=config,
                    replay_size=self.settings.session_replay_buffer,
                    out_queue_size=self.settings.session_out_queue,
                )
                if self._history is not None:
                    # open the durable conversation, then tap emit() — both are
                    # non-blocking enqueues onto the recorder's writer thread
                    self._history.open_conversation(
                        state.session_id, "realtime",
                        config=config.model_dump(exclude_none=True),
                        created_at=state.created_at)
                    state.record = self._history.realtime_sink(state.session_id)
                memory = None
                rollover = None
                if self._memory_store is not None and self._memory_writer is not None:
                    from ..memory import MemorySession  # local: optional plane
                    memory = MemorySession(
                        state.session_id, self.settings, self._memory_store, self._memory_writer,
                        default_lang=(config.asr_language or self.settings.asr_language or "zh"),
                        facts=self._memory_facts)
                    if reseat_factory is not None:
                        from ..memory.rollover import RolloverManager  # local: optional plane
                        rollover = RolloverManager(
                            self.settings, self._memory_store, state.session_id,
                            plane=self._memory_plane,
                            lifetime=memory.lifetime,
                            base_system_prompt=config.system_prompt or "",
                            lang_getter=lambda m=memory: m.language,
                            # interrupted turns ride the compact journal only
                            journal_extra=lambda m=memory: m.uncommitted_turns(),
                            # share the background-job semaphore with fact
                            # extraction (design §7: concurrency 1-2 on the sidecar)
                            semaphore=getattr(self._memory_facts, "semaphore", None))
                orchestrator = Orchestrator(state, engines, self.settings, memory=memory,
                                            rollover=rollover, reseat_factory=reseat_factory)
                state.orchestrator = orchestrator
                orchestrator.start()
            except Exception:
                await asyncio.to_thread(engines.close)
                raise
            self._sessions[state.session_id] = state
            self._arm_grace(state)
            log.info("session created: %s (grace=%.0fs)", state.session_id, self.settings.session_grace_seconds)
            return state

    def get(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"session not found: {session_id}")
        return state

    def list_snapshots(self) -> List[dict]:
        return [state.snapshot() for state in self._sessions.values()]

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------ WS binding

    async def attach_ws(
        self, session_id: str, last_seq: int = 0
    ) -> Tuple[SessionState, str, asyncio.Event, List[OutboundItem]]:
        """Bind a socket: cancel grace, supersede any old socket, hand back replay."""
        async with self._lock:
            state = self.get(session_id)
            self._cancel_grace(session_id)
            state.grace_deadline = None

            old_event = self._close_events.get(session_id)
            if old_event is not None:
                old_event.set()  # release the superseded pump

            token = uuid.uuid4().hex
            close_event = asyncio.Event()
            state.ws_token = token
            state.phase = PHASE_ACTIVE
            state.last_acked_seq = max(state.last_acked_seq, int(last_seq or 0))
            self._close_events[session_id] = close_event

            # the ring is the catch-up source; stale live-queue items would only
            # duplicate it (or leak another pump's leftovers), so discard them
            while True:
                try:
                    state.out_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            replay = state.replay_after(last_seq or 0)
            log.info("session %s attached (last_seq=%s, replaying %d events)",
                     session_id, last_seq, len(replay))
            return state, token, close_event, replay

    async def detach_ws(self, session_id: str, token: str) -> None:
        """Unbind a socket; the session survives for the grace window."""
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.ws_token != token:
                return  # session gone, or this socket was already superseded
            state.ws_token = None
            state.phase = PHASE_DETACHED
            event = self._close_events.pop(session_id, None)
            if event is not None:
                event.set()
            self._arm_grace(state)
            log.info("session %s detached; grace GC in %.0fs",
                     session_id, self.settings.session_grace_seconds)

    # ------------------------------------------------------------------ teardown

    async def close(self, session_id: str, reason: str = "client") -> bool:
        async with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is None:
                return False
            self._cancel_grace(session_id)
            event = self._close_events.pop(session_id, None)
            if event is not None:
                event.set()
            state.phase = PHASE_CLOSED
            state.ws_token = None
        orchestrator: Optional[Any] = state.orchestrator
        try:
            if orchestrator is not None:
                await orchestrator.close()
        finally:
            if self._history is not None:
                self._history.finalize(session_id, end_reason=reason)
        log.info("session %s closed (%s)", session_id, reason)
        return True

    async def aclose(self) -> None:
        self._closed = True
        for session_id in list(self._sessions):
            try:
                await self.close(session_id, reason="shutdown")
            except Exception as exc:  # noqa: BLE001
                log.warning("closing session %s at shutdown failed: %s", session_id, exc)

    # ------------------------------------------------------------------ grace timers

    def _arm_grace(self, state: SessionState) -> None:
        self._cancel_grace(state.session_id)
        grace = max(1.0, self.settings.session_grace_seconds)
        state.grace_deadline = time.time() + grace
        loop = asyncio.get_running_loop()
        self._grace_timers[state.session_id] = loop.call_later(
            grace, self._grace_fired, state.session_id)

    def _cancel_grace(self, session_id: str) -> None:
        handle = self._grace_timers.pop(session_id, None)
        if handle is not None:
            handle.cancel()

    def _grace_fired(self, session_id: str) -> None:
        self._grace_timers.pop(session_id, None)
        asyncio.get_running_loop().create_task(
            self.close(session_id, reason="grace_expired"),
            name=f"grace-gc-{session_id[:12]}",
        )
