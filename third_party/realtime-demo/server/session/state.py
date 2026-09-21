"""Per-session state: config, seq, out-queue + replay ring (backend_overhaul.md §B2).

`SessionState.emit()` is the single server→client egress: it assigns the monotonic
`seq`, appends the event to the replay ring (unless transient), and pushes it onto
the live out-queue the WS pump drains. Emit is event-loop-only — thread producers
(TTS worker, model threads) must bridge via `loop.call_soon_threadsafe`.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, List, Optional

from ..logging_conf import get_logger
from ..protocol import PROTOCOL_VERSION, Seq, encode_event
from ..schemas import SessionConfig

log = get_logger(__name__)

# lifecycle phases
PHASE_CREATED = "created"    # minted via REST, no WS yet
PHASE_ACTIVE = "active"      # WS attached
PHASE_DETACHED = "detached"  # WS gone, grace timer running
PHASE_CLOSED = "closed"


@dataclass
class OutboundItem:
    """One server→client event: a JSON text frame + optional trailing binary frame."""

    seq: int
    text: str
    binary: Optional[bytes] = None


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:24]}"


@dataclass
class SessionState:
    session_id: str
    config: SessionConfig
    created_at: float = field(default_factory=time.time)
    replay_size: int = 1024
    out_queue_size: int = 4096

    phase: str = PHASE_CREATED
    seq: Seq = field(default_factory=Seq)
    last_acked_seq: int = 0
    grace_deadline: Optional[float] = None

    # transport binding — the token identifies the *current* WS so a superseded
    # connection's teardown can't detach its replacement
    ws_token: Optional[str] = None

    orchestrator: Any = None            # set by the manager (session.orchestrator.Orchestrator)

    # history sink (HistoryRecorder.realtime_sink): called with (type, json_text)
    # for every non-transient event; must be non-blocking (it only enqueues)
    record: Optional[Callable[[str, str], None]] = None

    out_queue: "asyncio.Queue[OutboundItem]" = field(init=False)
    replay: Deque[OutboundItem] = field(init=False)
    events_dropped: int = 0

    def __post_init__(self) -> None:
        self.out_queue = asyncio.Queue(maxsize=max(16, self.out_queue_size))
        self.replay = deque(maxlen=max(16, self.replay_size))

    # ---- egress ----

    def emit(self, type_: str, *, binary: Optional[bytes] = None, transient: bool = False, **fields: Any) -> int:
        """Assign seq, buffer for replay, enqueue for the live WS pump.

        `transient` events (status/pong/session.created) are not replayed after a
        reconnect — they describe a moment, not session history.
        """
        seq = self.seq.next()
        item = OutboundItem(seq=seq, text=encode_event(type_, seq=seq, **fields), binary=binary)
        if not transient:
            self.replay.append(item)
            if self.record is not None:
                try:
                    self.record(type_, item.text)  # non-blocking enqueue
                except Exception:  # noqa: BLE001 — history must never break egress
                    pass
        while True:
            try:
                self.out_queue.put_nowait(item)
                break
            except asyncio.QueueFull:
                try:
                    dropped = self.out_queue.get_nowait()
                    self.events_dropped += 1
                    log.warning("session %s out-queue full; dropped seq=%d", self.session_id, dropped.seq)
                except asyncio.QueueEmpty:  # racing consumer freed space
                    continue
        return seq

    def replay_after(self, last_seq: int) -> List[OutboundItem]:
        return [item for item in self.replay if item.seq > int(last_seq)]

    # ---- snapshots ----

    def created_event_fields(self) -> dict:
        """Payload for `session.created` (sent on every WS attach)."""
        return {
            "session_id": self.session_id,
            "protocol_version": PROTOCOL_VERSION,
            "limits": {
                "replay_buffer": self.replay.maxlen,
                "out_queue": self.out_queue.maxsize,
            },
        }

    def snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "phase": self.phase,
            "created_at": self.created_at,
            "config": self.config.model_dump(exclude_none=True),
            "seq": self.seq.last,
            "last_acked_seq": self.last_acked_seq,
            "grace_deadline": self.grace_deadline,
            "events_dropped": self.events_dropped,
        }
