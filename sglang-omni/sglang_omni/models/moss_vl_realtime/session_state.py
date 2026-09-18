"""Scheduler-owned event ordering for MOSS-VL realtime requests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from sglang_omni.models.moss_vl_realtime.payload_types import FramePromptEvent


@dataclass(slots=True)
class MossVLRealtimeSession:
    """Ordering and lifecycle state for one request's incremental events."""

    request_id: str
    session_id: str
    next_seq_no: int = 0
    last_timestamp: float | None = None
    final_received: bool = False
    pending_events: deque[FramePromptEvent] = field(default_factory=deque)

    def accept(self, event: FramePromptEvent) -> None:
        if event.request_id != self.request_id:
            raise ValueError(
                f"request mismatch: expected {self.request_id!r}, "
                f"received {event.request_id!r}"
            )
        if event.session_id != self.session_id:
            raise ValueError(
                f"session mismatch: expected {self.session_id!r}, "
                f"received {event.session_id!r}"
            )
        if self.final_received:
            raise RuntimeError(
                f"request {self.request_id} already received final event"
            )
        if event.seq_no != self.next_seq_no:
            raise ValueError(
                f"out-of-order event for {self.request_id}: "
                f"expected seq_no {self.next_seq_no}, received {event.seq_no}"
            )
        if self.last_timestamp is not None and event.timestamp < self.last_timestamp:
            raise ValueError(
                f"timestamp moved backwards for {self.request_id}: "
                f"{event.timestamp} < {self.last_timestamp}"
            )

        self.pending_events.append(event)
        self.next_seq_no += 1
        self.last_timestamp = float(event.timestamp)
        self.final_received = event.final

    def drain(self, max_events: int | None = None) -> list[FramePromptEvent]:
        if max_events is not None and max_events <= 0:
            raise ValueError("max_events must be positive")
        count = len(self.pending_events)
        if max_events is not None:
            count = min(count, max_events)
        return [self.pending_events.popleft() for _ in range(count)]


class MossVLRealtimeSessionController:
    """Own realtime event queues in the same process as the model scheduler."""

    def __init__(self) -> None:
        self._sessions: dict[str, MossVLRealtimeSession] = {}

    def open(self, request_id: str, session_id: str) -> MossVLRealtimeSession:
        if request_id in self._sessions:
            raise ValueError(f"request {request_id} already has a realtime session")
        session = MossVLRealtimeSession(
            request_id=request_id,
            session_id=session_id,
        )
        self._sessions[request_id] = session
        return session

    def ingest(
        self, request_id: str, data: FramePromptEvent | dict[str, Any]
    ) -> FramePromptEvent:
        session = self._sessions.get(request_id)
        if session is None:
            raise KeyError(f"request {request_id} has no realtime session")
        event = (
            data
            if isinstance(data, FramePromptEvent)
            else FramePromptEvent.from_dict(data)
        )
        session.accept(event)
        return event

    def get(self, request_id: str) -> MossVLRealtimeSession | None:
        return self._sessions.get(request_id)

    def close(self, request_id: str) -> bool:
        return self._sessions.pop(request_id, None) is not None

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._sessions
