"""In-process metrics registry + usage ledger for the gateway plane
(GATEWAY_PLAN.md §2-P3/P4).

GatewayMetrics is a thread-safe gauge/counter registry. All updates are O(1)
under a short threading.Lock critical section, so the passthrough path
(receiver-thread `_track` calls) never blocks on anything heavier. Replica
gauges are refreshed from the pool at scrape time (update_pool_gauges) instead
of on every pool transition — the pool is the source of truth and cannot drift.

UsageLog appends one JSONL reconciliation record per session teardown (or
reset) to settings.gateway_usage_log ("" → {data_dir}/gateway_usage.jsonl).
Writes happen ONLY on session-terminal paths, never in the forwarding loop,
and a write failure is logged, never raised — metering must not break sessions.

Scraped by GET /v1/realtime/metrics (rest.py); alert rules over these series
live in docs/gateway_alerting.md.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

from ..logging_conf import get_logger

log = get_logger(__name__)

GAUGE_ACTIVE_SESSIONS = "gateway_active_sessions"
GAUGE_SLOTS_TOTAL = "gateway_replica_slots_total"
GAUGE_SLOTS_USED = "gateway_replica_slots_used"
GAUGE_REPLICAS_DOWN = "gateway_replicas_down"

COUNTER_SESSIONS_CREATED = "gateway_sessions_created_total"
COUNTER_FRAMES_ACCEPTED = "gateway_frames_accepted_total"
COUNTER_TEXT_CHARS = "gateway_text_chars_total"
COUNTER_ABNORMAL_DISCONNECTS = "gateway_abnormal_disconnects_total"
COUNTER_ATTACH_TIMEOUTS = "gateway_attach_timeouts_total"
COUNTER_ERRORS = "gateway_errors_total"  # per-code series, see inc_error()

GAUGES = (GAUGE_ACTIVE_SESSIONS, GAUGE_SLOTS_TOTAL, GAUGE_SLOTS_USED,
          GAUGE_REPLICAS_DOWN)
COUNTERS = (COUNTER_SESSIONS_CREATED, COUNTER_FRAMES_ACCEPTED,
            COUNTER_TEXT_CHARS, COUNTER_ABNORMAL_DISCONNECTS,
            COUNTER_ATTACH_TIMEOUTS)

# session teardown reasons (P3 usage ledger `end_reason` enum)
END_CLIENT_DISCONNECT = "client_disconnect"   # client WS ended (run())
END_CLIENT_DELETE = "client_delete"           # DELETE /v1/realtime/sessions/{id}
END_RESET = "reset"                           # reset(): old connection's meter closes
END_OMNI_DEAD = "omni_dead"                   # downstream transport died mid-session
END_SESSION_DONE = "session_done"             # omni closed cleanly after session.done
END_ATTACH_TIMEOUT = "attach_timeout"         # janitor GC of a never-attached session
END_SHUTDOWN = "shutdown"                     # gateway shutdown drained the session
END_REASONS = (END_CLIENT_DISCONNECT, END_CLIENT_DELETE, END_RESET,
               END_OMNI_DEAD, END_SESSION_DONE, END_ATTACH_TIMEOUT, END_SHUTDOWN)


class GatewayMetrics:
    """Thread-safe gauge/counter registry; one instance per GatewayRegistry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gauges: Dict[str, float] = {name: 0 for name in GAUGES}
        self._counters: Dict[str, int] = {name: 0 for name in COUNTERS}
        self._errors: Dict[str, int] = {}

    # ------------------------------------------------------------ writers

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def add_gauge(self, name: str, delta: float) -> None:
        with self._lock:
            self._gauges[name] = self._gauges.get(name, 0) + delta

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def inc_error(self, code: str) -> None:
        with self._lock:
            self._errors[code or "unknown"] = self._errors.get(code or "unknown", 0) + 1

    def update_pool_gauges(self, pool) -> None:
        """Refresh the replica gauges from the pool (source of truth). Called
        at scrape time, so the gauges can never drift from pool transitions."""
        slots_total, slots_used, replicas_down = pool.gauge_counts()
        with self._lock:
            self._gauges[GAUGE_SLOTS_TOTAL] = slots_total
            self._gauges[GAUGE_SLOTS_USED] = slots_used
            self._gauges[GAUGE_REPLICAS_DOWN] = replicas_down

    # ------------------------------------------------------------ reader

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters: Dict[str, Any] = dict(self._counters)
            counters[COUNTER_ERRORS] = dict(self._errors)
            return {"gauges": dict(self._gauges), "counters": counters}


class UsageLog:
    """Append-only JSONL reconciliation ledger. One record per session
    teardown/reset; failures are logged and swallowed (metering must never
    break a session)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings) -> "UsageLog":
        path = settings.gateway_usage_log or os.path.join(
            settings.data_dir, "gateway_usage.jsonl")
        return cls(path)

    def record(self, record: Dict[str, Any]) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False)
            with self._lock:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:  # noqa: BLE001 — metering must not break sessions
            log.exception("gateway usage ledger write failed (%s)", self.path)
