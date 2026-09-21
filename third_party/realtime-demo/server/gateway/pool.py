"""sglang-omni replica pool for the gateway plane.

Same READY/BUSY/DOWN discipline as adapters/vlm/moss_vl_sglang_omni/pool.py but
deliberately thinner: this plane owns the WS handshake itself (session.py), so
the pool only does slot bookkeeping, health probing, and capacity accounting.

Capacity model: each replica hosts up to `sglang_omni_sessions_per_replica`
live sessions (the remote omni instance's --max-running-requests; the two MUST
match). READY = healthy with a free slot, BUSY = healthy but full, DOWN =
unhealthy/quarantined.

- acquire() reserves a slot on the READY replica with the FEWEST used slots
  (ties → lowest index; raises GatewayCapacityError when none); the caller
  runs the WS handshake and MUST pair every acquire with exactly one
  release().
- mark_full(index): omni rejected the handshake with
  session_capacity_exceeded — that instance is full server-side (it may host
  sessions we don't track), so acquire skips it; the prober clears the mark
  once /health answers again.
- release(index, transport_dead=True) quarantines the replica (DOWN) because
  a dead transport means the server may be wedged; the prober flips it back
  once GET /health answers again (READY if a slot is free, else BUSY).
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from ..config import Settings
from ..logging_conf import get_logger

log = get_logger(__name__)

READY, BUSY, DOWN = "READY", "BUSY", "DOWN"


class GatewayCapacityError(RuntimeError):
    """No READY replica slot — the REST layer maps this to 503."""

    def __init__(self, capacity: int, busy: int):
        super().__init__(f"no READY sglang-omni replica ({busy}/{capacity} slots busy)")
        self.capacity = capacity
        self.busy = busy


class GatewayUnavailableError(ConnectionError):
    """No reachable replica or the overall creation deadline was exceeded."""


@dataclass
class GatewayReplica:
    url: str
    state: str = DOWN  # Not routable until the first successful health probe.
    slots: int = 1            # sessions this replica may host (= omni --max-running-requests)
    used: int = 0             # slots held by live tracked sessions
    capacity_limited: bool = False  # omni reported session_capacity_exceeded;
                                    # the prober re-checks and clears the mark
    health: Dict[str, Any] = field(default_factory=dict)
    health_epoch: int = 0


class GatewayPool:
    def __init__(self, settings: Settings):
        self.s = settings
        if settings.gateway_max_frame_bytes <= 0:
            raise ValueError("GATEWAY_MAX_FRAME_BYTES must be positive")
        if not math.isfinite(settings.gateway_create_timeout_s) or settings.gateway_create_timeout_s <= 0:
            raise ValueError("GATEWAY_CREATE_TIMEOUT_S must be finite and positive")
        urls = [u.strip().rstrip("/")
                for u in str(settings.sglang_omni_urls or "").split(",") if u.strip()]
        slots = max(1, int(settings.sglang_omni_sessions_per_replica or 1))
        self._replicas: List[GatewayReplica] = [
            GatewayReplica(url=u, slots=slots) for u in urls]
        self._lock = threading.Lock()
        self._prober_stop = threading.Event()
        self._prober: Optional[threading.Thread] = None
        versions = json.loads(settings.gateway_model_versions or "{}")
        if not isinstance(versions, dict) or any(
            not isinstance(url, str) or not isinstance(version, str) or not version.strip()
            for url, version in versions.items()
        ):
            raise ValueError("GATEWAY_MODEL_VERSIONS must map replica URLs to non-empty version strings")
        self._versions = {url.rstrip("/"): version.strip() for url, version in versions.items()}
        if set(self._versions) - set(urls):
            raise ValueError("GATEWAY_MODEL_VERSIONS contains an unconfigured replica URL")

    # ------------------------------------------------------------ introspection

    @property
    def capacity(self) -> int:
        return sum(r.slots for r in self._replicas)

    @property
    def busy(self) -> int:
        return sum(r.used for r in self._replicas)

    @property
    def replicas(self) -> List[GatewayReplica]:
        return self._replicas

    def replica_url(self, index: int) -> str:
        return self._replicas[index].url

    def replica_version(self, index: int) -> Optional[str]:
        return self._versions.get(self.replica_url(index)) or self.s.gateway_model_version.strip() or None

    def routable_replicas(self) -> list[tuple[int, str]]:
        with self._lock:
            return [(i, r.url) for state in (READY, BUSY)
                    for i, r in enumerate(self._replicas) if r.state == state]

    def first_ready_url(self) -> Optional[str]:
        """First READY replica, else a BUSY one (a live session does not make a
        replica unable to answer GET /v1/models). DOWN replicas never serve."""
        with self._lock:
            for state in (READY, BUSY):
                for r in self._replicas:
                    if r.state == state:
                        return r.url
        return None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            replicas = [{"url": r.url, "state": r.state} for r in self._replicas]
            return {
                "instances": len(self._replicas),
                "active_sessions": self.busy,
                "capacity": self.capacity,
                "replicas": replicas,
            }

    def gauge_counts(self) -> tuple:
        """(slots_total, slots_used, replicas_down) for the P4 metrics gauges.
        The pool is the source of truth; the registry scrapes at read time."""
        with self._lock:
            return (sum(r.slots for r in self._replicas),
                    sum(r.used for r in self._replicas),
                    sum(1 for r in self._replicas if r.state == DOWN))

    # ------------------------------------------------------------ slots

    def acquire(self, *, exclude: Optional[set[int]] = None) -> int:
        """Reserve a slot on the least-loaded READY replica (ties → lowest
        index); returns its index."""
        with self._lock:
            picked: Optional[int] = None
            for i, r in enumerate(self._replicas):
                if exclude and i in exclude:
                    continue
                if r.state == READY and r.used < r.slots and (
                        picked is None or r.used < self._replicas[picked].used):
                    picked = i
            if picked is None:
                raise GatewayCapacityError(self.capacity, self.busy)
            r = self._replicas[picked]
            r.used += 1
            if r.used >= r.slots:
                r.state = BUSY
            return picked

    def mark_full(self, index: int) -> None:
        """omni rejected the handshake with session_capacity_exceeded: the
        instance is full server-side (sessions we don't track hold slots) —
        hand back the failed reservation and mark the replica BUSY so acquire
        skips it. The prober re-probes marked replicas and clears the mark once
        /health answers, so a transient server-side full state cannot wedge
        the replica forever."""
        with self._lock:
            r = self._replicas[index]
            r.used = max(0, r.used - 1)  # the rejected acquire held nothing
            r.capacity_limited = True
            r.state = BUSY
            r.health_epoch += 1
        log.info("gateway replica %d (%s) at session capacity — marked full",
                 index, r.url)

    def release(self, index: int, transport_dead: bool = False) -> None:
        with self._lock:
            r = self._replicas[index]
            r.used = max(0, r.used - 1)
            r.capacity_limited = False  # a tracked slot drained — fullness may have eased
            # a dead transport means the server may be wedged — quarantine the
            # replica until the prober's next health poll clears it
            if transport_dead:
                r.state = DOWN
                r.health_epoch += 1
            elif r.state != DOWN:
                r.state = READY if r.used < r.slots else BUSY
        log.info("gateway replica %d (%s) released%s", index, r.url,
                 " (transport dead — quarantined)" if transport_dead else "")

    # ------------------------------------------------------------ health

    def mark_unhealthy(self, index: int) -> None:
        with self._lock:
            replica = self._replicas[index]
            replica.state = DOWN
            replica.health_epoch += 1

    def probe_all(self) -> None:
        for replica in self._replicas:
            if self._prober_stop.is_set():
                return
            with self._lock:
                epoch = replica.health_epoch
            health = self._probe_health(replica.url)
            with self._lock:
                # A handshake failure/full mark after this probe started wins.
                if epoch != replica.health_epoch:
                    continue
                replica.health = health or {}
                if health is None:
                    replica.state = DOWN
                else:
                    replica.capacity_limited = False
                    replica.state = READY if replica.used < replica.slots else BUSY

    def _probe_health(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            resp = requests.get(f"{url}/health",
                                timeout=max(1.0, self.s.sglang_omni_connect_timeout_s))
            if resp.ok:
                health = resp.json() if resp.content else {"ok": True}
                return health if isinstance(health, dict) else None
        except (requests.RequestException, ValueError):
            pass
        return None

    def start_prober(self) -> None:
        if self._prober is not None:
            return
        interval = max(0.2, float(self.s.sglang_omni_health_interval_s))

        def prober() -> None:
            while not self._prober_stop.is_set():
                self.probe_all()
                if self._prober_stop.wait(interval):
                    break

        self._prober = threading.Thread(
            target=prober, name="gateway-omni-health", daemon=True)
        self._prober.start()

    def close(self) -> None:
        self._prober_stop.set()
        if self._prober is not None:
            self._prober.join(timeout=2 * max(1.0, self.s.sglang_omni_connect_timeout_s) + 1)
            self._prober = None
