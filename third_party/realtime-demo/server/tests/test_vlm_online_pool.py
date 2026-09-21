"""Unit tests: VlmReplicaPool state machine (acquire/release/reclaim/crash).

Run:  <repo>/.venv/bin/python -m server.tests.test_vlm_online_pool

Regression focus: a session that dies worker-side WITHOUT a gateway stop()
(kv_exhausted, VLM-loop error) is reclaimed by the monitor; its deferred
stop() — grace GC fires up to ~45 s later — must NOT free or quarantine the
slot after a NEW session has leased it (generation bump on reclaim).
"""
from __future__ import annotations

import sys
import uuid
from typing import Any, Dict

from server.adapters.vlm.moss_vl_hf.online_pool import BUSY, DOWN, READY, NoFreeReplica, VlmReplicaPool
from server.config import Settings
from server.gpu.placement import PlacementPlan, WorkerSpec

LOADED = {"loaded": True}


class DummyInner:
    """Stands in for WorkerVlmSession: just the surface _PooledSession touches."""

    def __init__(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.active = True
        self.transport_dead = False
        self.stop_calls = 0

    @property
    def worker_transport_dead(self) -> bool:
        return self.transport_dead

    def stop(self, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        self.stop_calls += 1
        self.active = False
        return {"active": False}


def make_pool(inners: list) -> VlmReplicaPool:
    plan = PlacementPlan(
        gpus=(), asr_device="cuda:0", tts=(),
        workers=(WorkerSpec(worker_id=0, gpu_index=0, port=19999, attn_impl="sdpa"),))
    pool = VlmReplicaPool(Settings(), plan)
    pool.set_replica_health(0, dict(LOADED))  # STARTING → READY
    queued = list(inners)
    pool.replicas[0].proxy.start_realtime_session = lambda **kw: queued.pop(0)
    return pool


def test_acquire_release_and_capacity() -> None:
    inner = DummyInner()
    pool = make_pool([inner])
    assert pool.is_loaded() and pool.capacity == 1 and pool.busy == 0

    session = pool.start_realtime_session()
    assert pool.busy == 1 and pool.replicas[0].state == BUSY
    assert session.session_id == inner.session_id  # delegation

    try:
        pool.start_realtime_session()
    except NoFreeReplica as exc:
        assert "1/1" in str(exc)
    else:
        raise AssertionError("second acquire on a busy pool must raise NoFreeReplica")

    session.stop()
    assert inner.stop_calls == 1
    assert pool.busy == 0 and pool.replicas[0].state == READY
    print("acquire/release/capacity: OK")


def test_stale_release_after_reclaim() -> None:
    """The Bug-1 regression: reclaim must bump generation so the dead session's
    late stop() can't free the NEXT lease."""
    a_inner, b_inner = DummyInner(), DummyInner()
    pool = make_pool([a_inner, b_inner])

    a = pool.start_realtime_session()
    gen_before = pool.replicas[0].generation

    # session A dies worker-side (kv_exhausted / loop error) — no gateway stop
    a_inner.active = False
    pool.set_replica_health(0, dict(LOADED))  # monitor tick reclaims the slot
    r = pool.replicas[0]
    assert r.state == READY and r.session is None
    assert r.generation == gen_before + 1, "reclaim must bump the generation"

    # a new session leases the reclaimed replica
    b = pool.start_realtime_session()
    assert r.state == BUSY and r.session is b_inner

    # A's deferred teardown (grace GC) — must be a no-op for the slot
    a.stop()
    assert r.state == BUSY, "stale release must not free a re-leased replica"
    assert r.session is b_inner, "stale release must not clobber the live session"

    b.stop()
    assert r.state == READY and pool.busy == 0
    print("stale release after reclaim: OK")


def test_crash_quarantine_and_recovery() -> None:
    inner = DummyInner()
    pool = make_pool([inner])
    session = pool.start_realtime_session()

    # transport died with the worker → stop() quarantines the slot
    inner.transport_dead = True
    session.stop()
    r = pool.replicas[0]
    assert r.state == DOWN

    # respawned worker reports healthy → generation bump + READY
    gen = r.generation
    pool.set_replica_health(0, dict(LOADED))
    assert r.state == READY and r.generation == gen + 1

    # health lost entirely → DOWN again, and is_loaded reflects it
    pool.set_replica_health(0, None)
    assert r.state == DOWN and not pool.is_loaded()
    print("crash quarantine + recovery: OK")


def main() -> int:
    test_acquire_release_and_capacity()
    test_stale_release_after_reclaim()
    test_crash_quarantine_and_recovery()
    print("\nVLM POOL TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
