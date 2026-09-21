"""Multi-slot pool tests: SglangOmniPool with sglang_omni_sessions_per_replica=N.

Covers the P2 generalization from "one replica = one session" to "one replica
= N slots" (N = the remote omni instance's --max-running-requests):

- one 2-slot replica hosts two sessions; both slots busy → BUSY
- least-loaded routing across replicas (ties → lowest index)
- a session past every slot lands on the next replica / raises NoFreeReplica
- stop() hands the slot back (BUSY → READY)
- omni capacity rejection marks that replica full and falls through to the
  next READY replica; all replicas full → NoFreeReplica after one grace retry

Run:  <repo>/.venv/bin/python -m pytest server/tests/test_pool_multislot.py -q
"""
from __future__ import annotations

import sys
import time

from server.adapters.vlm.moss_vl_hf.online_pool import BUSY, READY, NoFreeReplica
from server.adapters.vlm.moss_vl_sglang_omni.pool import SglangOmniPool
from server.config import Settings
from server.tests.test_sglang_omni_adapter import FakeSglangOmniServer


def make_pool(urls: str, **overrides) -> SglangOmniPool:
    kwargs = dict(sglang_omni_urls=urls, sglang_omni_connect_timeout_s=5.0,
                  sglang_omni_health_interval_s=600.0,  # prober off in tests
                  sglang_omni_sessions_per_replica=2)
    kwargs.update(overrides)
    pool = SglangOmniPool(Settings(**kwargs))
    pool.capacity_retry_delay_s = 0.05
    return pool


def test_two_slots_one_replica() -> None:
    """A single 2-slot replica hosts two sessions; both land on replica 0."""
    server = FakeSglangOmniServer(max_sessions=2).start()
    try:
        pool = make_pool(server.url)
        pool.load("", -1, "online_streaming")
        assert pool.capacity == 2 and pool.busy == 0

        s1 = pool.start_realtime_session(prompt="")
        assert pool.replicas[0].state == READY  # one slot still free
        assert pool.busy == 1
        s2 = pool.start_realtime_session(prompt="")
        assert pool.replicas[0].state == BUSY  # slots exhausted
        assert pool.busy == 2 and server.active_sessions == 2

        # slot release: BUSY → READY, the used count is recycled
        s1.stop(timeout_seconds=2.0)
        assert pool.busy == 1 and pool.replicas[0].state == READY
        s2.stop(timeout_seconds=2.0)
        assert pool.busy == 0 and pool.replicas[0].state == READY
        status = pool.status()
        assert status["capacity"] == 2 and status["busy"] == 0
        assert status["replicas"][0]["slots"] == 2
        assert status["replicas"][0]["used"] == 0
        print("one replica, two slots, release recycles: OK")
    finally:
        server.close()


def test_least_loaded_across_replicas() -> None:
    """Two replicas × 2 slots: sessions spread least-loaded, ties → low index,
    the session past every slot raises NoFreeReplica."""
    server = FakeSglangOmniServer(max_sessions=4).start()
    try:
        # both pool replicas point at the same fake omni (URL identity is what
        # the pool keys on; the fake enforces 4 concurrent sessions total)
        pool = make_pool(f"{server.url},{server.url}")
        pool.load("", -1, "online_streaming")
        assert pool.capacity == 4

        s1 = pool.start_realtime_session(prompt="")  # tie at 0 → replica 0
        assert pool.replicas[0].used == 1
        s2 = pool.start_realtime_session(prompt="")  # r1 (0) < r0 (1) → replica 1
        assert pool.replicas[1].used == 1
        s3 = pool.start_realtime_session(prompt="")  # tie at 1 → replica 0
        assert pool.replicas[0].used == 2 and pool.replicas[0].state == BUSY
        s4 = pool.start_realtime_session(prompt="")  # replica 1 fills too
        assert all(r.state == BUSY for r in pool.replicas)
        assert pool.busy == 4

        try:
            pool.start_realtime_session(prompt="")
        except NoFreeReplica as exc:
            assert "4/4" in str(exc), str(exc)
        else:
            raise AssertionError("full pool must raise NoFreeReplica")

        for s in (s1, s2, s3, s4):
            s.stop(timeout_seconds=2.0)
        assert pool.busy == 0
        assert all(r.state == READY for r in pool.replicas)
        print("least-loaded spread + all-full NoFreeReplica: OK")
    finally:
        server.close()


def test_capacity_reject_falls_to_next_replica() -> None:
    """Omni-side session_capacity_exceeded marks that replica full (BUSY) and
    the session lands on the next READY replica — no DOWN quarantine."""
    wedged = FakeSglangOmniServer(reject_capacity=True).start()
    good = FakeSglangOmniServer().start()
    try:
        pool = make_pool(f"{wedged.url},{good.url}", sglang_omni_sessions_per_replica=1)
        pool.load("", -1, "online_streaming")

        session = pool.start_realtime_session(prompt="")
        assert pool.replicas[0].state == BUSY  # full, not DOWN
        assert pool.replicas[1].state == BUSY  # slots=1, hosting the session
        assert wedged.rejected_connects == 1
        session.stop(timeout_seconds=2.0)
        assert pool.replicas[1].state == READY
        print("capacity reject → next READY replica: OK")
    finally:
        wedged.close()
        good.close()


def test_all_replicas_capacity_rejected() -> None:
    """Every replica capacity-rejects → one teardown-grace retry, then
    NoFreeReplica; replicas stay BUSY(full), never DOWN."""
    server = FakeSglangOmniServer(reject_capacity=True).start()
    try:
        pool = make_pool(f"{server.url},{server.url}", sglang_omni_sessions_per_replica=1)
        pool.load("", -1, "online_streaming")
        try:
            pool.start_realtime_session(prompt="")
        except NoFreeReplica as exc:
            assert "2/2" in str(exc), str(exc)
        else:
            raise AssertionError("all-full pool must raise NoFreeReplica")
        assert all(r.state == BUSY for r in pool.replicas)
        # r0, r1 → grace → r0, r1 again (full marks reset after the grace)
        assert server.rejected_connects == 4, server.rejected_connects
        print("all replicas capacity-rejected → grace retry → NoFreeReplica: OK")
    finally:
        server.close()


def test_full_mark_cleared_by_prober() -> None:
    """A capacity-rejection full mark is not a life sentence: once omni answers
    /health and has room again, the prober clears the mark (DOWN replicas keep
    their existing recovery path)."""
    server = FakeSglangOmniServer(reject_capacity=True).start()
    try:
        pool = make_pool(server.url, sglang_omni_sessions_per_replica=1,
                         sglang_omni_health_interval_s=1.0)  # prober min cadence
        pool.load("", -1, "online_streaming")
        try:
            pool.start_realtime_session(prompt="")
        except NoFreeReplica:
            pass
        else:
            raise AssertionError("expected NoFreeReplica")
        assert pool.replicas[0].state == BUSY  # full mark, not DOWN

        server.reject_capacity = False  # omni has room again
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if pool.replicas[0].state == READY:
                break
            time.sleep(0.1)
        assert pool.replicas[0].state == READY, "prober must clear the stale full mark"
        session = pool.start_realtime_session(prompt="")  # and the slot works
        session.stop(timeout_seconds=2.0)
        print("stale full mark cleared by prober: OK")
    finally:
        server.close()


def main() -> int:
    test_two_slots_one_replica()
    test_least_loaded_across_replicas()
    test_capacity_reject_falls_to_next_replica()
    test_all_replicas_capacity_rejected()
    test_full_mark_cleared_by_prober()
    print("\nPOOL MULTISLOT TESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
