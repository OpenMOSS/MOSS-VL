"""Local handshake ownership must survive remote capacity recovery."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

from server.adapters.vlm.moss_vl_hf.online_pool import BUSY, DOWN, READY, NoFreeReplica
from server.adapters.vlm.moss_vl_sglang_omni import pool as pool_module
from server.adapters.vlm.moss_vl_sglang_omni.client import SessionCapacityExceeded
from server.adapters.vlm.moss_vl_sglang_omni.pool import SglangOmniPool


class Session:
    worker_transport_dead = False

    def __init__(self, name):
        self.session_id = name
        self.stops = 0

    def stop(self, timeout_seconds):
        self.stops += 1
        return {"stopped": True}

    def status(self):
        return {}


@pytest.fixture
def pool():
    p = SglangOmniPool(SimpleNamespace(
        sglang_omni_urls="http://unused.invalid",
        sglang_omni_sessions_per_replica=4,
        sglang_omni_health_interval_s=1.0,
    ))
    p.replicas[0].state = READY
    p._probe_health = lambda url: {"ok": True}
    return p


def test_real_prober_retains_inflight_reservation(pool):
    pool.replicas[0].slots = 1
    entered = threading.Event()
    finish = threading.Event()

    class OneProbe:
        calls = 0

        def wait(self, interval):
            self.calls += 1
            return self.calls > 1

    def start(replica, params):
        entered.set()
        assert finish.wait(5)
        return Session("inflight")

    pool._start_on_replica = start
    pool._prober_stop = OneProbe()
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(pool.start_realtime_session)
        try:
            assert entered.wait(5)
            assert pool.busy == 1
            pool._start_prober()
            pool._prober.join(5)
            assert not pool._prober.is_alive()
            assert pool.busy == 1
            assert pool.replicas[0].state == BUSY
        finally:
            finish.set()
            session = future.result(timeout=5)
            session.stop()


@pytest.mark.parametrize("count", [1, 4, 6])
def test_probe_during_concurrent_handshakes_preserves_slots(pool, count):
    r = pool.replicas[0]
    r.slots = count
    entered = threading.Barrier(count + 1)
    finish = threading.Event()

    def start(replica, params):
        entered.wait(timeout=5)
        assert finish.wait(5)
        return Session(params["name"])

    pool._start_on_replica = start
    with ThreadPoolExecutor(max_workers=count) as workers:
        futures = [workers.submit(pool.start_realtime_session, name=str(i)) for i in range(count)]
        try:
            entered.wait(timeout=5)
            assert r.pending == count and pool.busy == count
            pool._probe_replicas()
            assert r.pending == count and pool.busy == count
            assert r.state == BUSY
            with pytest.raises(NoFreeReplica):
                pool.start_realtime_session(name="excess")
        finally:
            finish.set()
        sessions = [f.result(timeout=5) for f in futures]
    assert r.pending == 0 and len(r.sessions) == count and pool.busy == count
    for session in sessions:
        session.stop()
        session.stop()
        assert session._inner.stops == 1
    assert pool.busy == 0 and r.state == READY


def test_expired_remote_mark_does_not_clear_pending_or_live_slots(pool, monkeypatch):
    monkeypatch.setattr(pool_module, "time", SimpleNamespace(monotonic=lambda: 20.0))
    r = pool.replicas[0]
    r.sessions.add(Session("live"))
    r.pending = 2
    r.remote_full_until = 10.0
    r.state = BUSY
    pool._probe_replicas()
    assert r.remote_full_until == 0.0
    assert r.pending == 2 and r.used == 3 and r.state == READY
    status = pool.status()["replicas"][0]
    assert status["pending"] == 2 and status["tracked_sessions"] == 1


def test_probe_cannot_clear_a_newer_remote_rejection(pool, monkeypatch):
    monkeypatch.setattr(pool_module, "time", SimpleNamespace(monotonic=lambda: 20.0))
    r = pool.replicas[0]
    r.remote_full_until = 10.0
    r.state = BUSY

    def probe(url):
        with pool._lock:
            r.remote_full_until = 25.0
        return {"ok": True}

    pool._probe_health = probe
    pool._probe_replicas()
    assert r.remote_full_until == 25.0 and r.state == BUSY


@pytest.mark.parametrize("error,state", [(ValueError("config"), READY), (ConnectionError("ws"), DOWN)])
def test_failed_handshake_releases_only_its_reservation(pool, error, state):
    r = pool.replicas[0]
    r.pending = 1
    r.sessions.add(Session("live"))

    def fail(replica, params):
        raise error

    pool._start_on_replica = fail
    with pytest.raises(type(error)):
        pool.start_realtime_session()
    assert r.pending == 1 and len(r.sessions) == 1 and r.used == 2
    assert r.state == state


def test_capacity_retry_preserves_another_handshake(pool, monkeypatch):
    r = pool.replicas[0]
    r.slots = 2
    r.pending = 1
    now = [10.0]
    calls = []

    def sleep(seconds):
        assert r.pending == 1 and r.state == BUSY
        now[0] += seconds
        pool._probe_replicas()
        assert r.pending == 1 and r.used == 1

    monkeypatch.setattr(pool_module, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))

    def start(replica, params):
        calls.append(True)
        if len(calls) == 1:
            raise SessionCapacityExceeded("full")
        assert r.pending == 2
        return Session("retried")

    pool._start_on_replica = start
    session = pool.start_realtime_session()
    assert len(calls) == 2 and r.pending == 1 and r.used == 2 and r.state == BUSY
    session.stop()
    assert r.pending == 1 and r.used == 1 and r.state == READY


def test_success_does_not_unquarantine_a_concurrently_failed_replica(pool):
    r = pool.replicas[0]

    def start(replica, params):
        with pool._lock:
            r.state = DOWN
        return Session("successful")

    pool._start_on_replica = start
    session = pool.start_realtime_session()
    assert r.pending == 0 and r.used == 1 and r.state == DOWN
    pool._probe_replicas()
    assert r.used == 1 and r.state == READY
    session.stop()


def test_release_and_load_preserve_other_handshake(pool):
    r = pool.replicas[0]
    r.slots = 2
    live = Session("live")
    r.sessions.add(live)
    r.pending = 1
    r.state = BUSY
    pool._release(0, live)
    pool._start_prober = lambda: None
    pool.load("", -1, "online_streaming")
    assert r.pending == 1 and r.used == 1 and r.state == READY
