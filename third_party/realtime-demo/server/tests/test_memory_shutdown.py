"""Writer termination must precede store and native embedding teardown."""

import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from server.config import Settings
from server.memory.store import MemoryStore
from server.memory.writer import MemoryWriter


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert predicate()


@pytest.fixture
def make_writer(tmp_path):
    created = []

    def make(capacity=512):
        store = MemoryStore(Settings(memory_db_path=str(tmp_path / f"{len(created)}.db"),
                                     memory_late_interaction=False))
        embedder = SimpleNamespace(name="test", dim=4,
                                   encode=lambda texts: np.ones((len(texts), 4), dtype=np.float32))
        writer = MemoryWriter(Settings(memory_late_interaction=False), store,
                              text_embedder=embedder, image_embedder=object())
        writer._q = queue.Queue(maxsize=capacity)
        entered, release = threading.Event(), threading.Event()
        handled = []

        def handle(job):
            if job["text"] == "blocked":
                entered.set()
                assert release.wait(3), "test did not release the embedding job"
            handled.append(job["text"])

        writer._handle = handle
        created.append((writer, store, release))
        return writer, store, entered, release, handled

    yield make
    for writer, store, release in created:
        release.set()
        writer.stop(timeout=2)
        store.close()


def submit(writer, text):
    writer.note_utterance("session", "user", text, lang="en")


def close_in_thread(writer):
    errors = []

    def close():
        try:
            writer.stop()
        except BaseException as exc:
            errors.append(exc)

    closer = threading.Thread(target=close)
    closer.start()
    wait_for(lambda: writer._stopping)
    return closer, errors


def test_idle_stop_is_idempotent_and_writer_can_restart(make_writer):
    writer, _, _, _, handled = make_writer()
    writer.stop()
    writer.start()
    writer.start()
    worker = writer._thread
    writer.stop(timeout=1)
    writer.stop(timeout=0)
    assert writer._thread is None and not worker.is_alive()
    assert writer._q.unfinished_tasks == 0
    writer.start()
    submit(writer, "restarted")
    writer.drain(timeout=1)
    assert handled == ["restarted"]


def test_graceful_stop_waits_and_rejects_new_jobs(make_writer):
    writer, _, entered, release, handled = make_writer()
    writer.start()
    worker = writer._thread
    submit(writer, "blocked")
    assert entered.wait(1)
    closer, errors = close_in_thread(writer)
    assert closer.is_alive() and writer._thread is worker
    submit(writer, "late")
    assert ("user", "late") not in writer._seen_utterances.get("session", set())
    assert writer._q.qsize() == 0
    release.set()
    closer.join(1)
    assert not errors and not closer.is_alive()
    assert not worker.is_alive() and writer._thread is None
    assert handled == ["blocked"]


def test_timed_stop_retains_handle_and_is_retryable(make_writer):
    writer, _, entered, release, handled = make_writer()
    writer.start()
    worker = writer._thread
    submit(writer, "blocked")
    assert entered.wait(1)
    with pytest.raises(TimeoutError, match="has not stopped"):
        writer.stop(timeout=0.01)
    assert writer._thread is worker and worker.is_alive()
    with pytest.raises(RuntimeError, match="still stopping"):
        writer.start()
    submit(writer, "late")
    release.set()
    writer.stop(timeout=1)
    assert writer._thread is None and handled == ["blocked"]
    writer.start()
    submit(writer, "late")
    writer.drain(timeout=1)
    assert handled == ["blocked", "late"]


def test_stop_does_not_need_space_in_a_full_queue(make_writer):
    writer, _, entered, release, handled = make_writer(capacity=2)
    writer.start()
    submit(writer, "blocked")
    assert entered.wait(1)
    submit(writer, "queued-1")
    submit(writer, "queued-2")
    assert writer._q.full()
    closer, errors = close_in_thread(writer)
    assert writer._q.full()
    release.set()
    closer.join(1)
    assert not errors and not closer.is_alive()
    assert handled == ["blocked", "queued-1", "queued-2"]
    assert writer._q.unfinished_tasks == 0 and writer._thread is None


def test_concurrent_stop_callers_join_the_same_worker(make_writer):
    writer, _, entered, release, handled = make_writer()
    writer.start()
    submit(writer, "blocked")
    assert entered.wait(1)
    closers = [close_in_thread(writer) for _ in range(2)]
    release.set()
    for closer, errors in closers:
        closer.join(1)
        assert not errors and not closer.is_alive()
    assert writer._thread is None and handled == ["blocked"]


def test_drain_includes_an_in_flight_job_and_honors_timeout(make_writer):
    writer, _, entered, release, _ = make_writer()
    writer.start()
    submit(writer, "blocked")
    assert entered.wait(1) and writer._q.empty()
    begin = time.monotonic()
    with pytest.raises(TimeoutError, match="drain timed out"):
        writer.drain(timeout=0.02)
    assert time.monotonic() - begin < 0.5
    release.set()
    writer.drain(timeout=1)


def test_overflow_balances_task_accounting(make_writer):
    writer, _, entered, release, handled = make_writer(capacity=1)
    writer.start()
    submit(writer, "blocked")
    assert entered.wait(1)
    submit(writer, "oldest")
    assert writer.note_utterance("session", "user", "replacement", lang="en") is False
    release.set()
    wait_for(lambda: len(handled) == 2)
    writer.drain(timeout=0.1)
    assert writer._q.unfinished_tasks == 0
    assert handled == ["blocked", "oldest"]
    assert writer.stats["utterances_rejected"] == 1
    assert writer.stats["dropped"] == 0
    assert writer.note_utterance("session", "user", "replacement", lang="en") is True
    writer.drain(timeout=1)
    assert handled == ["blocked", "oldest", "replacement"]


def test_overflow_evicts_frame_not_accepted_text(make_writer):
    writer, _, entered, release, handled = make_writer(capacity=2)
    writer.start()
    submit(writer, "blocked")
    assert entered.wait(1)
    submit(writer, "oldest")
    writer.note_frame("session", b"frame")
    assert writer.note_utterance("session", "user", "replacement", lang="en") is True
    release.set()
    writer.drain(timeout=1)
    assert handled == ["blocked", "oldest", "replacement"]
    assert writer._q.unfinished_tasks == 0
    assert writer.stats["dropped"] == 1


def test_stop_cannot_overtake_an_accepted_enqueue(make_writer):
    writer, _, _, _, handled = make_writer()
    entered, release = threading.Event(), threading.Event()
    original_put = writer._q.put_nowait

    def put(job):
        entered.set()
        assert release.wait(2)
        original_put(job)

    writer._q.put_nowait = put
    writer.start()
    producer = threading.Thread(target=submit, args=(writer, "accepted"))
    producer.start()
    assert entered.wait(1)
    errors = []

    def close():
        try:
            writer.stop()
        except BaseException as exc:
            errors.append(exc)

    closer = threading.Thread(target=close)
    closer.start()
    release.set()
    producer.join(1)
    closer.join(1)
    assert not errors and not closer.is_alive() and not producer.is_alive()
    assert handled == ["accepted"] and writer._q.unfinished_tasks == 0


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
def test_invalid_timeouts_are_rejected(make_writer, timeout):
    writer, _, _, _, _ = make_writer()
    with pytest.raises(ValueError):
        writer.stop(timeout)
    with pytest.raises(ValueError):
        writer.drain(timeout)


def test_runtime_does_not_release_resources_if_stop_fails(monkeypatch):
    from server import deps

    events = []

    def stop(*, timeout):
        assert timeout is None
        events.append("stop")
        raise TimeoutError("still running")

    runtime = deps.Runtime.__new__(deps.Runtime)
    runtime.memory_writer = SimpleNamespace(stop=stop)
    runtime.memory_store = SimpleNamespace(close=lambda: events.append("store"))
    runtime.history = SimpleNamespace(close=lambda: events.append("history"))
    runtime.index = SimpleNamespace(close=lambda: events.append("index"))
    monkeypatch.setattr(deps, "set_media_store", lambda value: events.append("media"))
    with pytest.raises(TimeoutError):
        runtime.close_persistence()
    assert events == ["stop"]
    runtime.memory_writer.stop = lambda **kwargs: events.append("joined")
    runtime.close_persistence()
    assert events == ["stop", "joined", "media", "store", "history", "index"]
