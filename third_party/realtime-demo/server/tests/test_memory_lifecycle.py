"""Session cleanup races: real SQLite/threads, no model or external service."""
import asyncio
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from server.config import Settings
from server.memory.facts import FactExtractor
from server.memory.rollover import RolloverManager
from server.memory.session import MemorySession
from server.memory.store import MemoryStore
from server.memory.writer import MemoryWriter
from server.schemas import SessionConfig
from server.session.manager import SessionManager
from server.session.orchestrator import EngineSet, Orchestrator
from server.session.state import SessionState
from server.tests.fakes import FakeVlmSession


@pytest.fixture
def stack(tmp_path):
    settings = Settings(data_dir=str(tmp_path), memory_db_path=str(tmp_path / 'memory.db'),
                        memory_late_interaction=False, memory_decision_mode='vector',
                        memory_summary_provider='none', session_grace_seconds=1)
    store = MemoryStore(settings)
    store.open()
    embedder = SimpleNamespace(name='fake', dim=4,
                              encode=lambda texts: np.ones((len(texts), 4), dtype=np.float32))
    writer = MemoryWriter(settings, store, text_embedder=embedder, image_embedder=object())
    yield settings, store, writer
    writer.stop(timeout=5)
    store.close()


def make_session(stack, cid='a'):
    settings, store, writer = stack
    return MemorySession(cid, settings, store, writer)


def assert_clean(store, writer, cid):
    assert store.count(cid) == 0
    assert not any(key[0] == cid for key in store._idx)
    assert cid not in store._li and cid not in store._li_loaded
    assert cid not in writer._frames and cid not in writer._seen_utterances
    for table in ('memory_vectors', 'memory_item_keys'):
        assert not store._conn.execute(
            f'SELECT item_id FROM {table} WHERE item_id NOT IN (SELECT id FROM memory_items)').fetchall()


def spawn_close(session):
    errors = []
    done = threading.Event()
    def run():
        try:
            session.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()
    thread = threading.Thread(target=run)
    thread.start()
    return thread, done, errors


def test_atomic_delete_isolation_and_stale_ids(stack):
    _, store, writer = stack
    session = make_session(stack)
    item = store.add_item('a', 'utterance', text='old')
    store.add_vector('a', item, 'text', np.ones(4))
    store.add_vector_late('a', item, np.ones((2, 4)))
    store.put_key(item, 'old key')
    store.search('a', 'text', np.ones(4))
    store.search_late('a', np.ones((2, 4)))
    session.close()
    session.close()
    assert_clean(store, writer, 'a')
    # SQLite may reuse INTEGER PRIMARY KEY ids after deletion. A late fact
    # update for A must not write into B's newly allocated row.
    other = store.add_item('b', 'utterance', text='new')
    assert other == item
    store.update_vector('a', item, 'text', np.ones(4))
    store.add_vector_late('a', item, np.ones((2, 4)))
    store.put_key(item, 'stale', conversation_id='a')
    store.mark_injected(item, 'a')
    assert store._conn.execute('SELECT COUNT(*) FROM memory_vectors').fetchone()[0] == 0
    assert store.get_key(other) is None
    assert store.count('b') == 1


def test_discard_queued_jobs_and_reject_closed_producers(stack):
    _, store, writer = stack
    a, b = make_session(stack), make_session(stack, 'b')
    a.note_user_turn('discard a')
    b.note_user_turn('keep b')
    a.close()
    a.note_user_turn('late')
    a.note_frame(b'late-frame')
    assert writer._q.qsize() == writer._q.unfinished_tasks == 1
    writer.start()
    writer.drain(timeout=2)
    assert_clean(store, writer, 'a')
    assert store.count('b') == 1
    assert not a.recall_for_turn('anything')
    b.close()


def test_idle_writer_does_not_retain_completed_payload(stack):
    _, _, writer = stack
    class Payload:
        pass
    payload = Payload()
    ref = weakref.ref(payload)
    writer._handle = lambda job: None
    writer.start()
    writer._put({'conv': 'a', 'payload': payload}, droppable=False)
    del payload
    writer.drain(timeout=2)
    assert ref() is None


def test_inflight_writer_finishes_before_delete(stack):
    _, store, writer = stack
    a, b = make_session(stack), make_session(stack, 'b')
    entered, release = threading.Event(), threading.Event()
    def encode(texts):
        if texts == ['blocked']:
            entered.set()
            assert release.wait(3)
        return np.ones((len(texts), 4), dtype=np.float32)
    writer.text.encode = encode
    writer.start()
    a.note_user_turn('blocked')
    assert entered.wait(1)
    a.lifetime.seal()
    closer, done, errors = spawn_close(a)
    try:
        assert not done.wait(0.05)
        b.note_user_turn('other-session')
        a.note_user_turn('late')
    finally:
        release.set()
        closer.join(3)
    assert done.is_set() and not errors
    writer.drain(timeout=2)
    assert_clean(store, writer, 'a')
    assert store.count('b') == 1


def test_prefetch_inflight_cannot_repopulate_cache(stack):
    _, store, writer = stack
    session = make_session(stack)
    entered, release = threading.Event(), threading.Event()
    item = store.add_item('a', 'utterance', text='old')
    store.add_vector('a', item, 'text', np.ones(4))
    search = session.retriever.search
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return search(*args, **kwargs)
    session.retriever.search = blocked
    reader = threading.Thread(target=session.prefetch, args=('earlier fact',))
    reader.start()
    assert entered.wait(1)
    closer, done, errors = spawn_close(session)
    try:
        assert not done.wait(0.05)
    finally:
        release.set()
        reader.join(3)
        closer.join(3)
    assert done.is_set() and not errors and session._prefetch is None
    assert_clean(store, writer, 'a')


def test_cancelled_fact_thread_cannot_outlive_cleanup(stack):
    async def run():
        settings, store, writer = stack
        session = make_session(stack)
        item = store.add_item('a', 'utterance', text='fact')
        extractor = FactExtractor(settings, store, writer)
        extractor.available = lambda: True
        async def extract(*args):
            return ['key']
        extractor._extract = extract
        session._facts = extractor
        entered, release = threading.Event(), threading.Event()
        def encode(texts):
            entered.set()
            assert release.wait(3)
            return np.ones((len(texts), 4), dtype=np.float32)
        writer.text.encode = encode
        task = asyncio.create_task(session.maybe_extract_facts(item, 'fact', []))
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        closer = asyncio.create_task(asyncio.to_thread(session.close))
        try:
            await asyncio.sleep(0.05)
            assert not closer.done()
        finally:
            release.set()
            await asyncio.wait_for(closer, 3)
        assert_clean(store, writer, 'a')
    asyncio.run(run())


def test_compact_worker_discards_late_result(stack):
    settings, store, writer = stack
    session = make_session(stack)
    store.add_item('a', 'utterance', text='fact')
    object.__setattr__(settings, 'memory_summary_provider', 'pi')
    entered, release = threading.Event(), threading.Event()
    def compact(*args):
        entered.set()
        assert release.wait(3)
        return {'summary': 'fact', 'pins': []}
    rollover = RolloverManager(settings, store, 'a', lifetime=session.lifetime,
                               pi=SimpleNamespace(compact=compact))
    assert rollover.maybe_prefetch_compact(10000)
    assert entered.wait(1)
    rollover.seal()
    closer, done, errors = spawn_close(session)
    try:
        assert not done.wait(0.05)
        assert not rollover.maybe_prefetch_compact(10000)
    finally:
        release.set()
        closer.join(3)
    assert done.is_set() and not errors and rollover._prefetch is None
    assert_clean(store, writer, 'a')


def test_repeated_sessions_leave_no_runtime_rows(stack):
    _, store, writer = stack
    writer.start()
    for n in range(100):
        session = make_session(stack, f's{n}')
        session.note_user_turn(f'fact {n}')
        writer.drain(timeout=2)
        session.prefetch('fact')
        session.close()
    assert store.count() == 0
    assert not store._idx and not store._li and not store._li_loaded
    assert not writer._frames and not writer._seen_utterances
    assert writer._q.unfinished_tasks == 0


def test_grace_reconnect_and_expiry(stack):
    async def run():
        settings, store, writer = stack
        writer.start()
        history = []
        archive = SimpleNamespace(open_conversation=lambda *a, **kw: None,
                                  realtime_sink=lambda *a: lambda *a: None,
                                  finalize=lambda cid, **kw: history.append(cid))
        manager = SessionManager(settings, history=archive, memory=(store, writer))
        async def engines():
            return EngineSet(vlm=FakeVlmSession(), asr=None, tts=None)
        state = await manager.create(SessionConfig(), engines)
        memory = state.orchestrator.memory
        memory.note_user_turn('keep through grace')
        writer.drain(timeout=2)
        _, token, _, _ = await manager.attach_ws(state.session_id)
        await manager.detach_ws(state.session_id, token)
        assert store.count(state.session_id) == 1 and not memory.lifetime.closing
        _, token, _, _ = await manager.attach_ws(state.session_id)
        await asyncio.sleep(1.05)
        assert manager.active_count == 1 and store.count(state.session_id) == 1
        await manager.detach_ws(state.session_id, token)
        for _ in range(100):
            if memory._cleaned:
                break
            await asyncio.sleep(0.025)
        assert memory._cleaned and manager.active_count == 0
        assert_clean(store, writer, state.session_id)
        assert history == [state.session_id], 'archive is finalized, not deleted'
        assert not await manager.close(state.session_id)
        await manager.aclose()
    asyncio.run(run())


def test_cancelled_close_caller_does_not_cancel_cleanup(stack):
    async def run():
        settings, store, writer = stack
        memory = make_session(stack)
        store.add_item('a', 'utterance', text='fact')
        entered, release = threading.Event(), threading.Event()
        def operation():
            with memory.lifetime.operation() as admitted:
                assert admitted
                entered.set()
                assert release.wait(3)
        worker = threading.Thread(target=operation)
        worker.start()
        assert entered.wait(1)
        state = SessionState(session_id='a', config=SessionConfig(), replay_size=10, out_queue_size=10)
        orch = Orchestrator(state, EngineSet(vlm=FakeVlmSession(), asr=None, tts=None),
                            settings, memory=memory)
        caller = asyncio.create_task(orch.close())
        await asyncio.sleep(0.05)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert not orch._close_task.done()
        release.set()
        await asyncio.wait_for(orch.close(), 3)
        worker.join(1)
        assert_clean(store, writer, 'a')
    asyncio.run(run())


def test_failed_cleanup_can_retry(stack, monkeypatch):
    _, store, writer = stack
    session = make_session(stack)
    store.add_item('a', 'utterance', text='fact')
    delete = store.delete_session
    def fail(*args):
        raise RuntimeError('disk failure')
    monkeypatch.setattr(store, 'delete_session', fail)
    with pytest.raises(RuntimeError):
        session.close()
    assert session.lifetime.closing and not session._cleaned
    monkeypatch.setattr(store, 'delete_session', delete)
    session.close()
    assert_clean(store, writer, 'a')


def test_delete_transaction_rolls_back_all_tables(stack):
    _, store, writer = stack
    session = make_session(stack)
    item = store.add_item('a', 'utterance', text='fact')
    store.add_vector('a', item, 'text', np.ones(4))
    store.put_key(item, 'key')
    store._conn.execute("CREATE TRIGGER fail_key_delete BEFORE DELETE ON memory_item_keys "
                        "BEGIN SELECT RAISE(ABORT, 'test failure'); END")
    store._conn.commit()
    with pytest.raises(Exception, match='test failure'):
        session.close()
    assert store.count('a') == 1 and store.get_key(item) == 'key'
    assert store._conn.execute('SELECT COUNT(*) FROM memory_vectors').fetchone()[0] == 1
    store._conn.execute('DROP TRIGGER fail_key_delete')
    store._conn.commit()
    session.close()
    assert_clean(store, writer, 'a')
