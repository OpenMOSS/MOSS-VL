import json
from pathlib import Path
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from server.config import Settings
from server.memory.frames import SessionFrames
from server.memory.store import MemoryStore, MemoryBudgetExceeded
from server.memory.writer import MemoryWriter
from server.memory.lifecycle import SessionLifetime


def settings(tmp_path, **kwargs):
    return Settings(memory_db_path=str(tmp_path/'memory.db'), memory_min_free_bytes=0, **kwargs)


def test_session_and_global_item_limits_are_independent(tmp_path):
    store = MemoryStore(settings(tmp_path, memory_session_max_items=2, memory_total_max_items=3))
    try:
        store.add_item('a','utterance',text='one')
        store.add_item('a','utterance',text='two')
        with pytest.raises(MemoryBudgetExceeded, match='session_items'):
            store.add_item('a','utterance',text='three')
        store.add_item('b','utterance',text='three')
        with pytest.raises(MemoryBudgetExceeded, match='total_items'):
            store.add_item('c','utterance',text='four')
        store.delete_session('a')
        store.add_item('b','utterance',text='four')
        assert store.count()==2
    finally:
        store.close()


def test_vector_replacements_charge_only_the_delta(tmp_path):
    store = MemoryStore(settings(tmp_path, memory_session_max_bytes=40))
    try:
        item = store.add_item('a','utterance',text='test')
        for _ in range(3):
            store.add_vector('a',item,'text',np.ones(8))
            store.search('a','text',np.ones(8))
        assert store.resource_status('a')['data_bytes']==36
        assert len(store._idx[('a','text')].ids)==1
        with pytest.raises(MemoryBudgetExceeded):
            store.update_vector('a',item,'text',np.ones(16))
        assert store.resource_status('a')['data_bytes']==36
    finally:
        store.close()


def test_cache_cap_evicts_without_deleting_persistent_evidence(tmp_path):
    store = MemoryStore(settings(tmp_path, memory_cache_max_bytes=64))
    try:
        for cid in ('a','b','c'):
            item = store.add_item(cid,'utterance',text='fact')
            store.add_vector(cid,item,'text',np.ones(8))
            assert store.search(cid,'text',np.ones(8))
            assert store.resource_status()['cache_bytes']<=64
        assert store.count()==3
        assert store.search('a','text',np.ones(8))
    finally:
        store.close()


def test_frame_limits_preserve_existing_frames_and_release_on_close(tmp_path):
    frames = SessionFrames(tmp_path/'db', settings(tmp_path, memory_session_frame_bytes=5, memory_total_frame_bytes=8))
    handle = frames.put('a',b'1234')
    frames.put('a',b'1234')
    assert frames.usage()==4
    with pytest.raises(MemoryBudgetExceeded): frames.put('a',b'xx')
    frames.put('b',b'abcd')
    with pytest.raises(MemoryBudgetExceeded): frames.put('c',b'x')
    assert frames.load('a',handle)==b'1234'
    frames.delete_session('a')
    assert frames.usage()==4
    frames.put('c',b'x')


def test_queue_payload_cap_includes_pending_and_releases_on_discard(tmp_path):
    cfg = settings(tmp_path, memory_queue_max_bytes=8, memory_queue_session_bytes=4)
    store = MemoryStore(cfg)
    writer = MemoryWriter(cfg,store,text_embedder=object(),image_embedder=object())
    a,b = SessionLifetime(),SessionLifetime()
    assert writer._put({'t':'frame','conv':'a','jpeg':b'1234','lifetime':a},droppable=True)
    assert not writer._put({'t':'frame','conv':'a','jpeg':b'5','lifetime':a},droppable=True)
    assert writer._put({'t':'frame','conv':'b','jpeg':b'1234','lifetime':b},droppable=True)
    assert not writer._put({'t':'frame','conv':'c','jpeg':b'x'},droppable=True)
    a.seal()
    writer.discard_session('a',a)
    assert writer._payload_bytes==4 and 'a' not in writer._session_payload_bytes
    b.seal()
    writer.discard_session('b',b)
    assert writer._payload_bytes==0 and not writer._session_payload_bytes


def test_wal_reader_pressure_pauses_writes_and_recovers(tmp_path):
    store = MemoryStore(settings(tmp_path,memory_max_wal_bytes=1024))
    reader = None
    try:
        store.add_item('a','utterance',text='first')
        reader = sqlite3.connect(store.path)
        reader.execute('BEGIN')
        reader.execute('SELECT * FROM memory_items').fetchall()
        store.add_item('a','utterance',text='second')
        store.reclaim_space()
        assert store.refresh_storage_status()['paused_reason']=='wal_budget'
        with pytest.raises(MemoryBudgetExceeded): store.add_item('a','utterance',text='blocked')
        reader.close()
        reader = None
        store.reclaim_space()
        assert store.refresh_storage_status()['paused_reason']==''
        store.add_item('a','utterance',text='resumed')
    finally:
        if reader: reader.close()
        store.close()


def test_cleanup_failure_is_bounded_and_retryable(tmp_path):
    store = MemoryStore(settings(tmp_path,memory_tracked_sessions=1))
    store.frames = SessionFrames(store.path,store.settings)
    try:
        store.add_item('a','utterance',text='fact')
        with patch.object(store.frames,'delete_session',side_effect=OSError('disk unavailable')):
            with pytest.raises(OSError): store.delete_session('a')
        assert store.cleanup_candidates()==['a']
        with pytest.raises(MemoryBudgetExceeded,match='tracked_sessions'):
            store.add_item('b','utterance',text='fact')
        store.delete_session('a')
        store.add_item('b','utterance',text='fact')
    finally:
        store.close()


def test_repeated_sessions_leave_no_cache_or_usage_entries(tmp_path):
    store = MemoryStore(settings(tmp_path))
    try:
        for n in range(150):
            cid = str(n)
            item = store.add_item(cid,'utterance',text='retained during session')
            store.add_vector(cid,item,'text',np.ones(1024))
            store.add_vector_late(cid,item,np.ones((8,1024)))
            store.search(cid,'text',np.ones(1024))
            store.search_late(cid,np.ones((2,1024)))
            store.delete_session(cid)
            assert store.resource_status()['data_bytes']==0
            assert store.resource_status()['cache_bytes']==0
            assert not store._cache_order and not store._usage
        assert store.count()==0
    finally:
        store.close()


def test_failed_commit_rolls_back_and_does_not_corrupt_budget(tmp_path):
    store = MemoryStore(settings(tmp_path,memory_session_max_items=1))
    store.open()
    real = store._conn
    class FaultyCommit:
        failed = False
        def __getattr__(self,name): return getattr(real,name)
        def commit(self):
            if not self.failed:
                self.failed = True
                raise sqlite3.OperationalError('simulated commit failure')
            return real.commit()
    store._conn = FaultyCommit()
    try:
        with pytest.raises(sqlite3.OperationalError): store.add_item('a','utterance',text='not committed')
        assert store.count()==0 and store.resource_status()['data_bytes']==0
        store.add_item('a','utterance',text='committed')
        assert store.count()==1 and store.resource_status()['items']==1
    finally:
        store.close()
