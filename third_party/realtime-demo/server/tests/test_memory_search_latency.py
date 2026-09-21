from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import patch

import numpy as np

from server.config import Settings
from server.memory.store import MemoryStore
from server.memory.writer import MemoryWriter
from types import SimpleNamespace
import server.memory.store as store_module


def test_slow_capacity_query_does_not_hold_retrieval_lock(tmp_path):
    store = MemoryStore(Settings(memory_db_path=str(tmp_path / 'memory.db'), memory_min_free_bytes=0))
    store.open()
    item = store.add_item('s', 'utterance', text='fact')
    store.add_vector('s', item, 'text', np.ones(4))
    usage = store_module.shutil.disk_usage(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def slow(path):
        entered.set()
        assert release.wait(3)
        return usage
    try:
        with patch.object(store_module.shutil, 'disk_usage', slow), ThreadPoolExecutor(max_workers=2) as pool:
            monitoring = pool.submit(store.refresh_storage_status)
            assert entered.wait(1)
            reading = pool.submit(store.search, 's', 'text', np.ones(4))
            try:
                assert reading.result(timeout=0.5)[0][0] == item
            finally:
                release.set()
            monitoring.result()
    finally:
        store.close()


def test_writer_uses_cached_capacity_state(tmp_path, monkeypatch):
    settings = Settings(memory_db_path=str(tmp_path / 'memory.db'), memory_late_interaction=False)
    store = MemoryStore(settings)
    embedder = SimpleNamespace(name='fake', dim=4, encode=lambda texts: np.ones((len(texts), 4)))
    writer = MemoryWriter(settings, store, text_embedder=embedder, image_embedder=object())
    def unexpected():
        raise AssertionError('filesystem capacity lookup on the per-write path')
    monkeypatch.setattr(store, 'refresh_storage_status', unexpected)
    writer.start()
    try:
        writer.note_utterance('s', 'user', 'fact', lang='en')
        writer.drain(timeout=2)
        assert store.count('s') == 1
    finally:
        writer.stop()
        store.close()
