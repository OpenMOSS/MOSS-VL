"""Recovery, ownership, disk admission and private-frame cleanup tests."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from server.config import Settings
from server.memory.frames import SessionFrames
from server.memory.maintenance import MemoryMaintenance
from server.memory.session import MemorySession
from server.memory.store import MemoryStore
from server.memory.writer import MemoryWriter
from server.persistence.media import MediaStore
from server.persistence.store import IndexStore
from server.tests.test_persistence import jpeg_with_exif

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def store(tmp_path):
    settings = Settings(data_dir=str(tmp_path), memory_db_path=str(tmp_path / 'memory.db'),
                        history_db_path='', memory_late_interaction=False, memory_decision_mode='vector',
                        memory_min_free_bytes=0)
    instance = MemoryStore(settings)
    instance.frames = SessionFrames(instance.path)
    instance.open()
    yield instance
    instance.close()


def writer_for(store):
    embedder = SimpleNamespace(name='test', dim=4,
        encode=lambda texts: np.ones((len(texts), 4), dtype=np.float32),
        encode_images=lambda images: np.ones((len(images), 4), dtype=np.float32))
    return MemoryWriter(store.settings, store, text_embedder=embedder, image_embedder=embedder)


def test_private_frames_are_isolated_from_sessions_and_archives(store):
    raw = jpeg_with_exif()
    index = IndexStore(store.settings)
    index.open()
    media = MediaStore(store.settings, index)
    media.open()
    try:
        archived = media.put_bytes(raw)['hash']
        writer = writer_for(store)
        writer.start()
        a, b = MemorySession('a', store.settings, store, writer), MemorySession('b', store.settings, store, writer)
        try:
            a.note_frame(raw)
            b.note_frame(raw)
            writer.drain(timeout=2)
            handle_a = store.recent('a')[0].media_hash
            handle_b = store.recent('b')[0].media_hash
            assert handle_a != handle_b
            assert a._load_frame(handle_a) == raw and b._load_frame(handle_b) == raw
            assert a._load_frame(handle_b) is None
            a.close()
            assert not store.frames._directory('a').exists()
            assert b._load_frame(handle_b) == raw
            assert Path(media.blob_path(archived)).exists()
            b.close()
        finally:
            writer.stop(timeout=2)
    finally:
        index.close()


def test_frame_delete_failure_is_persistent_and_retryable(store, monkeypatch):
    writer = writer_for(store)
    session = MemorySession('a', store.settings, store, writer)
    store.register_session('a')
    store.frames.put('a', b'frame')
    store.add_item('a', 'frame', text='frame')
    remove = store.frames.delete_session
    def fail(*args):
        raise PermissionError('simulated unlink failure')
    monkeypatch.setattr(store.frames, 'delete_session', fail)
    with pytest.raises(PermissionError):
        session.close()
    row = store._conn.execute('SELECT * FROM memory_sessions WHERE conversation_id=?', ('a',)).fetchone()
    assert row['state'] == 'pending' and row['attempts'] == 1
    assert store.count('a') == 0
    monkeypatch.setattr(store.frames, 'delete_session', remove)
    assert MemoryMaintenance(store).sweep()['cleaned_sessions'] == 1
    assert not store.cleanup_candidates()
    assert not store.frames._directory('a').exists()


def test_background_retry_runs_without_restarting(store, monkeypatch):
    store.add_item('failed', 'utterance', text='retry')
    object.__setattr__(store.settings, 'memory_maintenance_interval_s', 1)
    remove = store.frames.delete_session
    monkeypatch.setattr(store.frames, 'delete_session', lambda cid: (_ for _ in ()).throw(PermissionError('blocked')))
    with pytest.raises(PermissionError):
        store.delete_session('failed')
    maintenance = MemoryMaintenance(store)
    maintenance.start()
    try:
        monkeypatch.setattr(store.frames, 'delete_session', remove)
        deadline = time.monotonic() + 3
        while store.cleanup_candidates() and time.monotonic() < deadline:
            time.sleep(0.025)
        assert not store.cleanup_candidates()
    finally:
        maintenance.stop()


def test_recovery_preserves_current_owner_and_legacy_rows(store):
    store.add_item('old', 'utterance', text='old')
    store.frames.put('old', b'frame')
    store.add_item('legacy', 'utterance', text='legacy')
    store._conn.execute("DELETE FROM memory_sessions WHERE conversation_id='legacy'")
    store._conn.commit()
    store.close()
    recovered = MemoryStore(store.settings)
    recovered.frames = store.frames
    recovered.open()
    try:
        recovered.add_item('active', 'utterance', text='active')
        assert MemoryMaintenance(recovered).sweep(recover=True)['cleaned_sessions'] == 1
        assert recovered.count('old') == 0 and not recovered.frames._directory('old').exists()
        assert recovered.count('active') == recovered.count('legacy') == 1
        assert recovered.cleanup_candidates(include_legacy=True) == ['legacy']
    finally:
        recovered.close()


def test_process_crash_before_item_insert_recovers_frame_directory(tmp_path):
    path = tmp_path / 'crash.db'
    code = """import os,sys
from server.config import Settings
from server.memory.store import MemoryStore
from server.memory.frames import SessionFrames
s=MemoryStore(Settings(memory_db_path=sys.argv[1], memory_min_free_bytes=0))
s.open()
s.register_session('crashed')
SessionFrames(s.path).put('crashed', b'frame')
os._exit(0)
"""
    subprocess.run([sys.executable, '-c', code, str(path)], check=True)
    instance = MemoryStore(Settings(memory_db_path=str(path), memory_min_free_bytes=0))
    instance.frames = SessionFrames(path)
    instance.open()
    try:
        assert instance.frames._directory('crashed').exists()
        assert MemoryMaintenance(instance).sweep(recover=True)['cleaned_sessions'] == 1
        assert not instance.frames._directory('crashed').exists()
    finally:
        instance.close()


def test_second_process_cannot_open_active_database(store):
    code = 'from server.config import Settings; from server.memory.store import MemoryStore; import sys; MemoryStore(Settings(memory_db_path=sys.argv[1])).open()'
    result = subprocess.run([sys.executable, '-c', code, store.path], capture_output=True, text=True)
    assert result.returncode != 0 and 'owned by another' in result.stderr
    store.add_item('still-active', 'utterance', text='works')


def test_disk_pressure_pauses_and_resumes_memory_only(store, monkeypatch):
    import server.memory.store as module
    object.__setattr__(store.settings, 'memory_min_free_bytes', 100)
    writer = writer_for(store)
    writer.start()
    try:
        monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
        assert MemoryMaintenance(store).sweep()['paused_reason'] == 'low_disk'
        assert not writer.note_utterance('a', 'user', 'blocked', lang='en')
        assert store.count() == 0
        monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1000))
        assert MemoryMaintenance(store).sweep()['paused_reason'] == ''
        assert writer.note_utterance('a', 'user', 'accepted', lang='en')
        writer.drain(timeout=2)
        assert store.count('a') == 1
    finally:
        writer.stop(timeout=2)


def test_logical_budget_resumes_after_cleanup_and_vacuum(store):
    item = store.add_item('a', 'utterance', text='large')
    store.add_vector('a', item, 'text', np.ones(300000, dtype=np.float32))
    before = store.refresh_storage_status()['occupied_bytes']
    object.__setattr__(store.settings, 'memory_max_db_bytes', before // 2)
    assert store.refresh_storage_status()['paused_reason'] == 'memory_budget'
    store.delete_session('a')
    status = store.refresh_storage_status()
    assert status['paused_reason'] == '' and status['reusable_bytes'] > 0
    store.reclaim_space(vacuum=True)
    assert store.refresh_storage_status()['reusable_bytes'] == 0
    assert Path(store.path).stat().st_size < before


def test_offline_cli_dry_run_backup_and_explicit_legacy_cleanup(store):
    path = Path(store.path)
    store.add_item('legacy', 'utterance', text='retain until explicit migration')
    store._conn.execute('DELETE FROM memory_sessions')
    store._conn.commit()
    store.close()
    command = [sys.executable, str(ROOT / 'scripts/memory_maint.py'), '--db', str(path)]
    subprocess.run(command, check=True, capture_output=True)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM memory_items').fetchone()[0] == 1
    assert not list(path.parent.glob('*.backup-*'))
    rejected = subprocess.run(command + ['--apply'], capture_output=True)
    assert rejected.returncode != 0
    subprocess.run(command + ['--apply', '--offline', '--include-legacy', '--vacuum'], check=True, capture_output=True)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM memory_items').fetchone()[0] == 0
    backups = list(path.parent.glob('*.backup-*'))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as db:
        assert db.execute('SELECT COUNT(*) FROM memory_items').fetchone()[0] == 1


def test_offline_cli_refuses_active_store(store):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/memory_maint.py'), '--db', store.path,
                             '--apply', '--offline'], capture_output=True, text=True)
    assert result.returncode != 0 and 'owned by another' in result.stderr
    assert not list(Path(store.path).parent.glob('*.backup-*'))


def test_shared_media_pruner_honors_legacy_memory_references(store):
    spec = importlib.util.spec_from_file_location('history_prune_test', ROOT / 'scripts/history_prune.py')
    prune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prune)
    index = IndexStore(store.settings)
    index.open()
    media = MediaStore(store.settings, index)
    media.open()
    try:
        digest = media.put_bytes(jpeg_with_exif())['hash']
        store.add_item('legacy', 'frame', media_hash=digest)
        assert prune.prune_unreferenced(index, media, None, False) == 0
        assert Path(media.blob_path(digest)).exists()
        store.delete_session('legacy')
        index.upsert_conversation('saved', 'chat', 1)
        index.insert_turn('saved', role='user', text='saved image', ts=1, media_hashes=[digest])
        assert prune.prune_unreferenced(index, media, None, False) == 0
        assert Path(media.blob_path(digest)).exists()
        index.delete_conversation('saved')
        assert prune.prune_unreferenced(index, media, None, True) == 1
        assert Path(media.blob_path(digest)).exists()
        assert prune.prune_unreferenced(index, media, None, False) == 1
        assert not Path(media.blob_path(digest)).exists() and index.get_media(digest) is None
    finally:
        index.close()


def test_private_frame_handle_and_symlink_safety(store, tmp_path):
    with pytest.raises(ValueError):
        store.frames.load('a', '../../secret')
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'keep').write_text('keep')
    directory = store.frames._directory('a')
    directory.parent.mkdir(parents=True)
    directory.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError):
        store.frames.delete_session('a')
    assert (external / 'keep').exists()
