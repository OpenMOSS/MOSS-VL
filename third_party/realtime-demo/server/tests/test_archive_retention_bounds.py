import json
from pathlib import Path
from unittest.mock import patch

import pytest

from server.config import Settings
from server.persistence.store import IndexStore
from server.persistence.recorder import HistoryRecorder
from server.persistence.retention import acquire_archive_lock, plan_retention
from server.memory.store import MemoryStore


def setup(tmp_path, **kwargs):
    cfg = Settings(data_dir=str(tmp_path), history_db_path='', **kwargs)
    index = IndexStore(cfg)
    index.open()
    return cfg, index, HistoryRecorder(cfg, index)


def test_retention_is_preview_and_never_selects_unclosed_sessions(tmp_path):
    cfg, index, recorder = setup(tmp_path)
    try:
        for cid, ended in [('old', 1), ('new', 990), ('active', None)]:
            recorder._do_conv(cid, 'realtime', 1, None)
            if ended is not None: recorder._do_finalize(cid, ended, 'client')
        before = {p: p.read_bytes() for p in tmp_path.rglob('*.jsonl')}
        assert plan_retention(cfg)['conversations']==[]
        plan = plan_retention(cfg, days=.005, now=1000)
        assert plan['conversations']==['old']
        assert before == {p:p.read_bytes() for p in tmp_path.rglob('*.jsonl')}
    finally:
        index.close()


def test_retention_capacity_selects_oldest_and_reports_unmet_target(tmp_path):
    cfg, index, recorder = setup(tmp_path)
    try:
        for cid, ended in [('old', 1), ('new', 2), ('active', None)]:
            recorder._do_conv(cid, 'realtime', 1, None)
            if ended: recorder._do_finalize(cid, ended, 'client')
        plan = plan_retention(cfg,max_bytes=1,limit=1)
        assert plan['conversations']==['old'] and plan['batch_limited'] and plan['budget_unmet']
    finally:
        index.close()


def test_retention_exclusive_lock_blocks_when_api_uses_archives(tmp_path):
    with acquire_archive_lock(tmp_path/'index.db'):
        with pytest.raises(RuntimeError,match='Archive is in use'):
            acquire_archive_lock(tmp_path/'index.db',exclusive=True)


def test_failed_journal_deletion_keeps_index_for_retry(tmp_path):
    cfg, index, recorder = setup(tmp_path)
    try:
        recorder._do_conv('old','realtime',1,None)
        recorder._do_finalize('old',2,'client')
        with patch('server.persistence.recorder.os.unlink', side_effect=OSError('disk failure')):
            with pytest.raises(OSError): recorder.delete_conversation('old')
        assert index.list_conversations()[0]['conversation_id']=='old'
        recorder.delete_conversation('old')
        assert not index.list_conversations()
        assert not list(tmp_path.rglob('*.jsonl'))
    finally:
        index.close()


def test_archive_queue_bounds_still_finalize_incomplete_conversation(tmp_path):
    cfg, index, recorder = setup(tmp_path,history_queue_items=1)
    try:
        recorder.open_conversation('old','realtime',created_at=1)
        recorder.record_turn('old',role='user',text='will not fit')
        assert recorder.status()['rejected']==1 and recorder.status()['incomplete_sessions']==1
        recorder.finalize('old','client',ended_at=2)
        recorder.open()
        recorder.close()
        row = index.list_conversations()[0]
        assert row['ended_at']==2 and 'archive_incomplete' in row['end_reason']
        assert recorder.status()['queue_bytes']==0
        assert recorder.status()['pending_sessions']==0
    finally:
        index.close()


def test_archive_queue_and_session_budget_cannot_accumulate_forever(tmp_path):
    cfg, index, recorder = setup(tmp_path,history_queue_bytes=128,history_pending_sessions=1)
    try:
        recorder.open_conversation('one','realtime',created_at=1)
        recorder.open_conversation('two','realtime',created_at=1)
        recorder.record_turn('one',role='user',text='x'*256)
        for _ in range(100): recorder.record_turn('one',role='user',text='x'*256)
        assert recorder.status()['queue_bytes']<=128
        assert recorder.status()['pending_sessions']==1
        recorder.finalize('one','client')
        recorder.open()
        recorder.close()
        assert not recorder._paths and not recorder._resp_text
    finally:
        index.close()


def test_shared_media_is_not_counted_as_reclaimable_while_referenced(tmp_path):
    cfg, index, recorder = setup(tmp_path)
    memory = MemoryStore(cfg)
    try:
        digest = 'a'*64
        index.upsert_media(dict(hash=digest,algo='sha256',mime='image/jpeg',kind='image',bytes=100,
            width=None,height=None,duration_s=None,created_at=1,orig_name=None,thumb_path=None,poster_path=None))
        media = tmp_path/'media/example'
        media.parent.mkdir()
        media.write_bytes(b'x'*100)
        for cid in ('old','active'):
            recorder._do_conv(cid,'realtime',1,None)
            index.insert_turn(cid,role='user',text='image',ts=1,media_hashes=[digest])
        recorder._do_finalize('old',2,'client')
        plan = plan_retention(cfg,max_bytes=1)
        assert plan['conversations']==['old']
        assert plan['projected_payload_bytes']>=100 and plan['budget_unmet']
        recorder._do_finalize('active',3,'client')
        memory.add_item('runtime','frame',media_hash=digest)
        plan = plan_retention(cfg,max_bytes=1)
        assert plan['projected_payload_bytes']>=100
    finally:
        memory.close()
        index.close()
