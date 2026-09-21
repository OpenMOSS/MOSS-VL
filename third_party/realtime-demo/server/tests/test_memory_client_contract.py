"""Initial memory instructions and the lifetime of an explicitly shared still."""
import asyncio
import time
from types import SimpleNamespace

import pytest

from server.config import Settings
from server.memory.inject import augment_system_prompt
from server.routers.sessions import _vlm_start_params
from server.schemas import SessionConfig
from server.session.orchestrator import EngineSet, Orchestrator
from server.session.state import SessionState
from server.tests.fakes import FakeVlmSession
from server.tests.test_memory_pi import _make_session
from server.memory.retrieval import Candidate


def test_initial_session_preserves_system_prompt_with_memory_enabled():
    rt = SimpleNamespace(settings=Settings(), memory_store=object())
    cfg = SessionConfig(system_prompt='Custom instructions', asr_language='zh')
    prompt = _vlm_start_params(rt, cfg)['system_prompt']
    assert prompt == 'Custom instructions'
    assert _vlm_start_params(rt, SessionConfig())['system_prompt'] is None


def test_memory_disabled_keeps_existing_system_prompt_behavior():
    rt = SimpleNamespace(settings=Settings(), memory_store=None)
    assert _vlm_start_params(rt, SessionConfig())['system_prompt'] is None
    assert _vlm_start_params(rt, SessionConfig(system_prompt='custom'))['system_prompt'] == 'custom'


@pytest.mark.parametrize('source,expected', [('image', True), ('camera', False), ('screen', False), ('none', False)])
def test_only_still_images_survive_live_frame_expiry(source, expected):
    async def run():
        engine = FakeVlmSession()
        state = SessionState(session_id='test', config=SessionConfig(video_source=source), replay_size=20, out_queue_size=20)
        orch = Orchestrator(state, EngineSet(vlm=engine, asr=None, tts=None), Settings())
        orch._latest_frame = (b'frame', 1.0, time.monotonic()-3600)
        try:
            await orch._user_turn('What did I show you?')
            assert bool(engine.prompt_frames) is expected
        finally:
            await orch.close()
    asyncio.run(run())


def test_source_switch_drops_old_still_before_new_frame_arrives():
    async def run():
        engine = FakeVlmSession()
        state = SessionState(session_id='test', config=SessionConfig(video_source='image'), replay_size=20, out_queue_size=20)
        orch = Orchestrator(state, EngineSet(vlm=engine, asr=None, tts=None), Settings())
        orch._latest_frame = (b'old-frame', 1.0, time.monotonic())
        try:
            orch._set_video_source({'kind': 'image', 'session_ts_start': 2, 'name': 'new image'})
            assert orch._latest_frame is None
            await orch._user_turn('What is in the new image?')
            assert not engine.prompt_frames
        finally:
            await orch.close()
    asyncio.run(run())


def test_retained_correction_is_verified_before_context_dedup(tmp_path):
    settings, store, writer, session = _make_session(str(tmp_path), MEMORY_DECISION_MODE='hybrid')
    try:
        session.note_user_turn('旧编号是1234')
        session.note_user_turn('编号改为5678')
        writer.drain()
        new, old = store.recent('c1', limit=2)
        session.note_rollover([new.id], text_tokens=100)
        candidates = [Candidate(old, 1, raw=0.9, late=True), Candidate(new, 0.9, raw=0.85, late=True)]
        session._candidates = lambda query: candidates
        session.retriever.diversify = lambda c, q, limit: c
        called = []
        def select(query, evidence, **kwargs):
            assert {row['id'] for row in evidence} == {old.id, new.id}
            assert '5678' in kwargs['recent_turns']
            called.append(True)
            return [new.id]
        session._pi = SimpleNamespace(reachable=lambda: True,
            decide=lambda *args: {'retrieve': True, 'query': '编号', 'reason': 'test'}, select=select)
        result = session.recall_for_turn('现在的编号是什么')
        assert not result and result.reason == 'context_present', 'already-present evidence is not missing evidence'
        assert called, 'normal hybrid path must also verify evidence'
    finally:
        session.close()
        writer.stop()
        store.close()


def test_native_context_marker_expires_with_rollover(tmp_path):
    settings, store, writer, session = _make_session(str(tmp_path), MEMORY_DECISION_MODE='vector')
    try:
        session.note_assistant_turn('ATK')
        writer.drain()
        item = store.recent('c1', limit=1)[0]
        candidate = Candidate(item, 1, raw=0.9)
        assert session._distance_gate([candidate], 100)
        session.note_context_turn('assistant', 'ATK')
        assert not session._distance_gate([candidate], 100)
        session.note_rollover([], text_tokens=20)
        assert session._distance_gate([candidate], 20)
    finally:
        session.close()
        assert not session._native_context
        writer.stop()
        store.close()
