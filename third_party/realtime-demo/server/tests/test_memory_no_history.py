import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from server import protocol as p
from server.config import Settings
from server.persistence.store import IndexStore
from server.persistence.recorder import _apply_event
from server.schemas import SessionConfig
from server.session.orchestrator import EngineSet, Orchestrator
from server.session.state import SessionState
from server.tests.fakes import FakeVlmSession
from server.tests.test_memory_pi import _make_session
from server.memory.retrieval import is_history_question, refers_to_dialogue
from server.adapters.vlm.moss_vl_sglang_omni.pool import SglangOmniPool
from server.memory.inject import augment_system_prompt
from server.memory.session import RecallResult


@pytest.mark.parametrize('frames_forwarded', [0, 1])
def test_candidate_miss_does_not_block_native_context_with_or_without_video(tmp_path, frames_forwarded):
    async def run():
        settings, store, writer, memory = _make_session(str(tmp_path), MEMORY_DECISION_MODE='vector')
        memory.note_context_turn('user', '鼠标是ATK')
        state = SessionState(session_id='c1', config=SessionConfig(), replay_size=20, out_queue_size=20)
        engine = FakeVlmSession()
        orch = Orchestrator(state, EngineSet(vlm=engine, asr=None, tts=None), settings, memory=memory)
        orch.metrics['frames_forwarded'] = frames_forwarded
        orch._recall_for_turn = AsyncMock(return_value=RecallResult(block='', items=[], ids=[], reason='no_evidence'))
        orch._pending_source_notes = ['source changed']
        try:
            await orch._user_turn('我之前说的银行卡密码是什么')
            events = [json.loads(item.text) for item in state.replay]
            assert not any(e['type'] == p.MEMORY_NOTICE for e in events)
            assert engine.prompts and 'source changed' in engine.prompts[0]
            assert orch._pending_source_notes == []
            assert orch._pending_fact_user_text == '我之前说的银行卡密码是什么'
        finally:
            await orch.close()
            writer.stop()
            store.close()
    asyncio.run(run())


def test_history_commands_are_not_missing_history_questions():
    assert is_history_question('我之前说的数字是什么')
    assert not is_history_question('之前说错了，现在数字改为42')
    assert not is_history_question('之前那个先不要说话')
    assert refers_to_dialogue('我之前让你记住的数字是多少')
    assert not refers_to_dialogue('刚才画面里是什么')


def test_empty_history_is_a_system_notice_not_a_model_answer(tmp_path):
    async def run():
        settings, store, writer, memory = _make_session(str(tmp_path), MEMORY_DECISION_MODE='vector')
        state = SessionState(session_id='c1', config=SessionConfig(), replay_size=20, out_queue_size=20)
        engine = FakeVlmSession()
        orch = Orchestrator(state, EngineSet(vlm=engine, asr=None, tts=None), settings, memory=memory)
        try:
            await orch._user_turn('我之前说的数字是什么')
            events = [json.loads(item.text) for item in state.replay]
            notice = next(e for e in events if e['type'] == p.MEMORY_NOTICE)
            assert notice['code'] == 'no_history'
            assert all(e.get('source') == 'system' for e in events if e['type'] == p.RESPONSE_CREATED)
            assert any(e['type'] == p.RESPONSE_DONE and e.get('source') == 'system' for e in events)
            assert not any(e['type'] == p.RESPONSE_TEXT_DONE and e.get('text') for e in events)
            assert not engine.prompts and not engine.prompt_frames
            writer.drain(timeout=2)
            assert store.count('c1') == 1, 'the unanswered user question remains in history'
            await orch._user_turn('1+1等于几')
            assert engine.prompts, 'ordinary questions must continue using the model'
        finally:
            await orch.close()
            writer.stop()
            store.close()
    asyncio.run(run())


def test_native_context_prevents_false_empty_history_notice(tmp_path):
    async def run():
        settings, store, writer, memory = _make_session(str(tmp_path), MEMORY_DECISION_MODE='vector')
        memory.note_context_turn('user', '数字是42')
        state = SessionState(session_id='c1', config=SessionConfig(), replay_size=20, out_queue_size=20)
        engine = FakeVlmSession()
        orch = Orchestrator(state, EngineSet(vlm=engine, asr=None, tts=None), settings, memory=memory)
        try:
            await orch._user_turn('我之前说的数字是什么')
            assert not any(json.loads(item.text)['type'] == p.MEMORY_NOTICE for item in state.replay)
            assert engine.prompts
        finally:
            await orch.close()
            writer.stop()
            store.close()
    asyncio.run(run())


def test_notice_is_archived_as_system_not_assistant(tmp_path):
    index = IndexStore(Settings(data_dir=str(tmp_path), history_db_path=''))
    index.open()
    try:
        index.upsert_conversation('c', 'realtime', 1)
        _apply_event(index, 'c', {'type': p.MEMORY_NOTICE, 'seq': 1, 'message': 'No history', 'code': 'no_history'}, {})
        row = index.get_transcript('c')[0]
        assert row['role'] == 'system' and row['source'] == 'memory_notice'
    finally:
        index.close()


def test_base_system_prompt_does_not_discard_compacted_memory():
    pool = SglangOmniPool(Settings(sglang_omni_urls=''))
    payload = pool._configure_payload({'system_prompt': 'BASE', 'prefill_messages': json.dumps([
        {'role': 'system', 'content': 'BASE\nPinned: mouse brand ATK'},
        {'role': 'user', 'content': 'recent question'},
        {'role': 'assistant', 'content': 'recent answer'},
    ])})
    assert payload['system_prompt'] == 'BASE'
    assert 'mouse brand ATK' in payload['prompt']
    assert 'recent question' in payload['prompt']


def test_compacted_facts_do_not_duplicate_memory_instructions():
    pool = SglangOmniPool(Settings(sglang_omni_urls=''))
    payload = pool._configure_payload({'system_prompt': 'BASE', 'prefill_messages': json.dumps([
        {'role': 'system', 'content': augment_system_prompt('BASE', 'zh') + '\n\nPinned: ATK'},
        {'role': 'user', 'content': 'recent question'},
    ])})
    assert payload['system_prompt'] == 'BASE'
    assert 'Pinned: ATK' in payload['prompt'] and '不要猜测' not in payload['prompt']


def test_native_prefill_is_only_sent_after_capability_negotiation():
    pool = SglangOmniPool(Settings(sglang_omni_urls=''))
    params = {'system_prompt': 'BASE', 'prompt': 'Active task', 'prefill_messages': json.dumps([
        {'role': 'system', 'content': augment_system_prompt('BASE', 'zh') + '\n\nPinned: ATK'},
        {'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'},
    ])}
    legacy = pool._configure_payload(params)
    assert 'prefill_messages' not in legacy
    native = pool._configure_payload(params, native_prefill=True)
    messages = native['prefill_messages']
    assert messages[0] == {'role': 'system', 'content': 'BASE'}
    assert any(m['role'] == 'user' and m['content'] == 'Active task' for m in messages)
    assert any(m['role'] == 'user' and 'Pinned: ATK' in m['content'] for m in messages)
    assert messages[-1] == {'role': 'assistant', 'content': 'answer'}
