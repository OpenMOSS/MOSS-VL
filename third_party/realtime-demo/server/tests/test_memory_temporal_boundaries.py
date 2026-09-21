import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from server.memory.retrieval import Candidate
from server.memory.session import RecallResult
from server.memory.inject import estimate_tokens, verbatim_chunks
from server.schemas import SessionConfig
from server.session.orchestrator import EngineSet, Orchestrator
from server.session.state import SessionState
from server.tests.fakes import FakeVlmSession
from server.tests.test_memory_pi import _make_session


@pytest.mark.parametrize('query,expected', [
    ('我最早做出的是什么手势', '五指张开'),
    ('我第一次做了什么手势', '五指张开'),
    ('我第一个做出的手势是什么', '五指张开'),
    ('我最后做出的是什么手势', '竖起大拇指'),
    ('What was my first gesture?', '五指张开'),
])
def test_order_query_finds_event_outside_semantic_topk(tmp_path, query, expected):
    settings, store, writer, memory = _make_session(str(tmp_path), MEMORY_DECISION_MODE='hybrid')
    try:
        for timestamp, text in [(10, '你好'), (180, '右手五指张开。随后握成拳头。'),
                                (300, '右手竖起大拇指。')]:
            writer.note_utterance('c1', 'assistant', text, lang='zh', session_ts=timestamp)
        writer.note_utterance('another', 'assistant', '五指张开，其他会话', lang='zh', session_ts=1)
        writer.drain()
        newest = store.recent('c1', limit=1)[0]
        memory._candidates = lambda query: [Candidate(newest, raw=.9, relevance=1)]
        calls = []
        def select(query, candidates, **kwargs):
            calls.extend(candidates)
            return [c['id'] for c in candidates if expected in c['text']]
        memory._pi = SimpleNamespace(reachable=lambda:True,
            decide=lambda *a:{'retrieve':True,'query':'手势'}, select=select)
        result = memory.recall_for_turn(query)
        assert result and expected in result.block
        assert all('其他会话' not in c['text'] for c in calls)
    finally:
        memory.close()
        writer.stop()
        store.close()


def test_empty_candidate_selection_does_not_veto_native_context(tmp_path):
    async def run():
        settings, store, writer, memory = _make_session(str(tmp_path), MEMORY_DECISION_MODE='vector')
        memory.note_context_turn('assistant', '右手五指张开。')
        state = SessionState(session_id='c1', config=SessionConfig(), replay_size=30, out_queue_size=30)
        engine = FakeVlmSession()
        orch = Orchestrator(state, EngineSet(vlm=engine, asr=None, tts=None), settings, memory=memory)
        orch._recall_for_turn = AsyncMock(return_value=RecallResult('', [], [], reason='no_evidence'))
        try:
            await asyncio.wait_for(orch._user_turn('我最早做出的是什么手势'), 5)
            assert engine.prompts
            events = [json.loads(item.text) for item in state.replay]
            assert not any(e['type']=='memory.notice' for e in events)
        finally:
            await asyncio.wait_for(orch.close(), 5)
            writer.stop()
            store.close()
    asyncio.run(run())


def test_retelling_old_event_is_not_a_new_latest_observation(tmp_path):
    settings, store, writer, memory = _make_session(str(tmp_path), MEMORY_DECISION_MODE='hybrid')
    try:
        first = store.add_item('c1','utterance',role='assistant',text='五指张开',session_ts=180)
        last = store.add_item('c1','utterance',role='assistant',text='竖起大拇指',session_ts=300)
        store.add_item('c1','utterance',role='assistant',text='最早的手势是五指张开',session_ts=600,historical_answer=True)
        items, complete = store.timeline('c1',latest=True)
        assert [i.id for i in items]==[last,first] and complete
    finally:
        memory.close()
        writer.stop()
        store.close()


def test_scan_exhaustion_is_not_proof_of_absence(tmp_path):
    settings, store, writer, memory = _make_session(str(tmp_path), MEMORY_DECISION_MODE='hybrid', MEMORY_TEMPORAL_SCAN_ITEMS=2)
    try:
        for i in range(4): store.add_item('c1','utterance',text='无关记录',session_ts=i)
        store.add_item('c1','utterance',text='五指张开',session_ts=500)
        memory._pi = SimpleNamespace(reachable=lambda:True, decide=lambda *a:{'retrieve':True}, select=lambda *a,**kw:[])
        result = memory.recall_for_turn('我最早做出的是什么手势')
        assert not result and result.reason=='temporal_budget'
    finally:
        memory.close()
        writer.stop()
        store.close()


def test_duplicate_observation_preserves_first_and_last_time(tmp_path):
    settings, store, writer, memory = _make_session(str(tmp_path))
    try:
        for stamp, text in [(180, '五指张开'), (300, '竖起大拇指'), (500, '五指张开')]:
            writer.note_utterance('c1','assistant',text,lang='zh',session_ts=stamp)
        writer.drain()
        assert store.count('c1')==2
        first, _ = store.timeline('c1')
        last, _ = store.timeline('c1',latest=True)
        assert first[0].text=='五指张开' and first[0].session_ts==180
        assert last[0].text=='五指张开' and last[0].session_ts==500
    finally:
        memory.close()
        writer.stop()
        store.close()


def test_foreign_id_cannot_enter_injected_context(tmp_path):
    settings, store, writer, memory = _make_session(str(tmp_path))
    try:
        other = store.add_item('other','utterance',text='另一会话的内容',session_ts=1)
        memory.mark_injected([other])
        assert other not in memory._injected
    finally:
        memory.close()
        writer.stop()
        store.close()


def test_long_record_uses_verbatim_spans_without_false_full_context_dedup(tmp_path):
    settings, store, writer, memory = _make_session(str(tmp_path), MEMORY_DECISION_MODE='hybrid')
    text = '背景环境保持不变。'*30 + '男子五指张开。' + '背景灯光依然明亮。'*30 + '男子最后竖起大拇指。'
    expected = ['五指', '大拇指']
    try:
        item = store.add_item('c1','utterance',role='assistant',text=text,session_ts=180)
        memory._pi = SimpleNamespace(reachable=lambda:True,decide=lambda *a:{'retrieve':True},
            select=lambda q,c,**kw:[i['id'] for i in c if expected[0] in i['text']])
        first = memory.recall_for_turn('我最早做出的是什么手势')
        assert first.ids==[item] and '五指' in first.block
        assert estimate_tokens(first.block)<=settings.memory_inject_max_tokens
        assert first.items[0]['text']==text
        memory.mark_injected(first.ids)
        assert not memory._injected[item].full
        expected.pop(0)
        last = memory.recall_for_turn('我最后做出的是什么手势')
        assert last.ids==[item] and '大拇指' in last.block
    finally:
        memory.close()
        writer.stop()
        store.close()


def test_verbatim_spans_preserve_source_and_budget():
    text = 'Hello. This is an English sentence!\n这里是中文描述。'*30
    chunks = verbatim_chunks(text, 40)
    assert len(chunks)>1
    assert all(c in text and estimate_tokens(c)<=40 for c in chunks)


@pytest.mark.parametrize('fail', [False, True])
def test_injection_is_committed_only_after_prompt_submission(tmp_path, fail):
    async def run():
        settings, store, writer, memory = _make_session(str(tmp_path))
        state = SessionState(session_id='c1',config=SessionConfig(),replay_size=30,out_queue_size=30)
        engine = FakeVlmSession()
        if fail:
            def reject(*args): raise RuntimeError('submission failed')
            engine.put_prompt = reject
        orch = Orchestrator(state,EngineSet(vlm=engine,asr=None,tts=None),settings,memory=memory)
        memory.mark_injected = Mock()
        orch._recall_for_turn = AsyncMock(return_value=RecallResult('<recall>fact</recall>',[{'id':1,'text':'fact'}],[1]))
        try:
            await asyncio.wait_for(orch._user_turn('你好'),5)
            assert memory.mark_injected.called is not fail
            events = [json.loads(item.text) for item in state.replay]
            assert any(e['type']=='memory.recalled' for e in events) is not fail
        finally:
            await asyncio.wait_for(orch.close(),5)
            writer.stop()
            store.close()
    asyncio.run(run())
