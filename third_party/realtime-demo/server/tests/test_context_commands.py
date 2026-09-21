"""Manual context commands never reach the model as user prompts."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from server import protocol as p
from server.memory.rollover import RolloverManager
from server.memory.session import MemorySession
from server.memory.store import MemoryStore
from server.memory.writer import MemoryWriter
from server.schemas import SessionConfig
from server.session.orchestrator import EngineSet, Orchestrator
from server.session.state import SessionState
from server.tests.fakes import FakeVlmSession
from server.tests.test_rollover import _settings, _jpeg


async def wait_command(orch):
    for _ in range(500):
        if orch._context_command is None:
            return
        await asyncio.sleep(.01)
    raise AssertionError("command did not finish")


def events(state):
    result = []
    while not state.out_queue.empty():
        result.append(json.loads(state.out_queue.get_nowait().text))
    return result


@pytest.mark.parametrize('command', ['/compact', '/clear'])
def test_commands(tmp_path, command):
    async def check():
        settings = _settings(str(tmp_path), MEMORY_SUMMARY_PROVIDER='pi')
        store = MemoryStore(settings)
        writer = MemoryWriter(settings, store)
        writer.start()
        memory = MemorySession('commands', settings, store, writer)
        ro = RolloverManager(settings, store, 'commands', lifetime=memory.lifetime,
                             base_system_prompt='SYSTEM', journal_extra=memory.uncommitted_turns)
        ro.pi = SimpleNamespace(compact=lambda *args: {'summary': 'Remember 011202', 'pins': []})
        old = FakeVlmSession()
        made = []
        async def factory(**kwargs):
            made.append((kwargs, FakeVlmSession()))
            return made[-1][1]
        state = SessionState(session_id='commands', config=SessionConfig(system_prompt='SYSTEM',
            initial_prompt='INITIAL'), replay_size=100, out_queue_size=300)
        orch = Orchestrator(state, EngineSet(vlm=old, asr=None, tts=None), settings,
                            memory=memory, rollover=ro, reseat_factory=factory)
        orch.start()
        try:
            memory.note_user_turn('Remember 011202')
            await asyncio.to_thread(writer.drain)
            await orch.push_frame(_jpeg(), 25.0)
            await orch.handle_event(p.CLIENT_TEXT_INPUT, {'text': command})
            await orch.handle_event(p.CLIENT_TEXT_INPUT, {'text': 'must not run'})
            await wait_command(orch)
            assert len(made) == 1 and not old.active
            assert not old.prompts and not made[0][1].prompts
            ev = events(state)
            assert [e['status'] for e in ev if e['type'] == 'context.command'] == ['running', 'completed']
            assert any(e.get('code') == 'context_busy' for e in ev)
            assert not any(e['type'] == p.TEXT_DONE for e in ev)
            prefix = json.loads(made[0][0]['prefill_messages'])
            if command == '/compact':
                assert 'Remember 011202' in str(prefix)
                assert orch.memory is memory and store.count('commands') > 0
                assert made[0][1].frames
            else:
                assert prefix == [{'role': 'system', 'content': 'SYSTEM'}]
                assert made[0][0]['prompt'] == ''
                assert orch.memory is not memory and memory.lifetime.closing
                assert orch.memory._t0 == memory._t0
                assert orch._rollover.lifetime is orch.memory.lifetime
                assert store.count('commands') == 0
                assert not made[0][1].frames
                memory.note_user_turn('late stale write')
                await asyncio.to_thread(writer.drain)
                assert store.count('commands') == 0
                orch.memory.note_user_turn('fresh')
                await asyncio.to_thread(writer.drain)
                assert store.count('commands') == 1
        finally:
            await orch.close()
            writer.stop()
            store.close()
    asyncio.run(check())


@pytest.mark.parametrize('failure', ['summary', 'factory'])
def test_compact_failure_retains_old_context(tmp_path, failure):
    async def check():
        settings = _settings(str(tmp_path), MEMORY_SUMMARY_PROVIDER='pi')
        store = MemoryStore(settings)
        writer = MemoryWriter(settings, store)
        writer.start()
        memory = MemorySession('failure', settings, store, writer)
        ro = RolloverManager(settings, store, 'failure', lifetime=memory.lifetime)
        ro.pi = SimpleNamespace(compact=lambda *args: None if failure == 'summary'
                                 else {'summary': 'fact', 'pins': []})
        async def factory(**kwargs):
            raise RuntimeError('factory failure')
        old = FakeVlmSession()
        state = SessionState(session_id='failure', config=SessionConfig(), replay_size=100, out_queue_size=300)
        orch = Orchestrator(state, EngineSet(vlm=old, asr=None, tts=None), settings,
                            memory=memory, rollover=ro, reseat_factory=factory)
        orch.start()
        try:
            memory.note_user_turn('fact')
            await asyncio.to_thread(writer.drain)
            await orch.handle_event(p.CLIENT_TEXT_INPUT, {'text': '/compact'})
            await wait_command(orch)
            assert old.active and orch.engines.vlm is old
            assert store.count('failure') == 1
            assert not orch._vlm_drain_task.done()
            assert [e['status'] for e in events(state) if e['type'] == 'context.command'] == ['running', 'failed']
        finally:
            await orch.close()
            writer.stop()
            store.close()
    asyncio.run(check())


def test_clear_without_memory(tmp_path):
    async def check():
        settings = _settings(str(tmp_path))
        old = FakeVlmSession()
        async def factory(**kwargs): return FakeVlmSession()
        state = SessionState(session_id='no-memory', config=SessionConfig(), replay_size=100, out_queue_size=300)
        orch = Orchestrator(state, EngineSet(vlm=old, asr=None, tts=None), settings, reseat_factory=factory)
        orch.start()
        try:
            await orch.handle_event(p.CLIENT_TEXT_INPUT, {'text': '/compact'})
            assert any(e.get('code') == 'context_unavailable' for e in events(state))
            await orch.handle_event(p.CLIENT_TEXT_INPUT, {'text': '/clear'})
            await wait_command(orch)
            assert not old.active and orch.engines.vlm.active
            assert any(e.get('status') == 'completed' for e in events(state))
        finally:
            await orch.close()
    asyncio.run(check())


def test_clear_waits_for_old_writes_and_preserves_other_sessions(tmp_path):
    async def check():
        settings = _settings(str(tmp_path))
        store = MemoryStore(settings)
        writer = MemoryWriter(settings, store)
        writer.start()
        memory = MemorySession('old', settings, store, writer)
        entered, release = threading.Event(), threading.Event()
        def old_job():
            with memory.lifetime.operation() as admitted:
                assert admitted
                entered.set()
                release.wait(5)
                store.add_item('old', 'utterance', text='late result', role='assistant')
        job = asyncio.create_task(asyncio.to_thread(old_job))
        await asyncio.to_thread(entered.wait, 5)
        store.add_item('other', 'utterance', text='untouched', role='user')
        clearing = asyncio.create_task(asyncio.to_thread(memory.cleared_copy))
        await asyncio.sleep(.05)
        assert not clearing.done()
        release.set()
        await job
        fresh = await clearing
        try:
            assert store.count('old') == 0 and store.count('other') == 1
            assert not fresh.lifetime.closing
        finally:
            fresh.close()
            writer.stop()
            store.close()
    asyncio.run(check())


def test_shutdown_during_command_reclaims_late_engine(tmp_path):
    async def check():
        settings = _settings(str(tmp_path))
        entered, release = asyncio.Event(), asyncio.Event()
        new = FakeVlmSession()
        async def factory(**kwargs):
            entered.set()
            await release.wait()
            return new
        state = SessionState(session_id='closing', config=SessionConfig(), replay_size=100, out_queue_size=300)
        orch = Orchestrator(state, EngineSet(vlm=FakeVlmSession(), asr=None, tts=None), settings,
                            reseat_factory=factory)
        orch.start()
        await orch.handle_event(p.CLIENT_TEXT_INPUT, {'text': '/clear'})
        await asyncio.wait_for(entered.wait(), 3)
        closing = asyncio.create_task(orch.close())
        await asyncio.sleep(.05)
        release.set()
        await asyncio.wait_for(closing, 3)
        assert not new.active
        assert not any(e.get('status') == 'completed' for e in events(state))
        assert not orch._tasks
    asyncio.run(check())
