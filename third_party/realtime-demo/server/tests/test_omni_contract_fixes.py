"""Contract regressions for the Demo adapter and current omni protocol."""
import asyncio
import queue
from types import SimpleNamespace

import pytest

from server.adapters.vlm.moss_vl_sglang_omni import session as adapter
from server.adapters.vlm.moss_vl_sglang_omni.pool import SglangOmniPool, READY, DOWN
from server.config import Settings
from server.gateway.session import GatewaySession, GatewayRegistry
from server.gateway.pool import GatewayPool
from server.gateway.tokens import TokenIssuer
from server.session.orchestrator import Orchestrator
from server.tests.test_sglang_omni_adapter import JPEG


class Transport:
    def __init__(self):
        self.sent = []
        self.closed = False
        self.reject = False
        self.no_ack = False

    def send_json(self, message):
        self.sent.append(message)
        if self.no_ack:
            return
        if self.reject:
            self.reject = False
            self.session._handle_event({'type': 'error', 'code': 'invalid_request', 'message': 'rejected'})
            return
        seq = message['seq_no']
        if message['type'] == 'input.frame':
            self.session._handle_event({'type': 'input.frame.ready', 'seq_no': seq})
        else:
            self.session._handle_event({'type': 'input.prompt.accepted', 'seq_no': seq})
            self.session._handle_event({'type': 'input.prompt.processed', 'seq_no': seq})

    def send_bytes(self, raw):
        seq = self.sent[-1]['seq_no']
        self.session._handle_event({'type': 'input.frame.accepted', 'seq_no': seq})
        self.session._handle_event({'type': 'input.frame.processed', 'seq_no': seq})

    def close(self):
        self.closed = True


def make_session():
    transport = Transport()
    session = adapter.SglangOmniSession(transport, {'session_id': 'test'})
    transport.session = session
    session._tokens._failed = True
    return session, transport


def test_soft_interrupt_preserves_session_and_filters_already_polled_output():
    session, transport = make_session()
    session._handle_event({'type': 'response.text.delta', 'turn_id': 0, 'delta': 'old'})
    polled = session.poll_output().chunk_events
    session._next_seq_no = 1  # a previously submitted prompt is still in flight
    session.request_turn_end()
    assert session.active and transport.sent == []
    assert all(not session.output_event_is_current(event) for event in polled)
    session._handle_event({'type': 'response.turn.interrupted', 'turn_id': 0, 'next_turn_id': 1, 'seq_no': 0})
    session._handle_event({'type': 'response.text.delta', 'turn_id': 1, 'delta': 'stale'})
    assert session.poll_output().chunks == []
    session.put_prompt('new question')
    session._handle_event({'type': 'response.turn.interrupted', 'turn_id': 1, 'next_turn_id': 2, 'seq_no': 1})
    session._handle_event({'type': 'response.text.delta', 'turn_id': 2, 'delta': 'new answer'})
    assert session.poll_output().chunks == ['<|eot_id|>', '<|round_start|>', 'new answer']
    session.request_turn_end()
    session.request_turn_end()
    assert session.active and all(msg['type'] != 'session.abort' for msg in transport.sent)


def test_orchestrator_round_start_cannot_unmute_local_interrupt():
    orch = Orchestrator.__new__(Orchestrator)
    orch.engines = SimpleNamespace(vlm=make_session()[0])
    orch._drop_model_tail = True
    opened = []
    orch._open_response = lambda: opened.append(True)
    orch._handle_control_token('<|round_start|>')
    assert orch._drop_model_tail and not opened
    orch._handle_control_token('<|eot_id|>')
    orch._handle_control_token('<|round_start|>')
    assert opened == [True]


@pytest.mark.parametrize('kind', ['frame', 'prompt'])
def test_error_without_seq_rejects_current_input_and_recovers(kind):
    session, transport = make_session()
    transport.reject = True
    put = (lambda: session.put_frame(JPEG, 1.0)) if kind == 'frame' else (lambda: session.put_prompt('hello'))
    with pytest.raises(ValueError, match='rejected'):
        put()
    assert session.active and not session._waiters
    put()
    assert [message['seq_no'] for message in transport.sent] == [0, 0]
    assert not session._waiters and session._sending_input is None


def test_unknown_ack_state_ends_connection_without_reusing_sequence(monkeypatch):
    session, transport = make_session()
    transport.no_ack = True
    monkeypatch.setattr(adapter, 'INPUT_ACK_TIMEOUT_S', 0.01)
    with pytest.raises(TimeoutError):
        session.put_prompt('may have been accepted')
    assert not session.active and transport.closed and not session._waiters
    with pytest.raises(RuntimeError):
        session.put_prompt('must not retry')
    assert len(transport.sent) == 1


@pytest.mark.parametrize('timestamp', [float('nan'), float('inf'), -1.0])
def test_bad_timestamp_never_consumes_sequence(timestamp):
    session, transport = make_session()
    with pytest.raises(ValueError):
        session.put_frame(JPEG, timestamp)
    assert not transport.sent and session._next_seq_no == 0
    session.put_frame(JPEG, 0.0)
    assert session.frames_consumed == 1


def test_oversize_frame_is_rejected_locally(monkeypatch):
    session, transport = make_session()
    monkeypatch.setattr(adapter, 'MAX_FRAME_BYTES', 4)
    with pytest.raises(ValueError):
        session.put_frame(JPEG, 0.0)
    assert not transport.sent and not session._waiters


def test_error_after_acceptance_is_not_rolled_back():
    session, transport = make_session()
    waiter = session._next_input('prompt', True)
    waiter.accepted.set()
    session._handle_event({'type': 'error', 'code': 'invalid_request', 'message': 'unowned error'})
    assert not session.active and transport.closed
    assert session._next_seq_no == 1


@pytest.mark.parametrize('all_bad', [False, True])
def test_handshake_transport_failure_tries_other_replicas_once(all_bad):
    pool = SglangOmniPool(Settings(sglang_omni_urls='http://bad,http://good'))
    for replica in pool.replicas:
        replica.state = READY
    attempts = []
    def start(replica, params):
        attempts.append(replica.url)
        if all_bad or replica.url.endswith('bad'):
            raise ConnectionError('handshake failed')
        return type('Session', (), {'session_id': 'good'})()
    pool._start_on_replica = start
    if all_bad:
        with pytest.raises(ConnectionError):
            pool.start_realtime_session()
        assert pool.busy == 0
    else:
        assert pool.start_realtime_session().session_id == 'good'
        assert pool.busy == 1
    assert attempts == ['http://bad', 'http://good']
    assert pool.replicas[0].state == DOWN


def test_invalid_configuration_does_not_quarantine_or_try_other_replicas():
    pool = SglangOmniPool(Settings(sglang_omni_urls='http://one,http://two'))
    for replica in pool.replicas:
        replica.state = READY
    attempts = []
    def start(replica, params):
        attempts.append(replica.url)
        raise ValueError('invalid parameter')
    pool._start_on_replica = start
    with pytest.raises(ValueError):
        pool.start_realtime_session()
    assert attempts == ['http://one'] and pool.busy == 0
    assert all(replica.state == READY for replica in pool.replicas)


def make_gateway():
    settings = Settings(sglang_omni_urls='http://fake')
    pool = GatewayPool(settings)
    pool._probe_health = lambda url: {"ok": True}
    pool.probe_all()
    registry = GatewayRegistry(settings, pool, TokenIssuer(), usage=SimpleNamespace(record=lambda record: None))
    session = GatewaySession(registry, 'reset-test')
    registry._sessions[session.session_id] = session
    session._index = pool.acquire()
    session._owns_slot = True
    session._loop = asyncio.get_running_loop()
    session._queue = asyncio.Queue()
    session._client = SimpleNamespace(close=lambda: None)
    async def open_client(index, epoch):
        session._client = SimpleNamespace(close=lambda: None)
    session._open_client = open_client
    return session, registry, pool


@pytest.mark.parametrize('done', [False, True])
def test_reset_ignores_already_queued_old_close(done):
    async def run():
        session, registry, pool = make_gateway()
        session._phase = 'done' if done else 'streaming'
        session._on_omni_close(0, 'old close')
        await session.reset()
        await asyncio.sleep(0.05)
        assert not session.destroyed and registry.get(session.session_id) is session
        assert pool.busy == 1
        await registry.aclose()
        assert pool.busy == 0
    asyncio.run(run())


def test_reset_old_pump_uses_old_queue_and_does_not_clear_new_attach():
    async def run():
        session, registry, pool = make_gateway()
        closed = []
        class Socket:
            async def receive(self):
                await asyncio.Event().wait()
            async def send_text(self, text):
                pass
            async def close(self, code):
                closed.append(code)
        assert session.try_attach()
        old_run = asyncio.create_task(session.run(Socket()))
        await asyncio.sleep(0)
        await session.reset()
        assert session.try_attach()
        await asyncio.wait_for(old_run, 1)
        assert closed == [1012]
        assert session.attached and not session.destroyed and pool.busy == 1
        await registry.aclose()
    asyncio.run(run())


def test_backend_usage_pauses_new_frames_and_requires_prompt_rollover():
    session, transport = make_session()
    session._handle_event({'type': 'session.usage', 'decoder_tokens': 300,
                          'encoder_tokens': 13700, 'token_space_used': 14000,
                          'context_limit': 16384})
    assert session.status()['text_tokens'] == 300
    assert session.status()['context']['rollover_required']
    assert session.put_frame(JPEG, 1.0)['drop_reason'] == 'context_rollover'
    with pytest.raises(RuntimeError, match='rollover'):
        session.put_prompt('wait for a fresh context')
    assert transport.sent == [] and session.active


def test_old_backend_frame_budget_advances_without_visible_text():
    session, transport = make_session()
    for i in range(100):
        status = session.put_frame(JPEG, float(i))
        if status.get('frame_dropped'):
            break
    assert session.status()['context']['rollover_required']
    assert session.status()['context']['source'] == 'conservative_estimate'
    assert session.frames_received < 100


def test_usage_negotiation_only_sends_new_parameter_to_capable_backend():
    for capable in (False, True):
        sent = []
        client = SimpleNamespace(
            configure=lambda payload, *args, **kwargs: sent.append(payload),
            start_receiver=lambda *args: None,
        )
        session = adapter.SglangOmniSession(client, {'capabilities': ['session.usage'] if capable else []})
        session.configure({'type': 'session.configure'}, 1)
        assert ('include_usage' in sent[0]) == capable


@pytest.mark.parametrize('factory_fails', [False, True])
def test_hard_context_rollover_stops_growth_and_serializes_rebuild(factory_fails):
    from server.schemas import SessionConfig
    from server.session.orchestrator import EngineSet
    from server.session.state import SessionState
    from server.tests.fakes import FakeVlmSession
    import time

    async def run():
        old = FakeVlmSession()
        old_status = old.status
        old.status = lambda *args, **kwargs: {**old_status(), 'context': {'rollover_required': True}}
        entered, proceed = asyncio.Event(), asyncio.Event()
        async def prefix():
            assert not old.active
            entered.set()
            await proceed.wait()
            return [{'role': 'system', 'content': 'preserved memory'}], [], 10
        made = []
        async def factory(**kwargs):
            if factory_fails:
                raise ConnectionError('replacement unavailable')
            new = FakeVlmSession()
            made.append(new)
            return new
        memory = SimpleNamespace(note_rollover=lambda *args, **kwargs: None,
                                 note_assistant_turn=lambda *args, **kwargs: None,
                                 close=lambda: None)
        rollover = SimpleNamespace(build_prefix=prefix,
                                   maybe_prefetch_compact=lambda *args: None,
                                   should_rollover=lambda *args, **kwargs: False)
        state = SessionState(session_id='hard-budget', config=SessionConfig(), replay_size=20, out_queue_size=100)
        orch = Orchestrator(state, EngineSet(vlm=old, asr=None, tts=None), Settings(tts_enabled=False),
                            memory=memory, rollover=rollover, reseat_factory=factory)
        orch._last_rollover_at = time.monotonic()  # Hard limits bypass text cooldown.
        orch._route_model_text('<|round_start|>unfinished answer')
        try:
            orch._maybe_rollover(idle=False)
            task = orch._reseat_task
            assert task is not None
            orch._maybe_rollover(idle=False)
            assert orch._reseat_task is task
            await asyncio.wait_for(entered.wait(), 1)
            await orch.push_frame(JPEG, 1.0)
            assert old.frames == []
            proceed.set()
            await asyncio.wait_for(task, 2)
            if factory_fails:
                assert orch._vlm_dead and not old.active and not made
            else:
                assert len(made) == 1 and orch.engines.vlm is made[0]
                assert not orch._vlm_dead and made[0].frames
                assert orch.metrics['rollovers'] == 1
                assert not orch._drop_model_tail
        finally:
            proceed.set()
            await orch.close()
    asyncio.run(run())


def test_gateway_delete_waits_for_reset_then_releases_once():
    async def run():
        session, registry, pool = make_gateway()
        entered, proceed = asyncio.Event(), asyncio.Event()
        closed = []
        async def opening(index, epoch):
            entered.set()
            await proceed.wait()
            session._client = SimpleNamespace(close=lambda: closed.append(epoch))
        session._open_client = opening
        reset = asyncio.create_task(session.reset())
        await entered.wait()
        delete = asyncio.create_task(session.adestroy('delete'))
        await asyncio.sleep(0)
        assert not delete.done()
        proceed.set()
        await reset
        await delete
        await session.adestroy('delete again')
        assert closed == [1] and pool.busy == 0 and registry.get(session.session_id) is None
    asyncio.run(run())


def test_pooled_stop_releases_a_lease_only_once():
    from server.adapters.vlm.moss_vl_sglang_omni.pool import _PooledSession
    released = []
    stopped = []
    inner = SimpleNamespace(session_id='r', stop=lambda timeout: stopped.append(timeout) or {}, status=lambda: {})
    session = _PooledSession(SimpleNamespace(_release=lambda *args, **kwargs: released.append(args)), 0, inner)
    session.stop(1)
    session.stop(1)
    assert stopped == [1] and len(released) == 1


def test_cancelled_reseat_reclaims_late_handshake_result():
    async def run():
        orch = Orchestrator.__new__(Orchestrator)
        orch.state = SimpleNamespace(config=SimpleNamespace(initial_prompt='', system_prompt=None))
        orch.settings = Settings()
        entered, proceed = asyncio.Event(), asyncio.Event()
        stopped = []
        async def factory(**kwargs):
            entered.set()
            await proceed.wait()
            return SimpleNamespace(stop=lambda timeout: stopped.append(timeout))
        orch._reseat_factory = factory
        task = asyncio.create_task(orch._call_reseat_factory('[]'))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped == [5.0]
    asyncio.run(run())


def test_orchestrator_rejects_bad_timestamp_without_poisoning_latest_frame():
    from server.schemas import SessionConfig
    from server.session.orchestrator import EngineSet
    from server.session.state import SessionState
    from server.tests.fakes import FakeVlmSession
    async def run():
        engine = FakeVlmSession()
        state = SessionState(session_id='invalid-frame', config=SessionConfig(), replay_size=10, out_queue_size=20)
        orch = Orchestrator(state, EngineSet(vlm=engine, asr=None, tts=None), Settings(tts_enabled=False))
        try:
            await orch.push_frame(JPEG, float('nan'))
            assert orch._latest_frame is None and not orch._vlm_dead and engine.active
            await orch.push_frame(JPEG, 1.0)
            assert len(engine.frames) == 1 and not orch._vlm_dead
        finally:
            await orch.close()
    asyncio.run(run())
