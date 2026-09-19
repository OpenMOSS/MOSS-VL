"""v2 protocol/accounting regressions, no model or GPU required."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from sglang_omni.models.moss_vl_realtime.accounting import RealtimeAccounting
from sglang_omni.serve.video_realtime import (
    VideoRealtimeSession,
    VideoRealtimeSessionManager,
)
from vl_api_adapter.adapter.schemas import Configure, Frame
from vl_api_adapter.adapter.session import V2Session
from vl_api_adapter.adapter.usage import UsageLedger, empty_usage


def counts(vision=0, inputs=0, outputs=0):
    return dict(
        vision_tokens=vision,
        text_input_tokens=inputs,
        text_output_tokens=outputs,
        text_tokens=inputs + outputs,
        total_tokens=vision + inputs + outputs,
    )


class Socket:
    def __init__(self):
        self.sent = []
        self.incoming = asyncio.Queue()
        self.application_state = self.client_state = WebSocketState.CONNECTED
        self.close_code = None

    async def send_json(self, payload):
        self.sent.append(payload)

    async def receive(self):
        return await self.incoming.get()

    async def close(self, code=1000):
        self.close_code = code
        self.application_state = WebSocketState.DISCONNECTED
        self.incoming.put_nowait({"type": "websocket.disconnect"})


class Store:
    def __init__(self):
        self.cleaned = []

    def cleanup(self, rid):
        self.cleaned.append(rid)


class Client:
    def __init__(self):
        self.final = counts()
        self.watermark = 0
        self.failure_code = None
        self.seq_no = None
        self.aborts = []
        self.fail_admin = False
        self.updates = []

    async def abort(self, rid):
        self.aborts.append(rid)

    async def update_request(self, rid, data):
        self.updates.append(data)

    async def admin(self, action, payload, **kwargs):
        if self.fail_admin:
            if isinstance(self.fail_admin, Exception):
                raise self.fail_admin
            raise RuntimeError("worker unavailable")
        return {
            "success": True,
            "results": [
                {
                    "data": {
                        **payload,
                        "usage": self.final,
                        "watermark": self.watermark,
                        "failure_code": self.failure_code,
                        "seq_no": self.seq_no,
                        "final": True,
                    }
                }
            ],
        }


def make_session(**kwargs):
    socket, client, store = Socket(), Client(), Store()
    session = V2Session(
        socket,
        client=client,
        model_name="moss-vl-realtime",
        frame_store=store,
        context_limit=131072,
        accounting_stage="moss_vl_realtime",
        **kwargs,
    )
    return session, socket, client, store


def record(session, value, watermark):
    session.record_accounting(
        dict(usage=value, watermark=watermark, context_limit=131072)
    )


def test_accounting_counts_positions_not_kv_reuse():
    ledger = RealtimeAccounting("session")
    ledger.commit(text_input=78)
    ledger.sampled()
    ledger.sampled()  # The same resolved step cannot double count.
    ledger.commit(vision=145, text_input=8)
    ledger.sampled()
    assert ledger.snapshot() == counts(145, 86, 2)
    ledger.commit()  # Re-feeding a sampled token is not another input.
    ledger.sampled()
    assert ledger.freeze() == counts(145, 86, 3)
    with pytest.raises(RuntimeError):
        ledger.commit()


def test_usage_deltas_and_monotonic_invariants():
    ledger = UsageLedger()
    ledger.record(counts(145, 78, 10), 1)
    first = ledger.settle()
    ledger.record(counts(290, 98, 30), 2)
    second = ledger.settle()
    assert (
        first["total_tokens"] + second["total_tokens"]
        == second["cumulative"]["total_tokens"]
    )
    with pytest.raises(ValueError, match="regressed"):
        ledger.record(counts(145, 98, 30), 3)
    with pytest.raises(ValueError):
        ledger.record({**counts(), "total_tokens": 10}, 3)


@pytest.mark.asyncio
async def test_same_turn_multiple_responses_and_silence_only():
    session, socket, _, _ = make_session()
    silence = dict(
        type="response.turn.silence", turn_id=1, seq_no=0, timestamp=0, silence_seq=0
    )
    record(session, counts(145, 78, 1), 1)
    await session.send(silence)
    assert not [m for m in socket.sent if m["type"] == "response.done"]
    for index in (2, 3):
        record(session, counts(145, 78, index), index)
        await session.send(
            dict(type="response.text.delta", turn_id=1, delta=f"answer {index}")
        )
        await session.send({**silence, "silence_seq": index})
        await session.send({**silence, "silence_seq": index + 10})
    done = [m for m in socket.sent if m["type"] == "response.done"]
    assert len(done) == 2 and all(m["turn_id"] == 1 for m in done)
    assert done[0]["response_id"] != done[1]["response_id"]
    assert [m["response_seq"] for m in done] == [1, 2]
    assert all(
        m["finish_reason"] == "stop" and m["boundary"] == "silence" for m in done
    )
    assert sum(m["usage"]["total_tokens"] for m in done) == 226


@pytest.mark.asyncio
async def test_interruption_carries_usage_to_next_response():
    session, socket, _, _ = make_session()
    record(session, counts(145, 78, 10), 10)
    await session.send(dict(type="response.text.delta", turn_id=1, delta="unfinished"))
    await session.send(
        dict(type="response.turn.interrupted", turn_id=1, next_turn_id=2, seq_no=1)
    )
    assert not [m for m in socket.sent if m["type"] == "response.done"]
    record(session, counts(290, 90, 15), 15)
    await session.send(dict(type="response.text.delta", turn_id=2, delta="new answer"))
    await session.send(dict(type="response.turn.silence", turn_id=2))
    done = [m for m in socket.sent if m["type"] == "response.done"]
    assert len(done) == 1 and done[0]["usage"]["total_tokens"] == 395


@pytest.mark.asyncio
@pytest.mark.parametrize("wants", [False, True])
async def test_telemetry_preference_does_not_disable_accounting(wants):
    session, socket, _, _ = make_session()
    session.wants_usage = wants
    for step in range(1, 5):
        record(session, counts(145, 78, step), step)
        await session.send(dict(type="session.usage", token_space_used=223 + step))
    assert len([m for m in socket.sent if m["type"] == "session.usage"]) == int(wants)
    assert session.ledger.latest["text_output_tokens"] == 4


@pytest.mark.asyncio
async def test_finalization_uses_backend_total_and_emits_once():
    session, socket, client, store = make_session()
    session._request_started = True
    record(session, counts(145, 78, 1), 1)
    await session.send(dict(type="response.text.delta", turn_id=1, delta="answer"))
    client.final, client.watermark = counts(290, 90, 20), 20
    session.request_finished = True
    await session.send(dict(type="response.done", finish_reason="stop", turn_id=1))
    await asyncio.gather(session.teardown(), session.teardown())
    assert [m["type"] for m in socket.sent][-2:] == ["response.done", "session.done"]
    assert socket.sent[-1]["usage"] == client.final
    assert len([m for m in socket.sent if m["type"] == "session.done"]) == 1
    before = list(socket.sent)
    await session.send(dict(type="response.text.delta", turn_id=1, delta="late"))
    assert socket.sent == before and store.cleaned


@pytest.mark.asyncio
async def test_response_task_and_teardown_share_finalizer_without_deadlock():
    session, socket, _, store = make_session()
    session.request_finished = True
    session.response_task = asyncio.create_task(session.send({"type": "session.done"}))
    await asyncio.wait_for(asyncio.gather(session.response_task, session.teardown()), 1)
    assert session._finish_task.done() and not session._finish_waiters
    assert len([m for m in socket.sent if m["type"] == "session.done"]) == 1
    assert store.cleaned == [session.request_id]


@pytest.mark.asyncio
async def test_binary_timeout_does_not_wait_on_its_own_timer():
    session, socket, _, store = make_session(binary_timeout_s=0.01)
    session.configured = session.ready = True
    await session.prepare_frame(
        Frame(type="input.frame", seq_no=0, timestamp=0.0, mime_type="image/jpeg")
    )
    timer = session._binary_timer
    await asyncio.wait_for(timer, 1)
    assert session._finish_task.done() and store.cleaned == [session.request_id]
    assert (
        socket.sent[-1]["type"] == "session.done"
        and socket.sent[-1]["reason"] == "error"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,reason",
    [
        ("context_exhausted", "context_exhausted"),
        ("session_timeout", "error"),
        ("response_failed", "error"),
    ],
)
async def test_backend_error_classification_survives_teardown(code, reason):
    session, socket, client, _ = make_session()
    session._request_started = True
    client.failure_code, client.seq_no = code, 7
    client.final, client.watermark = counts(145, 78, 3), 3
    await session.send(
        dict(type="error", code="response_failed", message="engine failure")
    )
    await session.teardown()
    assert socket.sent[-2]["code"] == code and socket.sent[-2]["seq_no"] == 7
    assert (
        socket.sent[-1]["reason"] == reason and socket.sent[-1]["usage"] == client.final
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [RuntimeError("worker unavailable"), TimeoutError("finalization timed out")],
)
async def test_missing_snapshot_is_not_faked_as_zero_success(failure):
    session, socket, client, store = make_session()
    session._request_started = True
    client.fail_admin = failure
    await session.teardown()
    assert not [m for m in socket.sent if m["type"] == "session.done"]
    assert socket.close_code == 1011 and store.cleaned


@pytest.mark.asyncio
async def test_configuration_timeout_has_error_reason():
    session, socket, _, _ = make_session(configure_timeout_s=0.01)
    await asyncio.wait_for(session.run(), 1)
    assert socket.sent[-2]["code"] == "session_timeout"
    assert socket.sent[-1]["reason"] == "error"
    assert socket.sent[-1]["usage"] == empty_usage()


@pytest.mark.asyncio
async def test_corrupt_binary_keeps_seq_and_releases_input():
    session, socket, _, _ = make_session()
    session.configured = session.ready = True
    await session.prepare_frame(
        Frame(type="input.frame", seq_no=0, timestamp=0.0, mime_type="image/jpeg")
    )
    session._cancel_binary_timer()
    queue = asyncio.Queue()
    queue.put_nowait(("bytes", b"bad-jpeg"))
    worker = asyncio.create_task(session._process_inputs(queue))
    try:
        # PIL may import image plugins on the first malformed-image decode.
        # This is a lifecycle check, not a one-second decode benchmark.
        await asyncio.wait_for(queue.join(), 10)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
    rejected = [m for m in socket.sent if m["type"] == "input.frame.rejected"]
    assert rejected[0]["seq_no"] == 0
    assert session.pending_frame is None and not session.outstanding_seq_nos
    assert session.next_seq_no == 0


def test_shared_capacity_and_native_default_factory():
    manager = VideoRealtimeSessionManager(
        client=Client(), model_name="model", max_sessions=1
    )
    native = manager.open(Socket())
    assert type(native) is VideoRealtimeSession
    with pytest.raises(RuntimeError):
        manager.open(Socket(), session_factory=V2Session)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -1.0, True])
def test_v2_rejects_invalid_timestamps(timestamp):
    with pytest.raises(ValidationError):
        Frame(type="input.frame", seq_no=0, timestamp=timestamp, mime_type="image/jpeg")


def test_v2_keeps_native_config_fields():
    config = Configure(
        type="session.configure",
        prompt="user",
        system_prompt="system",
        max_new_tokens=128,
        include_usage=False,
    )
    assert config.prompt == "user" and config.system_prompt == "system"
    with pytest.raises(ValidationError):
        Configure(type="session.configure", unknown=True)


@pytest.mark.asyncio
async def test_v2_forwards_history_without_changing_telemetry_preference():
    session, _, _, _ = make_session()
    captured = []

    async def capture(request):
        captured.append(request)

    session.stream_response = capture
    history = [
        {"role": "user", "content": "Remember 42"},
        {"role": "assistant", "content": "OK"},
    ]
    try:
        await session.configure(
            Configure(
                type="session.configure", prefill_messages=history, include_usage=False
            )
        )
        await session.response_task
        assert captured[0].prompt["prefill_messages"] == history
        assert captured[0].extra_params["include_usage"] is True
        assert session.wants_usage is False
    finally:
        await session.teardown()


def test_history_prefill_accounting_counts_history_as_input(monkeypatch):
    import torch

    from sglang_omni.models.moss_vl_realtime import model_runner as module
    from sglang_omni.models.moss_vl_realtime.accounting import ACCOUNTING_PARAM
    from sglang_omni.models.moss_vl_realtime.request_builders import (
        make_moss_vl_realtime_scheduler_adapters,
    )
    from sglang_omni.proto import OmniRequest, StagePayload

    class Tokenizer:
        eos_token_id = 2
        vocab_size = 1000

        def __len__(self):
            return self.vocab_size

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return torch.tensor([[101, 102, 103]])

    tokenizer = Tokenizer()
    builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer, max_new_tokens=128
    )
    history = [
        {"role": "user", "content": "Remember 42"},
        {"role": "assistant", "content": "OK"},
    ]
    data = builder(
        StagePayload(
            request_id="history",
            request=OmniRequest(
                inputs={"prefill_messages": history}, params={ACCOUNTING_PARAM: True}
            ),
            data={},
        )
    )
    state = data.runtime_state
    data.req.req_pool_idx = 0  # Assigned by admission before the prefill callback.
    assert history[1]["content"] == "OK"
    assert tokenizer.messages[2]["content"] == "<|silence|>OK"
    assert state.accounting.snapshot() == counts()

    def commit(batch):
        state.decoder_length = len(data.initial_input_ids)

    monkeypatch.setattr(module, "is_moss_vl_realtime_batch", lambda _: True)
    monkeypatch.setattr(module, "commit_moss_vl_realtime_batch", commit)
    runner = object.__new__(module.MossVLRealtimeModelRunner)
    batch = SimpleNamespace(reqs=[data.req], seq_lens_cpu=torch.tensor([3]))
    runner.post_prefill(None, None, batch, [])
    assert state.accounting.snapshot() == counts(inputs=3)
    state.accounting.sampled()
    assert state.accounting.snapshot() == counts(inputs=3, outputs=1)


@pytest.mark.asyncio
async def test_native_done_event_remains_unchanged():
    socket = Socket()
    session = VideoRealtimeSession(
        socket, client=Client(), model_name="model", frame_store=Store()
    )
    event = dict(type="response.done", finish_reason="stop", turn_id=1)
    await session.send(event)
    assert socket.sent == [event]


@pytest.mark.asyncio
async def test_private_accounting_is_requested_and_not_forwarded():
    from sglang_omni.client.types import GenerateChunk
    from sglang_omni.models.moss_vl_realtime.accounting import (
        ACCOUNTING_EVENT,
        ACCOUNTING_PARAM,
    )

    session, _, client, _ = make_session()
    request = SimpleNamespace(extra_params={})

    async def generate(req, **kwargs):
        assert req.extra_params[ACCOUNTING_PARAM] is True
        yield GenerateChunk(
            request_id="r",
            control_event=ACCOUNTING_EVENT,
            control_data=dict(
                usage=counts(0, 78, 1), watermark=1, context_limit=131072
            ),
        )
        yield GenerateChunk(request_id="r", control_event="session.ready")

    client.generate = generate
    chunks = [chunk async for chunk in session.client.generate(request)]
    assert [chunk.control_event for chunk in chunks] == ["session.ready"]
    assert session.ledger.latest == counts(0, 78, 1)


def test_prefill_does_not_rebill_pending_generated_token(monkeypatch):
    from sglang_omni.models.moss_vl_realtime import model_runner as module
    from sglang_omni.models.moss_vl_realtime.batch_adapter import RUNTIME_STATE_ATTR
    from sglang_omni.models.moss_vl_realtime.runtime_state import (
        MossVLRealtimeRuntimeState,
    )

    ledger = RealtimeAccounting("s")
    ledger.commit(vision=5, text_input=10)
    ledger.sampled()
    state = MossVLRealtimeRuntimeState(
        "r",
        "s",
        req_pool_index=0,
        encoder_length=5,
        decoder_length=10,
        pending_token_id=7,
        accounting=ledger,
    )
    req = SimpleNamespace(rid="r")
    setattr(req, RUNTIME_STATE_ATTR, state)
    batch = SimpleNamespace(reqs=[req])

    def commit(batch):
        state.encoder_length = 8
        state.decoder_length = 15  # One old sampled token plus four fresh inputs.

    monkeypatch.setattr(module, "is_moss_vl_realtime_batch", lambda _: True)
    monkeypatch.setattr(module, "commit_moss_vl_realtime_batch", commit)
    runner = object.__new__(module.MossVLRealtimeModelRunner)
    runner.post_prefill(None, None, batch, [])
    assert ledger.snapshot() == counts(8, 14, 1)
    ledger.sampled()
    assert ledger.snapshot() == counts(8, 14, 2)


def test_finalization_resolves_target_before_snapshot():
    from sglang_omni.models.moss_vl_realtime.accounting import FINALIZE_ACTION
    from sglang_omni.models.moss_vl_realtime.scheduler import MossVLRealtimeScheduler

    ledger = RealtimeAccounting("s")
    ledger.commit(text_input=78)
    ledger.sampled()
    calls = []
    scheduler = SimpleNamespace(
        _accounting_records={"r": ledger},
        _async_pending=(SimpleNamespace(reqs=[SimpleNamespace(rid="r")]),),
        _find_request_data=lambda rid: object(),
    )

    def resolve():
        calls.append("resolve")
        ledger.commit()
        ledger.sampled()
        scheduler._async_pending = None

    def abort(rid, **kwargs):
        calls.append("abort")

    scheduler._resolve_pending_async = resolve
    scheduler.abort = abort
    result = MossVLRealtimeScheduler._run_admin_action(
        scheduler, FINALIZE_ACTION, dict(request_id="r", session_id="s")
    )
    assert calls == ["resolve", "abort"]
    assert result["data"]["usage"] == counts(0, 78, 2)
    assert ledger.frozen


def test_finished_lookahead_does_not_change_frozen_accounting():
    from sglang_omni.models.moss_vl_realtime.batch_adapter import RUNTIME_STATE_ATTR
    from sglang_omni.models.moss_vl_realtime.model_runner import (
        MossVLRealtimeModelRunner,
    )
    from sglang_omni.models.moss_vl_realtime.runtime_state import (
        MossVLRealtimeRuntimeState,
    )

    ledger = RealtimeAccounting("s")
    ledger.commit(text_input=78)
    ledger.sampled()
    expected = ledger.freeze()
    state = MossVLRealtimeRuntimeState("r", "s", decoder_length=78, accounting=ledger)
    req = SimpleNamespace()
    setattr(req, RUNTIME_STATE_ATTR, state)
    runner = object.__new__(MossVLRealtimeModelRunner)
    runner.post_decode(None, None, SimpleNamespace(reqs=[req]), [])
    assert ledger.snapshot() == expected
