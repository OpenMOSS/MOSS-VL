from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_vl_realtime.batch_adapter import RUNTIME_STATE_ATTR
from sglang_omni.models.moss_vl_realtime.config import MossVLRealtimePipelineConfig
from sglang_omni.models.moss_vl_realtime.request_builders import (
    make_moss_vl_realtime_scheduler_adapters,
    make_moss_vl_realtime_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _Tokenizer:
    eos_token_id = 2
    vocab_size = 900

    def __len__(self) -> int:
        return 1000

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "Track the cup"}
        assert kwargs["add_generation_prompt"] is True
        return {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join({10: "A", 11: "B", 12: "C"}.get(i, "") for i in token_ids)


class _TurnBoundaryTokenizer(_Tokenizer):
    def decode(self, token_ids, **kwargs):
        del kwargs
        values = list(token_ids)
        if values == [10]:
            return "old"
        if values == [11]:
            return "new"
        if values == [10, 11]:
            return "boundary-changed"
        return ""


class _SilenceTokenizer(_Tokenizer):
    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join({10: "A", 11: "B"}.get(i, "") for i in token_ids)


class _ByteFragmentTokenizer(_Tokenizer):
    def decode(self, token_ids, **kwargs):
        del kwargs
        pieces = {10: b"\xe9\xbe", 11: b"\x98", 13: b"\xef\xbf\xbd", 14: b"new"}
        return b"".join(pieces.get(i, b"") for i in token_ids).decode(
            "utf-8", errors="replace"
        )


def _byte_stream():
    tokenizer = _ByteFragmentTokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer, max_new_tokens=100
    )
    data = request_builder(_payload(stream=True))
    builder = make_moss_vl_realtime_stream_output_builder(
        tokenizer=tokenizer, silence_token_ids=(12,)
    )
    return data, builder


def _stream_text(messages):
    return "".join(message.data.get("text", "") for message in messages)


def test_stream_builder_buffers_utf8_byte_fragments() -> None:
    data, builder = _byte_stream()
    first = builder("req-1", data, SimpleNamespace(data=10))
    assert _stream_text(first) == ""
    assert data.turn_emitted_text == ""
    second = builder("req-1", data, SimpleNamespace(data=11))
    assert _stream_text(second) == "\u9f98"
    assert data.turn_emitted_text == "\u9f98"
    assert builder.flush("req-1", data) == []


@pytest.mark.parametrize("token_id", [10, 13])
@pytest.mark.parametrize("boundary", ["flush", "eos", "silence"])
def test_stream_builder_flushes_replacement_once(token_id, boundary) -> None:
    data, builder = _byte_stream()
    messages = builder("req-1", data, SimpleNamespace(data=token_id))
    if boundary == "flush":
        messages += builder.flush("req-1", data)
    else:
        messages += builder(
            "req-1", data, SimpleNamespace(data=2 if boundary == "eos" else 12)
        )
    assert _stream_text(messages) == "\ufffd"
    if boundary == "silence":
        text_index = next(i for i, m in enumerate(messages) if "text" in m.data)
        silence_index = next(
            i
            for i, m in enumerate(messages)
            if m.data.get("event") == "response.turn.silence"
        )
        assert text_index < silence_index
    assert builder.flush("req-1", data) == []


def test_stream_builder_drops_incomplete_bytes_at_turn_interrupt() -> None:
    data, builder = _byte_stream()
    first = builder("req-1", data, SimpleNamespace(data=10))
    data.req._moss_vl_realtime_processed_events = [
        {
            "seq_no": 4,
            "timestamp": 2.0,
            "prompt": "Interrupt",
            "final": False,
            "interrupted_turn_id": 0,
            "turn_id": 1,
        }
    ]
    data.runtime_state.turn_id = 1
    second = builder("req-1", data, SimpleNamespace(data=14))
    assert _stream_text(first + second) == "new"
    assert data.turn_emitted_text == "new"


def _payload(*, stream: bool, max_tokens_per_turn: float | None = None) -> StagePayload:
    params = {"stream": stream, "max_new_tokens": 20}
    if max_tokens_per_turn is not None:
        params["max_tokens_per_turn"] = max_tokens_per_turn
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs={"initial_prompt": "Track the cup", "session_id": "session-1"},
            params=params,
        ),
        data={},
    )


def test_request_builder_creates_pure_text_persistent_request() -> None:
    tokenizer = _Tokenizer()
    request_builder, result_adapter = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )

    data = request_builder(_payload(stream=False))

    assert data.initial_input_ids == [101, 102, 103]
    assert data.req.multimodal_inputs is None
    assert data.req.sampling_params.max_new_tokens == 20
    assert data.req.sampling_params.ignore_eos is True
    assert data.req.vocab_size == 1000
    state = getattr(data.req, RUNTIME_STATE_ATTR)
    assert state is data.runtime_state
    assert state.session_id == "session-1"
    assert state.max_tokens_per_turn == 86400.0

    data.generated_token_ids[:] = [10, 11]
    data.finish_reason = "stop"
    result = result_adapter(data)
    assert result.data["text"] == "AB"
    assert result.data["session_id"] == "session-1"
    assert result.data["completion_tokens"] == 2


def test_request_builder_sets_token_rate_on_runtime_state() -> None:
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=_Tokenizer(),
        max_new_tokens=100,
    )

    data = request_builder(_payload(stream=False, max_tokens_per_turn=12.5))

    assert data.runtime_state.max_tokens_per_turn == 12.5


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_request_builder_rejects_invalid_token_rate(value: float) -> None:
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=_Tokenizer(),
        max_new_tokens=100,
    )

    with pytest.raises(ValueError, match="max_tokens_per_turn"):
        request_builder(_payload(stream=False, max_tokens_per_turn=value))


def test_stream_builder_accumulates_sampled_tokens_not_injected_context() -> None:
    tokenizer = _Tokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )
    data = request_builder(_payload(stream=True))
    stream_builder = make_moss_vl_realtime_stream_output_builder(
        tokenizer=tokenizer,
        silence_token_ids=(12,),
    )
    data.req._moss_vl_realtime_processed_events = [
        {
            "seq_no": 3,
            "timestamp": 1.5,
            "frame_ref": "shm://frame",
            "final": False,
        },
    ]

    first = stream_builder("req-1", data, SimpleNamespace(data=10))
    second = stream_builder("req-1", data, SimpleNamespace(data=11))
    eos = stream_builder("req-1", data, SimpleNamespace(data=2))

    assert data.generated_token_ids == [10, 11]
    assert first[0].data == {
        "event": "input.frame.processed",
        "seq_no": 3,
        "timestamp": 1.5,
        "final": False,
        "modality": "control",
    }
    text_messages = [
        message for message in first + second if message.data.get("modality") == "text"
    ]
    assert [message.data["text"] for message in text_messages] == ["A", "B"]
    ready_messages = [
        message
        for message in first + second
        if message.data.get("event") == "session.ready"
    ]
    assert [message.data["session_id"] for message in ready_messages] == ["session-1"]
    assert eos == []


def test_stream_builder_emits_prompt_processed_control_event() -> None:
    tokenizer = _Tokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )
    data = request_builder(_payload(stream=True))
    data.req._moss_vl_realtime_processed_events = [
        {
            "seq_no": 4,
            "timestamp": 2.0,
            "prompt": "How many?",
            "final": True,
            "interrupted_turn_id": 0,
            "turn_id": 1,
        },
    ]
    stream_builder = make_moss_vl_realtime_stream_output_builder(
        tokenizer=tokenizer,
        silence_token_ids=(12,),
    )

    messages = stream_builder("req-1", data, SimpleNamespace(data=None))

    assert messages[0].data == {
        "event": "response.turn.interrupted",
        "turn_id": 0,
        "next_turn_id": 1,
        "seq_no": 4,
        "modality": "control",
    }
    assert messages[1].data["event"] == "input.prompt.processed"
    assert messages[1].data["seq_no"] == 4
    assert messages[1].data["final"] is True
    assert messages[1].data["interrupted_turn_id"] == 0
    assert messages[1].data["turn_id"] == 1
    assert data.turn_generated_token_ids == []
    assert data.turn_emitted_text == ""


def test_stream_builder_decodes_text_within_each_turn() -> None:
    tokenizer = _TurnBoundaryTokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )
    data = request_builder(_payload(stream=True))
    stream_builder = make_moss_vl_realtime_stream_output_builder(
        tokenizer=tokenizer,
        silence_token_ids=(12,),
    )

    first = stream_builder("req-1", data, SimpleNamespace(data=10))
    data.req._moss_vl_realtime_processed_events = [
        {
            "seq_no": 4,
            "timestamp": 2.0,
            "prompt": "Interrupt",
            "final": False,
            "interrupted_turn_id": 0,
            "turn_id": 1,
        },
    ]
    data.runtime_state.turn_id = 1
    second = stream_builder("req-1", data, SimpleNamespace(data=11))

    assert [message.data["text"] for message in first if "text" in message.data] == [
        "old"
    ]
    assert [message.data["text"] for message in second if "text" in message.data] == [
        "new"
    ]
    assert data.generated_token_ids == [10, 11]
    assert data.emitted_text == "boundary-changed"
    assert data.turn_generated_token_ids == [11]
    assert data.turn_emitted_text == "new"


def test_stream_builder_emits_every_silence_without_deduplication() -> None:
    tokenizer = _SilenceTokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )
    data = request_builder(_payload(stream=True))
    data.req._moss_vl_realtime_processed_events = [
        {
            "seq_no": 5,
            "timestamp": 3.0,
            "frame_ref": "shm://frame",
            "final": False,
        },
    ]
    stream_builder = make_moss_vl_realtime_stream_output_builder(
        tokenizer=tokenizer,
        silence_token_ids=(12,),
    )

    first = stream_builder("req-1", data, SimpleNamespace(data=12))
    second = stream_builder("req-1", data, SimpleNamespace(data=12))

    silence_events = [
        message.data
        for message in first + second
        if message.data.get("event") == "response.turn.silence"
    ]
    assert silence_events == [
        {
            "event": "response.turn.silence",
            "turn_id": 0,
            "seq_no": 5,
            "timestamp": 3.0,
            "silence_seq": 0,
            "modality": "control",
        },
        {
            "event": "response.turn.silence",
            "turn_id": 0,
            "seq_no": 5,
            "timestamp": 3.0,
            "silence_seq": 1,
            "modality": "control",
        },
    ]


def test_stream_builder_silence_tracks_input_and_prompt_turn() -> None:
    tokenizer = _SilenceTokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )
    data = request_builder(_payload(stream=True))
    stream_builder = make_moss_vl_realtime_stream_output_builder(
        tokenizer=tokenizer,
        silence_token_ids=(12,),
    )

    data.req._moss_vl_realtime_processed_events = [
        {
            "seq_no": 0,
            "timestamp": 0.0,
            "frame_ref": "shm://frame-0",
            "final": False,
        },
    ]
    frame_silence = stream_builder("req-1", data, SimpleNamespace(data=12))
    data.req._moss_vl_realtime_processed_events = [
        {
            "seq_no": 1,
            "timestamp": 1.0,
            "frame_ref": "shm://frame-1",
            "prompt": "What changed?",
            "final": False,
            "interrupted_turn_id": 0,
            "turn_id": 1,
        },
    ]
    data.runtime_state.turn_id = 1
    prompt_silence = stream_builder("req-1", data, SimpleNamespace(data=12))

    events = [
        message.data
        for message in frame_silence + prompt_silence
        if message.data.get("event") == "response.turn.silence"
    ]
    assert [
        (event["seq_no"], event["timestamp"], event["turn_id"]) for event in events
    ] == [
        (0, 0.0, 0),
        (1, 1.0, 1),
    ]


def test_stream_builder_waits_for_complete_multitoken_silence_marker() -> None:
    tokenizer = _SilenceTokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )
    data = request_builder(_payload(stream=True))
    stream_builder = make_moss_vl_realtime_stream_output_builder(
        tokenizer=tokenizer,
        silence_token_ids=(70, 77),
    )

    first = stream_builder("req-1", data, SimpleNamespace(data=70))
    second = stream_builder("req-1", data, SimpleNamespace(data=77))

    assert not [
        message
        for message in first
        if message.data.get("event") == "response.turn.silence"
    ]
    silence = [
        message.data
        for message in second
        if message.data.get("event") == "response.turn.silence"
    ]
    assert len(silence) == 1
    assert silence[0]["silence_seq"] == 0


def test_pipeline_config_targets_realtime_stage() -> None:
    config = MossVLRealtimePipelineConfig(model_path="/models/moss-vl")
    assert config.entry_stage == "moss_vl_realtime"
    assert type(config).supports_video_realtime is True
    assert "MossVLForConditionalGeneration" not in type(config).architecture_aliases
    assert config.stages[0].factory.endswith(
        "moss_vl_realtime.stages.create_sglang_moss_vl_realtime_executor"
    )
    factory_args = config.stages[0].factory_args
    assert factory_args["context_length"] == 262144
    assert factory_args["mem_fraction_static"] == 0.40
    assert factory_args["disable_cuda_graph"] is False
    assert factory_args["page_size"] == 1
    assert factory_args["enable_async_decode"] is False


def test_request_builder_marks_benchmark_ignore_eos() -> None:
    tokenizer = _Tokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )

    payload = _payload(stream=True)
    payload.request.params["benchmark_ignore_eos"] = True
    benchmark_data = request_builder(payload)
    assert benchmark_data.req._moss_vl_realtime_keep_ignore_eos is True
    assert benchmark_data.req.sampling_params.ignore_eos is True

    default_data = request_builder(_payload(stream=True))
    assert not hasattr(default_data.req, "_moss_vl_realtime_keep_ignore_eos")


def test_request_builder_keeps_realtime_requests_out_of_radix_cache() -> None:
    # A radix-matched prefix lands in the decoder region of the realtime page
    # row while upstream release protects cache_protected_len slots at the row
    # head; skipping inserts from birth keeps the tree empty so matches always
    # miss and release only ever frees request-owned slots.
    tokenizer = _Tokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )

    warmup_payload = _payload(stream=True)
    warmup_payload.request.params["realtime_warmup"] = True

    assert request_builder(warmup_payload).req.skip_radix_cache_insert is True
    assert request_builder(_payload(stream=True)).req.skip_radix_cache_insert is True
