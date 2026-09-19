from types import SimpleNamespace

from sglang_omni.models.moss_vl_realtime.request_builders import (
    MossVLRealtimeRequestData,
    make_moss_vl_realtime_stream_output_builder,
)
from sglang_omni.models.moss_vl_realtime.runtime_state import MossVLRealtimeRuntimeState
from sglang_omni.proto import OmniRequest, StagePayload


class Tokenizer:
    eos_token_id = 2

    def decode(self, ids, **kwargs):
        return "".join("a" for token in ids if token == 10)


def data(include_usage):
    state = MossVLRealtimeRuntimeState(
        request_id="r",
        session_id="s",
        encoder_length=20,
        appended_encoder_length=1000,
        decoder_length=100,
    )
    return MossVLRealtimeRequestData(
        runtime_state=state,
        req=SimpleNamespace(),
        stage_payload=StagePayload(
            request_id="r",
            data={},
            request=OmniRequest(
                inputs={},
                params={"stream": True, "include_usage": include_usage},
            ),
        ),
    )


def test_usage_counts_historical_positions_not_just_live_kv():
    build = make_moss_vl_realtime_stream_output_builder(
        tokenizer=Tokenizer(),
        silence_token_ids=(77,),
        context_length=2048,
    )
    request = data(True)
    for token in (10, 77, 2):
        messages = build("r", request, SimpleNamespace(data=token))
        usage = messages[0].data
        assert usage["event"] == "session.usage"
        assert usage["decoder_tokens"] == 101
        assert usage["encoder_tokens"] == 1000
        assert usage["encoder_kv_tokens"] == 20
        assert usage["token_space_used"] == 1101
        assert usage["context_remaining"] == 947
        assert usage["context_limit"] == 2048


def test_usage_is_opt_in_and_does_not_change_legacy_event_stream():
    build = make_moss_vl_realtime_stream_output_builder(
        tokenizer=Tokenizer(),
        silence_token_ids=(77,),
        context_length=2048,
    )
    messages = build("r", data(False), SimpleNamespace(data=10))
    assert [message.data.get("event") for message in messages] == [
        "session.ready",
        None,
    ]
