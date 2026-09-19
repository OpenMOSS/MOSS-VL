import pytest
import torch
from pydantic import ValidationError

from sglang_omni.models.moss_vl_realtime.request_builders import (
    make_moss_vl_realtime_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.serve.video_realtime import VideoSessionConfigure


class Tokenizer:
    eos_token_id = 2
    vocab_size = 1000

    def __len__(self):
        return 1000

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return torch.tensor([[101, 102, 103]])


def test_native_prefill_preserves_roles_and_opens_an_idle_turn():
    tokenizer = Tokenizer()
    builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer, max_new_tokens=100
    )
    messages = [
        {"role": "system", "content": "BASE"},
        {"role": "user", "content": "Remember 42"},
        {"role": "assistant", "content": "OK"},
    ]
    payload = StagePayload(
        request_id="r",
        request=OmniRequest(
            inputs={"prefill_messages": messages}, params={"max_new_tokens": 20}
        ),
        data={},
    )
    builder(payload)
    assert tokenizer.messages[:2] == messages[:2]
    assert tokenizer.messages[2] == {"role": "assistant", "content": "<|silence|>OK"}
    assert messages[2]["content"] == "OK"
    assert tokenizer.messages[-1] == {"role": "user", "content": ""}


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "tool", "content": "x"}],
        [{"role": "user", "content": 42}],
        [{"role": "user", "content": "x"}] * 65,
        [{"role": "user", "content": "x" * 70000}] * 2,
    ],
)
def test_prefill_schema_rejects_invalid_or_unbounded_input(messages):
    with pytest.raises(ValidationError):
        VideoSessionConfigure(type="session.configure", prefill_messages=messages)
