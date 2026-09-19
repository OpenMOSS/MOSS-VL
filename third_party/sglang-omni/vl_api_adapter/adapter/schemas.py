"""v2 retains native input fields while enforcing finite, strict values."""

from pydantic import ConfigDict, Field

from sglang_omni.serve.video_realtime import (
    VideoFrameMetadata,
    VideoPromptInput,
    VideoSessionConfigure,
)

CAPABILITIES = [
    "session.usage",
    "response.done.usage",
    "session.done.usage",
    "error.seq_no",
    "input.frame.rejected",
    "response.done.per_response",
    "response.id",
    "usage.text_input_output",
]


class Configure(VideoSessionConfigure):
    model_config = ConfigDict(extra="forbid", strict=True)
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)


class Frame(VideoFrameMetadata):
    model_config = ConfigDict(extra="forbid", strict=True)
    timestamp: float = Field(ge=0, allow_inf_nan=False)


class Prompt(VideoPromptInput):
    model_config = ConfigDict(extra="forbid", strict=True)
