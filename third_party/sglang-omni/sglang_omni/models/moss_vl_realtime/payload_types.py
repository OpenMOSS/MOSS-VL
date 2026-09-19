"""Input contract shared by MOSS-VL realtime preprocessing and serving."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
IMAGE_PLACEHOLDER = "<|image|>"
TIME_START_TOKEN = "<|time_start|>"
TIME_END_TOKEN = "<|time_end|>"
SILENCE_TOKEN = "<|silence|>"


@dataclass(frozen=True, slots=True)
class FramePromptEvent:
    """One ordered frame, prompt, or combined update for a live request.

    ``frame_ref`` identifies data owned by the media relay. The control plane
    must not serialize image bytes into this event.
    """

    request_id: str
    session_id: str
    seq_no: int
    timestamp: float
    frame_ref: str | None
    prompt: str | None = None
    final: bool = False
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in ("request_id", "session_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value:
                raise ValueError(f"{name} must not be empty")
        if isinstance(self.seq_no, bool) or not isinstance(self.seq_no, int):
            raise TypeError("seq_no must be an integer")
        if self.seq_no < 0:
            raise ValueError("seq_no must be non-negative")
        if isinstance(self.timestamp, bool) or not isinstance(
            self.timestamp, (int, float)
        ):
            raise TypeError("timestamp must be a number")
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if self.frame_ref is not None:
            if not isinstance(self.frame_ref, str):
                raise TypeError("frame_ref must be a string")
            if not self.frame_ref:
                raise ValueError("frame_ref must not be empty")
        if self.prompt is not None:
            if not isinstance(self.prompt, str):
                raise TypeError("prompt must be a string")
            if not self.prompt:
                raise ValueError("prompt must not be empty")
        if self.frame_ref is None and self.prompt is None:
            raise ValueError("event must contain a frame or prompt")
        if not isinstance(self.final, bool):
            raise TypeError("final must be a boolean")
        if self.fingerprint is not None and not isinstance(self.fingerprint, str):
            raise TypeError("fingerprint must be a string")

    def to_dict(self) -> dict[str, Any]:
        """Return the msgpack-safe representation used on the control plane."""
        data: dict[str, Any] = {
            "_type": "FramePromptEvent",
            "request_id": self.request_id,
            "session_id": self.session_id,
            "seq_no": self.seq_no,
            "timestamp": float(self.timestamp),
            "final": self.final,
        }
        if self.frame_ref is not None:
            data["frame_ref"] = self.frame_ref
        if self.prompt is not None:
            data["prompt"] = self.prompt
        if self.fingerprint is not None:
            data["fingerprint"] = self.fingerprint
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FramePromptEvent:
        """Parse and validate one control-plane frame event."""
        if not isinstance(data, dict):
            raise TypeError("frame event must be a dictionary")
        event_type = data.get("_type", "FramePromptEvent")
        if event_type != "FramePromptEvent":
            raise ValueError(f"unexpected frame event type: {event_type!r}")
        required = (
            "request_id",
            "session_id",
            "seq_no",
            "timestamp",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"frame event is missing fields: {', '.join(missing)}")
        return cls(
            request_id=data["request_id"],
            session_id=data["session_id"],
            seq_no=data["seq_no"],
            timestamp=data["timestamp"],
            frame_ref=data.get("frame_ref"),
            prompt=data.get("prompt"),
            final=data.get("final", False),
            fingerprint=data.get("fingerprint"),
        )

    @property
    def has_frame(self) -> bool:
        return self.frame_ref is not None


def build_realtime_frame_text(timestamp: float) -> str:
    """Build the exact timestamped image placeholder used during training."""
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise TypeError("timestamp must be a number")
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("timestamp must be finite and non-negative")
    return (
        f"{VISION_START_TOKEN}{TIME_START_TOKEN}{float(timestamp):.1f} seconds"
        f"{TIME_END_TOKEN}{IMAGE_PLACEHOLDER}{VISION_END_TOKEN}"
    )


def build_realtime_append_text(
    *,
    prompts: Iterable[str] = (),
    frame_timestamps: Iterable[float] = (),
) -> str:
    """Build a drain-cycle segment with the trained assistant opener consumed.

    Mirrors the reference ``_realtime_update_context`` drain semantics: prompts
    come first in arrival order (each closes the current assistant turn and
    opens the next user/assistant turn). Every prompt is followed by
    ``<|silence|>``; prompt-only turns must not sample that mandatory opener
    as an idle decision. Frame wrappers come last, in the
    caller-provided (timestamp-sorted) order.
    """
    timestamps = tuple(frame_timestamps)
    text = ""
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt:
            raise ValueError("prompt must not be empty")
        text += (
            "<|im_end|>\n<|im_start|>user\n"
            f"{prompt}"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        # This is the trained assistant opener, not a generated idle decision.
        # Letting a prompt-only turn generate it would immediately park the turn.
        text += SILENCE_TOKEN
    return text + "".join(build_realtime_frame_text(ts) for ts in timestamps)
