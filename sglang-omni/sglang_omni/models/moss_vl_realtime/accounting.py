"""Opt-in logical-position accounting, independent of the wire protocol."""

from __future__ import annotations

import time
from dataclasses import dataclass

ACCOUNTING_PARAM = "realtime_accounting_v2"
ACCOUNTING_EVENT = "realtime.accounting"
FINALIZE_ACTION = "realtime_finalize_usage"


class ContextExhaustedError(RuntimeError):
    code = "context_exhausted"


def error_code(exc: BaseException) -> str:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ContextExhaustedError):
            return "context_exhausted"
        if isinstance(exc, TimeoutError):
            return "session_timeout"
        exc = exc.__cause__
    return "response_failed"


@dataclass
class RealtimeAccounting:
    session_id: str
    vision: int = 0
    text_input: int = 0
    text_output: int = 0
    step: int = 0
    output_step: int = 0
    frozen: bool = False
    retired_at: float | None = None
    failure_code: str | None = None
    failure_seq: int | None = None

    def commit(self, *, vision: int = 0, text_input: int = 0) -> None:
        if self.frozen:
            raise RuntimeError("cannot commit positions after accounting finalization")
        if vision < 0 or text_input < 0:
            raise RuntimeError("negative committed position increment")
        self.vision += vision
        self.text_input += text_input
        self.step += 1

    def sampled(self) -> None:
        if self.frozen:
            raise RuntimeError("cannot record output after accounting finalization")
        if self.step <= 0:
            raise RuntimeError("sampled output has no committed model step")
        if self.output_step != self.step:
            self.text_output += 1
            self.output_step = self.step

    def snapshot(self) -> dict:
        text = self.text_input + self.text_output
        return {
            "vision_tokens": self.vision,
            "text_input_tokens": self.text_input,
            "text_output_tokens": self.text_output,
            "text_tokens": text,
            "total_tokens": self.vision + text,
        }

    def freeze(self) -> dict:
        self.frozen = True
        if self.retired_at is None:
            self.retired_at = time.monotonic()
        return self.snapshot()
