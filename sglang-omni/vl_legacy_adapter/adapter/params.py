"""Mapping from the legacy ``start`` message to omni ``session.configure``.

The omni configure schema is ``extra="forbid"``: any unknown field is
rejected with ``error[invalid_request]``. Only the whitelisted fields below
may be forwarded; everything else is stripped here.

Mapping (legacy -> omni):
    prompt                 -> NOT set here; attached to the final frame's
                              ``input.frame.prompt`` by session.py. Measured
                              behavior: with the question only in
                              ``configure.prompt`` the model often stays
                              silent on fast batched frames (SFT monitoring
                              behavior), while a prompt on the final frame
                              reliably triggers the answer turn.
    frame_queue_size       -> input_queue_capacity (clamped to [1, 256])
    max_new_tokens         -> max_new_tokens
    max_tokens_per_second  -> max_tokens_per_turn (tokens/s pacing knob)
    temperature            -> temperature (clamped to [0, 2])
    top_p                  -> top_p (clamped to (0, 1])
    do_sample=false        -> forces temperature=0.0

Dropped (unsupported upstream, silently stripped):
    do_sample, top_k, repetition_penalty
"""

from __future__ import annotations

import math
from typing import Any

MAX_INPUT_QUEUE_CAPACITY = 256  # mirrors omni MAX_INPUT_QUEUE_CAPACITY

DROPPED_FIELDS = ("do_sample", "top_k", "repetition_penalty")


def _finite_float(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def map_start_to_configure(start: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {"type": "session.configure"}

    do_sample = start.get("do_sample")
    temperature = start.get("temperature")
    if temperature is not None:
        temperature = _finite_float(temperature, "temperature")
    if do_sample is False:
        cfg["temperature"] = 0.0
    elif temperature is not None:
        cfg["temperature"] = min(max(temperature, 0.0), 2.0)

    top_p = start.get("top_p")
    if top_p is not None:
        # omni requires 0 < top_p <= 1
        cfg["top_p"] = min(max(_finite_float(top_p, "top_p"), 1e-6), 1.0)

    max_new_tokens = start.get("max_new_tokens")
    if max_new_tokens is not None:
        _finite_float(max_new_tokens, "max_new_tokens")
        cfg["max_new_tokens"] = max(int(max_new_tokens), 1)

    max_tokens_per_second = start.get("max_tokens_per_second")
    if max_tokens_per_second is not None:
        cfg["max_tokens_per_turn"] = max(
            _finite_float(max_tokens_per_second, "max_tokens_per_second"), 1e-6
        )

    frame_queue_size = start.get("frame_queue_size")
    if frame_queue_size is not None:
        _finite_float(frame_queue_size, "frame_queue_size")
        cfg["input_queue_capacity"] = min(
            max(int(frame_queue_size), 1), MAX_INPUT_QUEUE_CAPACITY
        )

    return cfg
