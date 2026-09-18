"""NPU deployment defaults; importing this module does not load a runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping


def recommended_deploy_params(tp_size: int, env: Mapping[str, str]) -> dict:
    if type(tp_size) is not int or tp_size < 1:
        raise ValueError("tp_size must be a positive integer")
    context = int(env.get("CONTEXT_LENGTH", "8192" if tp_size == 2 else "32768"))
    memory = float(env.get("MEM_FRACTION", "0.70"))
    capacity = int(env.get("MAX_RUNNING_REQUESTS", str(tp_size)))
    if context <= 0 or capacity <= 0:
        raise ValueError("context length and session capacity must be positive")
    if not math.isfinite(memory) or not 0 < memory <= 1:
        raise ValueError("MEM_FRACTION must be finite and in (0, 1]")
    return dict(
        context_length=context,
        mem_fraction_static=memory,
        max_running_requests=capacity,
    )
