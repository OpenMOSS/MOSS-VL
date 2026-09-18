"""Environment-driven configuration for the legacy VL adapter."""

from __future__ import annotations

import math
import os


def _positive_timeout(name: str, default: str) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


# Downstream (legacy contract) listener.
LISTEN_HOST = os.environ.get("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "18600"))
LISTEN_PATH = os.environ.get("LISTEN_PATH", "/v1/realtime")

# Upstream sglang-omni video realtime endpoint. Must be configurable per
# deployment (910B topology may place omni on another host).
OMNI_WS_URL = os.environ.get("OMNI_WS_URL", "ws://127.0.0.1:18500/v1/video/realtime")

# Adapter-level concurrency cap for active analysis rounds. The legacy
# contract assumes a single active session and retries on the busy message.
# 0 disables the adapter-level cap (pass through to omni capacity).
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "1"))

# Legacy contract: at most 8 buffered frames per round, each <= 10 MiB.
FRAME_BUFFER_CAP = int(os.environ.get("FRAME_BUFFER_CAP", "8"))
MAX_FRAME_BYTES = 10 * 1024 * 1024

# Timeouts aligned with the legacy caller's protection values.
READY_TIMEOUT_S = float(os.environ.get("READY_TIMEOUT_S", "10.0"))
ACK_TIMEOUT_S = float(os.environ.get("ACK_TIMEOUT_S", "10.0"))

# A downstream connection that has not sent `start` within this window is
# closed (code 1008) so it cannot squat on the MAX_INFLIGHT slot forever.
START_TIMEOUT_S = float(os.environ.get("START_TIMEOUT_S", "10.0"))

# Overall start-to-ready deadline, including connection and configuration.
SETUP_TIMEOUT_S = _positive_timeout("SETUP_TIMEOUT_S", "10.0")

# Deadline from receiving frame metadata to receiving its complete binary.
FRAME_RECEIVE_TIMEOUT_S = _positive_timeout("FRAME_RECEIVE_TIMEOUT_S", "10.0")

# websockets-library receive cap. Kept well above MAX_FRAME_BYTES so an
# oversize frame reaches the adapter's own size check (friendly legacy error)
# instead of being cut by the library with a bare 1009 close.
WS_MAX_SIZE = MAX_FRAME_BYTES + 1024 * 1024

# A round has no explicit "generate" signal in the legacy contract. When the
# buffered frames are drained and no new frame arrives within this quiet
# window, the adapter marks the last frame final upstream to close the round.
FINALIZE_QUIET_S = float(os.environ.get("FINALIZE_QUIET_MS", "300")) / 1000.0

# Omni keeps generating <|silence|> tokens after the visible answer until the
# token budget is exhausted (observed finish_reason="length"). The legacy
# caller itself wraps up after ~1s of output silence; the adapter mirrors
# that: once the round is final and no text delta has arrived for this long,
# it forges the end marker (or the "no visible output" error) and aborts the
# upstream session so total round latency stays well inside the 10s budget.
SILENCE_FINALIZE_S = float(os.environ.get("SILENCE_FINALIZE_MS", "1000")) / 1000.0

# Legacy busy semantics: callers match this substring and retry 3 times
# (0.5/1.0/1.5 s).
BUSY_MESSAGE = "realtime session is already active"
