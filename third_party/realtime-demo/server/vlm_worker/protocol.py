"""Gateway↔worker wire envelope (shared by worker WS handler and proxy).

Binary frames:  u32 big-endian header length | JSON header (utf-8) | payload.
Text frames are plain JSON. Every message carries `"t"` (type):

gateway→worker binary : frame {"t":"frame","ts","size"} + JPEG payload
                        prompt_frame {"t":"prompt_frame","text","ts","drop_pending"} + JPEG
gateway→worker text   : {"t":"prompt","text"} {"t":"turn_end"} {"t":"stop"} {"t":"ping"}
worker→gateway text   : {"t":"out","active","chunks","chunk_events","status"}
                        {"t":"status",...} {"t":"kv_warning",...}
                        {"t":"ended","reason":"stopped|error|kv_exhausted"} {"t":"pong"}
"""
from __future__ import annotations

import json
import struct
from typing import Any, Dict, Tuple

_LEN = struct.Struct(">I")

# frame acks never round-trip; drop/queue accounting rides the 1 Hz status frame
T_FRAME = "frame"
T_PROMPT_FRAME = "prompt_frame"
T_PROMPT = "prompt"
T_TURN_END = "turn_end"
T_STOP = "stop"
T_PING = "ping"
T_PONG = "pong"
T_OUT = "out"
T_STATUS = "status"
T_KV_WARNING = "kv_warning"
T_ENDED = "ended"


def pack_msg(header: Dict[str, Any], payload: bytes = b"") -> bytes:
    head = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _LEN.pack(len(head)) + head + payload


def unpack_msg(raw: bytes) -> Tuple[Dict[str, Any], bytes]:
    if len(raw) < _LEN.size:
        raise ValueError(f"envelope too short: {len(raw)} bytes")
    (head_len,) = _LEN.unpack_from(raw)
    end = _LEN.size + head_len
    if len(raw) < end:
        raise ValueError(f"truncated envelope: header wants {head_len} bytes")
    header = json.loads(raw[_LEN.size:end].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("envelope header must be a JSON object")
    return header, raw[end:]
