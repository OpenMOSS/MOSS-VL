"""Wire protocol for the single session WebSocket.

One place that encodes/decodes every frame on `WS /api/session/{sid}/ws`:

- **Text frames** are JSON control/events. Envelope: `{"v": 1, "type": "<noun.verb>", ...}`.
  Client events may carry an `event_id` (echoed back in `error`); every server event
  carries a monotonic `seq` (the resumable-reconnect cursor for `?last_seq=N`).
- **Binary frames** are media. Byte 0 is the stream tag:
    0x01  client→server  mic PCM16 mono @16 kHz LE (rest of frame = samples)
    0x02  client→server  video JPEG (bytes 1..4 = optional u32 BE capture-ts ms, then JPEG)
    0x11  server→client  TTS PCM16 LE (always preceded by a `response.audio.delta` descriptor)

Documented extensions beyond the frozen §2 set (clients may ignore unknown types):
- client `playback.status {buffered_s}` — reports the PCM player's buffered-but-unplayed
  seconds so the server's back-pressure gate tracks real playback, not an estimate.
- server `session.updated {config}` — acks a `session.update` with the effective config.
- server `pong {ping_seq}` — echoes the client ping's `seq` in `ping_seq`; the envelope
  `seq` stays the server-monotonic counter like every other server event.
- server `input.text.done {text, item_id}` — echoes an accepted typed turn (the voice
  twin is `input.transcription.done`), so reconnect replay and the history journal
  carry typed turns too.
- server `response.done` additionally carries `ttft_ms`/`ttfa_ms` when measured.
- client `video.attach {media, name?, duration_s?}` — links an uploaded CAS video
  (`sha256:<hex>` from POST /api/media) to the session so its history replay can
  show the streamed file; acked as server `input.video.attached` (journaled).
- turn-bearing server events (`input.transcription.done` / `input.text.done` /
  `response.done`) may carry `media_ts` — the capture timestamp (seconds) of the
  newest client frame, i.e. the video position in file-streaming sessions. The
  history recorder keeps it per turn so a replay can seek the stored video.
- client `input.video.source {kind, session_ts_start, name?, media?, duration_s?,
  media_offset?}` — announces a (mid-session) video source change: kind is one of
  camera|screen|file|image|none; session_ts_start is the session-clock second the
  new segment starts at (the client's frame sampler re-bases file media time onto
  one monotone session clock — live_control_overhaul.md §12). The orchestrator
  records the segment (file positions derive from it), buffers a timeline note
  for the model's next turn, and acks as server `input.video.source.changed`
  (journaled → renders as a dedicated transcript bubble, live and in history).
"""
from __future__ import annotations

import itertools
import json
import struct
import threading
from typing import Any, Dict, Optional, Tuple, Union

PROTOCOL_VERSION = 1

# --------------------------- binary stream tags ---------------------------

TAG_MIC_PCM = 0x01
TAG_VIDEO_JPEG = 0x02
TAG_TTS_PCM = 0x11

_JPEG_SOI = b"\xff\xd8"  # JPEG start-of-image marker

# --------------------------- client -> server ---------------------------

CLIENT_SESSION_UPDATE = "session.update"
CLIENT_AUDIO_START = "input.audio.start"
CLIENT_AUDIO_COMMIT = "input.audio.commit"
CLIENT_RESPONSE_CANCEL = "response.cancel"
CLIENT_TEXT_INPUT = "text.input"
CLIENT_PLAYBACK_STATUS = "playback.status"
CLIENT_VIDEO_ATTACH = "video.attach"
CLIENT_VIDEO_SOURCE = "input.video.source"
CLIENT_PING = "ping"

CLIENT_EVENT_TYPES = frozenset(
    {
        CLIENT_SESSION_UPDATE,
        CLIENT_AUDIO_START,
        CLIENT_AUDIO_COMMIT,
        CLIENT_RESPONSE_CANCEL,
        CLIENT_TEXT_INPUT,
        CLIENT_PLAYBACK_STATUS,
        CLIENT_VIDEO_ATTACH,
        CLIENT_VIDEO_SOURCE,
        CLIENT_PING,
    }
)

# --------------------------- server -> client ---------------------------

SESSION_CREATED = "session.created"
SESSION_UPDATED = "session.updated"
TRANSCRIPTION_DELTA = "input.transcription.delta"
TRANSCRIPTION_DONE = "input.transcription.done"
TEXT_DONE = "input.text.done"
VIDEO_ATTACHED = "input.video.attached"
VIDEO_SOURCE_CHANGED = "input.video.source.changed"
SPEECH_STARTED = "turn.speech_started"
RESPONSE_CREATED = "response.created"
RESPONSE_TEXT_DELTA = "response.text.delta"
RESPONSE_TEXT_DONE = "response.text.done"
RESPONSE_AUDIO_DELTA = "response.audio.delta"
RESPONSE_AUDIO_DONE = "response.audio.done"
RESPONSE_DONE = "response.done"
MEMORY_RECALLED = "memory.recalled"
MEMORY_NOTICE = "memory.notice"
CONTEXT_COMMAND = "context.command"
STATUS = "status"
ERROR = "error"
PONG = "pong"

# response.done stop reasons
STOP_END_TURN = "end_turn"
STOP_INTERRUPTED = "interrupted"
STOP_CANCELLED = "cancelled"
STOP_ERROR = "error"


class ProtocolError(ValueError):
    """A frame that does not decode under this protocol."""


class Seq:
    """Thread-safe monotonic per-session event counter (starts at 1)."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._lock = threading.Lock()
        self._last = 0

    def next(self) -> int:
        with self._lock:
            self._last = next(self._counter)
            return self._last

    @property
    def last(self) -> int:
        return self._last


# --------------------------- text (JSON) frames ---------------------------


def encode_event(type_: str, seq: Optional[int] = None, **fields: Any) -> str:
    """Encode a server event as a JSON text frame."""
    payload: Dict[str, Any] = {"v": PROTOCOL_VERSION, "type": type_}
    if seq is not None:
        payload["seq"] = seq
    payload.update(fields)
    return json.dumps(payload, ensure_ascii=False)


def parse_client_text(message: str) -> Tuple[str, Dict[str, Any]]:
    """Parse a client JSON text frame into (type, payload).

    Bare "ping"/"pong" strings are tolerated (gateway keepalive tools send them).
    """
    if message in ("ping", "pong"):
        return CLIENT_PING, {}
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("client event must be a JSON object")
    type_ = payload.get("type")
    if not isinstance(type_, str) or not type_:
        raise ProtocolError("client event missing 'type'")
    return type_, payload


# --------------------------- binary frames ---------------------------


def mic_binary(pcm16: bytes) -> bytes:
    """Client-side encode of a mic PCM chunk (used by tests/tooling)."""
    return bytes((TAG_MIC_PCM,)) + pcm16


def video_binary(jpeg: bytes, timestamp_ms: Optional[int] = None) -> bytes:
    """Client-side encode of a video JPEG frame (used by tests/tooling)."""
    if timestamp_ms is None:
        return bytes((TAG_VIDEO_JPEG,)) + jpeg
    return bytes((TAG_VIDEO_JPEG,)) + struct.pack(">I", int(timestamp_ms) & 0xFFFFFFFF) + jpeg


def audio_binary(pcm: bytes) -> bytes:
    """Server-side encode of a TTS PCM chunk (tag 0x11)."""
    return bytes((TAG_TTS_PCM,)) + pcm


def parse_binary(frame: Union[bytes, bytearray, memoryview]) -> Tuple[int, Optional[float], bytes]:
    """Decode a binary media frame -> (tag, timestamp_seconds | None, payload).

    - 0x01: payload = PCM16 samples, no timestamp.
    - 0x02: payload = JPEG bytes; the optional u32 BE capture timestamp (ms) is
      detected via the JPEG SOI marker and returned in **seconds**.
    - 0x11: payload = PCM (accepted for symmetry/testing).
    """
    data = bytes(frame)
    if len(data) < 2:
        raise ProtocolError(f"binary frame too short: {len(data)} bytes")
    tag = data[0]
    rest = data[1:]

    if tag == TAG_MIC_PCM or tag == TAG_TTS_PCM:
        return tag, None, rest

    if tag == TAG_VIDEO_JPEG:
        if rest[:2] == _JPEG_SOI:
            return tag, None, rest
        if len(rest) >= 6 and rest[4:6] == _JPEG_SOI:
            (ts_ms,) = struct.unpack(">I", rest[:4])
            return tag, ts_ms / 1000.0, rest[4:]
        raise ProtocolError("0x02 frame payload is not JPEG (missing SOI marker)")

    raise ProtocolError(f"unknown binary stream tag: 0x{tag:02x}")
