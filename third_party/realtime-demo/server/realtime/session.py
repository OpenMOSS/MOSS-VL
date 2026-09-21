"""Realtime session mechanics — ported from board inference.py.

Owns the per-session queues, back-pressure (drop-oldest pure frame), output
timestamping, counters and the put_*/poll_output/stop API the realtime WS router
drives. The model + daemon thread are wired in by the VLM adapter
(server/adapters/vlm_hf.py); this module is model-agnostic.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..adapters.base import OutputBatch
from ..logging_conf import get_logger

log = get_logger(__name__)

REALTIME_HIDDEN_OUTPUT_TOKENS = ("<|response|>", "<|assistant|>")
SILENCE_TOKENS = ("<|silence|>", "<|...|>")


def _float_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def clean_realtime_output_text(text: Any) -> str:
    cleaned = str(text)
    for token in REALTIME_HIDDEN_OUTPUT_TOKENS:
        cleaned = cleaned.replace(token, "")
    return cleaned


# --------------------------- queues ---------------------------


class RealTimeFrameQueue(queue.Queue):
    """Queue that reports when the model loop actually drains an event."""

    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize=maxsize)
        self.on_get = None

    def get(self, block: bool = True, timeout: Optional[float] = None):
        if timeout is None:
            item = super().get(block=block)
        else:
            item = super().get(block=block, timeout=timeout)
        if self.on_get is not None:
            self.on_get(item)
        return item

    def get_nowait(self):
        return self.get(block=False)

    def drop_oldest(self):
        return queue.Queue.get(self, block=False)

    def drop_oldest_matching(self, predicate):
        with self.mutex:
            for index, item in enumerate(self.queue):
                if predicate(item):
                    del self.queue[index]
                    self.not_full.notify()
                    return item
        raise queue.Empty


class RealTimeOutputQueue(queue.Queue):
    """Queue that preserves when realtime model text was emitted."""

    def __init__(self, created_at: float, maxsize: int = 0):
        super().__init__(maxsize=maxsize)
        self.created_at = float(created_at)

    def put(self, item, block: bool = True, timeout: Optional[float] = None):
        super().put(self._wrap_output(item), block=block, timeout=timeout)

    def _wrap_output(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, dict) and ("text" in item or "chunk" in item):
            return item
        now = time.time()
        text = clean_realtime_output_text(item)
        return {
            "type": "output",
            "text": text,
            "chunk": text,
            "emitted_at": now,
            "emitted_elapsed_seconds": max(0.0, now - self.created_at),
        }


# --------------------------- event builders ---------------------------


def decode_image(image: Any) -> Any:
    """JPEG bytes → RGB PIL image (PIL images pass through untouched).

    The gateway ships frames as encoded JPEG (cheap to move across the worker
    process boundary); the decode happens here, next to the model, on the
    caller's worker thread — never on the gateway event loop.
    """
    if isinstance(image, (bytes, bytearray, memoryview)):
        from io import BytesIO

        from PIL import Image

        decoded = Image.open(BytesIO(bytes(image))).convert("RGB")
        decoded.load()
        return decoded
    return image


def frame_event(image: Any, timestamp: float, byte_size: Optional[int] = None) -> Dict[str, Any]:
    return {"type": "frame", "image": image, "timestamp": float(timestamp), "byte_size": byte_size}


def prompt_event(prompt: str) -> Dict[str, Any]:
    return {"type": "prompt", "prompt": prompt}


def batch_event(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "batch", "events": events}


def event_frame_timestamps(event: Any) -> List[float]:
    if isinstance(event, dict):
        et = event.get("type")
        if et == "batch":
            ts: List[float] = []
            for child in event.get("events") or []:
                ts.extend(event_frame_timestamps(child))
            return ts
        if et == "frame":
            timestamp = event.get("timestamp")
            return [float(timestamp)] if timestamp is not None else []
        return []
    if isinstance(event, tuple) and len(event) >= 2:
        try:
            return [float(event[1])]
        except (TypeError, ValueError):
            return []
    return []


def event_prompt_count(event: Any) -> int:
    if isinstance(event, dict):
        et = event.get("type")
        if et == "batch":
            return sum(event_prompt_count(child) for child in event.get("events") or [])
        if et == "prompt":
            return 1
    return 0


def event_is_pure_frame(event: Any) -> bool:
    return bool(event_frame_timestamps(event)) and event_prompt_count(event) == 0


def output_event(item: Any, session: "RealtimeSession") -> Dict[str, Any]:
    now = time.time()
    if isinstance(item, dict):
        text = clean_realtime_output_text(item.get("text", item.get("chunk", "")))
        emitted_at = _float_or_none(item.get("emitted_at")) or now
        elapsed = _float_or_none(item.get("emitted_elapsed_seconds"))
        if elapsed is None:
            elapsed = max(0.0, emitted_at - session.created_at)
    else:
        text = clean_realtime_output_text(item)
        emitted_at = now
        elapsed = max(0.0, now - session.created_at)

    video_timestamp = _float_or_none(session.last_consumed_timestamp)
    return {
        "text": text,
        "chunk": text,
        "emitted_at": emitted_at,
        "emitted_elapsed_seconds": elapsed,
        "output_second": int(max(0.0, elapsed)),
        "wall_time": time.strftime("%H:%M:%S", time.localtime(emitted_at)),
        "video_timestamp": video_timestamp,
        "video_second": int(video_timestamp) if video_timestamp is not None and video_timestamp >= 0 else None,
    }


# --------------------------- session ---------------------------


@dataclass
class RealtimeSession:
    session_id: str
    gpu_id: int
    frame_queue: RealTimeFrameQueue
    prompt_queue: "queue.Queue[str]"
    output_queue: RealTimeOutputQueue
    stop_event: threading.Event
    created_at: float
    model: Any = None                       # loaded streaming model (for turn_interrupt)
    thread: Optional[threading.Thread] = None
    frames_received: int = 0
    frames_consumed: int = 0
    frames_dropped: int = 0
    prompts_received: int = 0
    prompts_consumed: int = 0
    prompts_dropped: int = 0
    bytes_received: int = 0
    outputs_emitted: int = 0
    silence_outputs: int = 0
    non_silence_outputs: int = 0
    last_frame_timestamp: Optional[float] = None
    last_frame_size: Optional[Tuple[int, int]] = None
    last_frame_at: Optional[float] = None
    last_consumed_timestamp: Optional[float] = None
    last_consumed_at: Optional[float] = None
    last_output_at: Optional[float] = None
    _stopper: Any = None                     # callback(session_id, timeout) -> dict, set by adapter

    # ----- internal helpers -----

    @staticmethod
    def _frame_size(image: Any) -> Optional[Tuple[int, int]]:
        size = getattr(image, "size", None)
        if isinstance(size, tuple) and len(size) == 2:
            return size
        return None

    def mark_consumed(self, item: Any) -> None:
        timestamps = event_frame_timestamps(item)
        prompt_count = event_prompt_count(item)
        now = time.time()
        if timestamps:
            self.frames_consumed += len(timestamps)
            self.last_consumed_timestamp = timestamps[-1]
            self.last_consumed_at = now
        if prompt_count:
            self.prompts_consumed += prompt_count

    def _put_event(self, event: Any, drop_pending: bool = False) -> Tuple[int, bool]:
        dropped_frames = 0
        dropped_any = False

        def _drop_one(preserve_prompts: bool = True) -> bool:
            nonlocal dropped_frames, dropped_any
            try:
                if preserve_prompts:
                    dropped = self.frame_queue.drop_oldest_matching(event_is_pure_frame)
                else:
                    dropped = self.frame_queue.drop_oldest()
            except queue.Empty:
                return False
            dropped_frames += len(event_frame_timestamps(dropped))
            dropped_any = True
            return True

        if drop_pending:
            while _drop_one(preserve_prompts=True):
                pass

        while True:
            try:
                self.frame_queue.put_nowait(event)
                break
            except queue.Full:
                if _drop_one(preserve_prompts=True):
                    continue
                if not _drop_one(preserve_prompts=False):
                    continue

        if dropped_frames:
            self.frames_dropped += dropped_frames
        return dropped_frames, dropped_any

    def status_payload(self) -> Dict[str, Any]:
        active = bool(self.thread and self.thread.is_alive() and not self.stop_event.is_set())
        return {
            "session_id": self.session_id,
            "gpu_id": self.gpu_id,
            "active": active,
            "created_at": self.created_at,
            "frame_queue_size": self.frame_queue.qsize(),
            "prompt_queue_size": self.prompt_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "frames_received": self.frames_received,
            "frames_consumed": self.frames_consumed,
            "frames_dropped": self.frames_dropped,
            "prompts_received": self.prompts_received,
            "prompts_consumed": self.prompts_consumed,
            "bytes_received": self.bytes_received,
            "outputs_emitted": self.outputs_emitted,
            "silence_outputs": self.silence_outputs,
            "non_silence_outputs": self.non_silence_outputs,
            "last_frame_timestamp": self.last_frame_timestamp,
            "last_consumed_timestamp": self.last_consumed_timestamp,
            "last_output_at": self.last_output_at,
            # exact text-KV token count from the token-counter patch (None until
            # the first generation step, or when the patch is absent) — the
            # rollover trigger reads this via the worker's 1 Hz status push
            "text_tokens": getattr(self.model, "_rt_text_tokens", None),
        }

    # ----- public API (matches VlmRealtimeSession Protocol) -----

    def put_frame(self, image: Any, timestamp: Optional[float] = None, byte_size: Optional[int] = None) -> Dict[str, Any]:
        if self.stop_event.is_set():
            raise RuntimeError(f"Realtime session is no longer active: {self.session_id}")
        if byte_size is None and isinstance(image, (bytes, bytearray, memoryview)):
            byte_size = len(image)
        try:
            image = decode_image(image)
        except Exception as exc:  # noqa: BLE001 — a corrupt frame must not kill the session
            self.frames_received += 1
            self.frames_dropped += 1
            log.warning("session %s: undecodable frame dropped (%s)", self.session_id, exc)
            payload = self.status_payload()
            payload.update({"timestamp": timestamp, "dropped_oldest": False,
                            "dropped_frames": 1, "bad_frame": True})
            return payload
        ts = float(timestamp) if timestamp is not None else time.time() - self.created_at
        # The model's real_time_generate drains `new_video_frames` expecting
        # `(PIL.Image, float ts)` TUPLES (modeling_moss_vl: `item[1]`,
        # `[img for img, _ in frames_to_process]`); dicts crash it with
        # `KeyError: 1`. The drop policy's event_* helpers handle tuples too.
        dropped_frames, dropped_any = self._put_event((image, ts))
        self.frames_received += 1
        if byte_size is not None:
            try:
                self.bytes_received += int(byte_size)
            except (TypeError, ValueError):
                pass
        self.last_frame_timestamp = ts
        self.last_frame_size = self._frame_size(image)
        self.last_frame_at = time.time()
        payload = self.status_payload()
        payload.update({"timestamp": ts, "dropped_oldest": dropped_any, "dropped_frames": dropped_frames})
        return payload

    def put_prompt(self, prompt: str) -> Dict[str, Any]:
        if self.stop_event.is_set():
            raise RuntimeError(f"Realtime session is no longer active: {self.session_id}")
        # Prompts go on `new_prompts` (the model reads it as a queue of strings),
        # NOT the frame queue — a dict/prompt on new_video_frames crashes the sort.
        self.prompt_queue.put(prompt)
        self.prompts_received += 1
        payload = self.status_payload()
        payload["prompt_queue_size"] = self.prompt_queue.qsize()
        return payload

    def put_prompt_frame(self, prompt: str, image: Any, timestamp: Optional[float] = None,
                         byte_size: Optional[int] = None, drop_pending: bool = True) -> Dict[str, Any]:
        if self.stop_event.is_set():
            raise RuntimeError(f"Realtime session is no longer active: {self.session_id}")
        if byte_size is None and isinstance(image, (bytes, bytearray, memoryview)):
            byte_size = len(image)
        try:
            image = decode_image(image)
        except Exception as exc:  # noqa: BLE001 — keep the user's turn, drop only the frame
            self.frames_dropped += 1
            log.warning("session %s: undecodable turn frame dropped — prompt sent without it (%s)",
                        self.session_id, exc)
            return self.put_prompt(prompt)
        ts = float(timestamp) if timestamp is not None else time.time() - self.created_at
        # Frame before prompt: the aligned frame goes on new_video_frames (tuple)
        # and the prompt on new_prompts; the model drains both in one cycle and
        # splices the prompt turn then the frame (inserting <|silence|>), matching
        # the old single-batch semantics without the crashing dict format.
        dropped_frames, dropped_any = self._put_event((image, ts), drop_pending=drop_pending)
        self.prompt_queue.put(prompt)
        self.prompts_received += 1
        self.frames_received += 1
        if byte_size is not None:
            try:
                self.bytes_received += int(byte_size)
            except (TypeError, ValueError):
                pass
        self.last_frame_timestamp = ts
        self.last_frame_size = self._frame_size(image)
        self.last_frame_at = time.time()
        payload = self.status_payload()
        payload.update({"timestamp": ts, "dropped_oldest": dropped_any, "dropped_frames": dropped_frames})
        return payload

    def request_turn_end(self) -> Dict[str, Any]:
        if self.stop_event.is_set():
            raise RuntimeError(f"Realtime session is no longer active: {self.session_id}")
        if self.model is None:
            raise RuntimeError(f"Realtime model is no longer loaded for session: {self.session_id}")
        try:
            self.model.turn_interrupt_requested = True
            log.info("Realtime session %s soft turn interrupt requested", self.session_id)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to request realtime turn interrupt: {exc}") from exc
        return self.status_payload()

    def poll_output(self, timeout_seconds: float = 0.0, max_items: int = 128) -> OutputBatch:
        chunks: List[str] = []
        chunk_events: List[Dict[str, Any]] = []

        def append_item(item: Any) -> None:
            event = output_event(item, self)
            if event["text"] == "":
                return
            chunks.append(event["text"])
            chunk_events.append(event)

        if timeout_seconds and timeout_seconds > 0:
            try:
                append_item(self.output_queue.get(timeout=timeout_seconds))
            except queue.Empty:
                pass
        while len(chunks) < max_items:
            try:
                append_item(self.output_queue.get_nowait())
            except queue.Empty:
                break

        if chunk_events:
            now = time.time()
            for event in chunk_events:
                text = str(event.get("text") or "").strip()
                self.outputs_emitted += 1
                if text in SILENCE_TOKENS:
                    self.silence_outputs += 1
                else:
                    self.non_silence_outputs += 1
            self.last_output_at = now

        status = self.status_payload()
        return OutputBatch(active=status["active"], chunks=chunks, chunk_events=chunk_events, status=status)

    def status(self) -> Dict[str, Any]:
        return self.status_payload()

    def stop(self, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        if self._stopper is not None:
            return self._stopper(self.session_id, timeout_seconds)
        self.stop_event.set()
        return self.status_payload()
