"""`SglangOmniSession` — VlmRealtimeSession over one sglang-omni realtime WS.

Two design loads carry the whole adapter:

1. **Event → control-token mapping**. The
   sglang-omni structured event stream is translated back into the exact
   control-token text stream the demo orchestrator already parses
   (server/session/orchestrator.py:47-58):

     sglang-omni event            emitted chunk
     -------------------------    ------------------------------
     first text.delta of a turn   "<|round_start|>" then the delta
     response.text.delta          the delta verbatim
     response.turn.silence        "<|silence|>"   (model end-of-turn / idle)
     response.turn.interrupted    "<|eot_id|>"    (barge-in ack)
     response.done / session.done active=False    (session over, once ever)
     recv thread exits (WS drop)  active=False    (orchestrator vlm_dead path)
     error (fatal)                "[ERROR] ..." chunk + active=False

   Deltas are filtered by turn_id: after response.turn.interrupted advances
   the turn, stragglers from the old turn are dropped.

2. **Credit backpressure** (ported from board's `_await_input_credit`).
   `_waiters` tracks in-flight inputs (added when metadata is sent, popped on
   `input.*.processed` — which is also when the server frees a queue slot).
   Pure frames wait at most `input_drop_wait_seconds` for a slot, then drop
   (protocol-safe: no metadata was sent yet); prompt-carrying inputs NEVER
   drop and bypass the gate.

seq_no rules: one dense sequence shared by frames and prompts, starting at 0.
On `error[invalid_request]` the rejected seq is rolled back (the server never
consumed it) and the put call raises — the session itself stays alive.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ....logging_conf import get_logger
from ...base import OutputBatch
from .client import SglangOmniClient

log = get_logger(__name__)

INPUT_ACK_TIMEOUT_S = 30.0  # waiting for frame.ready / *.accepted must not hang forever
MAX_FRAME_BYTES = 32 * 1024 * 1024


class InputRejected(ValueError):
    """An input was explicitly rejected before the server accepted its sequence."""


class ContextRolloverRequired(RuntimeError):
    """The current connection needs a fresh context before more inputs."""


def _encode_image(image: Any) -> bytes:
    """bytes pass through; PIL images (tests, legacy callers) are encoded."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    from io import BytesIO

    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _mime_type(raw: bytes) -> str:
    """sglang-omni requires an explicit mime_type enum on input.frame."""
    if raw.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unsupported realtime frame encoding; expected JPEG, PNG or WebP")


@dataclass
class _InputWaiter:
    seq_no: int
    kind: str                      # "frame" | "prompt"
    prompt: bool = False
    ready: threading.Event = field(default_factory=threading.Event)
    accepted: threading.Event = field(default_factory=threading.Event)
    processed: threading.Event = field(default_factory=threading.Event)
    error: Optional[str] = None


class _TextTokenMirror:
    """Local text-token estimate for status()["text_tokens"] (rollover feed).

    Fallback for older backends without negotiated session.usage: configure
    prompts + input prompts + output deltas. This is only an estimate, even
    with a tokenizer: per-delta tokenization is not the generated history.
    Lazy + failure-proof: a broken tokenizer degrades to the heuristic once,
    never raises into the session path.
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._tokenizer: Any = None
        self._failed = False
        self._lock = threading.Lock()
        self.total = 0

    def add(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self.total += self._count(text)

    def _count(self, text: str) -> int:
        tok = self._load()
        if tok is not None:
            try:
                return len(tok.encode(text))
            except Exception:  # noqa: BLE001
                pass
        return max(1, len(text) // 2)

    def _load(self) -> Any:
        if self._tokenizer is not None or self._failed:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("text-token mirror: tokenizer unavailable (%s) — "
                        "falling back to len//2 estimates", exc)
            self._failed = True
        return self._tokenizer


class SglangOmniSession:
    """VlmRealtimeSession facade over one sglang-omni realtime connection."""

    turn_interrupt_is_local = True

    def __init__(self, client: SglangOmniClient, created: Dict[str, Any], *,
                 input_queue_capacity: int = 4,
                 input_drop_wait_seconds: float = 0.5,
                 fallback_context_length: int = 131072,
                 fallback_frame_tokens: int = 2048,
                 context_reserve_tokens: int = 4096,
                 model_path: str = ""):
        self._client = client
        self.session_id = str(created.get("session_id") or "")
        self.gpu_id = -1  # remote server; no gateway-side GPU index
        self.created_at = time.time()
        self._active = True
        self._ended_reason: Optional[str] = None
        self._error: Optional[str] = None

        self._state_lock = threading.RLock()
        # signaled when the server frees an input slot (input.*.processed)
        self._input_credit = threading.Condition(self._state_lock)
        self._input_capacity = max(1, int(input_queue_capacity))
        self._input_drop_wait = max(0.0, float(input_drop_wait_seconds))
        self._waiters: Dict[int, _InputWaiter] = {}
        self._sending_input: Optional[_InputWaiter] = None
        self._next_seq_no = 0
        self._input_lock = threading.Lock()   # serializes the two-phase frame send
        self._last_timestamp = 0.0
        self._current_turn_id = int(created.get("turn_id") or 0)
        self._round_open_turn: Optional[int] = None  # turn_id whose <|round_start|> went out
        self._output_epoch = 0
        self._muted = False
        self._resume_after_seq = 0
        self._usage_supported = "session.usage" in created.get("capabilities", ())
        self._usage: Optional[Dict[str, int]] = None
        self._fallback_context_length = max(1, int(fallback_context_length))
        self._fallback_frame_tokens = max(1, int(fallback_frame_tokens))
        self._context_reserve_tokens = max(1, int(context_reserve_tokens))
        self._largest_extend = 512
        self._context_rollover = False
        # Upper-bound generation rate used ONLY by the context-space estimator
        # (frames + prompts + elapsed×rate ≤ limit). Deliberately decoupled from
        # the actual pacing cap: when the request caps generation at N tok/s the
        # true bound IS N; uncapped (86400) we assume the realistic ceiling of
        # free-running decode (~64 tok/s on one H200) so the KV rollover warning
        # still fires early enough. The backend-reported usage path above makes
        # this moot whenever usage snapshots stream.
        self._generation_rate = 4.0

        self._outputs: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._tokens = _TextTokenMirror(model_path)

        # counters (status surface — names match the HF worker's where it has one)
        self.frames_received = 0
        self.frames_consumed = 0
        self.frames_dropped = 0
        self.prompts_received = 0
        self.prompts_consumed = 0
        self.outputs_emitted = 0
        self.bytes_received = 0

    # ------------------------------------------------------------ lifecycle

    def configure(self, payload: Dict[str, Any], timeout_s: float) -> None:
        """Complete the handshake, then hand the socket to the recv thread."""
        payload = dict(payload)
        if self._usage_supported:
            payload["include_usage"] = True
        rate = float(payload.get("max_tokens_per_turn") or 0.0)
        # see __init__: 86400 = "uncapped" sentinel — then 64 tok/s estimates
        # free-running decode; an explicit small cap bounds generation exactly
        self._generation_rate = min(512.0, rate) if rate and rate < 86400.0 else 64.0
        self._client.configure(payload, timeout_s, on_event=self._handle_event)
        for key in ("prompt", "system_prompt"):
            text = payload.get(key)
            if isinstance(text, str):
                self._tokens.add(text)
        self._client.start_receiver(self._handle_event, self._on_transport_close)

    @property
    def active(self) -> bool:
        return self._active

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError(f"Realtime session is no longer active: {self.session_id}")

    def _mark_ended(self, reason: str) -> None:
        with self._state_lock:
            if not self._active:
                return
            self._active = False
            self._ended_reason = self._ended_reason or reason
            for waiter in self._waiters.values():
                waiter.error = waiter.error or f"session ended ({reason})"
                waiter.ready.set()
                waiter.accepted.set()
                waiter.processed.set()
            self._waiters.clear()
            self._input_credit.notify_all()

    def _on_transport_close(self, reason: str) -> None:
        if not self._active:
            return
        log.warning("sglang-omni session %s transport closed: %s", self.session_id, reason)
        self._emit_error_chunk(f"sglang-omni connection closed ({reason})")
        self._mark_ended("ws_closed")

    # ------------------------------------------------------------ VlmRealtimeSession API

    def put_frame(self, image: Any, timestamp: Optional[float] = None,
                  byte_size: Optional[int] = None) -> Dict[str, Any]:
        self._ensure_active()
        raw = _encode_image(image)
        with self._input_lock:
            return self._put_frame_locked(raw, timestamp, byte_size, prompt="")

    def put_prompt(self, prompt: str) -> Dict[str, Any]:
        self._ensure_active()
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        with self._input_lock:
            self._ensure_active()
            if self._context_status()["rollover_required"]:
                raise ContextRolloverRequired("realtime context needs rollover before accepting a prompt")
            waiter = self._next_input("prompt", prompt=True)
            def submit():
                self._client.send_json({
                    "type": "input.prompt", "seq_no": waiter.seq_no,
                    "prompt": prompt, "final": False,
                })
                self._wait(waiter.accepted, waiter, "prompt.accepted")
            self._submit_input(waiter, submit)
            with self._state_lock:
                self.prompts_received += 1
            self._tokens.add(prompt)
            return self.status(extra={"prompt_seq_no": waiter.seq_no})

    def put_prompt_frame(self, prompt: str, image: Any, timestamp: Optional[float] = None,
                         byte_size: Optional[int] = None, drop_pending: bool = True) -> Dict[str, Any]:
        # Accepted remote inputs cannot be retracted by this protocol. Local
        # credit drops only unsent pure frames; the prompt itself is never dropped.
        del drop_pending
        self._ensure_active()
        raw = _encode_image(image)
        with self._input_lock:
            return self._put_frame_locked(raw, timestamp, byte_size, prompt=str(prompt or ""))

    def request_turn_end(self) -> Dict[str, Any]:
        """Mute locally until a subsequently submitted prompt changes the turn."""
        with self._state_lock:
            self._ensure_active()
            self._muted = True
            self._resume_after_seq = self._next_seq_no
            self._output_epoch += 1
            self._round_open_turn = None
        return self.status(extra={"turn_interrupt_pending": True})

    def output_event_is_current(self, event: Dict[str, Any]) -> bool:
        with self._state_lock:
            return event.get("output_epoch", self._output_epoch) == self._output_epoch

    def poll_output(self, timeout_seconds: float = 0.0, max_items: int = 128) -> OutputBatch:
        events: List[Dict[str, Any]] = []
        if timeout_seconds and timeout_seconds > 0:
            try:
                events.append(self._outputs.get(timeout=timeout_seconds))
            except queue.Empty:
                pass
        while len(events) < max(1, int(max_items)):
            try:
                events.append(self._outputs.get_nowait())
            except queue.Empty:
                break
        events = [ev for ev in events if self.output_event_is_current(ev)]
        chunks = [str(ev.get("text") or "") for ev in events]
        return OutputBatch(active=self._active, chunks=chunks,
                           chunk_events=events, status=self.status())

    def status(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._state_lock:
            payload: Dict[str, Any] = {
                "session_id": self.session_id,
                "gpu_id": self.gpu_id,
                "active": self._active,
                "backend": "sglang_omni",
                "created_at": self.created_at,
                "frames_received": self.frames_received,
                "frames_consumed": self.frames_consumed,
                "frames_dropped": self.frames_dropped,
                "prompts_received": self.prompts_received,
                "prompts_consumed": self.prompts_consumed,
                "outputs_emitted": self.outputs_emitted,
                "bytes_received": self.bytes_received,
                "text_tokens": self._usage["decoder_tokens"] if self._usage else self._tokens.total,
                "context": self._context_status(),
                "frame_queue_size": sum(1 for w in self._waiters.values() if not w.prompt),
                "prompt_queue_size": sum(1 for w in self._waiters.values() if w.prompt),
                "output_queue_size": self._outputs.qsize(),
                "turn_id": self._current_turn_id,
                "last_frame_timestamp": self._last_timestamp,
                "error": self._error,
            }
        if self._ended_reason:
            payload["ended_reason"] = self._ended_reason
        if extra:
            payload.update(extra)
        return payload

    @property
    def worker_transport_dead(self) -> bool:
        """Mirrors WorkerVlmSession: the pool quarantines the replica slot when
        the session died from a dropped transport rather than a clean stop."""
        return self._ended_reason == "ws_closed"

    def stop(self, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        if self._active and not self._ended_reason:
            # a caller-initiated stop is "stopped" unless the transport already died
            try:
                self._client.send_json({"type": "session.abort"})
            except Exception:  # noqa: BLE001 — socket may already be gone
                pass
            # let the server finish its teardown (response.done → session.done →
            # close) before we drop the transport from under it
            deadline = time.monotonic() + max(0.0, timeout_seconds)
            while self._active and time.monotonic() < deadline:
                time.sleep(0.05)
        self._mark_ended("stopped")
        self._client.close()
        return self.status(extra={"stopped": True})

    # ------------------------------------------------------------ input path

    def _context_status(self) -> Dict[str, Any]:
        with self._state_lock:
            pending = len(self._waiters)
            if self._usage is not None:
                limit = self._usage["context_limit"]
                used = self._usage["token_space_used"]
                # Pending inputs may not yet be represented in the last snapshot.
                used += pending * self._largest_extend
                exact = True
            else:
                limit = self._fallback_context_length
                generated_bound = int(max(0, time.time() - self.created_at) * self._generation_rate)
                used = ((self.frames_received + pending) * self._fallback_frame_tokens
                        + self._tokens.total + generated_bound)
                exact = False
            reserve = min(limit // 2, max(self._context_reserve_tokens, limit // 10,
                                          (self._input_capacity + 2) * self._largest_extend))
            remaining = max(0, limit - used)
            if remaining <= reserve:
                self._context_rollover = True
            return {"context_limit": limit, "token_space_used": used,
                    "context_remaining": remaining, "reserve_tokens": reserve,
                    "source": "backend_with_pending_reserve" if exact else "conservative_estimate",
                    "rollover_required": self._context_rollover}

    def _next_input(self, kind: str, prompt: bool) -> _InputWaiter:
        with self._state_lock:
            seq_no = self._next_seq_no
            self._next_seq_no += 1
            waiter = _InputWaiter(seq_no=seq_no, kind=kind, prompt=prompt)
            self._waiters[seq_no] = waiter
            self._sending_input = waiter
        return waiter

    def _await_input_credit(self) -> bool:
        """Wait for a server-side input slot; False = drop the (pure) frame.

        len(_waiters) tracks the server's outstanding input count: a waiter is
        added right before its metadata is sent and popped on
        input.*.processed, which is exactly when server capacity frees up.
        Dropping here is protocol-safe because no metadata has been sent yet.
        """
        deadline = time.monotonic() + self._input_drop_wait
        with self._input_credit:
            while self._active and len(self._waiters) >= self._input_capacity:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._input_credit.wait(timeout=remaining)
        if not self._active:
            raise RuntimeError(f"Realtime session is no longer active: {self.session_id}")
        return True

    def _put_frame_locked(self, raw: bytes, timestamp: Optional[float],
                          byte_size: Optional[int], *, prompt: str) -> Dict[str, Any]:
        self._ensure_active()
        mime = _mime_type(raw)
        if not raw or len(raw) > MAX_FRAME_BYTES:
            raise ValueError("frame exceeds max_frame_bytes or is empty")
        requested_ts = float(timestamp) if timestamp is not None else max(0.0, time.time() - self.created_at)
        if not math.isfinite(requested_ts) or requested_ts < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if self._context_status()["rollover_required"]:
            if prompt:
                raise ContextRolloverRequired("realtime context needs rollover before accepting a prompt")
            self.frames_dropped += 1
            return self.status(extra={"frame_dropped": True, "drop_reason": "context_rollover"})
        # credit gate applies to PURE frames only — prompt-carrying inputs must
        # never drop (a full queue of slow frames would otherwise starve them)
        if not prompt and not self._await_input_credit():
            with self._state_lock:
                self.frames_dropped += 1
            return self.status(extra={"frame_dropped": True,
                                      "drop_reason": "sglang_omni_input_backpressure"})
        with self._state_lock:
            # the server requires monotonic transport timestamps; browser frame
            # encoding can deliver an older capture after a newer one
            ts = max(requested_ts, self._last_timestamp)
        waiter = self._next_input("frame", prompt=bool(prompt))
        def submit():
            self._client.send_json({
                "type": "input.frame", "seq_no": waiter.seq_no, "timestamp": ts,
                "prompt": prompt or None, "final": False, "mime_type": mime,
            })
            self._wait(waiter.ready, waiter, "frame.ready")
            self._client.send_bytes(raw)
            self._wait(waiter.accepted, waiter, "frame.accepted")
        self._submit_input(waiter, submit)
        with self._state_lock:
            self._last_timestamp = ts
            self.frames_received += 1
            self.bytes_received += int(byte_size if byte_size is not None else len(raw))
            if prompt:
                self.prompts_received += 1
        if prompt:
            self._tokens.add(prompt)
        return self.status(extra={"timestamp": ts, "frame_seq_no": waiter.seq_no})

    def _submit_input(self, waiter: _InputWaiter, submit: Any) -> None:
        try:
            submit()
        except InputRejected:
            raise
        except Exception as exc:
            # ACK loss is ambiguous: never reuse a sequence the server may own.
            self._error = str(exc)
            self._emit_error_chunk(str(exc))
            self._mark_ended("input_uncertain")
            self._client.close()
            raise
        finally:
            with self._state_lock:
                if self._sending_input is waiter:
                    self._sending_input = None

    def _wait(self, event: threading.Event, waiter: _InputWaiter, label: str) -> None:
        if not event.wait(INPUT_ACK_TIMEOUT_S):
            raise TimeoutError(
                f"timed out waiting for sglang-omni {label} (seq_no={waiter.seq_no})")
        if waiter.error:
            if self._active:
                raise InputRejected(waiter.error)
            raise RuntimeError(waiter.error)

    # ------------------------------------------------------------ event handling

    def _handle_event(self, message: Dict[str, Any]) -> None:
        event_type = str(message.get("type") or "")
        if event_type == "session.usage":
            fields = ("decoder_tokens", "encoder_tokens", "token_space_used", "context_limit")
            if not all(isinstance(message.get(key), int) and not isinstance(message[key], bool)
                       and message[key] >= 0 for key in fields) or message["context_limit"] <= 0:
                self._handle_error({"code": "invalid_usage", "message": "invalid backend context usage"}, None, None)
                return
            with self._state_lock:
                if self._usage is not None and message["encoder_tokens"] > self._usage["encoder_tokens"]:
                    self._largest_extend = max(self._largest_extend,
                        message["token_space_used"] - self._usage["token_space_used"])
                self._usage = {key: message[key] for key in fields}
                self._context_status()
            return
        seq_no_raw = message.get("seq_no")
        seq_no = int(seq_no_raw) if isinstance(seq_no_raw, int) else None
        with self._state_lock:
            waiter = self._waiters.get(seq_no) if seq_no is not None else None

        if event_type == "input.frame.ready":
            if waiter is not None:
                waiter.ready.set()
            return
        if event_type in ("input.frame.accepted", "input.prompt.accepted"):
            if waiter is not None:
                waiter.accepted.set()
            return
        if event_type in ("input.frame.processed", "input.prompt.processed"):
            if waiter is not None:
                with self._state_lock:
                    if waiter.kind == "frame":
                        self.frames_consumed += 1
                    if waiter.prompt:
                        self.prompts_consumed += 1
                waiter.processed.set()
                with self._input_credit:
                    self._waiters.pop(waiter.seq_no, None)
                    self._input_credit.notify_all()
            return
        if event_type == "response.text.delta":
            text = str(message.get("delta") or "")
            if not text:
                return
            turn_id = int(message.get("turn_id") or 0)
            with self._state_lock:
                if turn_id != self._current_turn_id:
                    return  # straggler from an interrupted turn
                self._tokens.add(text)
                if self._muted:
                    return
                if self._round_open_turn != turn_id:
                    self._round_open_turn = turn_id
                    self._emit("<|round_start|>", message)
                self._emit(text, message)
            return
        if event_type == "response.turn.silence":
            # the model went idle = the spoken round's end-of-turn signal
            with self._state_lock:
                self._round_open_turn = None
            self._emit("<|silence|>", message)
            return
        if event_type == "response.turn.interrupted":
            old_turn = int(message.get("turn_id") or self._current_turn_id)
            next_turn = int(message.get("next_turn_id") or old_turn + 1)
            with self._state_lock:
                self._current_turn_id = next_turn
                self._round_open_turn = None
                if self._muted:
                    if seq_no is None or seq_no < self._resume_after_seq:
                        return
                    self._muted = False
                self._emit("<|eot_id|>", {**message, "turn_id": old_turn,
                                          "next_turn_id": next_turn})
            return
        if event_type == "response.done":
            # once per session lifetime; session.done + server close follow
            self._mark_ended("done")
            return
        if event_type == "session.done":
            self._mark_ended("done")
            return
        if event_type == "error":
            self._handle_error(message, seq_no, waiter)
            return
        log.debug("unhandled sglang-omni event: %s", event_type)

    def _handle_error(self, message: Dict[str, Any], seq_no: Optional[int],
                      waiter: Optional[_InputWaiter]) -> None:
        code = str(message.get("code") or "")
        text = str(message.get("message") or "sglang-omni error")
        if code == "invalid_request":
            # single event rejected, session survives: fail this waiter, free
            # its credit, and roll the dense seq counter back over the rejected
            # seq_no (the server never consumed it)
            with self._input_credit:
                if seq_no is None:
                    waiter = self._sending_input
                can_reject = waiter is not None and not waiter.accepted.is_set()
                if can_reject:
                    waiter.error = text
                    self._waiters.pop(waiter.seq_no, None)
                    if waiter.seq_no == self._next_seq_no - 1:
                        self._next_seq_no = waiter.seq_no
                    waiter.ready.set()
                    waiter.accepted.set()
                    waiter.processed.set()
                    self._input_credit.notify_all()
            if can_reject:
                log.warning("sglang-omni rejected input seq_no=%s: %s", waiter.seq_no, text)
                return
            # No unique unaccepted input owns this error. End instead of guessing.
        # input_submission_failed / response_failed / unknown: session-fatal
        self._error = text
        self._emit_error_chunk(text)
        self._mark_ended(f"error:{code or 'unknown'}")
        self._client.close()

    def _emit(self, text: str, message: Dict[str, Any]) -> None:
        if not text:
            return
        event = {
            "text": text,
            "chunk": text,
            "emitted_at": time.time(),
            "turn_id": message.get("turn_id", self._current_turn_id),
            "sglang_event_type": message.get("type"),
            "output_epoch": self._output_epoch,
        }
        for key in ("seq_no", "finish_reason", "next_turn_id", "silence_seq"):
            if message.get(key) is not None:
                event[key] = message[key]
        self._outputs.put(event)
        with self._state_lock:
            self.outputs_emitted += 1

    def _emit_error_chunk(self, text: str) -> None:
        # the orchestrator's [ERROR]-prefix path surfaces this as vlm_error
        self._outputs.put({"text": f"[ERROR] {text}", "chunk": f"[ERROR] {text}",
                           "emitted_at": time.time(), "sglang_event_type": "error"})
