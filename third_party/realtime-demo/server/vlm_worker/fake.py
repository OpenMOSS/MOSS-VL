"""Fake worker VLM — the real RPC surface with a scripted token stream.

`VLM_WORKER_FAKE=1` swaps this in for `HfMossVlAdapter` so the whole
multi-worker plane (supervisor, proxy, pool, manager, WS) can be exercised on
boxes that cannot fit N model replicas (the 4090 dev box, CI containers).
Mirrors the board realtime semantics the orchestrator expects: each prompt
plays `<|round_start|>` → text pieces → `<|silence|>`; `request_turn_end`
injects `<|eot_id|>`.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from ..adapters.base import OutputBatch, VlmCaps

DEFAULT_REPLY = "这是一个占位回复，来自 fake VLM worker。第二句紧随其后。"


class FakeWorkerSession:
    def __init__(self, reply: str = DEFAULT_REPLY, piece_size: int = 4,
                 token_interval: float = 0.02, text_tokens: int = 0):
        self.session_id = uuid.uuid4().hex
        self.gpu_id = 0
        self.reply = reply
        self.piece_size = piece_size
        self.token_interval = token_interval
        # scripted stand-in for the token-counter patch's _rt_text_tokens;
        # rollover tests set it to drive the trigger through the status plane
        self.text_tokens = text_tokens
        self.created_at = time.time()
        self.active = True
        self.frames_received = 0
        self.frames_consumed = 0
        self.frames_dropped = 0
        self.prompts_received = 0
        self.outputs_emitted = 0
        self.bytes_received = 0
        self._out: "queue.Queue[str]" = queue.Queue()
        self._interrupt = threading.Event()
        self._producer: Optional[threading.Thread] = None

    # ---- inputs (VlmRealtimeSession protocol) ----

    def _ensure_active(self) -> None:
        if not self.active:
            raise RuntimeError(f"Realtime session is no longer active: {self.session_id}")

    def put_frame(self, image: Any, timestamp: Optional[float] = None,
                  byte_size: Optional[int] = None) -> Dict[str, Any]:
        self._ensure_active()
        self.frames_received += 1
        self.frames_consumed += 1
        if byte_size is None and isinstance(image, (bytes, bytearray)):
            byte_size = len(image)
        self.bytes_received += int(byte_size or 0)
        return self.status()

    def put_prompt(self, prompt: str) -> Dict[str, Any]:
        self._ensure_active()
        self.prompts_received += 1
        self._start_round()
        return self.status()

    def put_prompt_frame(self, prompt: str, image: Any, timestamp: Optional[float] = None,
                         byte_size: Optional[int] = None, drop_pending: bool = True) -> Dict[str, Any]:
        self.put_frame(image, timestamp, byte_size)
        return self.put_prompt(prompt)

    def request_turn_end(self) -> Dict[str, Any]:
        self._interrupt.set()
        return self.status()

    def _start_round(self) -> None:
        if self._producer is not None and self._producer.is_alive():
            self._interrupt.set()
            self._producer.join(timeout=2.0)
        self._interrupt.clear()
        pieces = [self.reply[i:i + self.piece_size]
                  for i in range(0, len(self.reply), self.piece_size)]

        def produce() -> None:
            self._out.put("<|round_start|>")
            for piece in pieces:
                if self._interrupt.is_set() or not self.active:
                    self._out.put("<|eot_id|>")
                    return
                if self.token_interval > 0:
                    time.sleep(self.token_interval)
                self._out.put(piece)
            self._out.put("<|silence|>")

        self._producer = threading.Thread(target=produce, daemon=True)
        self._producer.start()

    def poll_output(self, timeout_seconds: float = 0.0, max_items: int = 128) -> OutputBatch:
        chunks: List[str] = []
        if timeout_seconds and timeout_seconds > 0:
            try:
                chunks.append(self._out.get(timeout=timeout_seconds))
            except queue.Empty:
                pass
        while len(chunks) < max_items:
            try:
                chunks.append(self._out.get_nowait())
            except queue.Empty:
                break
        self.outputs_emitted += len(chunks)
        events = [{"text": c, "chunk": c, "emitted_at": time.time()} for c in chunks]
        return OutputBatch(active=self.active, chunks=chunks,
                           chunk_events=events, status=self.status())

    def status(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "gpu_id": self.gpu_id,
            "active": self.active,
            "created_at": self.created_at,
            "frames_received": self.frames_received,
            "frames_consumed": self.frames_consumed,
            "frames_dropped": self.frames_dropped,
            "prompts_received": self.prompts_received,
            "outputs_emitted": self.outputs_emitted,
            "bytes_received": self.bytes_received,
            "frame_queue_size": 0,
            "output_queue_size": self._out.qsize(),
            "text_tokens": self.text_tokens,
        }

    def stop(self, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        self.active = False
        self._interrupt.set()
        return self.status()


class FakeWorkerVlmAdapter:
    """`HfMossVlAdapter` stand-in: same surface the worker app drives."""

    def __init__(self, settings: Any = None):
        self.s = settings
        self.caps = VlmCaps(modes=("online_streaming", "offline"))
        self.model_path = "fake"
        self.gpu_id = 0
        self.hf_mode = "online_streaming"
        self.model_config: dict = {}
        self._attn_impl: Optional[str] = "fake"
        self._loaded = True
        self._sessions: Dict[str, FakeWorkerSession] = {}
        self.start_params: List[Dict[str, Any]] = []

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, model_path: str, gpu_id: int, hf_mode: str,
             attn_impl_override: Optional[str] = None) -> None:
        self.model_path = model_path or "fake"
        self.hf_mode = hf_mode or "online_streaming"
        self._loaded = True

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "model_path": self.model_path,
            "gpu_id": self.gpu_id,
            "hf_mode": self.hf_mode,
            "attn_impl": self._attn_impl,
            "active_sessions": len(self._sessions),
            "modes": list(self.caps.modes),
            "fake": True,
        }

    def start_realtime_session(self, **params: Any) -> FakeWorkerSession:
        if any(s.active for s in self._sessions.values()):
            raise RuntimeError("A realtime session is already running on this model")
        # record the start params (prompt / system_prompt / prefill_messages)
        # so rollover tests can assert what the re-seat actually sent
        self.start_params.append(dict(params))
        session = FakeWorkerSession()
        self._sessions = {session.session_id: session}
        return session

    async def generate_stream(self, req: Any) -> AsyncIterator[str]:
        import asyncio

        for piece in ("fake ", "offline ", "chat ", "reply"):
            await asyncio.sleep(0.01)
            yield piece
