"""Scripted fake engines for GPU-free session-layer tests.

Shape-compatible with the adapter Protocols (`server/adapters/base.py`) and the
board realtime session semantics: prompts trigger a scripted token stream
(`<|round_start|>` → text pieces → `<|round_end|>`), `request_turn_end` injects
`<|eot_id|>`, TTS yields PCM chunks per segment.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..adapters.base import AsrCaps, OutputBatch, TtsCaps

DEFAULT_REPLY = "你好，这是一段测试回复。第二句话紧随其后。"


# --------------------------- VLM ---------------------------


class FakeVlmSession:
    def __init__(self, reply: str = DEFAULT_REPLY, token_interval: float = 0.0,
                 piece_size: int = 4):
        self.session_id = "fake-vlm"
        self.gpu_id = 0
        self.reply = reply
        self.token_interval = token_interval
        self.piece_size = piece_size
        self.created_at = time.time()
        self.active = True
        # scripted stand-in for the worker token-counter patch's text_tokens
        # status field; rollover tests set it to drive the trigger
        self.text_tokens: Optional[int] = None
        self.prompts: List[str] = []
        self.frames: List[Any] = []
        self.prompt_frames: List[Any] = []
        self.interrupts = 0
        self._out: "queue.Queue[str]" = queue.Queue()
        self._interrupt = threading.Event()
        self._producer: Optional[threading.Thread] = None

    # ---- inputs ----

    def put_frame(self, image: Any, timestamp: Optional[float] = None,
                  byte_size: Optional[int] = None) -> Dict[str, Any]:
        if not self.active:
            raise RuntimeError("Realtime session is no longer active: fake-vlm")
        self.frames.append((image, timestamp))
        return self.status()

    def put_prompt(self, prompt: str) -> Dict[str, Any]:
        if not self.active:
            raise RuntimeError("Realtime session is no longer active: fake-vlm")
        self.prompts.append(prompt)
        self._start_round()
        return self.status()

    def put_prompt_frame(self, prompt: str, image: Any, timestamp: Optional[float] = None,
                         byte_size: Optional[int] = None, drop_pending: bool = True) -> Dict[str, Any]:
        if not self.active:
            raise RuntimeError("Realtime session is no longer active: fake-vlm")
        self.prompts.append(prompt)
        self.prompt_frames.append((prompt, image, timestamp))
        self._start_round()
        return self.status()

    def request_turn_end(self) -> Dict[str, Any]:
        self.interrupts += 1
        self._interrupt.set()
        return self.status()

    # ---- scripted output ----

    def _start_round(self) -> None:
        # the real model is a single generation thread: a new prompt is only
        # consumed after the current round ends, so end any in-flight round first
        if self._producer is not None and self._producer.is_alive():
            self._interrupt.set()
            self._producer.join(timeout=2.0)
        self._interrupt.clear()
        pieces = [self.reply[i:i + self.piece_size] for i in range(0, len(self.reply), self.piece_size)]

        def produce() -> None:
            self._out.put("<|round_start|>")
            for piece in pieces:
                if self._interrupt.is_set() or not self.active:
                    self._out.put("<|eot_id|>")
                    return
                if self.token_interval > 0:
                    time.sleep(self.token_interval)
                self._out.put(piece)
            # board-faithful: the real model closes a spoken round by going
            # idle with <|silence|> — it never emits round_end on the output
            self._out.put("<|silence|>")

        if self.token_interval > 0:
            self._producer = threading.Thread(target=produce, daemon=True)
            self._producer.start()
        else:
            produce()

    def narrate(self, text: str) -> None:
        """Script a SPONTANEOUS narration round: bare text + trailing silence.

        The real model emits <|round_start|> only on prompt events — unprompted
        narration arrives as plain text and ends when the model goes idle.
        """
        for i in range(0, len(text), self.piece_size):
            if self.token_interval > 0:
                time.sleep(self.token_interval)
            self._out.put(text[i:i + self.piece_size])
        self._out.put("<|silence|>")

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
        events = [{"text": c} for c in chunks]
        return OutputBatch(active=self.active, chunks=chunks, chunk_events=events, status=self.status())

    def status(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "active": self.active,
            "frames_received": len(self.frames) + len(self.prompt_frames),
            "frames_consumed": len(self.frames),
            "frames_dropped": 0,
            "outputs_emitted": 0,
            "frame_queue_size": 0,
            "output_queue_size": self._out.qsize(),
            "text_tokens": self.text_tokens,
        }

    def stop(self, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        self.active = False
        self._interrupt.set()
        return self.status()


class FakeVlmAdapter:
    """Adapter-shaped wrapper so the REST layer can build sessions from fakes."""

    def __init__(self, reply: str = DEFAULT_REPLY, token_interval: float = 0.0):
        self.reply = reply
        self.token_interval = token_interval
        self.model_path = "/fake/model"
        self.gpu_id = 0
        self.hf_mode = "online_streaming"
        self.sessions: List[FakeVlmSession] = []
        # every start_realtime_session call's kwargs (rollover re-seats included)
        self.start_params: List[Dict[str, Any]] = []

    def is_loaded(self) -> bool:
        return True

    def status(self) -> Dict[str, Any]:
        return {"loaded": True, "model_path": self.model_path, "gpu_id": self.gpu_id,
                "hf_mode": self.hf_mode, "active_sessions": len(self.sessions)}

    def start_realtime_session(self, **params: Any) -> FakeVlmSession:
        self.start_params.append(dict(params))
        session = FakeVlmSession(reply=self.reply, token_interval=self.token_interval)
        self.sessions.append(session)
        return session

    async def generate_stream(self, req: Any):
        for piece in ("离线", "回复", "。"):
            yield piece


# --------------------------- ASR ---------------------------


class FakeAsrStream:
    def __init__(self, text: str, finalize_delay: float = 0.0):
        self.text = text
        self.finalize_delay = finalize_delay
        self.pcm = bytearray()
        self.closed = False

    def send_pcm(self, pcm16: bytes) -> None:
        self.pcm.extend(pcm16)

    def commit(self, final: bool) -> None:
        return

    def finalize(self) -> str:
        if self.finalize_delay:
            time.sleep(self.finalize_delay)
        return self.text if self.pcm else ""

    def close(self) -> None:
        self.closed = True


class FakeAsrAdapter:
    def __init__(self, text: str = "画面里发生了什么？", sample_rate: int = 16000):
        self.caps = AsrCaps(streaming_partials=False, needs_server_vad=False, sample_rate=sample_rate)
        self.text = text
        self.ready = True
        self.streams: List[FakeAsrStream] = []

    def start(self) -> None:
        return

    def status(self) -> Dict[str, Any]:
        return {"enabled": True, "ready": True, "provider": "fake", "message": "ready"}

    def transcribe_pcm(self, pcm: bytes, allow_short: bool = False) -> str:
        """One-shot decode (the POST /api/asr dictation path)."""
        return self.text if pcm else ""

    def open_stream(self, on_partial: Optional[Callable[[str], None]] = None) -> FakeAsrStream:
        stream = FakeAsrStream(self.text)
        self.streams.append(stream)
        return stream


# --------------------------- TTS ---------------------------


class FakeTtsEngine:
    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 chunk_bytes: int = 3200, chunks_per_segment: int = 2,
                 synth_delay: float = 0.005):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_bytes = chunk_bytes
        self.chunks_per_segment = chunks_per_segment
        self.synth_delay = synth_delay
        self.segments: List[str] = []

    def synthesize_pcm(self, text: str, voice: Optional[str] = None) -> Iterable[bytes]:
        self.segments.append(text)
        for _ in range(self.chunks_per_segment):
            if self.synth_delay:
                time.sleep(self.synth_delay)
            yield b"\x01\x00" * (self.chunk_bytes // 2)


class FakeTtsAdapter:
    def __init__(self, engine: Optional[FakeTtsEngine] = None):
        self.engine = engine or FakeTtsEngine()
        self.caps = TtsCaps(token_streaming_input=False,
                            sample_rate=self.engine.sample_rate, channels=self.engine.channels)

    def start(self) -> None:
        return

    def status(self) -> Dict[str, Any]:
        return {"enabled": True, "ready": True, "provider": "fake", "message": "ready"}


class FakeOfflineVlm:
    """Offline-plane stand-in for routing tests (routers/chat.py:_chat_vlm)."""

    def __init__(self, loaded: bool = True, reply=("sglang ", "回复", "。")):
        self.loaded = loaded
        self.reply = reply
        self.calls = 0

    def is_loaded(self) -> bool:
        return self.loaded

    def status(self) -> Dict[str, Any]:
        return {"loaded": self.loaded, "provider": "fake_sglang", "modes": ["offline"]}

    async def generate_stream(self, req: Any):
        self.calls += 1
        for piece in self.reply:
            yield piece


# --------------------------- helpers ---------------------------

# 160 ms of loud-ish 16 kHz PCM16 (amplitude 3000 ≫ default RMS threshold 350)
LOUD_CHUNK = (3000).to_bytes(2, "little", signed=True) * 2560


def loud_pcm_chunks(n: int = 4) -> List[bytes]:
    return [LOUD_CHUNK for _ in range(n)]
