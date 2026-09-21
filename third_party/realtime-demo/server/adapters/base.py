"""Pluggable adapter contracts.

Every engine that could be swapped (ASR model, TTS model, VLM backend) implements
one of these Protocols. Concrete adapters live alongside this file and are wired up
by registry.py from Settings, so changing an engine is a config change, not a code
change. Capability dataclasses let the routers adapt behaviour (e.g. partial ASR
results, token-streaming TTS) without knowing the concrete engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol, runtime_checkable

# ----------------------------- ASR -----------------------------


@dataclass
class AsrCaps:
    streaming_partials: bool  # emits on_partial during a turn
    needs_server_vad: bool    # engine expects the server to segment turns
    sample_rate: int          # required input PCM sample rate


class AsrStream(Protocol):
    """A single utterance/turn. PCM is fed as it arrives; finalize() decodes."""

    def send_pcm(self, pcm16: bytes) -> None: ...

    def commit(self, final: bool) -> None: ...

    def finalize(self) -> str: ...

    def close(self) -> None: ...


@runtime_checkable
class AsrAdapter(Protocol):
    caps: AsrCaps

    def start(self) -> None: ...          # load + warm; called in lifespan (sync, run in thread)

    def status(self) -> Dict[str, Any]: ...

    def open_stream(self, on_partial: Optional[Callable[[str], None]] = None) -> AsrStream: ...


# ----------------------------- TTS -----------------------------


@dataclass
class TtsCaps:
    token_streaming_input: bool  # can consume incremental text (vs sentence-level)
    sample_rate: int
    channels: int


class TtsEngine(Protocol):
    """Low-level synthesis: text segment -> PCM byte chunks (streaming)."""

    sample_rate: int
    channels: int

    def synthesize_pcm(self, text: str, voice: Optional[str] = None): ...  # -> Iterable[bytes]


@runtime_checkable
class TtsAdapter(Protocol):
    caps: TtsCaps
    engine: TtsEngine

    def start(self) -> None: ...

    def status(self) -> Dict[str, Any]: ...


# ----------------------------- VLM -----------------------------


@dataclass
class OutputBatch:
    active: bool
    chunks: List[str]
    chunk_events: List[Dict[str, Any]]
    status: Dict[str, Any]


class VlmRealtimeSession(Protocol):
    session_id: str

    def put_frame(self, image: Any, timestamp: Optional[float] = None, byte_size: Optional[int] = None) -> Dict[str, Any]: ...

    def put_prompt(self, prompt: str) -> Dict[str, Any]: ...

    def put_prompt_frame(self, prompt: str, image: Any, timestamp: Optional[float] = None,
                         byte_size: Optional[int] = None, drop_pending: bool = True) -> Dict[str, Any]: ...

    def request_turn_end(self) -> Dict[str, Any]: ...

    def poll_output(self, timeout_seconds: float = 0.0, max_items: int = 128) -> OutputBatch: ...

    def status(self) -> Dict[str, Any]: ...

    def stop(self, timeout_seconds: float = 10.0) -> Dict[str, Any]: ...


@dataclass
class VlmCaps:
    modes: tuple  # ("online_streaming", "offline")


@runtime_checkable
class VlmAdapter(Protocol):
    caps: VlmCaps

    def load(self, model_path: str, gpu_id: int, hf_mode: str,
             attn_impl_override: Optional[str] = None) -> None: ...

    def is_loaded(self) -> bool: ...

    def status(self) -> Dict[str, Any]: ...

    def start_realtime_session(self, **params: Any) -> VlmRealtimeSession: ...

    async def generate_stream(self, req: Any) -> AsyncIterator[str]: ...
