"""Shared infrastructure for the TTS provider adapters.

`read_chunk_bytes()` centralises the streaming read-size knob so every sidecar
HTTP client (nano protocol, OpenAI-speech engines, the MOSS-TTS-Realtime native
session client) behaves identically.
"""
from __future__ import annotations

import os
from typing import Any


def read_chunk_bytes(default: int = 8192) -> int:
    """Cap (bytes) for one streaming read from a sidecar's audio stream.

    Env: ``MOSS_TTS_READ_BYTES`` — applies to every TTS provider (realtime,
    nano, cosyvoice, vllm-omni), not just Nano. The legacy
    ``MOSS_TTS_NANO_READ_BYTES`` is still honoured as a fallback.

    Paired with ``HTTPResponse.read1()`` (not ``.read()``): the reader returns
    the moment ANY PCM is buffered, so this value bounds chunk *size*, never the
    first-chunk *latency* — a large ``.read(32768)`` used to block ~0.68 s (24
    kHz mono) before the first frame reached the pipeline.
    """
    raw = os.getenv("MOSS_TTS_READ_BYTES") or os.getenv("MOSS_TTS_NANO_READ_BYTES")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def stream_read(response: Any, cap: int) -> bytes:
    """Read the next available audio slice from an HTTP stream.

    Uses ``read1`` so a slow producer's first bytes are delivered immediately
    instead of blocking until ``cap`` bytes accumulate. Falls back to ``read``
    on the rare stream object without ``read1``.
    """
    reader = getattr(response, "read1", None)
    if reader is None:
        return response.read(cap)
    return reader(cap)
