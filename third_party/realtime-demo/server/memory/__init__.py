"""L2 memory: multimodal vector memory for the realtime session plane.

    write   orchestrator ──enqueue──▶ MemoryWriter (thread) ──▶ MemoryStore
              user/assistant turns, deduped keyframes            rows + vectors
    match   ASR partial ──prefetch──▶ Retriever (fuse → rank → diversify → gate)
    insert  _user_turn ──▶ <recall> block prepended to the model-facing text only

Two invariants worth
restating here because breaking either is silent:

1. Retrieval is scoped to one `conversation_id`. There is no cross-session read
   path in this package.
2. Injected tokens are permanent residents of the session KV cache, so the gate
   is skip-biased; re-injection is allowed only past a post-ranking distance
   gate with a per-session copy cap (design §5).

Everything is behind `MEMORY_ENABLED` (default off) and every failure path
degrades to "no memory" rather than to a broken turn.
"""
from __future__ import annotations

from typing import Any, Optional

from ..config import Settings
from ..logging_conf import get_logger
from .session import EMPTY_RECALL, MemorySession, RecallResult
from .store import MemoryStore
from .writer import MemoryWriter
from .frames import SessionFrames

log = get_logger(__name__)

__all__ = ["MemoryStore", "MemoryWriter", "MemorySession", "RecallResult", "EMPTY_RECALL",
           "build_memory"]


def build_memory(settings: Settings, media: Any = None):
    """(store, writer) for the Runtime, or (None, None) when disabled."""
    if not settings.memory_enabled:
        return None, None
    try:
        store = MemoryStore(settings)
        if settings.media_enabled:
            store.frames = SessionFrames(store.path, settings)
        writer = MemoryWriter(settings, store, media=media)
        return store, writer
    except Exception as exc:  # noqa: BLE001 — never block boot on memory
        log.warning("memory disabled: construction failed (%s)", exc)
        return None, None
