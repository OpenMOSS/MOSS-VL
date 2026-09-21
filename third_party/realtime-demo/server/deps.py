"""Runtime container + FastAPI dependency.

Holds the constructed adapters and the session manager for the life of the
process. Built in app.py's lifespan; routers reach it via `get_runtime()`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException

from .adapters.registry import build_asr, build_tts, build_vlm, build_vlm_offline
from .config import Settings, get_settings
from .logging_conf import get_logger
from .memory import build_memory
from .memory.facts import build_fact_extractor
from .persistence import HistoryRecorder, IndexStore, MediaStore, set_media_store
from .session.manager import SessionManager

log = get_logger(__name__)


class Runtime:
    def __init__(self, settings: Settings, plan: Optional[Any] = None):
        self.settings = settings
        # GPU placement plan (server/gpu/placement.py); None = pre-multi-GPU
        # construction (tests, bare build_runtime) → inproc-compatible defaults
        self.plan = plan
        self.asr = build_asr(settings, plan)
        self.tts = build_tts(settings, plan)
        # optional cloud TTS alongside the local pool (no GPU, no sidecar):
        # sessions pick it via config tts_engine=elevenlabs. Skipped when the
        # boot provider IS elevenlabs (self.tts already is it) or no API key.
        from .adapters.tts.providers import is_external_provider
        self.tts_elevenlabs: Optional[Any] = None
        if (settings.tts_enabled and settings.elevenlabs_api_key
                and not is_external_provider(settings.tts_provider)):
            try:
                from .adapters.tts.elevenlabs import ElevenLabsAdapter
                self.tts_elevenlabs = ElevenLabsAdapter(settings)
            except Exception as exc:  # noqa: BLE001 — optional lane never blocks boot
                log.warning("elevenlabs TTS unavailable: construction failed (%s)", exc)
        self.tts_minimax: Optional[Any] = None
        if (settings.tts_enabled and settings.minimax_api_key
                and not is_external_provider(settings.tts_provider)):
            try:
                from .adapters.tts.minimax import MiniMaxAdapter
                self.tts_minimax = MiniMaxAdapter(settings)
            except Exception as exc:  # noqa: BLE001 — optional lane never blocks boot
                log.warning("minimax TTS unavailable: construction failed (%s)", exc)
        self.vlm = build_vlm(settings, plan)
        # dedicated offline-chat plane (sglang sidecars); None = offline chat
        # falls back to the online pool (routers/chat.py:_chat_vlm)
        self.vlm_offline: Optional[Any] = build_vlm_offline(settings, plan)
        # process supervisors, attached by app.py's lifespan under workers mode
        self.vlm_supervisor: Optional[Any] = None
        self.sglang_supervisor: Optional[Any] = None
        self.tts_pool: Optional[Any] = None
        # durable history + media archives (server/persistence/); constructed
        # here but touch no disk until open_persistence() runs in the lifespan
        self.index: Optional[IndexStore] = None
        self.media: Optional[MediaStore] = None
        self.history: Optional[HistoryRecorder] = None
        if settings.history_enabled or settings.media_enabled:
            self.index = IndexStore(settings)
        if settings.media_enabled and self.index is not None:
            self.media = MediaStore(settings, self.index)
        if settings.history_enabled and self.index is not None:
            self.history = HistoryRecorder(settings, self.index)
        # L2 memory plane (server/memory/); (None, None) unless MEMORY_ENABLED.
        # Shared across sessions — per-session state lives in MemorySession.
        self.memory_store, self.memory_writer = build_memory(settings, media=self.media)
        from .memory.maintenance import MemoryMaintenance
        self.memory_maintenance = MemoryMaintenance(self.memory_store) if self.memory_store is not None else None
        # fact extraction (memory/facts.py) rides the offline sglang plane and
        # holds the semaphore future summary jobs will share; getattr because
        # tests build partial Runtimes via Runtime.__new__
        self.memory_facts = build_fact_extractor(
            settings, self.memory_store, self.memory_writer,
            plane=getattr(self, "vlm_offline", None))
        # server-orchestrated realtime sessions (backend_overhaul.md §3);
        # torn down via `await session_manager.aclose()` in the app lifespan
        self.session_manager = SessionManager(
            settings, history=self.history,
            capacity=lambda: getattr(self.vlm, "capacity", 1),
            memory=(self.memory_store, self.memory_writer),
            memory_facts=self.memory_facts,
            # the rollover summarizer rides the same offline plane as facts
            memory_plane=getattr(self, "vlm_offline", None))
        # ASR backs off its partial re-decodes under multi-session load
        if hasattr(self.asr, "session_count_provider"):
            self.asr.session_count_provider = lambda: self.session_manager.active_count

    # ---- lifecycle (blocking; run via asyncio.to_thread in lifespan) ----

    def open_persistence(self) -> None:
        if getattr(self, 'memory_maintenance', None) is not None:
            self.memory_store.open()  # fail ownership checks before opening any archive writers
        if self.index is not None:
            from .persistence.retention import acquire_archive_lock
            self._archive_guard = acquire_archive_lock(self.index.db_path)
            self.index.open()
        if self.media is not None:
            self.media.open()
            set_media_store(self.media)  # lets the VLM adapter resolve CAS handles
        if self.history is not None:
            self.history.open()
        # getattr: tests build partial Runtimes via Runtime.__new__
        if getattr(self, "memory_writer", None) is not None:
            if getattr(self, 'memory_maintenance', None) is not None:
                self.memory_maintenance.start()
            self.memory_writer.start()  # opens the store on its own thread
            # eager embedder load (still off-loop here): the first live turn
            # must not pay the BGE-M3/Chinese-CLIP weight load on recall
            self.memory_writer.warmup()

    def close_persistence(self) -> None:
        if getattr(self, "memory_writer", None) is not None:
            # Embeddings execute native code; never close their stores while a
            # worker still owns an in-flight job, even when shutdown is slow.
            self.memory_writer.stop(timeout=None)
        if getattr(self, 'memory_maintenance', None) is not None:
            self.memory_maintenance.stop()
        set_media_store(None)
        if getattr(self, "memory_store", None) is not None:
            self.memory_store.close()
        if self.history is not None:
            self.history.close()  # flush the writer queue first
        if self.index is not None:
            self.index.close()
        guard = getattr(self, '_archive_guard', None)
        if guard is not None:
            guard.close()
            self._archive_guard = None

    def start_voice(self) -> None:
        self.asr.start()
        self.tts.start()
        if getattr(self, "tts_elevenlabs", None) is not None:
            self.tts_elevenlabs.start()  # GET /v1/voices health + voice list
        if getattr(self, "tts_minimax", None) is not None:
            self.tts_minimax.start()  # POST /v1/get_voice health + voice list
        log.info("Voice runtime: asr=%s tts=%s", self.asr.status().get("message"), self.tts.status().get("message"))

    def maybe_load_vlm(self) -> None:
        s = self.settings
        # getattr: tests build partial Runtimes via Runtime.__new__
        if getattr(self, "vlm_supervisor", None) is not None:
            return  # workers mode: each worker autoloads at spawn
        if s.autoload_vlm and s.model_path:
            try:
                self.vlm.load(s.model_path, s.gpu_id, s.hf_mode)
            except Exception as exc:  # noqa: BLE001
                log.exception("Autoload VLM failed: %s", exc)

    def voice_status(self) -> Dict[str, Any]:
        eleven = getattr(self, "tts_elevenlabs", None)
        minimax = getattr(self, "tts_minimax", None)
        return {
            "asr": self.asr.status(),
            "tts": self.tts.status(),
            # the UI's TTS engine select keys off these: shown only when the
            # lane was actually built (API key present at boot)
            "tts_elevenlabs": eleven.status() if eleven is not None
            else {"available": False, "provider": "elevenlabs"},
            "tts_minimax": minimax.status() if minimax is not None
            else {"available": False, "provider": "minimax"},
        }


_runtime: Optional[Runtime] = None


def set_runtime(runtime: Optional[Runtime]) -> None:
    global _runtime
    _runtime = runtime


def get_runtime() -> Runtime:
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not initialized")
    return _runtime


def build_runtime(plan: Optional[Any] = None) -> Runtime:
    return Runtime(get_settings(), plan)
