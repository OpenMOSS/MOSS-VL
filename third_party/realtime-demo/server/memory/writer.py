"""The write path: one daemon thread owns every embedding and DB write.

The hot path (orchestrator, ASR thread, WS reader) only ever enqueues. Nothing
in a live turn may block on an embedder or on SQLite — an LLM-mediated write
path costs seconds per item and would stall captions or TTS.

Frame dedup is a cheap cascade, ordered by cost: a wall-clock throttle on the
hot path, then a 64-bit difference hash, then descriptor cosine. A static scene
still leaves one keyframe every `memory_keyframe_force_s` so "what was on the
table earlier" has something to hit.

Shutdown seals the input queue and drains accepted jobs before returning. A
separate wakeup event keeps the stop notification independent of queue capacity.
"""
from __future__ import annotations

import math
import hashlib
from collections import OrderedDict
from contextlib import nullcontext
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..config import Settings
from ..logging_conf import get_logger
from . import embed as embed_mod
from .store import KIND_FRAME, KIND_UTTERANCE, SPACE_IMAGE, SPACE_TEXT, MemoryStore, MemoryBudgetExceeded

log = get_logger(__name__)


@dataclass
class _FrameState:
    last_hash: Optional[int] = None
    last_vec: Optional[np.ndarray] = None
    last_kept_at: float = 0.0


class MemoryWriter:
    """Shared across sessions; every job carries its own conversation_id."""

    def __init__(self, settings: Settings, store: MemoryStore, *, media: Any = None,
                 text_embedder: Any = None, image_embedder: Any = None) -> None:
        self.settings = settings
        self.store = store
        self.media = media
        self.text = text_embedder or embed_mod.build_text_embedder(settings)
        self.image = image_embedder or embed_mod.build_image_embedder(settings)
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=512)
        self._thread: Optional[threading.Thread] = None
        self._lifecycle_lock = threading.RLock()
        self._wakeup = threading.Event()
        self._frames: Dict[str, _FrameState] = {}
        # conversation_id -> set((role, whitespace-normalized text)): rollover /
        # repeated output must not store the same utterance twice (board parity);
        # pinned items never pass through here and are unaffected
        self._seen_utterances: Dict[str, set] = {}
        self._seen_lock = threading.Lock()
        self._stopping = False
        self._payload_bytes = 0
        self._session_payload_bytes = {}
        self.stats = {"utterances": 0, "frames_kept": 0, "frames_skipped": 0,
                      "dropped": 0, "utterances_rejected": 0}

    def _mark_utterance_seen(self, conversation_id: str, role: str, text: str, historical_answer=False) -> bool:
        """True on first sight, False on an exact duplicate (whitespace-normalized)."""
        marker = self._utterance_key(role, text, historical_answer)
        with self._seen_lock:
            seen = self._seen_utterances.setdefault(str(conversation_id), OrderedDict())
            if marker in seen:
                return False
            seen[marker] = None
            while len(seen) > max(1, self.settings.memory_session_max_items):
                seen.popitem(last=False)
            return True

    @staticmethod
    def _utterance_key(role, text, historical_answer=False):
        return hashlib.sha256((str(role)+'\0'+str(historical_answer)+'\0'+' '.join(str(text).split())).encode()).digest()

    def _release_payload(self, job):
        size = job.get('_payload_bytes', 0)
        conv = job.get('conv')
        self._payload_bytes -= size
        left = self._session_payload_bytes.get(conv, 0) - size
        if left > 0:
            self._session_payload_bytes[conv] = left
        else:
            self._session_payload_bytes.pop(conv, None)

    def status(self):
        with self._lifecycle_lock:
            return {**self.stats, 'payload_bytes': self._payload_bytes,
                    'queued_items': self._q.qsize(), 'pending_sessions': len(self._session_payload_bytes)}

    # ---- lifecycle ----

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None:
                if self._stopping:
                    raise RuntimeError("memory writer is still stopping; retry stop() before restarting")
                return
            self.store.open()
            self._stopping = False
            if self._q.empty():
                self._wakeup.clear()
            else:
                self._wakeup.set()
            self._thread = threading.Thread(target=self._run, name="memory-writer", daemon=True)
            self._thread.start()
        log.info("memory writer started (text=%s dim=%s, image=%s dim=%s)",
                 getattr(self.text, "name", "?"), getattr(self.text, "dim", "?"),
                 getattr(self.image, "name", "?"), getattr(self.image, "dim", "?"))

    def stop(self, timeout: Optional[float] = None) -> None:
        """Seal and drain the writer; only return once the worker has exited.

        None is a graceful, unbounded wait. On an explicit timeout the writer
        stays sealed and its thread handle is retained. Call stop() again to
        finish joining before closing stores or releasing embedding resources.
        """
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and non-negative, or None")
        with self._lifecycle_lock:
            thread = self._thread
            if thread is threading.current_thread():
                raise RuntimeError("memory writer cannot join itself")
            self._stopping = True
            self._wakeup.set()
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError(f"memory writer has not stopped; {self._q.qsize()} jobs remain queued")
        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None

    # ---- producers (hot path: enqueue only) ----

    def _put(self, job: Dict[str, Any], *, droppable: bool) -> bool:
        with self._lifecycle_lock:
            lifetime = job.get("lifetime")
            if self._stopping or not self.store.accepting_writes or (lifetime is not None and lifetime.closing):
                return False
            size = len(job.get('jpeg', b'')) + len(job.get('text', '').encode('utf-8'))
            conv = job.get('conv')
            if (self._payload_bytes + size > max(1, self.settings.memory_queue_max_bytes) or
                    self._session_payload_bytes.get(conv, 0) + size > max(1, self.settings.memory_queue_session_bytes)):
                self.stats['dropped'] += 1
                self.stats['budget_rejected'] = self.stats.get('budget_rejected', 0) + 1
                return False
            job['_payload_bytes'] = size
            try:
                self._q.put_nowait(job)
            except queue.Full:
                if droppable:
                    self.stats["dropped"] += 1
                    return False
                # Replace only a frame; preserve FIFO order and unfinished_tasks.
                # The consumer shares lifecycle_lock, so it cannot race eviction.
                with self._q.mutex:
                    victim = next((i for i, queued in enumerate(self._q.queue)
                                   if queued.get("t") == "frame"), None)
                    if victim is None:
                        self.stats["utterances_rejected"] += 1
                        log.error("memory queue full: utterance NOT accepted for %s; "
                                  "accepted utterances retained", job.get("conv"))
                        return False
                    self._release_payload(self._q.queue[victim])
                    del self._q.queue[victim]
                    self._q.queue.append(job)
                    self.stats["dropped"] += 1
            self._payload_bytes += size
            self._session_payload_bytes[conv] = self._session_payload_bytes.get(conv, 0) + size
            self._wakeup.set()
            return True

    def note_utterance(self, conversation_id: str, role: str, text: str, *, lang: str,
                       session_ts: Optional[float] = None, media_ts: Optional[float] = None,
                       importance: float = 0.5, lifetime: Any = None, historical_answer: bool = False) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        if len(text) > max(1, self.settings.memory_item_max_chars):
            self.stats['utterances_rejected'] += 1
            return False
        with self._lifecycle_lock:
            if self._stopping or (lifetime is not None and lifetime.closing):
                return False
            if not self._mark_utterance_seen(conversation_id, role, text, historical_answer):
                log.debug("updating repeat timestamp for %s utterance in %s: %r",
                          role, conversation_id, text[:60])
                return self._put({'t':'repeat', 'conv':conversation_id, 'role':role, 'text':text,
                                  'lang':lang, 'session_ts':session_ts, 'media_ts':media_ts,
                                  'importance':importance, 'lifetime':lifetime,
                                  'historical_answer':historical_answer}, droppable=False)
            accepted = self._put({"t": "utterance", "conv": conversation_id, "role": role, "text": text,
                       "lang": lang, "session_ts": session_ts, "media_ts": media_ts,
                       "importance": importance, "lifetime": lifetime, "historical_answer": historical_answer}, droppable=False)
            if not accepted:
                with self._seen_lock:
                    self._seen_utterances.get(str(conversation_id), {}).pop(self._utterance_key(role, text, historical_answer), None)
            return accepted

    def note_frame(self, conversation_id: str, jpeg: bytes, *, session_ts: Optional[float] = None,
                   media_ts: Optional[float] = None, lang: str = "zh", lifetime: Any = None) -> None:
        if not jpeg:
            return
        if len(jpeg) > max(1, self.settings.memory_frame_max_bytes):
            self.stats['dropped'] += 1
            return
        self._put({"t": "frame", "conv": conversation_id, "jpeg": jpeg, "lang": lang,
                   "session_ts": session_ts, "media_ts": media_ts, "lifetime": lifetime}, droppable=True)

    # ---- consumer ----

    def _run(self) -> None:
        while True:
            self._wakeup.wait()
            with self._lifecycle_lock:
                try:
                    job = self._q.get_nowait()
                except queue.Empty:
                    if self._stopping:
                        return
                    self._wakeup.clear()
                    continue
                lifetime = job.get('lifetime')
                lease = lifetime.operation() if lifetime is not None else nullcontext(True)
                admitted = lease.__enter__()
                if not admitted:
                    self._release_payload(job)
                    job = None
                    self._q.task_done()
                    lease.__exit__(None, None, None)
                    continue
            try:
                if not self.store.accepting_writes:
                    self.stats['dropped'] += 1
                    continue
                self._handle(job)
            except Exception as exc:  # noqa: BLE001 — memory must never kill a session
                self.stats['write_errors'] = self.stats.get('write_errors', 0) + 1
                if isinstance(exc, MemoryBudgetExceeded):
                    self.stats['budget_rejected'] = self.stats.get('budget_rejected', 0) + 1
                log.warning("memory writer job %s failed session=%s: %s", job.get("t"), job.get('conv'), exc)
            finally:
                try:
                    with self._lifecycle_lock:
                        self._release_payload(job)
                    job = None
                    lifetime = None
                    self._q.task_done()
                finally:
                    # Session close waits for payload release too, not just SQL.
                    lease.__exit__(None, None, None)

    def _handle(self, job: Dict[str, Any]) -> None:
        kind = job.get("t")
        if kind == "utterance":
            self._handle_utterance(job)
        elif kind == 'repeat':
            marker = self._utterance_key(job['role'], job['text'], job.get('historical_answer', False))
            with self._seen_lock:
                item_id = self._seen_utterances.get(job['conv'], {}).get(marker)
            if not self.store.note_repeat(job['conv'], job['role'], job['text'], job.get('session_ts'),
                                          historical_answer=job.get('historical_answer', False), item_id=item_id):
                self._handle_utterance(job)
        elif kind == "frame":
            self._handle_frame(job)

    def _handle_utterance(self, job: Dict[str, Any]) -> None:
        conv, text = job["conv"], job["text"]
        item_id = self.store.add_item(
            conv, KIND_UTTERANCE, text=text, role=job.get("role"), lang=job.get("lang"),
            session_ts=job.get("session_ts"), media_ts=job.get("media_ts"),
            importance=float(job.get("importance", 0.5)), historical_answer=job.get('historical_answer', False))
        marker = self._utterance_key(job.get('role'), text, job.get('historical_answer', False))
        with self._seen_lock:
            seen = self._seen_utterances.get(conv, {})
            if marker in seen:
                seen[marker] = item_id
        vec = self.text.encode([text])[0]
        self.store.add_vector(conv, item_id, SPACE_TEXT, vec)
        # late interaction: token matrices alongside the pooled vector. Gated on
        # the capability, not the config flag — the first call lazily loads the
        # colbert head and flips `supports_late`; a missing head degrades to
        # pooled-only retrieval with no other change.
        encode_tokens = getattr(self.text, "encode_tokens", None)
        if self.settings.memory_late_interaction and callable(encode_tokens):
            try:
                self.store.add_vector_late(conv, item_id, encode_tokens([text])[0])
            except Exception as exc:  # noqa: BLE001 — pooled retrieval still works
                log.debug("memory: late-interaction encode failed (%s)", exc)
        self.stats["utterances"] += 1

    def _handle_frame(self, job: Dict[str, Any]) -> None:
        conv, jpeg = job["conv"], job["jpeg"]
        state = self._frames.setdefault(conv, _FrameState())
        now = time.monotonic()
        forced = (now - state.last_kept_at) >= max(1.0, float(self.settings.memory_keyframe_force_s))

        digest = embed_mod.dhash(jpeg)
        if not forced and embed_mod.hamming(digest, state.last_hash) <= 6:
            self.stats["frames_skipped"] += 1
            return
        vec = self.image.encode_images([jpeg])[0]
        if (not forced and state.last_vec is not None
                and float(vec @ state.last_vec) >= float(self.settings.memory_keyframe_sim_threshold)):
            self.stats["frames_skipped"] += 1
            return

        media_hash = None
        if self.store.frames is not None:
            # Record ownership BEFORE placing a frame: crash recovery can then
            # remove a directory even if the following item insert never ran.
            self.store.register_session(conv)
            media_hash = self.store.frames.put(conv, jpeg)
        elif self.media is not None:
            try:
                media_hash = self.media.put_bytes(jpeg, orig_name="keyframe.jpg").get("hash")
            except Exception as exc:  # noqa: BLE001 — CAS rejection must not lose the vector
                log.debug("memory: keyframe not stored in CAS (%s)", exc)
        item_id = self.store.add_item(
            conv, KIND_FRAME, text="", lang=job.get("lang"), session_ts=job.get("session_ts"),
            media_ts=job.get("media_ts"), media_hash=media_hash, importance=0.4)
        self.store.add_vector(conv, item_id, SPACE_IMAGE, vec)
        state.last_hash = digest
        state.last_vec = vec
        state.last_kept_at = now
        self.stats["frames_kept"] += 1

    # ---- test/manual helper ----

    def warmup(self) -> None:
        """Eagerly load the lazy embedders. Called from the app lifespan
        (off-loop, before "ready"): without it the FIRST live turn pays the
        full BGE-M3/Chinese-CLIP weight load on the recall path. The hashing/
        descriptor fallbacks make this a no-op; every failure degrades to the
        normal lazy path."""
        try:
            self.text.encode(["memory warmup"])
        except Exception as exc:  # noqa: BLE001
            log.debug("memory warmup: text embedder not preloaded (%s)", exc)
        encode_tokens = getattr(self.text, "encode_tokens", None)
        if self.settings.memory_late_interaction and callable(encode_tokens):
            try:
                encode_tokens(["memory warmup"])  # loads the colbert head too
            except Exception as exc:  # noqa: BLE001 — late lane is optional
                log.debug("memory warmup: late interaction not preloaded (%s)", exc)
        encode_texts = getattr(self.image, "encode_texts", None)
        if callable(encode_texts):
            try:
                encode_texts(["memory warmup"])
            except Exception as exc:  # noqa: BLE001
                log.debug("memory warmup: image embedder not preloaded (%s)", exc)

    def drain(self, timeout: Optional[float] = 10.0) -> None:
        """Wait for queued AND in-flight jobs, honoring the supplied deadline."""
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and non-negative, or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._q.all_tasks_done:
            while self._q.unfinished_tasks:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("memory writer drain timed out")
                self._q.all_tasks_done.wait(remaining)

    def discard_session(self, conversation_id: str, lifetime: Any) -> None:
        """Remove queued payloads for a sealed session without draining others."""
        with self._lifecycle_lock, self._q.mutex:
            kept = [job for job in self._q.queue
                    if not (job.get("conv") == conversation_id and job.get("lifetime") is lifetime)]
            removed = len(self._q.queue) - len(kept)
            for job in self._q.queue:
                if job.get('conv') == conversation_id and job.get('lifetime') is lifetime:
                    self._release_payload(job)
            self._q.queue.clear()
            self._q.queue.extend(kept)
            self._q.unfinished_tasks -= removed
            if not self._q.unfinished_tasks:
                self._q.all_tasks_done.notify_all()
            self._q.not_full.notify_all()

    def forget(self, conversation_id: str) -> None:
        self._frames.pop(conversation_id, None)
        with self._seen_lock:
            self._seen_utterances.pop(conversation_id, None)
