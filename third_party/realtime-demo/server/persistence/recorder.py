"""History recorder — JSONL journal (source of truth) + SQLite index writer.

One daemon writer thread with a bounded payload queue and reserved finalization
bookkeeping owns every history write. Shutdown drains accepted work. Producers —
`SessionState.emit`'s record sink on the event loop, the chat router — only
`queue.put_nowait`, so recording can never stall captions or TTS.

Journal layout: `journal/YYYY/MM/<conversation_id>.jsonl`, one JSON object per
line. Lines are either raw session-plane events (the `JOURNAL_TYPES` subset —
deltas/audio are excluded: `response.text.done` already carries the full text,
and PCM is deliberately not persisted) or synthetic `history.*` records
(`history.open` / `history.turn` / `history.finalize`). `replay_file()` applies
either shape, so `scripts/history_prune.py --rebuild` reconstructs the index
from journals alone.
"""
from __future__ import annotations

import glob
import json
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

from ..config import Settings
from ..logging_conf import get_logger
from .store import IndexStore

log = get_logger(__name__)

# the semantic subset of server→client events worth journaling (turn-level; the
# distiller below derives `turns` rows from exactly these)
JOURNAL_TYPES = frozenset({
    "input.transcription.done",
    "input.text.done",
    "input.video.attached",
    "input.video.source.changed",
    "response.created",
    "response.text.done",
    "response.done",
    "session.updated",
    "memory.notice",
    "context.command",
})

_TITLE_MAX = 80


def _month_dir(created_at: float) -> Tuple[str, str]:
    t = time.gmtime(created_at)
    return f"{t.tm_year:04d}", f"{t.tm_mon:02d}"


class HistoryRecorder:
    def __init__(self, settings: Settings, index: IndexStore):
        self.settings = settings
        self.index = index
        self.journal_root = os.path.join(settings.data_dir, "journal")
        self._queue = queue.Queue(maxsize=max(1, settings.history_queue_items))
        self._admission_lock = threading.RLock()
        self._wakeup = threading.Event()
        self._queued_bytes = 0
        self._tracked = set()
        self._failed_cids = set()
        self._pending_fin = {}
        self._stats = {'rejected': 0, 'write_errors': 0}
        self._worker: Optional[threading.Thread] = None
        self._closed = threading.Event()
        # writer-thread state
        self._paths: Dict[str, str] = {}                 # cid → journal file
        self._resp_text: Dict[Tuple[str, str], str] = {} # (cid, rid) → final text

    # ------------------------------------------------------------------ lifecycle

    def open(self) -> None:
        if self._worker is not None:
            return
        self._closed.clear()
        os.makedirs(self.journal_root, exist_ok=True)
        self._worker = threading.Thread(target=self._loop, name="history-writer", daemon=True)
        self._worker.start()
        log.info("history recorder open: %s", self.journal_root)

    def close(self, timeout=None) -> None:
        """Flush + stop the writer (blocking — call via asyncio.to_thread)."""
        self._closed.set()
        self._wakeup.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            if self._worker.is_alive():
                raise TimeoutError('History writer is still flushing; archive lock retained')
            self._worker = None
        self._paths.clear()
        self._resp_text.clear()

    def status(self):
        with self._admission_lock:
            return {**self._stats, 'queue_bytes': self._queued_bytes,
                    'pending_sessions': len(self._tracked), 'incomplete_sessions': len(self._failed_cids)}

    def _enqueue(self, op, args):
        cid = args[0]
        size = len(json.dumps(args, ensure_ascii=False).encode('utf-8'))
        with self._admission_lock:
            if self._closed.is_set():
                return False
            fresh = op == 'conv' and cid not in self._tracked
            if (not isinstance(cid, str) or len(cid) > 256 or
                    (fresh and len(self._tracked) >= max(1, self.settings.history_pending_sessions))):
                self._stats['rejected'] += 1
                log.error('history recording admission limit; conversation not archived')
                return False
            if fresh:
                self._tracked.add(cid)
            if cid not in self._tracked or cid in self._failed_cids:
                return False
            if self._queue.full() or self._queued_bytes + size > max(1, self.settings.history_queue_bytes):
                self._stats['rejected'] += 1
                if fresh:
                    self._tracked.remove(cid)
                else:
                    self._failed_cids.add(cid)
                log.error('history queue budget exceeded; archive incomplete for %s', cid)
                return False
            self._queue.put_nowait((op, args, size))
            if op == 'conv':
                self._pending_fin.pop(cid, None)
            self._queued_bytes += size
            self._wakeup.set()
            return True

    # ------------------------------------------------------------------ producers (thread-safe, non-blocking)

    def open_conversation(self, conversation_id: str, kind: str,
                          config: Optional[Dict[str, Any]] = None,
                          created_at: Optional[float] = None) -> None:
        self._enqueue("conv", (conversation_id, kind, created_at or time.time(), config))

    def realtime_sink(self, conversation_id: str) -> Callable[[str, str], None]:
        """The `SessionState.record` sink: filter to JOURNAL_TYPES, then enqueue."""
        def sink(type_: str, text: str) -> None:
            if type_ in JOURNAL_TYPES:
                self._enqueue("event", (conversation_id, time.time(), text))
        return sink

    def record_turn(self, conversation_id: str, *, role: str, text: str,
                    source: str = "chat", ts: Optional[float] = None,
                    media_hashes: Sequence[str] = (),
                    metrics: Optional[Dict[str, Any]] = None) -> None:
        """Direct turn commit (the chat path journals `history.turn` lines)."""
        self._enqueue("turn", (conversation_id, {
            "type": "history.turn", "ts": ts or time.time(), "role": role,
            "source": source, "text": text, "media": list(media_hashes),
            "metrics": metrics or None,
        }))

    def finalize(self, conversation_id: str, end_reason: Optional[str] = None,
                 ended_at: Optional[float] = None) -> None:
        with self._admission_lock:
            if conversation_id in self._tracked and not self._closed.is_set():
                self._pending_fin[conversation_id] = (conversation_id, ended_at or time.time(), (end_reason or '')[:512])
                self._wakeup.set()

    # ------------------------------------------------------------------ deletion (called via asyncio.to_thread)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Remove a conversation from index AND journal (user-initiated delete).

        The journal is the source of truth — leaving the file would resurrect
        the conversation on rebuild. Blobs stay until the prune tool collects
        ref_count==0 media.
        """
        if not conversation_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in conversation_id):
            raise ValueError('Unsafe conversation ID')
        deleted = False
        pattern = os.path.join(self.journal_root, "*", "*", f"{conversation_id}.jsonl")
        for path in glob.glob(pattern):
            try:
                os.unlink(path)
                deleted = True
            except FileNotFoundError:
                pass
        deleted = self.index.delete_conversation(conversation_id) or deleted
        self._paths.pop(conversation_id, None)
        return deleted

    # ------------------------------------------------------------------ writer thread

    def _loop(self) -> None:
        while True:
            self._wakeup.wait()
            queued = False
            with self._admission_lock:
                try:
                    op, args, size = self._queue.get_nowait()
                    queued = True
                except queue.Empty:
                    if self._pending_fin:
                        cid = next(iter(self._pending_fin))
                        args = self._pending_fin.pop(cid)
                        op, size = 'fin', 0
                    elif self._closed.is_set():
                        self._tracked.clear()
                        self._failed_cids.clear()
                        return
                    else:
                        self._wakeup.clear()
                        continue
            try:
                if op == "conv":
                    self._do_conv(*args)
                elif op == "event":
                    self._do_event(*args)
                elif op == "turn":
                    self._do_turn(*args)
                elif op == "fin":
                    self._do_finalize(*args)
            except Exception as exc:  # noqa: BLE001
                with self._admission_lock:
                    self._stats['write_errors'] += 1
                    self._failed_cids.add(args[0])
                log.exception("history write failed (%s): %s", op, exc)
            finally:
                with self._admission_lock:
                    self._queued_bytes -= size
                    if op == 'fin':
                        self._tracked.discard(args[0])
                        self._failed_cids.discard(args[0])
                        self._paths.pop(args[0], None)
                        for key in [k for k in self._resp_text if k[0] == args[0]]:
                            self._resp_text.pop(key, None)
                if queued:
                    self._queue.task_done()
                args = None

    def _journal_path(self, cid: str, ts: float) -> str:
        path = self._paths.get(cid)
        if path is None:
            yyyy, mm = _month_dir(ts)
            d = os.path.join(self.journal_root, yyyy, mm)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{cid}.jsonl")
            self._paths[cid] = path
        return path

    def _append(self, cid: str, ts: float, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        with open(self._journal_path(cid, ts), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _do_conv(self, cid: str, kind: str, created_at: float, config: Optional[Dict[str, Any]]) -> None:
        self.index.upsert_conversation(
            cid, kind, created_at,
            config_json=json.dumps(config, ensure_ascii=False) if config else None)
        # idempotent header: chat threads re-open per request — journal once
        if os.path.exists(self._journal_path(cid, created_at)):
            return
        self._append(cid, created_at, {
            "type": "history.open", "ts": created_at, "conversation_id": cid,
            "kind": kind, "config": config or None,
        })

    def _do_event(self, cid: str, ts: float, text: str) -> None:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return
        obj["ts"] = ts
        self._append(cid, ts, obj)
        _apply_event(self.index, cid, obj, self._resp_text)

    def _do_turn(self, cid: str, record: Dict[str, Any]) -> None:
        self._append(cid, float(record.get("ts") or time.time()), record)
        _apply_turn(self.index, cid, record)

    def _do_finalize(self, cid: str, ended_at: float, end_reason: Optional[str]) -> None:
        if cid in self._failed_cids:
            end_reason = (end_reason or '') + '|archive_incomplete'
        self.index.finalize_conversation(cid, ended_at, end_reason)
        self._append(cid, ended_at, {
            "type": "history.finalize", "ts": ended_at, "end_reason": end_reason,
        })
        self._paths.pop(cid, None)
        # drop any never-flushed response buffers for this conversation
        for key in [k for k in self._resp_text if k[0] == cid]:
            self._resp_text.pop(key, None)


# ---------------------------------------------------------------------- shared distiller
# (used live by the writer thread AND by --rebuild's journal replay)


def _apply_event(index: IndexStore, cid: str,
                 obj: Dict[str, Any], resp_text: Dict[Tuple[str, str], str]) -> None:
    """Distill one journaled session-plane event into `turns` rows."""
    type_ = obj.get("type")
    ts = float(obj.get("ts") or time.time())
    seq = obj.get("seq")
    if type_ == "input.transcription.done":
        text = str(obj.get("text") or "").strip()
        if text:
            metrics = {k: obj[k] for k in ("asr_ms", "media_ts") if obj.get(k) is not None}
            index.insert_turn(cid, role="user", text=text, ts=ts, seq=seq,
                              source="asr", metrics=metrics or None)
            index.set_title_if_empty(cid, text[:_TITLE_MAX])
    elif type_ == "input.text.done":
        text = str(obj.get("text") or "").strip()
        if text:
            metrics = {"media_ts": obj["media_ts"]} if obj.get("media_ts") is not None else None
            index.insert_turn(cid, role="user", text=text, ts=ts, seq=seq,
                              source="typed", metrics=metrics)
            index.set_title_if_empty(cid, text[:_TITLE_MAX])
    elif type_ == "input.video.attached":
        # the session's source video (file-streaming mode) — a media-only turn
        # row so the blob is ref-counted and the replay UI can find it
        hex_ = str(obj.get("media") or "").split(":", 1)[-1]
        if hex_:
            metrics = {"duration_s": obj["duration_s"]} if obj.get("duration_s") is not None else None
            index.insert_turn(cid, role="user", text="", ts=ts, seq=seq,
                              source="video", metrics=metrics, media_hashes=[hex_])
            name = str(obj.get("name") or "").strip()
            if name:
                index.set_title_if_empty(cid, name[:_TITLE_MAX])
    elif type_ == "input.video.source.changed":
        # a (mid-session) source change — a media-less system row so the
        # transcript (live and replayed) shows its dedicated bubble; kind and
        # timing live in metrics, the optional CAS handle rides video.attach
        kind = str(obj.get("kind") or "").strip()
        if kind:
            metrics = {"kind": kind}
            for key in ("session_ts_start", "duration_s", "media_offset"):
                if obj.get(key) is not None:
                    metrics[key] = obj[key]
            name = str(obj.get("name") or "").strip()
            if name:
                metrics["name"] = name
            index.insert_turn(cid, role="user", text="", ts=ts, seq=seq,
                              source="source_change", metrics=metrics)
    elif type_ == "memory.notice":
        index.insert_turn(cid, role="system", text=str(obj.get("message") or ""), ts=ts, seq=seq,
                          source="memory_notice", metrics={"code": obj.get("code")})
    elif type_ == "response.text.done":
        rid = str(obj.get("response_id") or "")
        resp_text[(cid, rid)] = str(obj.get("text") or "")
    elif type_ == "response.done":
        rid = str(obj.get("response_id") or "")
        text = resp_text.pop((cid, rid), "").strip()
        if text:
            metrics = {k: obj[k] for k in ("ttft_ms", "ttfa_ms", "media_ts",
                                           "gen_started_at", "gen_ended_at")
                       if obj.get(k) is not None}
            index.insert_turn(cid, role="assistant", text=text, ts=ts, seq=seq,
                              source="vlm", stop_reason=obj.get("stop_reason"),
                              metrics=metrics or None)
    # response.created / session.updated are journaled for fidelity, no turn row


def _apply_turn(index: IndexStore, cid: str, record: Dict[str, Any]) -> None:
    text = str(record.get("text") or "").strip()
    if not text and not record.get("media"):
        return
    index.insert_turn(
        cid, role=str(record.get("role") or "user"), text=text,
        ts=float(record.get("ts") or time.time()),
        source=record.get("source"), metrics=record.get("metrics"),
        media_hashes=[h for h in (record.get("media") or []) if h],
    )
    if record.get("role") == "user" and text:
        index.set_title_if_empty(cid, text[:_TITLE_MAX])


# ---------------------------------------------------------------------- rebuild


def iter_journal_files(journal_root: str) -> Iterable[str]:
    yield from sorted(glob.glob(os.path.join(journal_root, "*", "*", "*.jsonl")))


def replay_file(index: IndexStore, path: str) -> int:
    """Replay one journal file into the index; returns applied line count.

    Tolerates a truncated trailing line (crash mid-append).
    """
    cid = os.path.splitext(os.path.basename(path))[0]
    resp_text: Dict[Tuple[str, str], str] = {}
    applied = 0
    opened = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail
            type_ = obj.get("type")
            ts = float(obj.get("ts") or 0.0)
            if type_ != "history.open" and not opened:
                # lost/truncated header — stub the row so turns still land
                index.upsert_conversation(
                    cid, "realtime" if cid.startswith("sess_") else "chat", ts or time.time())
                opened = True
            if type_ == "history.open":
                opened = True
                config = obj.get("config")
                index.upsert_conversation(
                    cid, str(obj.get("kind") or "chat"), ts or time.time(),
                    config_json=json.dumps(config, ensure_ascii=False) if config else None)
            elif type_ == "history.turn":
                _apply_turn(index, cid, obj)
            elif type_ == "history.finalize":
                index.finalize_conversation(cid, ts, obj.get("end_reason"))
            elif type_ in JOURNAL_TYPES:
                _apply_event(index, cid, obj, resp_text)
            applied += 1
    return applied
