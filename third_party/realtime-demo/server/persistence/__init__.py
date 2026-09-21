"""Durable history + media store.

Two archives under `DATA_DIR` (default `<repo>/data`, on the shared FS that
survives pod restarts):

- **History** — append-only JSONL journal (`journal/YYYY/MM/<cid>.jsonl`, the
  source of truth) + a derived SQLite index (`index.db`: conversations / turns /
  FTS / media). The index is rebuildable via `scripts/history_prune.py --rebuild`.
- **Media** — content-addressed blob store (`media/blobs/<algo>/<ab>/<cd>/<hash>`)
  for uploaded images/videos: hash = identity → free dedup, immutable → HTTP
  `immutable` caching. Ingest is hardened (magic-byte sniff, allowlist, size cap,
  image re-encode strips EXIF/polyglots).

Both stores are durable (no auto-eviction); growth is bounded by the manual
prune tool only. Writes never run on the event loop: the recorder owns a daemon
writer thread (TtsSession pattern) and media ingest runs via `asyncio.to_thread`.
"""
from .media import MediaStore, maybe_get_media_store, set_media_store
from .recorder import HistoryRecorder, JOURNAL_TYPES
from .store import IndexStore

__all__ = [
    "IndexStore",
    "MediaStore",
    "HistoryRecorder",
    "JOURNAL_TYPES",
    "set_media_store",
    "maybe_get_media_store",
]
