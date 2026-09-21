"""SQLite index over the history journal + media blobs (WAL, FTS5 trigram).

The index is a *derived projection*: every row is reconstructible from the
JSONL journals (turn/conversation data) and a scan of `media/blobs`
(media rows). Losing or corrupting `index.db` is never data loss —
`scripts/history_prune.py --rebuild` regenerates it.

Threading: one writable connection guarded by a lock (writers are the
recorder's daemon thread and media ingest inside `asyncio.to_thread`); reads
open short-lived connections so WAL readers never block the writer. Nothing
here may be called on the event loop directly.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from ..config import Settings
from ..logging_conf import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  conversation_id TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,
  created_at  REAL NOT NULL,
  ended_at    REAL,
  title       TEXT,
  config_json TEXT,
  turn_count  INTEGER NOT NULL DEFAULT 0,
  end_reason  TEXT
);
CREATE INDEX IF NOT EXISTS conversations_by_time ON conversations(created_at DESC);

CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
  seq INTEGER,
  role TEXT NOT NULL,
  source TEXT,
  text TEXT,
  ts REAL NOT NULL,
  stop_reason TEXT,
  metrics_json TEXT
);
CREATE INDEX IF NOT EXISTS turns_by_conv ON turns(conversation_id, id);

-- trigram = substring semantics (matches the sidebar's .includes()) and works
-- for CJK, which unicode61 would lump into one token per run
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
  text, content='turns', content_rowid='id', tokenize='trigram');

CREATE TRIGGER IF NOT EXISTS turns_fts_ai AFTER INSERT ON turns BEGIN
  INSERT INTO turns_fts(rowid, text) VALUES (new.id, coalesce(new.text, ''));
END;
CREATE TRIGGER IF NOT EXISTS turns_fts_ad AFTER DELETE ON turns BEGIN
  INSERT INTO turns_fts(turns_fts, rowid, text) VALUES ('delete', old.id, coalesce(old.text, ''));
END;
CREATE TRIGGER IF NOT EXISTS turns_fts_au AFTER UPDATE ON turns BEGIN
  INSERT INTO turns_fts(turns_fts, rowid, text) VALUES ('delete', old.id, coalesce(old.text, ''));
  INSERT INTO turns_fts(rowid, text) VALUES (new.id, coalesce(new.text, ''));
END;

CREATE TABLE IF NOT EXISTS media (
  hash TEXT PRIMARY KEY,              -- bare hex digest (algo column names it)
  algo TEXT NOT NULL DEFAULT 'sha256',
  mime TEXT NOT NULL,
  kind TEXT NOT NULL,                 -- 'image' | 'video'
  bytes INTEGER NOT NULL,
  width INTEGER, height INTEGER, duration_s REAL,
  created_at REAL NOT NULL,
  last_access_at REAL,
  ref_count INTEGER NOT NULL DEFAULT 0,
  orig_name TEXT,
  thumb_path TEXT,                    -- relative to data_dir
  poster_path TEXT
);

CREATE TABLE IF NOT EXISTS turn_media (
  turn_id INTEGER NOT NULL REFERENCES turns(id),
  hash TEXT NOT NULL REFERENCES media(hash),
  ord INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (turn_id, hash)
);
"""


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _like_escape(q: str) -> str:
    return q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class IndexStore:
    """Owns `index.db`. All writes go through the internal lock."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_dir = settings.data_dir
        self.db_path = settings.history_db_path or os.path.join(self.data_dir, "index.db")
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn
        log.info("history index open: %s", self.db_path)

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            with self._lock:
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    conn.commit()
                finally:
                    conn.close()

    def _write(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("IndexStore is not open")
        return self._conn

    def _read_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ conversations

    def upsert_conversation(
        self, conversation_id: str, kind: str, created_at: float,
        config_json: Optional[str] = None, title: Optional[str] = None,
    ) -> None:
        with self._lock:
            conn = self._write()
            conn.execute(
                "INSERT INTO conversations(conversation_id, kind, created_at, config_json, title) "
                "VALUES (?,?,?,?,?) ON CONFLICT(conversation_id) DO NOTHING",
                (conversation_id, kind, created_at, config_json, title),
            )
            conn.commit()

    def finalize_conversation(
        self, conversation_id: str, ended_at: float, end_reason: Optional[str] = None,
    ) -> None:
        with self._lock:
            conn = self._write()
            conn.execute(
                "UPDATE conversations SET ended_at=?, end_reason=coalesce(?, end_reason) "
                "WHERE conversation_id=?",
                (ended_at, end_reason, conversation_id),
            )
            conn.commit()

    def set_title_if_empty(self, conversation_id: str, title: str) -> None:
        with self._lock:
            conn = self._write()
            conn.execute(
                "UPDATE conversations SET title=? WHERE conversation_id=? "
                "AND (title IS NULL OR title='')",
                (title, conversation_id),
            )
            conn.commit()

    # ------------------------------------------------------------------ turns

    def insert_turn(
        self, conversation_id: str, *, role: str, text: str, ts: float,
        seq: Optional[int] = None, source: Optional[str] = None,
        stop_reason: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None,
        media_hashes: Sequence[str] = (),
    ) -> int:
        metrics_json = json.dumps(metrics, ensure_ascii=False) if metrics else None
        with self._lock:
            conn = self._write()
            cur = conn.execute(
                "INSERT INTO turns(conversation_id, seq, role, source, text, ts, stop_reason, metrics_json) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (conversation_id, seq, role, source, text, ts, stop_reason, metrics_json),
            )
            turn_id = int(cur.lastrowid or 0)
            for i, h in enumerate(media_hashes):
                # skip handles that were never uploaded (FK would abort the turn)
                if conn.execute("SELECT 1 FROM media WHERE hash=?", (h,)).fetchone() is None:
                    continue
                changed = conn.execute(
                    "INSERT INTO turn_media(turn_id, hash, ord) VALUES (?,?,?) "
                    "ON CONFLICT(turn_id, hash) DO NOTHING",
                    (turn_id, h, i),
                ).rowcount
                if changed:
                    conn.execute("UPDATE media SET ref_count=ref_count+1 WHERE hash=?", (h,))
            conn.execute(
                "UPDATE conversations SET turn_count=turn_count+1 WHERE conversation_id=?",
                (conversation_id,),
            )
            conn.commit()
            return turn_id

    # ------------------------------------------------------------------ media

    def upsert_media(self, desc: Dict[str, Any]) -> None:
        with self._lock:
            conn = self._write()
            conn.execute(
                "INSERT INTO media(hash, algo, mime, kind, bytes, width, height, duration_s, "
                "created_at, last_access_at, orig_name, thumb_path, poster_path) "
                "VALUES (:hash,:algo,:mime,:kind,:bytes,:width,:height,:duration_s,"
                ":created_at,:created_at,:orig_name,:thumb_path,:poster_path) "
                "ON CONFLICT(hash) DO UPDATE SET last_access_at=excluded.created_at",
                desc,
            )
            conn.commit()

    def get_media(self, hash_: str) -> Optional[Dict[str, Any]]:
        conn = self._read_conn()
        try:
            row = conn.execute("SELECT * FROM media WHERE hash=?", (hash_,)).fetchone()
            return _row_to_dict(row) if row else None
        finally:
            conn.close()

    def touch_media(self, hash_: str) -> None:
        with self._lock:
            conn = self._write()
            conn.execute("UPDATE media SET last_access_at=? WHERE hash=?", (time.time(), hash_))
            conn.commit()

    def unreferenced_media(self, older_than_s: Optional[float] = None) -> List[Dict[str, Any]]:
        conn = self._read_conn()
        try:
            sql = "SELECT * FROM media WHERE ref_count<=0"
            args: list = []
            if older_than_s is not None:
                sql += " AND coalesce(last_access_at, created_at) < ?"
                args.append(time.time() - older_than_s)
            return [_row_to_dict(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()

    def delete_media_row(self, hash_: str) -> None:
        with self._lock:
            conn = self._write()
            conn.execute("DELETE FROM media WHERE hash=? AND ref_count<=0", (hash_,))
            conn.commit()

    # ------------------------------------------------------------------ queries (read path)

    def list_conversations(self, q: str = "", limit: int = 50, offset: int = 0,
                           kind: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self._read_conn()
        try:
            q = (q or "").strip()
            # optional kind facet ('chat' | 'realtime') — the sidebar shows the
            # two chat types in separate lists
            kind_sql = " AND kind=?" if kind else ""
            kind_args: tuple = (kind,) if kind else ()
            if not q:
                rows = conn.execute(
                    f"SELECT * FROM conversations WHERE 1=1{kind_sql} "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (*kind_args, limit, offset),
                ).fetchall()
            elif len(q) >= 3:
                # trigram FTS over turn text (quoted → no MATCH-syntax injection),
                # plus a LIKE over titles
                match = '"' + q.replace('"', '""') + '"'
                rows = conn.execute(
                    "SELECT * FROM conversations WHERE "
                    "(title LIKE ? ESCAPE '\\' OR conversation_id IN ("
                    "  SELECT t.conversation_id FROM turns_fts f JOIN turns t ON t.id=f.rowid "
                    f"  WHERE turns_fts MATCH ?)){kind_sql} "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (f"%{_like_escape(q)}%", match, *kind_args, limit, offset),
                ).fetchall()
            else:
                # trigram needs ≥3 chars — short queries fall back to LIKE
                like = f"%{_like_escape(q)}%"
                rows = conn.execute(
                    "SELECT * FROM conversations WHERE "
                    "(title LIKE ? ESCAPE '\\' OR conversation_id IN ("
                    f"  SELECT conversation_id FROM turns WHERE text LIKE ? ESCAPE '\\')){kind_sql} "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (like, like, *kind_args, limit, offset),
                ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        conn = self._read_conn()
        try:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
            return _row_to_dict(row) if row else None
        finally:
            conn.close()

    def get_transcript(self, conversation_id: str) -> List[Dict[str, Any]]:
        conn = self._read_conn()
        try:
            turns = [
                _row_to_dict(r)
                for r in conn.execute(
                    "SELECT * FROM turns WHERE conversation_id=? ORDER BY id", (conversation_id,)
                ).fetchall()
            ]
            for turn in turns:
                media = conn.execute(
                    "SELECT m.hash, m.mime, m.kind, m.width, m.height, m.duration_s, "
                    "       (m.thumb_path IS NOT NULL OR m.poster_path IS NOT NULL) AS has_thumb "
                    "FROM turn_media tm JOIN media m ON m.hash=tm.hash "
                    "WHERE tm.turn_id=? ORDER BY tm.ord",
                    (turn["id"],),
                ).fetchall()
                turn["media"] = [_row_to_dict(m) for m in media]
            return turns
        finally:
            conn.close()

    # ------------------------------------------------------------------ deletion (user-initiated)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            conn = self._write()
            exists = conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id=?", (conversation_id,)
            ).fetchone()
            if not exists:
                return False
            # one decrement per turn_media row (a blob may back several turns)
            conn.execute(
                "UPDATE media SET ref_count = ref_count - ("
                "  SELECT COUNT(*) FROM turn_media tm JOIN turns t ON t.id=tm.turn_id "
                "  WHERE t.conversation_id=? AND tm.hash=media.hash) "
                "WHERE hash IN ("
                "  SELECT tm.hash FROM turn_media tm JOIN turns t ON t.id=tm.turn_id "
                "  WHERE t.conversation_id=?)",
                (conversation_id, conversation_id),
            )
            conn.execute(
                "DELETE FROM turn_media WHERE turn_id IN "
                "(SELECT id FROM turns WHERE conversation_id=?)",
                (conversation_id,),
            )
            conn.execute("DELETE FROM turns WHERE conversation_id=?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE conversation_id=?", (conversation_id,))
            conn.commit()
            return True
