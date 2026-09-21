#!/usr/bin/env python
"""Maintenance for the durable history + media store (server/persistence/).

Both stores are archives — nothing auto-evicts. This tool is the only pruner:

  # delete media blobs no retained turn references (optionally: idle N+ days)
  .venv/bin/python scripts/history_prune.py --unreferenced --offline [--older-than 30]

  # drop whole conversations older than N days (index + journal; refs released)
  .venv/bin/python scripts/history_prune.py --drop-conversations --older-than 90 --offline

  # reconstruct index.db from journals + a blob scan (proves the index is derived)
  .venv/bin/python scripts/history_prune.py --rebuild --offline

Run with the backend STOPPED (the index writer would race). DATA_DIR /
HISTORY_DB_PATH env vars are honored (server/config.py). Add --dry-run to
preview, --vacuum to compact the DB afterwards.
"""
from __future__ import annotations

import argparse
import math
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.config import get_settings  # noqa: E402
from server.persistence.media import MediaStore, normalize_hash  # noqa: E402
from server.memory.store import MemoryStore  # noqa: E402
from server.persistence.recorder import HistoryRecorder, iter_journal_files, replay_file  # noqa: E402
from server.persistence.store import IndexStore  # noqa: E402
from server.persistence.retention import acquire_archive_lock


def prune_unreferenced(index: IndexStore, media: MediaStore,
                       older_than_days: float | None, dry_run: bool) -> int:
    memory_path = Path(media.settings.memory_db_path or Path(media.settings.data_dir) / 'memory.db').resolve()
    protected = set()
    if memory_path.exists():
        # Fail closed if memory references cannot be read. History ref_count
        # alone never proved that a legacy memory keyframe was unreferenced.
        with sqlite3.connect(memory_path.as_uri() + '?mode=ro', uri=True) as conn:
            protected = {normalize_hash(r[0]) for r in conn.execute(
                'SELECT DISTINCT media_hash FROM memory_items WHERE media_hash IS NOT NULL')}
    rows = index.unreferenced_media(
        older_than_s=older_than_days * 86400.0 if older_than_days else None)
    rows = [row for row in rows if row['hash'] not in protected]
    with closing(sqlite3.connect(Path(index.db_path).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        # Recheck actual references, not only the denormalized ref_count column.
        rows = [row for row in rows if not conn.execute(
            'SELECT 1 FROM turn_media WHERE hash=? LIMIT 1', (row['hash'],)).fetchone()]
    for row in rows:
        h = row["hash"]
        print(f"{'DRY ' if dry_run else ''}prune media {h[:12]}… "
              f"({row['kind']}, {row['bytes']} B, refs={row['ref_count']})")
        if not dry_run:
            media.delete_blob(h)
            index.delete_media_row(h)
    return len(rows)


def drop_conversations(index: IndexStore, recorder: HistoryRecorder,
                       older_than_days: float, dry_run: bool) -> int:
    if not math.isfinite(older_than_days) or older_than_days <= 0:
        raise ValueError('Retention days must be finite and positive')
    cutoff = time.time() - older_than_days * 86400.0
    victims = [c for c in index.list_conversations(limit=100000)
               if c['ended_at'] is not None and c['ended_at'] < cutoff]
    for c in victims:
        print(f"{'DRY ' if dry_run else ''}drop conversation {c['conversation_id']} "
              f"({c['kind']}, {c['turn_count']} turns, {c['title']!r})")
        if not dry_run:
            recorder.delete_conversation(c["conversation_id"])
    return len(victims)


def rebuild(settings) -> None:
    db = settings.history_db_path or os.path.join(settings.data_dir, "index.db")
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(db + suffix)
        except FileNotFoundError:
            pass
    index = IndexStore(settings)
    index.open()
    media = MediaStore(settings, index)
    media.open()
    blobs = media.rescan_blobs()
    print(f"rescanned {blobs} media blob(s)")
    files = applied = 0
    for path in iter_journal_files(os.path.join(settings.data_dir, "journal")):
        files += 1
        applied += replay_file(index, path)
    print(f"replayed {applied} journal line(s) from {files} file(s)")
    index.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unreferenced", action="store_true",
                    help="delete media with ref_count == 0")
    ap.add_argument("--drop-conversations", action="store_true",
                    help="delete conversations older than --older-than days")
    ap.add_argument("--rebuild", action="store_true",
                    help="reconstruct index.db from journals + blob scan")
    ap.add_argument("--older-than", type=float, default=None, metavar="DAYS")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vacuum", action="store_true")
    ap.add_argument("--offline", action="store_true", help="confirm APIs are stopped before modifying archives")
    args = ap.parse_args()
    if args.older_than is not None and (not math.isfinite(args.older_than) or args.older_than <= 0):
        ap.error('--older-than must be finite and positive')
    if not (args.unreferenced or args.drop_conversations or args.rebuild):
        ap.print_help()
        return 2

    settings = get_settings()
    if args.rebuild and args.dry_run:
        ap.error('--rebuild does not support --dry-run')
    if not args.dry_run and not args.offline:
        ap.error('destructive maintenance requires --offline; stop all APIs first')
    guard = None
    archive_guard = None
    try:
        if not args.dry_run:
            archive_guard = acquire_archive_lock(settings.history_db_path or os.path.join(settings.data_dir,'index.db'), exclusive=True)
            guard = MemoryStore(settings)
            guard.open()
        return run_prune(settings, args, ap)
    finally:
        if guard is not None:
            guard.close()
        if archive_guard is not None:
            archive_guard.close()


def run_prune(settings, args, ap):
    if args.rebuild:
        rebuild(settings)
        return 0
    if not (args.unreferenced or args.drop_conversations):
        ap.print_help()
        return 2
    if args.drop_conversations and args.older_than is None:
        ap.error("--drop-conversations requires --older-than DAYS")

    index = IndexStore(settings)
    index.open()
    media = MediaStore(settings, index)
    recorder = HistoryRecorder(settings, index)  # delete path is synchronous — no open()
    try:
        if args.drop_conversations:
            n = drop_conversations(index, recorder, args.older_than, args.dry_run)
            print(f"{n} conversation(s) {'would be ' if args.dry_run else ''}dropped")
        if args.unreferenced:
            n = prune_unreferenced(index, media, args.older_than, args.dry_run)
            print(f"{n} media blob(s) {'would be ' if args.dry_run else ''}pruned")
        if args.vacuum and not args.dry_run:
            with sqlite3.connect(index.db_path) as conn:
                conn.execute("VACUUM")
            print("vacuumed")
    finally:
        index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
