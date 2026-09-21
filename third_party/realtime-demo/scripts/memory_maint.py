"""Inspect runtime memory; deletion requires --apply --offline and creates a DB backup."""
import argparse
from contextlib import closing
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.config import Settings
from server.memory.frames import SessionFrames
from server.memory.store import MemoryStore


def inspect_database(path):
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        sessions = {r[0] for r in conn.execute('SELECT DISTINCT conversation_id FROM memory_items')}
        managed = {r[0] for r in conn.execute('SELECT conversation_id FROM memory_sessions')} if 'memory_sessions' in tables else set()
        return {'managed_sessions': len(managed), 'legacy_sessions': len(sessions - managed),
                'items': conn.execute('SELECT COUNT(*) FROM memory_items').fetchone()[0],
                'vectors': conn.execute('SELECT COUNT(*) FROM memory_vectors').fetchone()[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--offline', action='store_true', help='confirm all APIs using this database are stopped')
    parser.add_argument('--include-legacy', action='store_true', help='also delete pre-upgrade rows without ownership metadata')
    parser.add_argument('--vacuum', action='store_true', help='reclaim SQLite file space after cleanup')
    args = parser.parse_args()
    path = args.db.resolve()
    if args.apply and not args.offline:
        parser.error('--apply requires --offline; pre-upgrade APIs do not hold the new lock')
    print(json.dumps({'dry_run': not args.apply, **inspect_database(path)}))
    if not args.apply:
        return
    store = MemoryStore(replace(Settings(), memory_db_path=str(path)))
    store.frames = SessionFrames(path)
    store.open()
    try:
        backup = Path(str(path) + f'.backup-{time.time_ns()}')
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        try:
            with closing(sqlite3.connect(backup)) as destination:
                store._conn.backup(destination)
        except Exception:
            backup.unlink(missing_ok=True)
            raise
        print(json.dumps({'backup': str(backup)}))
        ids = store.cleanup_candidates(recover=True, include_legacy=args.include_legacy)
        for session_id in ids:
            store.delete_session(session_id)
        store.reclaim_space(vacuum=args.vacuum)
        print(json.dumps({'cleaned_sessions': len(ids), **store.refresh_storage_status()}))
    finally:
        store.close()


if __name__ == '__main__':
    main()
