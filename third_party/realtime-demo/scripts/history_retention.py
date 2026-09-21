#!/usr/bin/env python
"""Preview archive retention; deletion requires --apply --offline."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.config import get_settings
from server.memory.store import MemoryStore
from server.persistence.store import IndexStore
from server.persistence.recorder import HistoryRecorder
from server.persistence.media import MediaStore
from server.persistence.retention import acquire_archive_lock, payload_bytes, plan_retention
from history_prune import prune_unreferenced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=float)
    parser.add_argument('--max-bytes', type=int, help='Journal/media payload target, not SQLite file allocation')
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.apply and (not args.offline or (args.days is None and args.max_bytes is None)):
        parser.error('Deletion requires --offline and an explicit retention policy')
    settings = get_settings()
    guard = memory = index = None
    try:
        if args.apply:
            db = settings.history_db_path or str(Path(settings.data_dir)/'index.db')
            guard = acquire_archive_lock(db, exclusive=True)
            memory = MemoryStore(settings)
            memory.open()
        plan = plan_retention(settings, days=args.days, max_bytes=args.max_bytes, limit=args.limit)
        print(json.dumps({'apply':args.apply, **plan}))
        if args.apply:
            index = IndexStore(settings)
            index.open()
            recorder = HistoryRecorder(settings, index)
            for cid in plan['conversations']:
                recorder.delete_conversation(cid)
            prune_unreferenced(index, MediaStore(settings, index), args.days, False)
            remaining = payload_bytes(settings.data_dir)
            print(json.dumps({'remaining_payload_bytes':remaining}))
            if args.max_bytes is not None and remaining > args.max_bytes:
                return 1
        return 0
    finally:
        if index is not None:
            index.close()
        if memory is not None:
            memory.close()
        if guard is not None:
            guard.close()


if __name__ == '__main__':
    raise SystemExit(main())
