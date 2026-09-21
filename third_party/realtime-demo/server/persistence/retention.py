"""Offline, opt-in archive retention. Preview never modifies archive content."""
import fcntl
import math
import os
from pathlib import Path
import sqlite3
import time


def acquire_archive_lock(db_path, *, exclusive=False):
    path = Path(str(db_path) + '.archive.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    try:
        fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError('Archive is in use; stop its APIs before retention maintenance')
    return stream


def payload_bytes(data_dir):
    total = 0
    for name in ('journal', 'media'):
        root = Path(data_dir) / name
        if root.is_symlink():
            raise ValueError('Archive root must not be a symlink')
        if root.exists():
            for path in root.rglob('*'):
                if path.is_symlink():
                    raise ValueError('Archive files must not be symlinks')
                if path.is_file():
                    total += path.stat().st_size
    return total


def plan_retention(settings, *, days=None, max_bytes=None, limit=1000, now=None):
    if days is not None and (not math.isfinite(days) or days <= 0):
        raise ValueError('Retention days must be finite and positive')
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError('Archive payload budget must be positive')
    if limit <= 0:
        raise ValueError('Batch limit must be positive')
    now = time.time() if now is None else now
    db = Path(settings.history_db_path or Path(settings.data_dir)/'index.db')
    total = payload_bytes(settings.data_dir)
    result = dict(payload_bytes=total, projected_payload_bytes=total, conversations=[],
                  batch_limited=False, budget_unmet=False)
    if not db.exists() or (days is None and max_bytes is None):
        return result
    protected = set()
    memory = Path(settings.memory_db_path or Path(settings.data_dir)/'memory.db')
    if memory.exists():
        from .media import normalize_hash
        with sqlite3.connect(memory.resolve().as_uri()+'?mode=ro', uri=True) as conn:
            protected = {normalize_hash(r[0]) for r in conn.execute('SELECT media_hash FROM memory_items WHERE media_hash IS NOT NULL')}
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro', uri=True) as conn:
        conn.execute('BEGIN')
        rows = conn.execute('SELECT conversation_id,ended_at FROM conversations WHERE ended_at IS NOT NULL ORDER BY ended_at,conversation_id LIMIT ?', (limit+1,)).fetchall()
        freed_hashes, dropped_refs = set(), {}
        for cid, ended in rows:
            over_age = days is not None and ended < now-days*86400
            over_size = max_bytes is not None and result['projected_payload_bytes'] > max_bytes
            if not (over_age or over_size):
                break
            if len(result['conversations']) >= limit:
                result['batch_limited'] = True
                break
            if not cid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in cid):
                raise ValueError('Unsafe conversation ID in archive')
            journals = list((Path(settings.data_dir)/'journal').glob(f'*/*/{cid}.jsonl'))
            reclaim = sum(p.stat().st_size for p in journals)
            refs = conn.execute('SELECT m.hash,m.bytes,m.ref_count,COUNT(*) FROM turn_media tm JOIN turns t ON t.id=tm.turn_id JOIN media m ON m.hash=tm.hash WHERE t.conversation_id=? GROUP BY m.hash', (cid,)).fetchall()
            for digest, size, refs_total, refs_here in refs:
                dropped_refs[digest] = dropped_refs.get(digest, 0)+refs_here
                if digest not in protected and digest not in freed_hashes and dropped_refs[digest] >= refs_total:
                    reclaim += size
                    freed_hashes.add(digest)
            result['conversations'].append(cid)
            result['projected_payload_bytes'] = max(0, result['projected_payload_bytes']-reclaim)
    result['budget_unmet'] = max_bytes is not None and result['projected_payload_bytes'] > max_bytes
    return result
