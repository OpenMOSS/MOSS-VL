"""Session-private temporary frames, deliberately outside the history CAS."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import uuid
import threading

from .store import MemoryBudgetExceeded


_HANDLE = re.compile(r"memory:([0-9a-f]{64}):([0-9a-f]{64})\Z")


class SessionFrames:
    def __init__(self, db_path, settings=None):
        self.root = Path(str(db_path) + '.frames')
        self.settings = settings
        self._lock = threading.RLock()
        self._bytes = None
        self._total_bytes = 0

    def refresh_usage(self):
        with self._lock:
            if self._bytes is None:
                usage = {}
                if self.root.is_symlink():
                    raise ValueError('memory frame root must not be a symlink')
                if self.root.exists():
                    for directory in self.root.iterdir():
                        if directory.is_symlink():
                            raise ValueError('memory frame directory must not be a symlink')
                        if directory.is_dir():
                            usage[directory.name] = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
                self._bytes = usage
                self._total_bytes = sum(usage.values())
            return self._total_bytes

    def usage(self):
        return self._total_bytes

    def _directory(self, session_id):
        return self.root / hashlib.sha256(session_id.encode('utf-8')).hexdigest()

    def put(self, session_id, data):
        with self._lock:
            return self._put(session_id, data)

    def _put(self, session_id, data):
        directory = self._directory(session_id)
        if self.root.is_symlink() or directory.is_symlink():
            raise ValueError('memory frame directory must not be a symlink')
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.mkdir(mode=0o700, exist_ok=True)
        digest = hashlib.sha256(data).hexdigest()
        target = directory / (digest + '.jpg')
        if target.is_symlink():
            raise ValueError('memory frame must not be a symlink')
        self.refresh_usage()
        old = target.stat().st_size if target.exists() else 0
        delta = len(data) - old
        session_limit = max(1, getattr(self.settings, 'memory_session_frame_bytes', 128*1024**2))
        total_limit = max(1, getattr(self.settings, 'memory_total_frame_bytes', 512*1024**2))
        if self._bytes.get(directory.name, 0) + delta > session_limit or self.usage() + delta > total_limit:
            raise MemoryBudgetExceeded('frame_bytes')
        part = directory / (uuid.uuid4().hex + '.part')
        try:
            with os.fdopen(os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
                stream.write(data)
            part.replace(target)
            self._bytes[directory.name] = self._bytes.get(directory.name, 0) + delta
            self._total_bytes += delta
        finally:
            part.unlink(missing_ok=True)
        return f'memory:{directory.name}:{digest}'

    def load(self, session_id, handle):
        match = _HANDLE.fullmatch(handle or '')
        directory = self._directory(session_id)
        if not match or match[1] != directory.name:
            raise ValueError('frame does not belong to this session')
        path = directory / (match[2] + '.jpg')
        if self.root.is_symlink() or directory.is_symlink() or path.is_symlink():
            raise ValueError('memory frame path must not be a symlink')
        return path.read_bytes()

    def delete_session(self, session_id):
        with self._lock:
            self._delete_session(session_id)

    def _delete_session(self, session_id):
        directory = self._directory(session_id)
        if self.root.is_symlink() or directory.is_symlink():
            raise ValueError('memory frame directory must not be a symlink')
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        if self._bytes is not None:
            self._total_bytes -= self._bytes.pop(directory.name, 0)
