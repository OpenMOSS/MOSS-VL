"""Recovery and retry for runtime memory, never for durable history archives."""
import threading
import math

from ..logging_conf import get_logger

log = get_logger(__name__)


class MemoryMaintenance:
    def __init__(self, store):
        self.store = store
        self._stop = threading.Event()
        self._thread = None

    def sweep(self, *, recover=False):
        cleaned = 0
        for session_id in self.store.cleanup_candidates(recover=recover):
            try:
                self.store.delete_session(session_id)
                cleaned += 1
            except Exception:
                log.exception('memory cleanup pending: %s', session_id)
        if self.store.frames is not None:
            self.store.frames.refresh_usage()
        self.store.reclaim_space()
        status = self.store.refresh_storage_status()
        if cleaned:
            log.info('memory maintenance cleaned %d session(s)', cleaned)
        return {'cleaned_sessions': cleaned, **status}

    def start(self):
        if self._thread is not None:
            return
        # The DB process lock excludes a second live API. Unknown pre-upgrade
        # rows have no owner ledger and require explicit offline migration.
        self.sweep(recover=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name='memory-maintenance')
        self._thread.start()

    def _run(self):
        interval = self.store.settings.memory_maintenance_interval_s
        if not math.isfinite(interval) or interval <= 0:
            interval = 30.0
        interval = max(1.0, interval)
        while not self._stop.wait(interval):
            try:
                self.sweep()
            except Exception:
                log.exception('memory maintenance failed; retrying on next interval')

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
