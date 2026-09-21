"""Shared sidecar-pool adapter for all TTS providers.

One engine per sidecar/engine process; `acquire()` hands sessions the
least-loaded ready one, `release()` returns the lease (TtsSession on_close).
`self.engine` stays aliased to the first engine for the single-sidecar callers
(routers/speech.py, routers/sessions.py, server/samp) — with one sidecar the
pool behaves exactly like a bare engine.

Extracted verbatim from the original MossTtsNanoAdapter so every provider
(moss_tts_nano, vllm_omni, cosyvoice3, moss_tts_realtime, *_native) shares one
routing implementation; only `engine_cls` differs per provider.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from ....config import Settings
from ...base import TtsCaps


class SidecarPoolAdapter:
    """Pool of provider engines with least-loaded lease routing.

    Subclasses set `engine_cls` to a class constructed as
    `engine_cls(settings, base_url)` exposing the TtsEngine surface plus
    `.ready`, `.sample_rate`, `.channels`, `.start()`, `.status()`.
    """

    engine_cls: Any = None
    # True when the engine consumes incremental text (one warm session per turn,
    # item 4) — TtsSession drives streaming off the engine's supports_streaming.
    token_streaming_input: bool = False

    def __init__(self, settings: Settings, base_urls: Optional[list] = None):
        urls = list(base_urls) if base_urls else [settings.moss_tts_nano_base_url]
        self.engines = [self.engine_cls(settings, url) for url in urls]
        self.engine = self.engines[0]
        self._leases = [0] * len(self.engines)
        self._lock = threading.Lock()
        self.caps = TtsCaps(
            token_streaming_input=self.token_streaming_input,
            sample_rate=self.engine.sample_rate,
            channels=self.engine.channels,
        )

    def start(self) -> None:
        # Engines are independent sidecar processes (one per GPU) and start()
        # blocks on health-check + first-synthesis warmup (~3 min cold, ~20 s
        # warm each) — run them concurrently, not engine-by-engine (serialized,
        # 4 sidecars cost ~4 extra minutes of backend lifespan; 2026-08-07).
        # Each start() only mutates its own engine, so plain threads suffice.
        if len(self.engines) == 1:
            self.engine.start()
        else:
            threads = [
                threading.Thread(target=engine.start, name=f"tts-pool-start-{i}", daemon=True)
                for i, engine in enumerate(self.engines)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.caps = TtsCaps(False, self.engine.sample_rate, self.engine.channels)

    def status(self) -> Dict[str, Any]:
        status = self.engines[0].status()
        if len(self.engines) > 1:
            status["ready"] = any(e.ready for e in self.engines)
            status["sidecars"] = [
                {"base_url": e.base_url, "ready": e.ready, "leases": self._leases[i]}
                for i, e in enumerate(self.engines)]
        return status

    # ---- session routing (least-loaded ready sidecar) ----

    def acquire(self):
        """Lease an engine for one session; pair with release() (TtsSession
        on_close). Falls back to the primary engine when none report ready."""
        with self._lock:
            ready = [i for i, e in enumerate(self.engines) if e.ready]
            if not ready:
                return self.engine
            best = min(ready, key=lambda i: self._leases[i])
            self._leases[best] += 1
            return self.engines[best]

    def release(self, engine) -> None:
        with self._lock:
            for i, candidate in enumerate(self.engines):
                if candidate is engine and self._leases[i] > 0:
                    self._leases[i] -= 1
                    return
