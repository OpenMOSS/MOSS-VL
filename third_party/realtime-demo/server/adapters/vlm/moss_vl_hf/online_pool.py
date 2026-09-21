"""ONLINE VLM replica pool — the gateway's `rt.vlm` under VLM_DEPLOY=workers.

The realtime worker fleet (one HF vlm_worker process per online GPU). Its
`generate_stream` doubles as the offline-chat FALLBACK when the dedicated
sglang plane (`rt.vlm_offline`, adapters/vlm/moss_vl_sglang/adapter.py) is absent
or down — 1-GPU boxes and degraded states keep a working chat mode.

Implements the `VlmAdapter` protocol as a facade over one `WorkerVlmProxy` per
GPU, so `routers/sessions.py:_build_engines` (and the samp monkeypatch around
it) never changes. Capacity semantics: one live realtime session per replica;
`start_realtime_session` picks the lowest-index READY replica and raises
`NoFreeReplica` when all are busy/down — the router maps that to a clean 409.

`status()` keeps the `"loaded"` key (external scripts grep it) and adds
replicas/capacity/busy.
"""
from __future__ import annotations

import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from ....config import Settings
from ....gpu.placement import PlacementPlan
from ....logging_conf import get_logger
from ...base import VlmCaps
from .worker_proxy import WorkerVlmProxy, WorkerVlmSession

log = get_logger(__name__)

# replica lifecycle
STARTING = "starting"   # spawned, model not loaded yet
READY = "ready"         # loaded, no live session
BUSY = "busy"           # loaded, hosting the (single) live session
DOWN = "down"           # health checks failing; supervisor is respawning


class NoFreeReplica(RuntimeError):
    def __init__(self, capacity: int, busy: int):
        self.capacity = capacity
        self.busy = busy
        super().__init__(
            f"all {capacity} model replicas busy ({busy}/{capacity})"
            if capacity else "no model replicas available")


@dataclass
class _Replica:
    proxy: WorkerVlmProxy
    state: str = STARTING
    session: Optional[Any] = None  # the wrapped live session, while BUSY
    generation: int = 0            # bumped on respawn (stale-session guard)
    lock: threading.Lock = field(default_factory=threading.Lock)


class _PooledSession:
    """Delegates to a WorkerVlmSession; stop() releases the replica slot."""

    def __init__(self, pool: "VlmReplicaPool", index: int, generation: int,
                 inner: WorkerVlmSession):
        self._pool = pool
        self._index = index
        self._generation = generation
        self._inner = inner
        self.session_id = inner.session_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def stop(self, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        try:
            return self._inner.stop(timeout_seconds)
        finally:
            self._pool._release(
                self._index, self._generation,
                worker_dead=getattr(self._inner, "worker_transport_dead", False))


class VlmReplicaPool:
    def __init__(self, settings: Settings, plan: PlacementPlan):
        self.s = settings
        self.plan = plan
        self.caps = VlmCaps(modes=("online_streaming", "offline"))
        self._replicas: List[_Replica] = [
            _Replica(proxy=WorkerVlmProxy(spec, settings)) for spec in plan.workers]
        self._lock = threading.Lock()

    # ------------------------------------------------------------ introspection

    @property
    def capacity(self) -> int:
        return len(self._replicas)

    @property
    def busy(self) -> int:
        return sum(1 for r in self._replicas if r.state == BUSY)

    @property
    def replicas(self) -> List[_Replica]:
        return self._replicas

    def is_loaded(self) -> bool:
        return any(r.state in (READY, BUSY) for r in self._replicas)

    def status(self) -> Dict[str, Any]:
        replicas = []
        for r in self._replicas:
            st = r.proxy.status()
            st["state"] = r.state
            replicas.append(st)
        return {
            "loaded": self.is_loaded(),
            "deploy": "workers",
            "capacity": self.capacity,
            "busy": self.busy,
            "active_sessions": self.busy,
            "replicas": replicas,
            "modes": list(self.caps.modes),
        }

    # ------------------------------------------------------------ supervisor hooks

    def set_replica_health(self, index: int, health: Optional[Dict[str, Any]]) -> None:
        """Monitor callback: refresh cached health + derive the replica state.

        Never yanks BUSY→READY under a live session; DOWN is sticky until a
        healthy poll arrives (respawn), which also bumps `generation` so a
        stale session release can't free the fresh slot.
        """
        r = self._replicas[index]
        with self._lock:
            if health is None:
                if r.state != DOWN:
                    log.warning("VLM replica %d marked DOWN", index)
                r.state = DOWN
                r.session = None
                return
            r.proxy.cached_health = health
            if not health.get("loaded"):
                if r.state != BUSY:
                    r.state = STARTING
                return
            if r.state == DOWN:
                r.generation += 1
                log.info("VLM replica %d recovered (generation %d)", index, r.generation)
                r.state = READY
            elif r.state == BUSY:
                inner = r.session
                if inner is not None and not getattr(inner, "active", True):
                    # session died without a stop() (crash path) — reclaim. Bump
                    # the generation: the dead session's deferred stop() (grace
                    # GC fires up to ~45 s later) must not free/quarantine the
                    # slot after a NEW session has leased it.
                    log.info("VLM replica %d reclaimed from a dead session", index)
                    r.session = None
                    r.generation += 1
                    r.state = READY
            else:
                r.state = READY

    # ------------------------------------------------------------ VlmAdapter

    def load(self, model_path: str, gpu_id: int, hf_mode: str,
             attn_impl_override: Optional[str] = None) -> None:
        """Broadcast to every non-DOWN worker in parallel (blocking; via to_thread)."""
        targets = [r for r in self._replicas if r.state != DOWN]
        if not targets:
            raise RuntimeError("no live workers to load the model on")
        errors: List[str] = []

        def _load(r: _Replica) -> None:
            try:
                r.proxy.load(model_path, gpu_id, hf_mode, attn_impl_override)
                with self._lock:
                    if r.state != BUSY:
                        r.state = READY
            except Exception as exc:  # noqa: BLE001
                errors.append(f"worker {r.proxy.spec.worker_id}: {exc}")

        with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
            list(pool.map(_load, targets))
        if errors:
            raise RuntimeError("; ".join(errors))

    def start_realtime_session(self, **params: Any) -> _PooledSession:
        """Blocking (called via to_thread). Lowest-index READY replica wins."""
        with self._lock:
            picked: Optional[int] = None
            for i, r in enumerate(self._replicas):
                if r.state == READY:
                    picked = i
                    r.state = BUSY  # reserve before the (slow) HTTP call
                    break
            if picked is None:
                raise NoFreeReplica(self.capacity, self.busy)
            generation = self._replicas[picked].generation
        r = self._replicas[picked]
        try:
            inner = r.proxy.start_realtime_session(**params)
        except Exception:
            with self._lock:
                if r.state == BUSY and r.generation == generation:
                    r.state = READY
            raise
        session = _PooledSession(self, picked, generation, inner)
        with self._lock:
            r.session = inner
        log.info("session %s → VLM replica %d (gpu %d)",
                 inner.session_id, picked, r.proxy.spec.gpu_index)
        return session

    def _release(self, index: int, generation: int, worker_dead: bool = False) -> None:
        r = self._replicas[index]
        with self._lock:
            if r.generation != generation:
                return  # replica was respawned since; nothing to free
            r.session = None
            if r.state == BUSY:
                # a dead transport means the worker likely crashed — quarantine
                # the slot; the supervisor's next health poll (or respawn)
                # flips it back to READY
                r.state = DOWN if worker_dead else READY
        log.info("VLM replica %d released%s", index, " (worker dead — quarantined)" if worker_dead else "")

    async def generate_stream(self, req: Any) -> AsyncIterator[str]:
        """Offline chat → least-busy live worker (BUSY replicas are eligible —
        matches the single-GPU behavior where chat runs beside the realtime
        session). CAS media handles are resolved gateway-side (workers have no
        MediaStore)."""
        import asyncio

        with self._lock:
            candidates = [r for r in self._replicas if r.state == READY] or \
                         [r for r in self._replicas if r.state == BUSY]
        if not candidates:
            raise RuntimeError("no model replica available for chat")
        proxy = candidates[0].proxy
        resolved = await asyncio.to_thread(_resolve_media_handles, req)
        async for delta in proxy.generate_stream(resolved):
            yield delta


def _resolve_media_handles(req: Any) -> Any:
    """Replace image CAS handles (`sha256:<hex>`) in a ChatRequest with raw base64.

    Blocking (CAS reads hit disk) — call via to_thread. Data-URLs/base64
    payloads pass through untouched; the worker's `_decode_chat_image` handles
    plain base64 natively. VIDEO handles are deliberately left as handles:
    the worker resolves them to blob paths itself (persistence.media.
    resolve_blob_path is pure path math over the shared DATA_DIR — no
    MediaStore/index needed) and torchcodec decodes from the path, so the
    bytes never ride the JSON body.
    """
    from ....persistence.media import maybe_get_media_store, normalize_hash

    store = maybe_get_media_store()

    def resolve(payload: Any) -> Any:
        s = str(payload or "")
        if s.startswith("data:"):
            return payload
        hex_ = normalize_hash(s)
        if hex_ is None:
            return payload
        if store is None:
            raise RuntimeError("media store unavailable — cannot resolve media handle")
        return base64.b64encode(store.load_bytes(hex_)).decode("ascii")

    out = req.model_copy(deep=True) if hasattr(req, "model_copy") else req
    if getattr(out, "images", None):
        out.images = [resolve(s) for s in out.images]
    for message in getattr(out, "messages", []) or []:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                for key in ("media", "image", "data"):
                    if part.get(key):
                        part[key] = resolve(part[key])
                        break
    return out
