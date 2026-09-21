"""Cross-process VLM adapter: gateway-side proxy for one ONLINE vlm_worker.

`WorkerVlmProxy` implements the `VlmAdapter` protocol for a single worker
process; `WorkerVlmSession` implements `VlmRealtimeSession` over the worker's
duplex session WS. Design rules:

- `status()`/`is_loaded()` NEVER do I/O — they read the health blob the
  supervisor's monitor task refreshes (safe from the event loop).
- The outbound WS queue is bounded with drop-oldest-PURE-FRAME overflow,
  mirroring `RealtimeSession._put_event`: a stalled worker can never
  back-pressure a gateway WS handler; prompts/turn_end/stop never drop.
- A dropped WS or a worker `{"t":"ended"}` flips `active=False`, which the
  orchestrator's existing `_mark_vlm_dead` path picks up from `poll_output`.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple

import requests

from ....config import Settings
from ....gpu.placement import WorkerSpec
from ....logging_conf import get_logger
from ....vlm_worker import protocol as wp
from ...base import OutputBatch, VlmCaps

log = get_logger(__name__)

SESSION_START_TIMEOUT_S = 30.0
SESSION_STOP_TIMEOUT_S = 15.0
OUT_QUEUE_SIZE = 256  # ≈4 min of 1 fps frames; prompts never drop


def _encode_jpeg(image: Any) -> bytes:
    """bytes pass through; PIL images (tests, legacy callers) are encoded."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    from io import BytesIO

    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


class _OutboundQueue:
    """Bounded send queue with drop-oldest-pure-frame overflow."""

    def __init__(self, maxsize: int = OUT_QUEUE_SIZE):
        self._items: Deque[Tuple[bool, Any]] = deque()  # (is_pure_frame, wire msg)
        self._maxsize = maxsize
        self._cond = threading.Condition()
        self.frames_dropped = 0
        self.closed = False

    def put(self, message: Any, pure_frame: bool = False) -> None:
        with self._cond:
            if self.closed:
                return
            if len(self._items) >= self._maxsize:
                for i, (is_frame, _) in enumerate(self._items):
                    if is_frame:
                        del self._items[i]
                        self.frames_dropped += 1
                        break
                else:
                    # no droppable frame (queue full of control msgs — should
                    # never happen at sane sizes); drop the oldest anyway
                    self._items.popleft()
            self._items.append((pure_frame, message))
            self._cond.notify()

    def get(self, timeout: float = 0.5) -> Optional[Any]:
        with self._cond:
            if not self._items:
                self._cond.wait(timeout)
            if not self._items:
                return None
            return self._items.popleft()[1]

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._items.clear()
            self._cond.notify_all()


class WorkerVlmSession:
    """VlmRealtimeSession over the worker's /session/{sid}/io WebSocket."""

    def __init__(self, base_url: str, session_id: str, create_info: Dict[str, Any]):
        self.session_id = session_id
        self.gpu_id = -1  # filled from status frames (worker-local index)
        self.created_at = time.time()
        self._base_url = base_url.rstrip("/")
        self._active = True
        self._ended_reason: Optional[str] = None
        self._out = _OutboundQueue()
        self._inbound: Deque[Dict[str, Any]] = deque()
        self._inbound_event = threading.Event()
        self._status: Dict[str, Any] = {
            "session_id": session_id, "active": True,
            "kv_budget_tokens": create_info.get("kv_budget_tokens"),
            "est_max_seconds": create_info.get("est_max_seconds"),
        }
        self._status_lock = threading.Lock()
        self._stopped = threading.Event()

        ws_url = self._base_url.replace("http://", "ws://", 1) + f"/session/{session_id}/io"
        from websockets.sync.client import connect

        self._ws = connect(ws_url, max_size=None,
                           open_timeout=SESSION_START_TIMEOUT_S)
        self._send_thread = threading.Thread(
            target=self._send_loop, name=f"vlmproxy-send-{session_id[:8]}", daemon=True)
        self._recv_thread = threading.Thread(
            target=self._recv_loop, name=f"vlmproxy-recv-{session_id[:8]}", daemon=True)
        self._send_thread.start()
        self._recv_thread.start()

    # ------------------------------------------------------------ io threads

    def _send_loop(self) -> None:
        try:
            while not self._stopped.is_set():
                message = self._out.get(timeout=0.5)
                if message is None:
                    continue
                self._ws.send(message)
        except Exception as exc:  # noqa: BLE001 — socket died; recv loop reports
            if not self._stopped.is_set():
                log.warning("worker session %s send loop ended: %s", self.session_id, exc)
        finally:
            self._mark_ended("ws_closed")

    def _recv_loop(self) -> None:
        try:
            while True:
                raw = self._ws.recv()
                if isinstance(raw, (bytes, bytearray)):
                    continue  # worker→gateway is text-only today
                event = json.loads(raw)
                t = event.get("t")
                if t == wp.T_OUT:
                    if isinstance(event.get("status"), dict):
                        self._update_status(event["status"])
                    self._inbound.append(event)
                    self._inbound_event.set()
                    if not event.get("active", True):
                        self._mark_ended("stopped")
                elif t == wp.T_STATUS:
                    status = dict(event.get("status") or {})
                    if event.get("kv") is not None:
                        status["kv"] = event["kv"]
                    if event.get("gpu") is not None:
                        status["gpu"] = event["gpu"]
                    self._update_status(status)
                elif t == wp.T_KV_WARNING:
                    log.warning("session %s KV warning: used=%s/%s tokens (~%ss left)",
                                self.session_id, event.get("used_tokens"),
                                event.get("budget_tokens"), event.get("est_seconds_left"))
                    self._update_status({"kv_warning": True})
                elif t == wp.T_ENDED:
                    self._mark_ended(str(event.get("reason") or "stopped"))
        except Exception as exc:  # noqa: BLE001 — includes ConnectionClosed
            if not self._stopped.is_set():
                log.warning("worker session %s recv loop ended: %s", self.session_id, exc)
        finally:
            self._mark_ended("ws_closed")

    def _update_status(self, status: Dict[str, Any]) -> None:
        with self._status_lock:
            self._status.update(status)
            gpu = status.get("gpu_id")
            if isinstance(gpu, int):
                self.gpu_id = gpu

    def _mark_ended(self, reason: str) -> None:
        if not self._active:
            return
        self._active = False
        self._ended_reason = self._ended_reason or reason
        self._inbound_event.set()  # wake any poll_output waiter

    # ------------------------------------------------------------ protocol API

    @property
    def active(self) -> bool:
        return self._active

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError(f"Realtime session is no longer active: {self.session_id}")

    def put_frame(self, image: Any, timestamp: Optional[float] = None,
                  byte_size: Optional[int] = None) -> Dict[str, Any]:
        self._ensure_active()
        jpeg = _encode_jpeg(image)
        header = {"t": wp.T_FRAME, "ts": timestamp, "size": byte_size or len(jpeg)}
        self._out.put(wp.pack_msg(header, jpeg), pure_frame=True)
        return self.status()

    def put_prompt(self, prompt: str) -> Dict[str, Any]:
        self._ensure_active()
        self._out.put(json.dumps({"t": wp.T_PROMPT, "text": prompt}, ensure_ascii=False))
        return self.status()

    def put_prompt_frame(self, prompt: str, image: Any, timestamp: Optional[float] = None,
                         byte_size: Optional[int] = None, drop_pending: bool = True) -> Dict[str, Any]:
        self._ensure_active()
        jpeg = _encode_jpeg(image)
        header = {"t": wp.T_PROMPT_FRAME, "text": prompt, "ts": timestamp,
                  "size": byte_size or len(jpeg), "drop_pending": bool(drop_pending)}
        self._out.put(wp.pack_msg(header, jpeg))  # carries a prompt — never dropped
        return self.status()

    def request_turn_end(self) -> Dict[str, Any]:
        self._ensure_active()
        self._out.put(json.dumps({"t": wp.T_TURN_END}))
        return self.status()

    def poll_output(self, timeout_seconds: float = 0.0, max_items: int = 128) -> OutputBatch:
        if not self._inbound and timeout_seconds > 0 and self._active:
            self._inbound_event.wait(timeout_seconds)
        self._inbound_event.clear()
        chunks: List[str] = []
        chunk_events: List[Dict[str, Any]] = []
        while self._inbound and len(chunks) < max_items:
            event = self._inbound.popleft()
            chunks.extend(str(c) for c in event.get("chunks") or [])
            chunk_events.extend(event.get("chunk_events") or [])
        return OutputBatch(active=self._active, chunks=chunks,
                           chunk_events=chunk_events, status=self.status())

    def status(self) -> Dict[str, Any]:
        with self._status_lock:
            snapshot = dict(self._status)
        snapshot["active"] = self._active
        snapshot["proxy_frames_dropped"] = self._out.frames_dropped
        if self._ended_reason:
            snapshot["ended_reason"] = self._ended_reason
        return snapshot

    @property
    def worker_transport_dead(self) -> bool:
        """True when the session ended because the worker WS dropped (crash) —
        the replica pool quarantines the slot until a health poll clears it."""
        return self._ended_reason == "ws_closed"

    def stop(self, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        if not self._stopped.is_set():
            self._stopped.set()
            # a caller-initiated stop is "stopped" unless the transport already
            # died — the imminent ws.close must not race recv into "ws_closed"
            self._mark_ended("stopped")
            try:
                self._ws.send(json.dumps({"t": wp.T_STOP}))
            except Exception:  # noqa: BLE001
                pass
            try:
                requests.delete(f"{self._base_url}/session/{self.session_id}",
                                timeout=min(timeout_seconds, SESSION_STOP_TIMEOUT_S))
            except requests.RequestException as exc:
                log.warning("worker DELETE /session/%s failed: %s", self.session_id, exc)
            self._out.close()
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._send_thread.join(timeout=2.0)
            self._recv_thread.join(timeout=2.0)
        return self.status()


class WorkerVlmProxy:
    """VlmAdapter facade for ONE worker process."""

    def __init__(self, spec: WorkerSpec, settings: Settings):
        self.spec = spec
        self.s = settings
        self.caps = VlmCaps(modes=("online_streaming", "offline"))
        self.base_url = spec.base_url
        # refreshed by the supervisor monitor (and health-gate); read-only here
        self.cached_health: Dict[str, Any] = {}

    # ---- health (I/O — supervisor threads only, never the event loop) ----

    def fetch_health(self, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=timeout)
            if resp.ok:
                self.cached_health = resp.json()
                return self.cached_health
        except requests.RequestException:
            pass
        return None

    # ---- VlmAdapter protocol (loop-safe: cached reads only) ----

    def is_loaded(self) -> bool:
        return bool(self.cached_health.get("loaded"))

    def status(self) -> Dict[str, Any]:
        health = self.cached_health
        return {
            "loaded": bool(health.get("loaded")),
            "worker_id": self.spec.worker_id,
            "gpu_index": self.spec.gpu_index,
            "base_url": self.base_url,
            "model_path": health.get("model_path", ""),
            "hf_mode": health.get("hf_mode", ""),
            "attn_impl": health.get("attn_impl"),
            "kv": health.get("kv"),
            "gpu": health.get("gpu"),
            "session": health.get("session"),
        }

    def load(self, model_path: str, gpu_id: int, hf_mode: str,
             attn_impl_override: Optional[str] = None) -> None:
        """Blocking (call via to_thread). gpu_id is ignored — the worker's GPU
        is pinned by CUDA_VISIBLE_DEVICES at spawn."""
        resp = requests.post(
            f"{self.base_url}/load",
            json={"model_path": model_path, "hf_mode": hf_mode,
                  "attn_impl": attn_impl_override or self.spec.attn_impl},
            timeout=self.s.vlm_health_timeout_s)
        if not resp.ok:
            raise RuntimeError(f"worker {self.spec.worker_id} load failed "
                               f"({resp.status_code}): {resp.text[:300]}")
        self.cached_health = resp.json()

    def start_realtime_session(self, **params: Any) -> WorkerVlmSession:
        """Blocking (called via to_thread from _build_engines)."""
        resp = requests.post(f"{self.base_url}/session", json=_jsonable(params),
                             timeout=SESSION_START_TIMEOUT_S)
        if resp.status_code == 409:
            raise RuntimeError("A realtime session is already running on this model")
        if resp.status_code != 201:
            raise RuntimeError(f"worker {self.spec.worker_id} session create failed "
                               f"({resp.status_code}): {resp.text[:300]}")
        info = resp.json()
        return WorkerVlmSession(self.base_url, info["session_id"], info)

    async def generate_stream(self, req: Any) -> AsyncIterator[str]:
        """Offline chat: relay the worker's SSE stream (aiohttp, event-loop side)."""
        import aiohttp

        payload = req.model_dump() if hasattr(req, "model_dump") else dict(req)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=300)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.post(f"{self.base_url}/chat/stream", json=payload) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"worker chat stream failed ({resp.status}): "
                                       f"{(await resp.text())[:300]}")
                async for line in resp.content:
                    text = line.decode("utf-8").strip()
                    if not text.startswith("data:"):
                        continue
                    event = json.loads(text[5:].strip())
                    etype = event.get("type")
                    if etype == "generation_delta":
                        yield str(event.get("delta") or "")
                    elif etype == "generation_error":
                        raise RuntimeError(str(event.get("message") or "generation failed"))
                    elif etype == "generation_end":
                        return


def _jsonable(params: Dict[str, Any]) -> Dict[str, Any]:
    """start_realtime_session kwargs are all scalars/None — assert that here so
    a future non-serializable param fails loudly at the seam, not in requests."""
    for key, value in params.items():
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"non-JSON session param {key!r}: {type(value).__name__}")
    return params
