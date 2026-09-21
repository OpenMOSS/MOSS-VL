"""OFFLINE chat adapter — client pool over the dedicated sglang sidecars.

The offline half of the VlmAdapter protocol (`generate_stream` only): one
`sglang.launch_server` process per offline placement GPU (spawned/monitored by
`server/gpu/supervisor.py:SglangSidecarSupervisor`), each serving the OFFLINE
checkpoint via the fnlp-vision sglang fork's native moss_vl implementation.
`routers/chat.py` prefers this pool and falls back to the online worker fleet
(`rt.vlm`) whenever `is_loaded()` is False — degraded boxes keep a working
chat mode.

Request flow (board `_infer_via_sglang` parity, over HTTP instead of the
embedded Engine): normalize ChatRequest messages → flatten content to strings
with inline `<|image|>`/`<|video|>` placeholders → `apply_chat_template(...,
add_generation_prompt=True)` with the offline ckpt's own template → POST the
prompt + media (blob paths / base64) to the sidecar's native `/generate` with
`stream: true` → re-yield text deltas (sglang events carry CUMULATIVE text).

Health/state reads are cached (supervisor-refreshed) and loop-safe; the only
event-loop I/O is the streaming POST itself.
"""
from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import aiohttp

from ....config import Settings
from ....gpu.placement import OfflineSpec, PlacementPlan
from ....logging_conf import get_logger
from ...base import VlmCaps

log = get_logger(__name__)

# replica states (simpler than the online pool: sglang continuous-batches, so
# there is no BUSY — a replica is either serving or it isn't)
STARTING = "starting"
UP = "up"
DOWN = "down"

CONNECT_TIMEOUT_S = 10.0
# video prefill (decode + ViT over many frames) can stall the stream for a while
SOCK_READ_TIMEOUT_S = 300.0

# MOSS-VL vision placeholders (board `_flatten_content_with_vision_tokens`,
# MOSS branch — the fork's multimodal processor expands these)
IMAGE_TOKEN = "<|image|>"
VIDEO_TOKEN = "<|video|>"


class SglangReplica:
    """One sidecar. Blocking fetches are for supervisor threads ONLY."""

    def __init__(self, spec: OfflineSpec):
        self.spec = spec
        self.state = STARTING
        self.cached_health: Optional[Dict[str, Any]] = None
        self.inflight = 0  # in-flight generate_stream calls (least-busy pick)

    def _get_json(self, path: str, timeout: float) -> Optional[Dict[str, Any]]:
        try:
            with urllib.request.urlopen(self.spec.base_url + path, timeout=timeout) as resp:
                if resp.status != 200:
                    return None
                body = resp.read()
                return json.loads(body) if body.strip() else {"ok": True}
        except Exception:  # noqa: BLE001 — any failure = not healthy
            return None

    def fetch_health(self, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """GET /health (cheap liveness). 200 → healthy."""
        return self._get_json("/health", timeout)

    def fetch_health_generate(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """GET /health_generate — forces a real decode, so flashinfer JIT and
        warmup are done before the replica is marked UP."""
        return self._get_json("/health_generate", timeout)

    def fetch_model_info(self, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """GET /get_model_info — adoption identity check (shared boxes run
        other people's sglang servers; never adopt a foreign model)."""
        return self._get_json("/get_model_info", timeout)


class SglangOfflinePool:
    """VlmAdapter (offline modes only) over N sglang sidecars."""

    def __init__(self, settings: Settings, plan: PlacementPlan):
        self.s = settings
        self.plan = plan
        self.caps = VlmCaps(modes=("offline",))
        self._replicas: List[SglangReplica] = [SglangReplica(spec) for spec in plan.offline]
        self._lock = threading.Lock()
        # chat-template owner (processor or tokenizer), loaded lazily off-loop —
        # template-only, no weights/GPU touched in the gateway
        self._template_owner: Optional[Any] = None
        self._template_lock: Optional[asyncio.Lock] = None

    # ------------------------------------------------------------ introspection

    @property
    def capacity(self) -> int:
        return len(self._replicas)

    @property
    def replicas(self) -> List[SglangReplica]:
        return self._replicas

    def is_loaded(self) -> bool:
        return any(r.state == UP for r in self._replicas)

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": self.is_loaded(),
            "provider": "sglang",
            "deploy": "sidecars",
            "model_path": self.s.offline_model_path,
            "capacity": self.capacity,
            "replicas": [
                {"replica_id": r.spec.replica_id, "gpu": r.spec.gpu_index,
                 "port": r.spec.port, "state": r.state, "inflight": r.inflight}
                for r in self._replicas],
            "modes": list(self.caps.modes),
        }

    # ------------------------------------------------------------ supervisor hook

    def set_replica_health(self, index: int, health: Optional[Dict[str, Any]]) -> None:
        r = self._replicas[index]
        with self._lock:
            if health is None:
                if r.state != DOWN:
                    log.warning("offline sglang replica %d marked DOWN", index)
                r.state = DOWN
                r.cached_health = None
                return
            if r.state != UP:
                log.info("offline sglang replica %d UP (gpu %d :%d)",
                         index, r.spec.gpu_index, r.spec.port)
            r.state = UP
            r.cached_health = health

    # ------------------------------------------------------------ VlmAdapter (unsupported half)

    def load(self, model_path: str, gpu_id: int, hf_mode: str,
             attn_impl_override: Optional[str] = None) -> None:
        raise RuntimeError(
            "the offline sglang pool loads its model at sidecar spawn "
            "(OFFLINE_MODEL_PATH) — POST /api/models/load targets the online pool")

    def start_realtime_session(self, **params: Any) -> Any:
        raise RuntimeError("the offline sglang adapter has no realtime mode")

    # ------------------------------------------------------------ offline chat

    async def generate_stream(self, req: Any) -> AsyncIterator[str]:
        owner = await self._ensure_template_owner()
        messages, image_data, video_data = await asyncio.to_thread(
            _prepare_sglang_chat, req)
        prompt = await asyncio.to_thread(
            owner.apply_chat_template, messages,
            tokenize=False, add_generation_prompt=True)
        body: Dict[str, Any] = {
            "text": prompt,
            "sampling_params": _sampling_params(req.params),
            "stream": True,
        }
        if image_data:
            body["image_data"] = image_data
        if video_data:
            body["video_data"] = video_data

        replica = None
        tried: List[str] = []
        for candidate in self._pick_order():
            try:
                async for delta in self._stream_one(candidate, body):
                    replica = candidate
                    yield delta
                return
            except _ConnectFailed as exc:
                # connect-stage failure only — nothing streamed yet, safe to
                # try the next replica; mid-stream errors propagate (no retry
                # after the first yielded delta: the client already saw text)
                tried.append(f"{candidate.spec.base_url}: {exc}")
                continue
        if replica is None:
            raise RuntimeError(
                "offline backend unavailable"
                + (f" ({'; '.join(tried)})" if tried else " (no replicas up)"))

    def _pick_order(self) -> List[SglangReplica]:
        with self._lock:
            up = [r for r in self._replicas if r.state == UP]
        return sorted(up, key=lambda r: r.inflight)

    async def _stream_one(self, replica: SglangReplica,
                          body: Dict[str, Any]) -> AsyncIterator[str]:
        timeout = aiohttp.ClientTimeout(
            total=None, connect=CONNECT_TIMEOUT_S, sock_read=SOCK_READ_TIMEOUT_S)
        replica.inflight += 1
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    resp_ctx = session.post(replica.spec.base_url + "/generate", json=body)
                    resp = await resp_ctx.__aenter__()
                except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                    raise _ConnectFailed(str(exc) or type(exc).__name__) from exc
                try:
                    if resp.status != 200:
                        detail = (await resp.text())[:300]
                        raise RuntimeError(
                            f"sglang /generate HTTP {resp.status}: {detail}")
                    async for delta in _iter_sse_deltas(resp):
                        yield delta
                finally:
                    await resp_ctx.__aexit__(None, None, None)
        finally:
            replica.inflight -= 1

    async def _ensure_template_owner(self) -> Any:
        if self._template_owner is not None:
            return self._template_owner
        if self._template_lock is None:
            self._template_lock = asyncio.Lock()
        async with self._template_lock:
            if self._template_owner is None:
                self._template_owner = await asyncio.to_thread(
                    _load_template_owner, self.s.offline_model_path)
        return self._template_owner


class _ConnectFailed(RuntimeError):
    """Connect-stage failure (nothing streamed) — replica failover is safe."""


async def _iter_sse_deltas(resp: aiohttp.ClientResponse) -> AsyncIterator[str]:
    """sglang native /generate SSE → text deltas.

    Events carry the CUMULATIVE text so far (tokenizer_manager appends), so the
    delta is `text[len(prev):]`. Frames are split on the blank-line boundary by
    hand — `async for line in resp.content` caps lines at ~64 KB, which a long
    cumulative payload exceeds. The stream ends with `data: [DONE]`.
    """
    buf = b""
    prev = ""
    async for chunk in resp.content.iter_any():
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:"):].strip()
                if data == b"[DONE]":
                    return
                obj = json.loads(data)
                meta = obj.get("meta_info") or {}
                finish = meta.get("finish_reason") or {}
                if isinstance(finish, dict) and finish.get("type") == "abort":
                    raise RuntimeError(f"sglang aborted the request: {finish}")
                text = str(obj.get("text") or "")
                if len(text) < len(prev):
                    prev = ""  # engine restarted the stream — never slice negative
                delta = text[len(prev):]
                prev = text
                if delta:
                    yield delta


def _sampling_params(p: Any) -> Dict[str, Any]:
    """Board `_build_sglang_sampling_params` parity."""
    do_sample = bool(getattr(p, "do_sample", True))
    return {
        "max_new_tokens": int(getattr(p, "max_new_tokens", 4096)),
        "temperature": float(getattr(p, "temperature", 0.7)) if do_sample else 0.0,
        "top_p": float(getattr(p, "top_p", 0.8)),
        "top_k": int(getattr(p, "top_k", 20)),
        # `or 1.0`: the schema default is None (→ server default lives in the
        # REALTIME adapter; offline keeps stock 1.0) and getattr's fallback
        # only fires when the attribute is ABSENT, not when it is None
        "repetition_penalty": float(getattr(p, "repetition_penalty", None) or 1.0),
        "stop": ["<|im_end|>"],
        "skip_special_tokens": True,
    }


def _load_template_owner(model_path: str) -> Any:
    """Processor (preferred) or tokenizer from the OFFLINE ckpt — whichever
    provides `apply_chat_template`. CPU/template-only; same trust_remote_code
    pattern as vlm_hf's processor load."""
    try:
        from transformers import AutoProcessor

        owner = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        log.info("offline chat template: processor from %s", model_path)
        return owner
    except Exception as exc:  # noqa: BLE001 — processor class may not import
        log.warning("AutoProcessor load failed (%s) — falling back to AutoTokenizer", exc)
        from transformers import AutoTokenizer

        owner = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        log.info("offline chat template: tokenizer from %s", model_path)
        return owner


def _resolve_image_payload(payload: str) -> str:
    """Chat image payload → something sglang's load_image accepts.

    CAS handles and raw base64 become MIME-qualified data URLs. Bare JPEG
    base64 starts with /9j/, which some sglang loaders interpret as an
    absolute file path. Explicit data URLs avoid that ambiguity without
    re-encoding the image. Existing data URLs pass through unchanged.
    """
    import base64
    from io import BytesIO

    from PIL import Image

    from ....persistence.media import normalize_hash, resolve_blob_path

    s = (payload or "").strip()
    if s.startswith("data:"):
        return s
    if normalize_hash(s) is not None:
        path = resolve_blob_path(s)
        if path is None:
            raise ValueError(f"unknown image media: {s[:19]}…")
        with open(path, "rb") as f:
            data = f.read()
    else:
        data = base64.b64decode(s, validate=True)
    # Read only the image header to identify its MIME type; keep bytes intact.
    with Image.open(BytesIO(data)) as image:
        mime = Image.MIME.get(image.format)
    if not mime or not mime.startswith("image/"):
        raise ValueError("unsupported image format")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _resolve_video_payload(payload: str) -> str:
    """CAS handles ONLY (same policy as vlm_hf._resolve_chat_video: a chat
    request must not point the decoder at arbitrary files) → blob path."""
    from ....persistence.media import normalize_hash, resolve_blob_path

    s = (payload or "").strip()
    if normalize_hash(s) is None:
        raise ValueError(
            "videos must be CAS media handles (sha256:<hex> from POST /api/media)")
    path = resolve_blob_path(s)
    if path is None:
        raise ValueError(f"unknown video media: {s[:19]}…")
    return path


def _prepare_sglang_chat(req: Any) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Normalize a ChatRequest → (flattened messages, image_data, video_data).

    Same wire-shape contract as vlm_hf._prepare_chat_messages (legacy top-level
    images/videos prepend to the LAST user message; parts walked in document
    order) but flattens each message's content to a STRING with inline
    `<|image|>`/`<|video|>` placeholders (board `_flatten_content_with_vision_
    tokens`, MOSS branch) and collects media payloads for sglang instead of
    PIL/tensors. Placeholders the user already typed into text parts are
    honored: that many auto-insertions are skipped (board parity).

    Blocking (CAS path checks hit disk) — call via `asyncio.to_thread`.
    """
    messages = [m.model_dump() if hasattr(m, "model_dump") else dict(m)
                for m in req.messages]

    extra_images = list(getattr(req, "images", None) or [])
    extra_videos = [v for v in (getattr(req, "videos", None) or []) if isinstance(v, str)]
    if extra_images or extra_videos:
        target = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if target is None:
            raise ValueError("media provided but no user message to attach it to")
        body = target.get("content")
        media_parts = [{"type": "image", "media": s} for s in extra_images] + \
            [{"type": "video", "media": s} for s in extra_videos]
        if isinstance(body, list):
            target["content"] = media_parts + list(body)
        else:
            target["content"] = media_parts + [
                {"type": "text", "text": body if isinstance(body, str) else str(body or "")}]

    if _has_pixel_overrides(getattr(req, "params", None)):
        log.warning("sglang offline backend uses its own multimodal processor "
                    "defaults — per-request pixel/fps overrides are ignored")

    image_data: List[str] = []
    video_data: List[str] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            m["content"] = content if isinstance(content, str) else str(content or "")
            continue
        # pass 1: placeholders already typed in text parts suppress that many
        # auto-insertions (board's existing-token skip logic, MOSS branch)
        existing_text = "".join(
            str(part.get("text") or "") if isinstance(part, dict) else str(part)
            for part in content
            if not isinstance(part, dict) or part.get("type") in (None, "text"))
        img_skip = existing_text.count(IMAGE_TOKEN) + existing_text.count("<image>")
        vid_skip = existing_text.count(VIDEO_TOKEN) + existing_text.count("<video>")

        chunks: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                chunks.append(str(part))
                continue
            ptype = part.get("type")
            if ptype == "image":
                payload = part.get("media") or part.get("image") or part.get("data") or ""
                image_data.append(_resolve_image_payload(str(payload)))
                if img_skip > 0:
                    img_skip -= 1
                else:
                    chunks.append(IMAGE_TOKEN)
            elif ptype == "video":
                payload = part.get("media") or part.get("video") or ""
                video_data.append(_resolve_video_payload(str(payload)))
                if vid_skip > 0:
                    vid_skip -= 1
                else:
                    chunks.append(VIDEO_TOKEN)
            elif ptype == "text":
                chunks.append(str(part.get("text") or ""))
            # unknown part types are dropped (vlm_hf parity)
        m["content"] = "".join(chunks)
    return messages, image_data, video_data


def _has_pixel_overrides(params: Any) -> bool:
    if params is None:
        return False
    return any(
        getattr(params, k, None) is not None
        for k in ("min_pixels", "max_pixels", "video_fps", "min_frames",
                  "max_frames", "multi_image_max_pixels", "video_max_pixels"))
