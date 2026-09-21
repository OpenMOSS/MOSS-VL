"""sglang-omni remote replica pool — the gateway's `rt.vlm` under
VLM_DEPLOY=sglang_omni.

One replica per `SGLANG_OMNI_URLS` entry = one remote sglang-omni server = up
to `sglang_omni_sessions_per_replica` live realtime sessions (the remote
instance's --max-running-requests; sessions are isolated, parked ones still
hold a slot). The facade mirrors `VlmReplicaPool` so routers/sessions.py never
changes. Replica state semantics:

- READY   healthy with at least one free slot
- BUSY    healthy but every slot occupied
- DOWN    unhealthy / quarantined (dead transport)

- `start_realtime_session` picks the READY replica with the FEWEST used slots
  (ties → lowest index), reserves a slot under the lock, then runs the WS
  handshake; failure returns the slot. No READY replica → `NoFreeReplica`
  (imported from online_pool — the router's 409+Retry-After contract is
  untouched).
- NO `set_replica_health` method on purpose: server/app.py uses hasattr() to
  decide whether to build the local `VlmWorkerSupervisor`; this pool spawns no
  local worker processes, ever.
- `session_capacity_exceeded` from omni means THAT instance is full (with
  N>1 slots this is routine, not a fault): the replica is marked BUSY (full)
  and the next READY replica is tried. Only when every replica has
  capacity-rejected us does the pool wait out one 5 s teardown grace (a
  session being torn down frees a slot) and retry once before raising
  `NoFreeReplica`.
- Self-heal without a supervisor: a daemon thread re-probes DOWN replicas
  every `sglang_omni_health_interval_s` and flips them back when GET /health
  answers (READY if a slot is free, else BUSY).
- `generate_stream` is unsupported: this deploy shape has no offline-chat
  plane (MIGRATION_PLAN.md §6).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import requests

from ....config import Settings
from ....logging_conf import get_logger
from ...base import VlmCaps
from ..moss_vl_hf.online_pool import BUSY, DOWN, READY, STARTING, NoFreeReplica
from .client import SessionCapacityExceeded, SglangOmniClient
from .session import SglangOmniSession


def _decode_prefill_messages(raw: Any) -> Optional[list]:
    """Lazy wrapper around realtime.mossvl_patches.decode_prefill_messages.

    mossvl_patches imports torch at module scope, and this adapter must stay
    usable in torch-less gateway envs/tests: skip the import entirely when
    there is nothing to decode (the common path), and fall back to a minimal
    local validation with the SAME contract (clean [{"role","content"}] list,
    or None on any problem) when torch is absent.
    """
    if raw is None:
        return None
    try:
        from ....realtime.mossvl_patches import decode_prefill_messages

        return decode_prefill_messages(raw)
    except ImportError:
        pass
    import json

    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError) as exc:
        log.warning("prefill_messages rejected (bad JSON): %s", exc)
        return None
    if not isinstance(data, list) or not data:
        log.warning("prefill_messages rejected: not a non-empty list")
        return None
    out = []
    for entry in data:
        if not isinstance(entry, dict):
            return None
        role, content = entry.get("role"), entry.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            log.warning("prefill_messages rejected: bad role/content (%r)", role)
            return None
        out.append({"role": role, "content": content})
    return out

log = get_logger(__name__)

# grace for the server to finish tearing down a previous session before the
# one retry a capacity rejection gets
CAPACITY_RETRY_DELAY_S = 5.0
# session.configure → session.ready includes the initial prefill
CONFIGURE_TIMEOUT_S = 180.0

# SessionConfig params sglang-omni does not understand (extra=forbid → 422).
# Logged once per pool so a misconfigured deploy is visible, not silent.
# NOTE: max_tokens_per_turn IS supported (tokens/second rate cap, omni
# VideoSessionConfigure default 86400 = unthrottled) and mapped below.
_UNSUPPORTED_PARAMS = ("top_k", "do_sample", "repetition_penalty",
                       "frame_queue_size",
                       "min_pixels", "max_pixels", "video_fps",
                       "min_frames", "max_frames",
                       "multi_image_max_pixels", "video_max_pixels")


@dataclass
class _Replica:
    url: str
    state: str = STARTING
    slots: int = 1                 # concurrent sessions this replica may host
    sessions: set = field(default_factory=set)  # live SglangOmniSessions
    pending: int = 0               # local handshakes holding a slot
    remote_full_until: float = 0.0  # cooldown after remote capacity rejection
    health: Dict[str, Any] = field(default_factory=dict)

    @property
    def used(self) -> int:
        local = len(self.sessions) + self.pending
        return max(local, self.slots) if self.remote_full_until else local


class _PooledSession:
    """Delegates to a SglangOmniSession; stop() releases the replica slot."""

    def __init__(self, pool: "SglangOmniPool", index: int, inner: SglangOmniSession):
        self._pool = pool
        self._index = index
        self._inner = inner
        self.session_id = inner.session_id
        self._stop_lock = threading.Lock()
        self._released = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def stop(self, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        with self._stop_lock:
            if self._released:
                return {**self._inner.status(), "stopped": True}
            try:
                return self._inner.stop(timeout_seconds)
            finally:
                self._pool._release(
                    self._index, session=self._inner,
                    transport_dead=getattr(self._inner, "worker_transport_dead", False))
                self._released = True


class SglangOmniPool:
    def __init__(self, settings: Settings):
        self.s = settings
        self.caps = VlmCaps(modes=("online_streaming",))
        urls = [u.strip().rstrip("/")
                for u in str(settings.sglang_omni_urls or "").split(",") if u.strip()]
        slots = max(1, int(settings.sglang_omni_sessions_per_replica or 1))
        self._replicas: List[_Replica] = [_Replica(url=u, slots=slots) for u in urls]
        self._lock = threading.Lock()
        self._prober_stop = threading.Event()
        self._prober: Optional[threading.Thread] = None
        self._warned_unsupported = False
        # tests shrink this; production keeps the 5 s teardown grace
        self.capacity_retry_delay_s = CAPACITY_RETRY_DELAY_S

    # ------------------------------------------------------------ introspection

    @property
    def capacity(self) -> int:
        """Total session slots across healthy replicas (drives the router's 409)."""
        return sum(r.slots for r in self._replicas if r.state in (READY, BUSY))

    @property
    def busy(self) -> int:
        return sum(r.used for r in self._replicas)

    @property
    def replicas(self) -> List[_Replica]:
        return self._replicas

    def is_loaded(self) -> bool:
        return any(r.state in (READY, BUSY) for r in self._replicas)

    def status(self) -> Dict[str, Any]:
        replicas = [
            {"url": r.url, "state": r.state, "slots": r.slots, "used": r.used,
             "pending": r.pending, "tracked_sessions": len(r.sessions),
             "health": r.health or None}
            for r in self._replicas]
        return {
            "loaded": self.is_loaded(),
            "deploy": "sglang_omni",
            "capacity": self.capacity,
            "busy": self.busy,
            "active_sessions": self.busy,
            "replicas": replicas,
            "modes": list(self.caps.modes),
        }

    # ------------------------------------------------------------ VlmAdapter

    def load(self, model_path: str, gpu_id: int, hf_mode: str,
             attn_impl_override: Optional[str] = None) -> None:
        """Remote servers load their own models; load() = health probe."""
        if not self._replicas:
            raise RuntimeError("VLM_DEPLOY=sglang_omni but SGLANG_OMNI_URLS is empty")
        errors: List[str] = []
        for i, r in enumerate(self._replicas):
            health = self._probe_health(r.url)
            with self._lock:
                if health is not None:
                    r.health = health
                    if r.state != BUSY:
                        r.state = READY
                    self._refresh_capacity(r)
                else:
                    if r.state != BUSY:
                        r.state = DOWN
                    errors.append(f"replica {i} ({r.url}) unhealthy")
        self._start_prober()
        if errors and not self.is_loaded():
            raise RuntimeError("no sglang-omni replica reachable: " + "; ".join(errors))
        if errors:
            log.warning("sglang-omni load probe: %s", "; ".join(errors))

    def start_realtime_session(self, **params: Any) -> _PooledSession:
        """Blocking (called via to_thread). The least-loaded READY replica wins
        (fewest used slots; ties → lowest index)."""
        retried_after_grace = False
        attempted: set[int] = set()
        last_error: Optional[Exception] = None
        while True:
            with self._lock:
                picked: Optional[int] = None
                for i, r in enumerate(self._replicas):
                    self._refresh_capacity(r)
                    if i not in attempted and r.state == READY and r.used < r.slots and (
                            picked is None or r.used < self._replicas[picked].used):
                        picked = i
                if picked is None:
                    if last_error is not None:
                        raise ConnectionError("no reachable sglang-omni replica") from last_error
                    raise NoFreeReplica(self.capacity, self.busy)
                r = self._replicas[picked]
                attempted.add(picked)
                r.pending += 1
                self._refresh_capacity(r)
            try:
                inner = self._start_on_replica(r, params)
            except SessionCapacityExceeded:
                # multi-slot semantics: the instance is FULL, not wedged — mark
                # it full (omni may host sessions we don't track) and try the
                # next READY replica
                log.warning("sglang-omni %s at session capacity — trying next replica",
                            r.url)
                with self._lock:
                    r.pending -= 1
                    r.remote_full_until = time.monotonic() + self.capacity_retry_delay_s
                    self._refresh_capacity(r)
                    any_ready = any(i not in attempted and x.state == READY
                                    for i, x in enumerate(self._replicas))
                if any_ready:
                    continue
                if retried_after_grace:
                    raise NoFreeReplica(self.capacity, self.busy)
                # every replica capacity-rejected us: a session in teardown
                # frees its slot within seconds — one grace + one retry
                retried_after_grace = True
                attempted.clear()
                log.warning("all sglang-omni replicas at session capacity — "
                            "retrying in %.0fs", self.capacity_retry_delay_s)
                time.sleep(self.capacity_retry_delay_s)
                with self._lock:
                    for x in self._replicas:
                        self._refresh_capacity(x)
                continue
            except Exception as exc:
                with self._lock:
                    r.pending -= 1
                    if not isinstance(exc, ValueError):
                        r.state = DOWN
                    self._refresh_capacity(r)
                if isinstance(exc, ValueError):
                    raise
                last_error = exc
                log.warning("sglang-omni replica %d handshake failed; trying another: %s", picked, exc)
                continue
            session = _PooledSession(self, picked, inner)
            with self._lock:
                r.pending -= 1
                r.sessions.add(inner)
                self._refresh_capacity(r)
            log.info("session %s → sglang-omni replica %d (%s) [%d/%d slots used]",
                     inner.session_id, picked, r.url, r.used, r.slots)
            return session

    def _refresh_capacity(self, replica: _Replica) -> None:
        """Called under _lock; expiry never changes local slot ownership."""
        if replica.remote_full_until and time.monotonic() >= replica.remote_full_until:
            replica.remote_full_until = 0.0
        if replica.state in (READY, BUSY):
            replica.state = BUSY if replica.used >= replica.slots else READY

    def _start_on_replica(self, replica: _Replica, params: Dict[str, Any]) -> SglangOmniSession:
        capabilities = replica.health.get('video_realtime_capabilities')
        payload = self._configure_payload(params, native_prefill=(
            isinstance(capabilities, dict) and capabilities.get('prefill_messages') is True))
        client = SglangOmniClient(replica.url, self.s.sglang_omni_connect_timeout_s)
        try:
            created = client.open()
        except Exception:
            client.close()
            raise
        try:
            session = SglangOmniSession(
                client, created,
                input_queue_capacity=self.s.sglang_omni_input_queue_capacity,
                input_drop_wait_seconds=self.s.sglang_omni_input_drop_wait_seconds,
                fallback_context_length=self.s.sglang_omni_context_length,
                fallback_frame_tokens=self.s.sglang_omni_fallback_frame_tokens,
                context_reserve_tokens=self.s.sglang_omni_context_reserve_tokens,
                model_path=self.s.model_path)
            session.configure(payload, CONFIGURE_TIMEOUT_S)
            return session
        except Exception:
            client.close()
            raise

    def _release(self, index: int, session: Optional[Any] = None,
                 transport_dead: bool = False) -> None:
        r = self._replicas[index]
        with self._lock:
            if session is not None:
                r.sessions.discard(session)
            if transport_dead:
                # a dead transport means the server may be wedged — quarantine
                # the replica until the prober's next health poll clears it
                r.state = DOWN
            self._refresh_capacity(r)
        log.info("sglang-omni replica %d released [%d/%d slots used]%s",
                 index, r.used, r.slots,
                 " (transport dead — quarantined)" if transport_dead else "")

    async def generate_stream(self, req: Any) -> AsyncIterator[str]:
        raise RuntimeError("offline chat not supported by sglang_omni deploy")
        yield  # pragma: no cover — keeps this an async generator

    # ------------------------------------------------------------ configure mapping

    def _configure_payload(self, params: Dict[str, Any], *, native_prefill: bool = False) -> Dict[str, Any]:
        """Map the router's start kwargs (routers/sessions.py:_vlm_start_params)
        onto session.configure. sglang-omni is extra=forbid: ONLY the keys the
        server declares may be sent."""
        if not self._warned_unsupported:
            present = [k for k in _UNSUPPORTED_PARAMS if params.get(k) is not None]
            if present:
                log.info("sglang-omni session.configure does not support %s — ignored",
                         ", ".join(present))
            self._warned_unsupported = True

        prompt = str(params.get("prompt") or "")
        system_prompt = params.get("system_prompt")
        prefill = _decode_prefill_messages(params.get("prefill_messages"))
        native_messages = None
        if prefill and native_prefill:
            native_messages = _native_prefill(prefill, system_prompt)
            if prompt:
                native_messages[1:1] = [{'role': 'user', 'content': prompt},
                                        {'role': 'assistant', 'content': '<|silence|>'}]
        elif prefill:
            # rollover re-seat: the rebuilt memory prefix crosses as a JSON
            # string of chat messages; sglang-omni takes plain text, so system
            # messages fold into system_prompt and the rest renders as one
            # role-tagged prompt block
            prefill_system, rendered = _render_prefill(prefill)
            if not system_prompt:
                system_prompt = prefill_system
            elif prefill_system and prefill_system != system_prompt:
                # A caller-supplied base prompt must not discard the memory
                # summary/pins that the compactor placed in its system block.
                context = _compacted_context(prefill_system, system_prompt)
                if context:
                    rendered = f"<recall>\n{context}\n\n{rendered}\n</recall>"
            prompt = "\n\n".join(part for part in (prompt, rendered) if part)

        do_sample = bool(params.get("do_sample", True))
        temperature = float(params.get("temperature") or 0.0) if do_sample else 0.0
        temperature = max(0.0, min(2.0, temperature))
        top_p = max(1e-6, min(1.0, float(params.get("top_p") or 1.0)))
        return {
            "type": "session.configure",
            "prompt": prompt,
            "system_prompt": system_prompt or None,
            "max_new_tokens": max(1, int(params.get("max_new_tokens") or 4096)),
            # tokens/SECOND rate cap (omni default 86400 = unthrottled); the
            # demo plane always supplies one (session param or server default)
            "max_tokens_per_turn": max(0.1, float(
                params.get("max_tokens_per_turn") or 86400.0)),
            "temperature": temperature,
            "top_p": top_p,
            "input_queue_capacity": max(1, int(self.s.sglang_omni_input_queue_capacity)),
            **({'prefill_messages': native_messages} if native_messages is not None else {}),
        }

    # ------------------------------------------------------------ health

    def _probe_health(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            resp = requests.get(f"{url}/health",
                                timeout=max(1.0, self.s.sglang_omni_connect_timeout_s))
            if resp.ok:
                health = resp.json() if resp.content else {"ok": True}
                try:
                    caps = requests.get(f'{url}/v1/video/realtime/capabilities', timeout=2)
                    if caps.ok:
                        value = caps.json()
                        if isinstance(value, dict):
                            health['video_realtime_capabilities'] = value
                except Exception:
                    pass  # legacy servers keep the original configure contract
                return health
        except requests.RequestException:
            pass
        return None

    def _start_prober(self) -> None:
        if self._prober is not None:
            return
        interval = max(1.0, float(self.s.sglang_omni_health_interval_s))

        def prober() -> None:
            while not self._prober_stop.wait(interval):
                self._probe_replicas()

        self._prober = threading.Thread(
            target=prober, name="sglang-omni-health", daemon=True)
        self._prober.start()

    def _probe_replicas(self) -> None:
        for i, r in enumerate(self._replicas):
            with self._lock:
                needs_probe = r.state == DOWN or (
                    r.remote_full_until and time.monotonic() >= r.remote_full_until)
            if not needs_probe:
                continue
            health = self._probe_health(r.url)
            if health is None:
                continue
            with self._lock:
                r.health = health
                if r.state == DOWN:
                    r.state = READY
                    log.info("sglang-omni replica %d recovered (%s)", i, r.url)
                # A newer rejection during the probe retains its own cooldown.
                self._refresh_capacity(r)


def _native_prefill(messages: List[Dict[str, str]], base_system: Optional[str]) -> list:
    if not base_system:
        return messages
    system, _ = _render_prefill(messages)
    context = _compacted_context(system or '', base_system)
    result = [{'role': 'system', 'content': base_system}]
    if context:
        result.extend([{'role': 'user', 'content': f'<recall>\n{context}\n</recall>'},
                       {'role': 'assistant', 'content': '<|silence|>'}])
    result.extend(m for m in messages if m['role'] != 'system')
    return result


def _compacted_context(prefill_system: str, base_system: str) -> str:
    """Keep all facts, without duplicating the known system/instruction prefix."""
    from ....memory.inject import augment_system_prompt
    for prefix in (augment_system_prompt(base_system, 'zh'),
                   augment_system_prompt(base_system, 'en'), base_system):
        if prefill_system == prefix:
            return ''
        if prefill_system.startswith(prefix + '\n'):
            return prefill_system[len(prefix):].lstrip()
    return prefill_system


def _render_prefill(messages: List[Dict[str, str]]) -> tuple:
    """(system_prompt, rendered_prompt) from validated prefill messages."""
    system_parts = [m["content"] for m in messages
                    if m["role"] == "system" and m["content"].strip()]
    lines = [f"{m['role']}: {m['content']}"
             for m in messages if m["role"] != "system"]
    system = "\n\n".join(system_parts) or None
    return system, "\n".join(lines)
