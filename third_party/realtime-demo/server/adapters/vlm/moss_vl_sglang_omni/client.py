"""Sync WebSocket client for one sglang-omni `/v1/video/realtime` connection.

Thin wrapper over `websocket-client` (already in requirements.txt) in the
spirit of board's `SGLangRealtimeSession` transport: blocking handshake, a
single daemon receiver thread, and a lock-serialized send side — nothing here
touches the gateway's asyncio loop.

Wire facts (verified against the sglang-omni source):
  connect → server sends `session.created` (or `error[session_capacity_exceeded]`
  + close 1013) → client sends `session.configure` (exactly once; extra fields
  are forbidden) → server replies `session.configured` then `session.ready`.
  No input is accepted before `session.ready`.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from ....logging_conf import get_logger

log = get_logger(__name__)

# inbound deltas are tiny; the cap only guards against a runaway peer
MAX_INBOUND_FRAME_BYTES = 32 * 1024 * 1024


class SessionCapacityExceeded(RuntimeError):
    """The server rejected the session at accept time (close code 1013)."""


class SessionConfigurationError(ValueError):
    """Invalid session configuration; retrying another replica cannot fix it."""


def http_to_ws_url(base_url: str) -> str:
    """Derive the realtime WS endpoint from a replica's http(s) base URL."""
    url = base_url.strip().rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    elif not url.startswith(("ws://", "wss://")):
        url = "ws://" + url
    return url + "/v1/video/realtime"


class SglangOmniClient:
    """Owns the transport for ONE sglang-omni realtime session."""

    def __init__(self, base_url: str, connect_timeout_s: float = 10.0,
                 ping_interval_s: float = 20.0, ping_timeout_s: float = 20.0):
        self.base_url = base_url
        self.ws_url = http_to_ws_url(base_url)
        self.connect_timeout_s = max(1.0, float(connect_timeout_s))
        # Downstream keepalive (gateway doc §4: the gateway maintains the
        # downstream itself). websocket-client's create_connection ping_*
        # kwargs are dead store in this version — only WebSocketApp honors
        # them — so start_receiver() runs an explicit ping thread: send a
        # ping every ping_interval_s, declare the peer dead when no frame
        # (any frame counts, pongs included) arrived within ping_timeout_s
        # of the previous ping. 0 disables.
        self.ping_interval_s = max(0.0, float(ping_interval_s))
        self.ping_timeout_s = max(0.05, float(ping_timeout_s))
        self._last_seen = 0.0
        self._ping_thread: Optional[threading.Thread] = None
        self._ws: Any = None
        self._send_lock = threading.Lock()
        self._recv_thread: Optional[threading.Thread] = None
        self._closed = threading.Event()

    # ------------------------------------------------------------ handshake

    def open(self) -> Dict[str, Any]:
        """Connect and consume the mandatory first event (`session.created`)."""
        import websocket  # websocket-client (sync API)

        deadline = time.monotonic() + 2 * self.connect_timeout_s
        self._ws = websocket.create_connection(
            self.ws_url, timeout=self.connect_timeout_s, enable_multithread=True)
        self._last_seen = time.monotonic()
        try:
            first = self._recv_event(deadline=deadline)
        except BaseException:
            self.abort_transport()
            raise
        if first.get("type") == "error":
            if first.get("code") == "session_capacity_exceeded":
                self.abort_transport()
                raise SessionCapacityExceeded(
                    str(first.get("message") or "session capacity exceeded"))
            self.abort_transport()
            raise RuntimeError(str(first.get("message") or "sglang-omni rejected the session"))
        if first.get("type") != "session.created":
            self.abort_transport()
            raise RuntimeError(f"expected session.created, got: {first}")
        return first

    def configure(self, payload: Dict[str, Any], timeout_s: float,
                  on_event: Callable[[Dict[str, Any]], None]) -> None:
        """Send `session.configure` and block until `session.ready`.

        Non-handshake events arriving in the configure window (the initial
        prefill can already emit text) are forwarded to `on_event`.
        """
        self.send_json(payload)
        deadline = time.monotonic() + max(1.0, timeout_s)
        configured = False
        ready = False
        while time.monotonic() < deadline and not (configured and ready):
            self._ws.settimeout(max(0.1, deadline - time.monotonic()))
            try:
                message = self._recv_event()
            except Exception as exc:  # noqa: BLE001 — socket.timeout included
                if "timed out" in str(exc).lower():
                    continue
                raise
            event_type = message.get("type")
            if event_type == "session.configured":
                configured = True
            elif event_type == "session.ready":
                ready = True
            elif event_type == "error":
                if message.get("code") == "invalid_request":
                    raise SessionConfigurationError(str(message.get("message") or "invalid configuration"))
                raise RuntimeError(
                    str(message.get("message") or "sglang-omni session.configure failed"))
            else:
                on_event(message)
        if not configured or not ready:
            raise TimeoutError("sglang-omni session did not become ready in time")

    # ------------------------------------------------------------ io

    def start_receiver(self, on_event: Callable, on_close: Callable[[str], None],
                       pass_raw: bool = False) -> None:
        """Start the daemon receive loop; events stream until the socket dies.

        pass_raw=True (the gateway plane's verbatim passthrough) calls
        on_event(raw_text, parsed); the default calls on_event(parsed).
        """
        self._ws.settimeout(1.0)  # poll for local close while blocking on recv
        self._start_keepalive()

        def loop() -> None:
            reason = "ws_closed"
            try:
                while not self._closed.is_set():
                    try:
                        got = self._recv_message()
                    except Exception as exc:  # noqa: BLE001
                        if self._closed.is_set():
                            break
                        if "timed out" in str(exc).lower():
                            continue
                        reason = f"ws_closed: {exc}"
                        log.warning("sglang-omni recv failed (%s): %s", self.ws_url, exc)
                        break
                    if got is None:
                        continue  # pong — keepalive proof, not an event
                    raw, message = got
                    on_event(raw, message) if pass_raw else on_event(message)
            finally:
                self._closed.set()
                on_close(reason)

        self._recv_thread = threading.Thread(
            target=loop, name=f"sglang-omni-recv-{id(self) & 0xFFFF:04x}", daemon=True)
        self._recv_thread.start()

    def _start_keepalive(self) -> None:
        """Ping every ping_interval_s; if no frame arrived within
        ping_timeout_s of the previous ping, the peer is half-open — abort()
        so the receiver loop errors out and fires on_close (gateway: 1011 +
        replica quarantine). Parked sessions stay alive through this because
        the omni server auto-answers pings even while silent."""
        if self.ping_interval_s <= 0 or self._ping_thread is not None:
            return

        def keepalive() -> None:
            while not self._closed.wait(self.ping_interval_s):
                # capture BEFORE ping(): a fast pong may land before ping()
                # returns, and it must count as an answer to this ping
                sent_at = time.monotonic()
                try:
                    self._ws.ping()
                except Exception as exc:  # noqa: BLE001 — send side is dead
                    log.warning("sglang-omni %s ping failed: %s", self.ws_url, exc)
                    break
                # wait out the pong deadline (or an early local close)
                if self._closed.wait(self.ping_timeout_s):
                    return
                if self._last_seen < sent_at:
                    log.warning("sglang-omni %s missed pong for %.0fs — aborting",
                                self.ws_url, self.ping_timeout_s)
                    break
            try:
                self._ws.abort()
            except Exception:  # noqa: BLE001
                pass

        self._ping_thread = threading.Thread(
            target=keepalive, name=f"sglang-omni-ping-{id(self) & 0xFFFF:04x}",
            daemon=True)
        self._ping_thread.start()

    def send_json(self, payload: Dict[str, Any]) -> None:
        with self._send_lock:
            self._ws.send(json.dumps(payload, ensure_ascii=False))

    def send_text(self, raw: str) -> None:
        """Forward a client text frame verbatim (gateway plane passthrough)."""
        with self._send_lock:
            self._ws.send(raw)

    def send_bytes(self, data: bytes) -> None:
        with self._send_lock:
            self._ws.send_binary(data)

    def close(self) -> None:
        self._closed.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:  # noqa: BLE001
            pass
        if self._recv_thread is not None and self._recv_thread is not threading.current_thread():
            self._recv_thread.join(timeout=2.0)

    def _recv_message(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """(raw_text, parsed) for one inbound event; None for a pong
        (keepalive proof). Raises on binary/close. Frame-level recv so pongs
        are observable — websocket-client's recv() silently eats them."""
        from websocket._abnf import ABNF
        with self._ws.readlock:
            opcode, frame = self._ws.recv_data_frame(control_frame=True)
        self._last_seen = time.monotonic()  # any inbound frame proves liveness
        if opcode == ABNF.OPCODE_PING:
            # server keepalive (uvicorn pings every 20s): recv_data_frame
            # already auto-answered with a pong — treat as liveness proof
            return None
        if opcode == ABNF.OPCODE_PONG:
            return None
        if opcode == ABNF.OPCODE_CLOSE:
            raise ConnectionError("sglang-omni WebSocket closed")
        if opcode != ABNF.OPCODE_TEXT:
            data = frame.data
            if isinstance(data, (bytes, bytearray)) and len(data) > MAX_INBOUND_FRAME_BYTES:
                raise RuntimeError("oversized inbound frame from sglang-omni")
            raise RuntimeError("unexpected binary event from sglang-omni")
        data = frame.data
        raw = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
        if not raw:
            raise ConnectionError("sglang-omni WebSocket closed")
        message = json.loads(raw)
        if not isinstance(message, dict):
            raise RuntimeError("non-object event from sglang-omni")
        return raw, message

    def abort_transport(self) -> None:
        """Close a failed handshake without waiting on the peer's close reply."""
        self._closed.set()
        try:
            if self._ws is not None:
                self._ws.abort()
        except Exception:
            pass
        self.close()

    def _recv_event(self, deadline: Optional[float] = None) -> Dict[str, Any]:
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("sglang-omni handshake deadline exceeded")
                self._ws.settimeout(min(self.connect_timeout_s, remaining))
            got = self._recv_message()
            if got is not None:
                return got[1]
