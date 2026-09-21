"""One-shot ws_token issuer for the gateway plane.

POST /v1/realtime/sessions mints a token bound to the session_id; the WSS
attach consumes it exactly once. Tokens expire after `ttl_s` (default 60 s,
settings.gateway_ws_token_ttl_s). Thread-safe: REST handlers and the WSS
endpoint may run on different threads/loops of the same process.

`consume()` follows the spec shape (session_id or None); `consume_detailed()`
adds the failure reason so ws.py can pick ws_token_invalid vs
ws_token_expired. Expired-but-unconsumed entries are reaped by purge_expired()
(called from the session registry's janitor).
"""
from __future__ import annotations

import secrets
import threading
import time
from typing import Dict, Optional, Tuple

from ..logging_conf import get_logger

log = get_logger(__name__)


class TokenIssuer:
    def __init__(self, ttl_s: float = 60.0):
        self.ttl_s = max(0.05, float(ttl_s))
        self._entries: Dict[str, Tuple[str, int, float]] = {}
        self._lock = threading.Lock()

    def mint(self, session_id: str, ttl_s: Optional[float] = None, *, epoch: int = 0) -> str:
        token = secrets.token_urlsafe(32)
        expires = time.monotonic() + max(0.05, float(ttl_s if ttl_s is not None else self.ttl_s))
        with self._lock:
            self._entries[token] = (session_id, epoch, expires)
        return token

    def consume(self, token: Optional[str]) -> Optional[str]:
        """One-shot: a valid token is burned on use; anything else → None."""
        return self.consume_detailed(token)[0]

    def consume_detailed(self, token: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """→ (session_id, None) | (None, "invalid" | "expired")."""
        binding, reason = self.consume_binding(token)
        return (binding[0] if binding is not None else None), reason

    def consume_binding(self, token: Optional[str]) -> Tuple[Optional[Tuple[str, int]], Optional[str]]:
        if not token:
            return None, "invalid"
        with self._lock:
            entry = self._entries.pop(token, None)
        if entry is None:
            return None, "invalid"
        session_id, epoch, expires = entry
        if time.monotonic() > expires:
            return None, "expired"
        return (session_id, epoch), None

    def revoke_session(self, session_id: str) -> None:
        with self._lock:
            for token in [token for token, entry in self._entries.items() if entry[0] == session_id]:
                del self._entries[token]

    def purge_expired(self) -> int:
        """Drop expired-unconsumed entries; returns how many were reaped."""
        now = time.monotonic()
        with self._lock:
            stale = [t for t, (_, _, exp) in self._entries.items() if exp < now]
            for t in stale:
                del self._entries[t]
        if stale:
            log.info("gateway ws_token janitor reaped %d expired token(s)", len(stale))
        return len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
