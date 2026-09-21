"""Thin HTTP client for the pi_agent side of the memory protocol (board parity).

The gateway calls pi_agent for two LLM jobs: `/decide` (should this turn
retrieve memory at all) and `/compact` (compress the rollover journal into
{summary, pins}). Uses the gateway's existing aiohttp dependency. Every failure
mode (timeout, non-200, bad payload) returns None so the caller degrades to
the local-vector / verbatim-tail behavior without disturbing the turn, and is
logged at WARNING level throttled per endpoint (60s) to avoid per-turn spam.
"""
from __future__ import annotations

import asyncio
import math
import socket
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiohttp

from ..config import Settings
from ..logging_conf import get_logger

log = get_logger(__name__)

_WARN_EVERY_S = 60.0
_last_warn_at: Dict[str, float] = {}
_MAX_ATTEMPTS = 3  # 最多三次，含首次
_BACKOFF_S = (0.3, 0.8)


def _warn_throttled(url: str, message: str, *args: Any) -> None:
    now = time.monotonic()
    if now - _last_warn_at.get(url, 0.0) < _WARN_EVERY_S:
        return
    _last_warn_at[url] = now
    log.warning("pi_agent call %s failed: " + message + " (further failures suppressed for %ds)",
                url, *args, int(_WARN_EVERY_S))


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
    """Blocking/off-loop API; timeout covers connect, body, all attempts and backoff."""
    if not math.isfinite(timeout) or timeout <= 0:
        _warn_throttled(url, "invalid total timeout")
        return None
    return asyncio.run(_post_json_async(url, payload, timeout))


async def _post_json_async(url: str, payload: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
    deadline = time.monotonic() + timeout
    last_error = "deadline exceeded"
    try:
        async with asyncio.timeout(timeout):
            async with aiohttp.ClientSession(trust_env=False) as session:
                for attempt in range(_MAX_ATTEMPTS):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        async with session.post(url, json=payload,
                                headers={"X-Memory-Timeout-Ms": str(max(1, int(remaining * 1000)))},
                                timeout=aiohttp.ClientTimeout(total=remaining)) as response:
                            if 200 <= response.status < 300:
                                try:
                                    body = await response.json(content_type=None)
                                except ValueError:
                                    _warn_throttled(url, "invalid JSON (no retry)")
                                    return None
                                if isinstance(body, dict):
                                    return body
                                _warn_throttled(url, "non-object JSON (no retry)")
                                return None
                            last_error = f"HTTP {response.status}"
                            if 400 <= response.status < 500:
                                _warn_throttled(url, "%s (no retry)", last_error)
                                return None
                            if response.status not in (500, 502, 503, 504):
                                return None
                    except (aiohttp.ClientError, OSError, TimeoutError) as exc:
                        last_error = str(exc) or "deadline exceeded"
                    if attempt < _MAX_ATTEMPTS - 1:
                        delay = _BACKOFF_S[attempt]
                        if deadline - time.monotonic() <= delay:
                            break
                        await asyncio.sleep(delay)
    except TimeoutError:
        last_error = "total deadline exceeded"
    _warn_throttled(url, "%s (bounded attempts exhausted)", last_error)
    return None


class PiAgentClient:
    """Stateless caller; constructed from Settings, shared per session/manager."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._url = (settings.memory_pi_url or "").rstrip("/")
        self._reachability: tuple = (0.0, False)

    def select(self, query: str, candidates: list, *, timeout_s: float, recent_turns: str = '') -> Optional[list]:
        """Evidence IDs only; None means unavailable, not a positive verdict."""
        if not self._url or timeout_s <= 0:
            return None
        result = _post_json(f"{self._url}/select", {"query": query, "candidates": candidates,
                                                   "recent_turns": recent_turns}, timeout_s)
        if result is None:
            return None
        ids = result.get("ids")
        allowed = {c["id"] for c in candidates}
        if not isinstance(ids, list) or any(type(i) is not int or i not in allowed for i in ids):
            return None
        return list(dict.fromkeys(ids))

    def decide(
        self,
        conversation_id: str,
        recent_turns: str,
        pending_user_text: str,
    ) -> Optional[Dict[str, Any]]:
        """Ask pi_agent whether to retrieve; None means degrade to local gating."""
        if not self._url:
            return None
        result = _post_json(
            f"{self._url}/decide",
            {
                "conversation_id": str(conversation_id),
                "recent_turns": recent_turns,
                "pending_user_text": pending_user_text,
            },
            float(self._settings.memory_pi_decide_timeout_s),
        )
        if result is not None and (type(result.get("retrieve")) is not bool
                or (result.get("query") is not None and not isinstance(result["query"], str))
                or (isinstance(result.get("query"), str) and len(result["query"]) > 512)
                or not isinstance(result.get("reason"), str) or len(result["reason"]) > 512):
            _warn_throttled(f"{self._url}/decide", "invalid decision schema (no retry)")
            return None
        return result

    def compact(self, conversation_id: str, journal: str) -> Optional[Dict[str, Any]]:
        """Ask pi_agent to compress the journal; None means caller must fall back."""
        if not self._url:
            return None
        return _post_json(
            f"{self._url}/compact",
            {"conversation_id": str(conversation_id), "journal": journal,
             "summary_max_tokens": min(200, max(16, int(self._settings.memory_summary_max_tokens)))},
            float(self._settings.memory_pi_compact_timeout_s),
        )

    def reachable(self, cache_seconds: float = 5.0) -> bool:
        """Best-effort TCP reachability probe, cached for a few seconds."""
        if not self._url:
            return False
        cached_at, cached = self._reachability
        if time.monotonic() - cached_at < cache_seconds:
            return cached
        parsed = urlparse(self._url)
        ok = False
        try:
            with socket.create_connection(
                (parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=0.5
            ):
                ok = True
        except OSError:
            ok = False
        self._reachability = (time.monotonic(), ok)
        return ok
