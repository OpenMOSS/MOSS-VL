#!/usr/bin/env python3
"""Legacy-protocol entry router: one client port, N upstream adapters.

Session-level data-parallel routing for the legacy /v1/realtime contract:

* a client connection is bound to exactly one upstream adapter on its first
  ``start`` message (least-loaded upstream with free capacity);
* an upstream ``busy`` reply during the start phase is absorbed internally and
  the ``start`` is retried on the next upstream, so the router exposes the
  pooled capacity (N x ROUTER_UPSTREAM_CAPACITY) behind a single port;
* after the upstream is bound the connection is pumped through unchanged
  (text and binary frames) until either side closes;
* the client-facing capacity is therefore DP_REPLICAS x CAPACITY sessions.

Environment:
  ROUTER_LISTEN_HOST      bind address          (default 0.0.0.0)
  ROUTER_LISTEN_PORT      client-facing port    (default 18610)
  ROUTER_UPSTREAMS        comma-separated upstream adapter ws URLs (required)
  ROUTER_UPSTREAM_CAPACITY  per-upstream concurrent sessions (default 2)
  ROUTER_START_TIMEOUT_S  per-upstream start->ready deadline (default 15)
  ROUTER_MAX_MESSAGE_MB   websocket max message size (default 32)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed, WebSocketException

LISTEN_HOST = os.environ.get("ROUTER_LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("ROUTER_LISTEN_PORT", "18610"))
UPSTREAM_URLS = [
    u.strip() for u in os.environ.get("ROUTER_UPSTREAMS", "").split(",") if u.strip()
]
CAPACITY = int(os.environ.get("ROUTER_UPSTREAM_CAPACITY", "2"))
START_TIMEOUT_S = float(os.environ.get("ROUTER_START_TIMEOUT_S", "15"))
MAX_MSG = int(os.environ.get("ROUTER_MAX_MESSAGE_MB", "32")) << 20
BUSY_SUBSTRING = "realtime session is already active"
logger = logging.getLogger("adapter.router")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_rr_counter = 0  # tie-break 轮转计数：所有上游同空闲时轮转，避免重连风暴全撞第一个实例


def process_request(connection, request):
    """Defend against plain HTTP probes: /health returns 200 (health check),
    anything else 404 — mirroring the upstream adapter's behaviour so that
    port probes never crash the router with InvalidUpgrade."""
    path = request.path.split("?", 1)[0]
    if request.headers.get("Connection", "").lower() == "upgrade":
        return None  # genuine websocket upgrade, continue handshake
    if path == "/health":
        return connection.respond(200, "router ok\n")
    return connection.respond(404, "unknown realtime endpoint\n")


class Upstream:
    __slots__ = ("url", "capacity", "active")

    def __init__(self, url: str, capacity: int) -> None:
        self.url = url
        self.capacity = capacity
        self.active = 0

    @property
    def free(self) -> int:
        return self.capacity - self.active


UPSTREAMS = [Upstream(u, CAPACITY) for u in UPSTREAM_URLS]


def pick(exclude: frozenset = frozenset()) -> Upstream | None:
    """Least-loaded upstream with free capacity; tie-break rotates (round-robin)."""
    global _rr_counter
    cands = [u for u in UPSTREAMS if u.free > 0 and id(u) not in exclude]
    if not cands:
        return None
    least = min(u.active for u in cands)
    cands = [u for u in cands if u.active == least]
    u = cands[_rr_counter % len(cands)]
    _rr_counter += 1
    return u


async def _try_upstream(up: Upstream, start_text: str):
    """Send `start` to one upstream.

    Returns (upstream, ws, first_reply) when the session is bound (ready, or a
    non-busy error that should be surfaced through the bound session), or
    (None, None, reason) when the caller should try the next upstream.
    """
    ws = None
    try:
        ws = await asyncio.wait_for(
            connect(up.url, max_size=MAX_MSG, open_timeout=3.0), timeout=5.0
        )
        await ws.send(start_text)
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=START_TIMEOUT_S)
            if not isinstance(raw, str):
                continue  # adapters only send text JSON before ready
            data = json.loads(raw)
            kind = data.get("type")
            if kind == "ready":
                up.active += 1
                logger.info("bind client -> %s (active=%d)", up.url, up.active)
                return up, ws, raw
            if kind == "error":
                message = data.get("message", "")
                if BUSY_SUBSTRING in message:
                    logger.info("upstream %s busy, trying next", up.url)
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    return None, None, message
                # Real (non-busy) failure: keep the session bound so the client
                # sees the error over the live connection, mirroring adapter
                # behaviour; the slot is released when the connection closes.
                up.active += 1
                return up, ws, raw
    except (
        asyncio.TimeoutError,
        ConnectionClosed,
        OSError,
        json.JSONDecodeError,
        WebSocketException,  # e.g. upstream is not the adapter (bad handshake)
    ) as exc:
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        return None, None, f"upstream {up.url} unavailable: {exc!r}"


async def bind_upstream(start_text: str):
    """Try every upstream once per client start; return the first that accepts."""
    tried: set[int] = set()
    reason = "no upstream configured"
    while len(tried) < len(UPSTREAMS):
        up = pick(tried)
        if up is None:
            # Every upstream is at capacity in the router's own view; surface
            # the legacy busy message so callers keep their retry behaviour.
            reason = BUSY_SUBSTRING
            break
        tried.add(id(up))
        up2, ws, reply = await _try_upstream(up, start_text)
        if up2 is not None:
            return up2, ws, reply
        reason = reply
    return None, None, reason


async def _pump(src, dst, tag: str) -> None:
    try:
        async for message in src:
            await dst.send(message)
    except (ConnectionClosed, OSError) as e:
        logger.info("pump %s ended: %r", tag, e)
    except Exception:
        logger.exception("pump %s failed", tag)
    finally:
        logger.debug("pump %s cleanup", tag)


async def handler(websocket) -> None:
    try:
        first = await asyncio.wait_for(websocket.recv(), timeout=30)
    except (asyncio.TimeoutError, ConnectionClosed, OSError):
        return
    if not isinstance(first, str):
        await websocket.close(code=1003)
        return
    try:
        data = json.loads(first)
    except json.JSONDecodeError:
        await websocket.close(code=1003)
        return
    if data.get("type") != "start":
        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": "router: first message must be type=start",
                    }
                )
            )
        except Exception:
            pass
        await websocket.close()
        return

    logger.info("client %s connected, start received", websocket.remote_address)
    up, upstream_ws, reply = await bind_upstream(first)
    if upstream_ws is None:
        try:
            await websocket.send(
                json.dumps({"type": "error", "message": f"router: {reply}"})
            )
        except Exception:
            pass
        await websocket.close()
        return

    await websocket.send(reply)
    # Relay in both directions, but finish as soon as EITHER side ends: the
    # other pump is cancelled and both sockets are closed below, so a client
    # that drops mid-session (or skips the legacy `stop`) propagates its close
    # to the upstream adapter instead of squatting on the capacity slot.
    pumps = [
        asyncio.create_task(_pump(websocket, upstream_ws, "client->up")),
        asyncio.create_task(_pump(upstream_ws, websocket, "up->client")),
    ]
    try:
        _, pending = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        up.active -= 1
        logger.info(
            "client %s disconnected, %s released (active=%d)",
            websocket.remote_address,
            up.url,
            up.active,
        )
        for w in (websocket, upstream_ws):
            try:
                await w.close()
            except Exception:
                pass


async def main() -> None:
    if not UPSTREAMS:
        raise SystemExit(
            "ROUTER_UPSTREAMS is required, e.g. 'ws://127.0.0.1:18611,ws://127.0.0.1:18612'"
        )
    print(
        f"router listening on {LISTEN_HOST}:{LISTEN_PORT} -> "
        + ", ".join(f"{u.url}(cap={u.capacity})" for u in UPSTREAMS),
        flush=True,
    )
    async with serve(
        handler,
        LISTEN_HOST,
        LISTEN_PORT,
        max_size=MAX_MSG,
        process_request=process_request,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
