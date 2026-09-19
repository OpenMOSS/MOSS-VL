"""Downstream WebSocket server speaking the legacy VL contract.

Listens on LISTEN_HOST:LISTEN_PORT, accepts connections whose path ends with
/v1/realtime (gateway prefixes and the legacy empty ``session_id=`` query
parameter are tolerated and ignored). Each accepted connection drives one
BridgeSession.

Adapter-level busy semantics: with MAX_INFLIGHT > 0, connections beyond the
cap immediately receive the legacy busy error so the caller's existing
0.5/1.0/1.5 s retry logic keeps working. The slot is released as soon as the
client round ends (stop/disconnect), before upstream teardown, so an
immediate back-to-back round never sees a stale busy.
"""

from __future__ import annotations

import asyncio
import json
import logging

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

from . import config
from .session import BridgeSession

logger = logging.getLogger(__name__)


class AdapterServer:
    def __init__(self) -> None:
        self._inflight = 0
        self._inflight_lock = asyncio.Lock()

    def _check_path(self, connection: ServerConnection, request) -> Response | None:
        path = request.path.split("?", 1)[0]
        if path == config.LISTEN_PATH or path.endswith(config.LISTEN_PATH):
            return None
        return connection.respond(404, "unknown realtime endpoint\n")

    async def _handler(self, websocket: ServerConnection) -> None:
        busy = False
        if config.MAX_INFLIGHT > 0:
            async with self._inflight_lock:
                busy = self._inflight >= config.MAX_INFLIGHT
                if not busy:
                    self._inflight += 1
        if busy:
            try:
                await websocket.send(
                    json.dumps({"type": "error", "message": config.BUSY_MESSAGE})
                )
                # Drain outside the admission lock so a rejected client cannot
                # delay another session's release. Preserve busy delivery.
                await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except (asyncio.TimeoutError, OSError, ConnectionClosed):
                pass
            finally:
                await websocket.close()
            logger.info("rejected connection: %s", config.BUSY_MESSAGE)
            return
        released = False

        async def release_slot() -> None:
            nonlocal released
            if config.MAX_INFLIGHT > 0 and not released:
                async with self._inflight_lock:
                    if not released:
                        self._inflight -= 1
                        released = True

        try:
            session = BridgeSession(websocket, on_client_done=release_slot)
            await session.run()
        except Exception:
            logger.exception("bridge session failed")
        finally:
            await release_slot()

    async def run(self) -> None:
        logger.info(
            "legacy VL adapter listening on %s:%d%s -> %s (max_inflight=%d)",
            config.LISTEN_HOST,
            config.LISTEN_PORT,
            config.LISTEN_PATH,
            config.OMNI_WS_URL,
            config.MAX_INFLIGHT,
        )
        async with serve(
            self._handler,
            config.LISTEN_HOST,
            config.LISTEN_PORT,
            process_request=self._check_path,
            max_size=config.WS_MAX_SIZE,
        ):
            await asyncio.Future()  # run forever
