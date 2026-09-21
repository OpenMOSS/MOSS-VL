"""Gateway plane WSS data surface: WS /v1/realtime?ws_token=...

Token gate (one-shot, 60 s TTL by default):
  unknown/consumed token → error{code:ws_token_invalid} + close 1008
  expired token          → error{code:ws_token_expired} + close 1008
  token whose session is gone → error{code:session_not_found} + close 1008
  second attach on a live session → error{code:session_already_attached} + close 1008

After the gate the socket is a pure passthrough (see session.py): the cached
`session.created` is replayed first, then omni events stream verbatim. Either
side ending the socket destroys the session — no grace, no reconnect.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket

from ..logging_conf import get_logger
from .deps import get_gateway_registry, get_gateway_tokens
from .session import WS_CLOSE_POLICY

log = get_logger(__name__)
router = APIRouter(tags=["gateway"])


async def _reject(websocket: WebSocket, code: str, message: str) -> None:
    await websocket.send_text(json.dumps(
        {"type": "error", "code": code, "message": message}, ensure_ascii=False))
    await websocket.close(code=WS_CLOSE_POLICY)


@router.websocket("/v1/realtime")
async def realtime_ws(websocket: WebSocket):
    await websocket.accept()
    tokens = get_gateway_tokens(websocket)
    registry = get_gateway_registry(websocket)

    binding, reason = tokens.consume_binding(websocket.query_params.get("ws_token"))
    if binding is None:
        if reason == "expired":
            registry.metrics.inc_error("ws_token_expired")
            await _reject(websocket, "ws_token_expired", "ws_token has expired")
        else:
            registry.metrics.inc_error("ws_token_invalid")
            await _reject(websocket, "ws_token_invalid", "missing or unknown ws_token")
        return

    session_id, epoch = binding
    session = registry.get(session_id)
    if session is None:
        registry.metrics.inc_error("session_not_found")
        await _reject(websocket, "session_not_found", f"no such session: {session_id}")
        return
    if not session.token_epoch_is_current(epoch):
        registry.metrics.inc_error("ws_token_invalid")
        await _reject(websocket, "ws_token_invalid", "ws_token belongs to an earlier session generation")
        return
    if not session.try_attach(epoch):
        registry.metrics.inc_error("session_already_attached")
        await _reject(websocket, "session_already_attached",
                      f"session {session_id} already has a live data socket")
        return

    log.info("gateway session %s: client attached (trace_id=%s)",
             session_id, session.trace_id)
    await session.run(websocket)
