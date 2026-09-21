"""Offline (non-realtime) multimodal chat — WS + SSE.

Message shape mirrors board's generate_stream:
  in : {messages, images?, videos?, params, conversation_id?}
  out: generation_start | generation_delta{delta} | generation_end | generation_error

`images` entries may be data-URL/base64 strings or CAS handles
(`sha256:<hex>` from POST /api/media); `videos` entries (and
`{"type":"video"}` message parts) must be CAS handles — the adapter resolves
them to blob paths and samples frames for the VLM. When `conversation_id` is set the
user/assistant turn pair is committed to the durable history
(server/persistence/) — enqueue-only, so recording never delays streaming.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ..deps import Runtime, get_runtime
from ..logging_conf import get_logger
from ..persistence.media import normalize_hash
from ..schemas import ChatRequest

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

# client-minted thread ids become journal filenames — constrain hard
_CID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _chat_vlm(rt: Runtime):
    """Offline chats prefer the dedicated sglang plane; fall back to the online
    worker pool (1-GPU boxes, sidecars down/not built) — exactly the pre-split
    behavior. `is_loaded()` reads supervisor-cached health only (loop-safe);
    the degradation is visible as `vlm_offline.loaded == false` in /api/status.
    """
    offline = getattr(rt, "vlm_offline", None)  # tests build partial Runtimes
    if offline is not None:
        if offline.is_loaded():
            return offline
        log.warning("offline sglang pool unavailable — chat falls back to the online pool")
    return rt.vlm


def _last_user_turn(req: ChatRequest) -> Tuple[str, List[str]]:
    """Text + in-content media handles of the LAST user message.

    Multi-turn requests resend the whole conversation — only the final user
    message is a NEW turn; earlier messages were recorded when first sent.
    Content may be a plain string or a parts list (see
    vlm_hf._prepare_chat_messages for the wire shapes).
    """
    for m in reversed(req.messages):
        if m.role != "user":
            continue
        content = m.content
        if isinstance(content, str):
            return content, []
        if isinstance(content, list):
            texts: List[str] = []
            handles: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    texts.append(str(part))
                elif part.get("type") == "text":
                    texts.append(str(part.get("text") or ""))
                elif part.get("type") == "image":
                    handles.append(str(part.get("media") or part.get("image") or ""))
                elif part.get("type") == "video":
                    handles.append(str(part.get("media") or part.get("video") or ""))
            return " ".join(t for t in texts if t).strip(), handles
        return str(content or ""), []
    return "", []


def _recording_ctx(rt: Runtime, req: ChatRequest) -> Optional[Tuple[str, str, List[str]]]:
    """Validate conversation_id and record the user turn.

    Returns (cid, user_text, media_hashes) when recording, None when the
    request is stateless. Raises ValueError on a malformed conversation_id.
    """
    cid = (req.conversation_id or "").strip()
    if not cid or rt.history is None:
        return None
    if not _CID_RE.match(cid):
        raise ValueError("conversation_id must match [A-Za-z0-9_-]{8,64}")
    user_text, part_handles = _last_user_turn(req)
    # media handles come from the last user message's image/video parts plus
    # the legacy top-level images/videos (all of which also feed the VLM —
    # see vlm_hf._prepare_chat_messages)
    handles = part_handles + list(req.images or []) + \
        [v for v in (req.videos or []) if isinstance(v, str)]
    hashes = [h for h in (normalize_hash(s) for s in handles) if h]
    rt.history.open_conversation(cid, "chat")
    rt.history.record_turn(cid, role="user", text=user_text, source="chat",
                           media_hashes=hashes)
    return cid, user_text, hashes


def _record_reply(rt: Runtime, cid: str, text: str, t0: float) -> None:
    if text.strip() and rt.history is not None:
        rt.history.record_turn(
            cid, role="assistant", text=text, source="vlm",
            metrics={"latency_ms": round((time.monotonic() - t0) * 1000.0, 1)})


@router.websocket("/chat/stream")
async def chat_stream_ws(websocket: WebSocket):
    await websocket.accept()
    rt: Runtime = get_runtime()
    try:
        first = await websocket.receive_text()
        payload = json.loads(first)
        if payload.get("type") == "start" and "request" in payload:
            payload = payload["request"]
        req = ChatRequest(**payload)
        recording = _recording_ctx(rt, req)
    except Exception as exc:  # noqa: BLE001
        await websocket.send_text(json.dumps({"type": "generation_error", "message": f"bad request: {exc}"}))
        await websocket.close()
        return

    await websocket.send_text(json.dumps({"type": "generation_start"}))
    t0 = time.monotonic()
    parts: List[str] = []
    try:
        async for delta in _chat_vlm(rt).generate_stream(req):
            parts.append(delta)
            await websocket.send_text(json.dumps({"type": "generation_delta", "delta": delta}, ensure_ascii=False))
        await websocket.send_text(json.dumps({"type": "generation_end"}))
        if recording:
            _record_reply(rt, recording[0], "".join(parts), t0)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("chat stream failed: %s", exc)
        try:
            await websocket.send_text(json.dumps({"type": "generation_error", "message": str(exc)}))
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


@router.post("/chat/stream")
async def chat_stream_sse(req: ChatRequest, rt: Runtime = Depends(get_runtime)):
    try:
        recording = _recording_ctx(rt, req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def event_gen():
        yield _sse({"type": "generation_start"})
        t0 = time.monotonic()
        parts: List[str] = []
        try:
            async for delta in _chat_vlm(rt).generate_stream(req):
                parts.append(delta)
                yield _sse({"type": "generation_delta", "delta": delta})
            yield _sse({"type": "generation_end"})
            if recording:
                _record_reply(rt, recording[0], "".join(parts), t0)
        except Exception as exc:  # noqa: BLE001
            log.exception("chat SSE failed: %s", exc)
            yield _sse({"type": "generation_error", "message": str(exc)})

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
        # every proxy on the path (vite preview, notebook gateway) must flush
        # each event immediately — nginx-family proxies otherwise batch 4-8 KB
        # and the client sees the stream in bursts instead of token-by-token
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    })


def _sse(obj: Any) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
