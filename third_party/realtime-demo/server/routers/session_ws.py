"""The single multiplexed session WebSocket (backend_overhaul.md §B4).

WS /api/session/{sid}/ws?last_seq=N
  in : JSON control events (§2.2) · binary 0x01 mic PCM · 0x02 video JPEG
  out: JSON events (§2.3, every one carries `seq`) · binary 0x11 TTS PCM
       (each preceded by its `response.audio.delta` descriptor)

Sessions are minted via POST /api/sessions; this socket only attaches. On connect
the router sends `session.created`, replays ring-buffered events with seq >
`last_seq`, then pumps live. On disconnect the session is *detached* (grace
window), not closed. A second socket for the same session supersedes the first
(close code 4000); the replay ring makes the handover lossless for the client.

Wire note: `seq` is monotonic but not gapless (transient events are never
replayed), and delivery order can very occasionally trail allocation order by an
event or two around an attach — clients must track `max(seq)`.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import protocol as p
from ..deps import Runtime, get_runtime
from ..logging_conf import get_logger
from ..session.state import OutboundItem, SessionState

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["session"])

WS_CLOSE_SUPERSEDED = 4000
WS_CLOSE_NOT_FOUND = 4404


@router.websocket("/session/{sid}/ws")
async def session_ws(websocket: WebSocket, sid: str):
    await websocket.accept()
    rt: Runtime = get_runtime()
    manager = rt.session_manager

    try:
        last_seq = int(websocket.query_params.get("last_seq") or 0)
    except ValueError:
        last_seq = 0

    try:
        state, token, close_event, replay = await manager.attach_ws(sid, last_seq=last_seq)
    except KeyError as exc:
        await websocket.send_text(p.encode_event(p.ERROR, code="not_found", message=str(exc)))
        await websocket.close(code=WS_CLOSE_NOT_FOUND)
        return

    send_lock = asyncio.Lock()

    async def send_item(item: OutboundItem) -> None:
        async with send_lock:
            await websocket.send_text(item.text)
            if item.binary is not None:
                await websocket.send_bytes(item.binary)

    orchestrator = state.orchestrator

    async def outbound() -> None:
        # per-connection welcome, then history catch-up, then live pump
        created = p.encode_event(
            p.SESSION_CREATED,
            seq=state.seq.next(),
            **state.created_event_fields(),
            replayed=len(replay),
            audio_out=_audio_out_format(rt),
            audio_in={"sample_rate": rt.settings.asr_sample_rate, "channels": 1},
        )
        async with send_lock:
            await websocket.send_text(created)

        last_sent = last_seq
        for item in replay:
            if item.seq > last_sent:
                await send_item(item)
                last_sent = item.seq

        closer = asyncio.create_task(close_event.wait())
        getter = None
        try:
            while True:
                getter = asyncio.create_task(state.out_queue.get())
                done, _ = await asyncio.wait({getter, closer}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    item = getter.result()
                    if closer in done:
                        # superseded mid-fetch: drop the item — the successor
                        # recovers non-transient events from the replay ring
                        return
                    if item.seq > last_sent:
                        await send_item(item)
                        last_sent = item.seq
                else:
                    getter.cancel()
                    await asyncio.gather(getter, return_exceptions=True)
                    return
        finally:
            closer.cancel()
            if getter is not None:
                getter.cancel()
            await asyncio.gather(closer, *([getter] if getter is not None else []),
                                 return_exceptions=True)

    async def inbound() -> None:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return

            text = message.get("text")
            if text is not None:
                try:
                    type_, payload = p.parse_client_text(text)
                except p.ProtocolError as exc:
                    state.emit(p.ERROR, code="bad_request", message=str(exc))
                    continue
                try:
                    await orchestrator.handle_event(type_, payload)
                except Exception as exc:  # noqa: BLE001
                    log.exception("event %s failed for %s: %s", type_, sid, exc)
                    state.emit(p.ERROR, code="internal", message=str(exc),
                               event_id=payload.get("event_id"))
                continue

            raw = message.get("bytes")
            if raw is None:
                continue
            try:
                tag, timestamp, payload_bytes = p.parse_binary(raw)
            except p.ProtocolError as exc:
                state.emit(p.ERROR, code="bad_frame", message=str(exc))
                continue
            if tag == p.TAG_MIC_PCM:
                orchestrator.push_pcm(payload_bytes)
            elif tag == p.TAG_VIDEO_JPEG:
                await orchestrator.push_frame(payload_bytes, timestamp)

    outbound_task = asyncio.create_task(outbound(), name=f"ws-out-{sid[:12]}")
    inbound_task = asyncio.create_task(inbound(), name=f"ws-in-{sid[:12]}")
    superseded = False
    try:
        done, pending = await asyncio.wait(
            {outbound_task, inbound_task}, return_when=asyncio.FIRST_COMPLETED)
        superseded = close_event.is_set() and state.ws_token != token
        for task in done:
            try:
                task.result()
            except (WebSocketDisconnect, asyncio.CancelledError):
                pass
            except Exception as exc:  # noqa: BLE001
                log.warning("session ws task failed for %s: %s", sid, exc)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        await manager.detach_ws(sid, token)
        if superseded:
            try:
                await websocket.close(code=WS_CLOSE_SUPERSEDED)
            except Exception:  # noqa: BLE001
                pass
        log.info("session ws disconnected: %s%s", sid, " (superseded)" if superseded else "")


def _audio_out_format(rt: Runtime) -> dict:
    engine = getattr(rt.tts, "engine", None)
    return {
        "sample_rate": getattr(engine, "sample_rate", rt.settings.tts_sample_rate),
        "channels": getattr(engine, "channels", rt.settings.tts_channels),
    }
