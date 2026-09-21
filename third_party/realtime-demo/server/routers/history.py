"""Conversation history — list/search, transcript, delete.

GET    /api/history?q=&limit=&offset=   sidebar list; q → FTS5 (trigram) search
GET    /api/history/{cid}               conversation + turns (+ media handles)
DELETE /api/history/{cid}               remove from index AND journal

All reads hit short-lived SQLite read connections via asyncio.to_thread (WAL —
readers never block the recorder's writer thread).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import Runtime, get_runtime
from ..logging_conf import get_logger
from ..persistence.store import IndexStore

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["history"])


def _index(rt: Runtime) -> IndexStore:
    if rt.index is None or rt.history is None:
        raise HTTPException(status_code=503, detail="history is disabled")
    return rt.index


def _load_json(s: Any) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except (TypeError, json.JSONDecodeError):
        return None


def _conversation_out(row: Dict[str, Any]) -> Dict[str, Any]:
    # realtime sessions carry their capture source in the session config —
    # surfaced so the live sidebar can tell camera chats from file streams
    config = _load_json(row.get("config_json")) or {}
    return {
        "conversation_id": row["conversation_id"],
        "kind": row["kind"],
        "created_at": row["created_at"],
        "ended_at": row["ended_at"],
        "title": row["title"],
        "turn_count": row["turn_count"],
        "end_reason": row["end_reason"],
        "video_source": config.get("video_source") if isinstance(config, dict) else None,
    }


def _turn_out(row: Dict[str, Any]) -> Dict[str, Any]:
    media = []
    for m in row.get("media") or []:
        media.append({
            "hash": f"sha256:{m['hash']}",
            "mime": m["mime"],
            "kind": m["kind"],
            "width": m.get("width"),
            "height": m.get("height"),
            "duration_s": m.get("duration_s"),
            "url": f"/api/media/{m['hash']}",
            "thumb_url": f"/api/media/{m['hash']}/thumb" if m.get("has_thumb") else None,
        })
    return {
        "role": row["role"],
        "source": row["source"],
        "text": row["text"],
        "ts": row["ts"],
        "seq": row["seq"],
        "stop_reason": row["stop_reason"],
        "metrics": _load_json(row.get("metrics_json")),
        "media": media,
    }


@router.get("/history")
async def list_history(
    q: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    kind: str = Query(default="", pattern="^(chat|realtime)?$"),
    rt: Runtime = Depends(get_runtime),
):
    index = _index(rt)
    rows = await asyncio.to_thread(index.list_conversations, q, limit, offset, kind or None)
    return {"conversations": [_conversation_out(r) for r in rows]}


@router.get("/history/{cid}")
async def get_history(cid: str, rt: Runtime = Depends(get_runtime)):
    index = _index(rt)
    conv = await asyncio.to_thread(index.get_conversation, cid)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    turns = await asyncio.to_thread(index.get_transcript, cid)
    out = _conversation_out(conv)
    out["config"] = _load_json(conv.get("config_json"))
    out["turns"] = [_turn_out(t) for t in turns]
    return out


@router.delete("/history/{cid}")
async def delete_history(cid: str, rt: Runtime = Depends(get_runtime)):
    _index(rt)
    try:
        live = rt.session_manager.get(cid)
    except KeyError:
        live = None
    if live is not None:
        raise HTTPException(
            status_code=409,
            detail="session is still live — delete it via DELETE /api/sessions/{sid} first")
    deleted = await asyncio.to_thread(rt.history.delete_conversation, cid)
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"deleted": cid}
