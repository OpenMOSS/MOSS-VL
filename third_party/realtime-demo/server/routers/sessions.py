"""REST session lifecycle — create/config/teardown off the socket (§B5).

POST   /api/sessions        {config} → {session_id, ws_url, expires_at, config}
GET    /api/sessions/{sid}  → live snapshot (state + orchestrator + VLM status)
DELETE /api/sessions/{sid}  → close (stops the VLM realtime loop, frees the replica)
POST   /api/models/load     → load/replace the VLM (409 while sessions are live)

Capacity = the VLM replica pool (one live session per GPU worker). When every
replica is busy, POST /api/sessions returns 409 with a Retry-After hint sized
to the reconnect grace window.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from ..adapters.vlm.moss_vl_hf.online_pool import NoFreeReplica
from ..deps import Runtime, get_runtime
from ..logging_conf import get_logger
from ..schemas import CreateSessionRequest, CreateSessionResponse, LoadModelRequest, SessionConfig
from ..session.manager import SessionConflict
from ..session.orchestrator import EngineSet
from ..voice.tts_session import TtsSession

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["sessions"])


def _noop_emit(payload: Dict[str, Any]) -> None:
    """Placeholder TTS emit; the orchestrator rebinds it on construction."""


def _vlm_start_params(rt: Runtime, cfg: SessionConfig) -> Dict[str, Any]:
    """The scalar start_realtime_session kwargs, shared by session create and
    the rollover re-seat factory so a reseat rides the exact same sampling /
    vision budget the session was created with."""
    s = rt.settings
    params = cfg.params
    return dict(
        prompt=cfg.initial_prompt if cfg.initial_prompt is not None else s.initial_prompt,
        system_prompt=cfg.system_prompt,  # Preserve the model's trained/default or user-supplied system prompt.
        temperature=params.temperature, top_k=params.top_k, top_p=params.top_p,
        do_sample=params.do_sample, repetition_penalty=params.repetition_penalty,
        max_new_tokens=params.max_new_tokens,
        # per-session rate cap wins; None → the server-wide default
        max_tokens_per_turn=(params.max_tokens_per_turn
                             if params.max_tokens_per_turn is not None
                             else s.max_tokens_per_turn),
        frame_queue_size=s.frame_queue_size,
        min_pixels=params.min_pixels, max_pixels=params.max_pixels,
        video_fps=params.video_fps, min_frames=params.min_frames, max_frames=params.max_frames,
        multi_image_max_pixels=params.multi_image_max_pixels,
        video_max_pixels=params.video_max_pixels,
    )


def _make_reseat_factory(rt: Runtime, cfg: SessionConfig):
    """Rollover re-seat (design §6): start a replacement VLM realtime session
    whose prefill is the rebuilt memory prefix. `prefill_messages` is a JSON
    STRING — it crosses the worker proxy's scalar-only seam unchanged."""
    async def reseat(*, prompt: str = "", system_prompt: Any = None,
                     prefill_messages: Any = None) -> Any:
        params = _vlm_start_params(rt, cfg)
        params["prompt"] = prompt
        if prefill_messages and not system_prompt and getattr(rt, 'memory_store', None) is not None:
            from ..memory.rollover import _default_system_prompt
            system_prompt = cfg.system_prompt or _default_system_prompt()
        params["system_prompt"] = system_prompt
        params["prefill_messages"] = prefill_messages
        return await asyncio.to_thread(rt.vlm.start_realtime_session, **params)

    return reseat


async def _build_engines(rt: Runtime, cfg: SessionConfig) -> EngineSet:
    s = rt.settings
    vlm_session = await asyncio.to_thread(
        rt.vlm.start_realtime_session, **_vlm_start_params(rt, cfg))

    try:
        tts_session = None
        # creation-time engine pick (SessionConfig.tts_engine): the cloud lane
        # exists only when an API key was present at boot (deps.py); a request
        # for it without the lane falls back to the local pool with a warning
        tts_pool = rt.tts
        engine_pick = (cfg.tts_engine or "").strip().lower()
        if engine_pick in ("elevenlabs", "eleven", "el", "minimax", "mini_max", "mm"):
            lane = getattr(rt, f"tts_{'minimax' if engine_pick in ('minimax', 'mini_max', 'mm') else 'elevenlabs'}", None)
            if lane is not None:
                # boot-time probe can lose to a flaky egress path — one lazy
                # retry at session creation before falling back
                if not lane.status().get("ready"):
                    await asyncio.to_thread(lane.start)
                tts_pool = lane
            else:
                log.warning("tts_engine=%s requested but the lane is not built "
                            "(no API key at boot?) — using the local pool", engine_pick)
        tts_ready = bool(tts_pool.status().get("ready")) if s.tts_enabled else False
        if tts_ready:
            # lease the least-loaded pooled sidecar; released via on_close
            acquire = getattr(tts_pool, "acquire", None)
            engine = acquire() if callable(acquire) else tts_pool.engine
            release = getattr(tts_pool, "release", None)
            on_close = (lambda e=engine: release(e)) if callable(release) else None
            tts_session = TtsSession(engine, "session", _noop_emit, on_close=on_close)
            if cfg.tts_voice:
                tts_session.set_voice(cfg.tts_voice)
        else:
            log.warning("TTS not ready — session will stream captions without audio")

        asr = rt.asr if s.asr_enabled else None
        return EngineSet(vlm=vlm_session, asr=asr, tts=tts_session)
    except Exception:
        # never leak the exclusive VLM realtime session (it holds the infer lock)
        try:
            await asyncio.to_thread(vlm_session.stop, 5.0)
        except Exception:  # noqa: BLE001
            pass
        raise


@router.post("/sessions", status_code=201, response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest, rt: Runtime = Depends(get_runtime)) -> CreateSessionResponse:
    if not rt.vlm.is_loaded():
        raise HTTPException(status_code=503,
                            detail="VLM not loaded — POST /api/models/load first")
    cfg = req.config
    retry_after = str(max(1, int(rt.settings.session_grace_seconds)))
    try:
        state = await rt.session_manager.create(cfg, lambda: _build_engines(rt, cfg),
                                                reseat_factory=_make_reseat_factory(rt, cfg))
    except NoFreeReplica as exc:
        # all GPU replicas host a live session; abandoned ones free up after grace
        raise HTTPException(status_code=409, detail=str(exc),
                            headers={"Retry-After": retry_after})
    except SessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc),
                            headers={"Retry-After": retry_after})
    except RuntimeError as exc:
        status = 409 if "already running" in str(exc) else 500
        raise HTTPException(status_code=status, detail=str(exc))
    return CreateSessionResponse(
        session_id=state.session_id,
        ws_url=f"/api/session/{state.session_id}/ws",
        expires_at=state.grace_deadline or 0.0,
        config=cfg,
    )


@router.get("/sessions/{sid}")
async def get_session(sid: str, rt: Runtime = Depends(get_runtime)) -> Dict[str, Any]:
    try:
        state = rt.session_manager.get(sid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    snapshot = state.snapshot()
    if state.orchestrator is not None:
        snapshot["orchestrator"] = state.orchestrator.status_payload()
    return snapshot


@router.delete("/sessions/{sid}")
async def delete_session(sid: str, rt: Runtime = Depends(get_runtime)) -> Dict[str, Any]:
    closed = await rt.session_manager.close(sid, reason="client_delete")
    if not closed:
        raise HTTPException(status_code=404, detail=f"session not found: {sid}")
    return {"ok": True, "session_id": sid}


@router.post("/models/load")
async def load_model(req: LoadModelRequest, rt: Runtime = Depends(get_runtime)) -> Dict[str, Any]:
    """Load/replace the VLM. Refused while sessions are live (they hold the GPU)."""
    if rt.session_manager.active_count > 0:
        raise HTTPException(status_code=409,
                            detail="sessions are live; DELETE /api/sessions/{id} first")
    try:
        await asyncio.to_thread(rt.vlm.load, req.model_path, req.gpu_id, req.hf_mode)
    except Exception as exc:  # noqa: BLE001
        log.exception("model load failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "vlm": rt.vlm.status()}
