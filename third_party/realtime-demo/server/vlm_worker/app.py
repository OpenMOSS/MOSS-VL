"""VLM worker FastAPI app — REST control plane + one duplex session WS.

RPC surface (127.0.0.1 only; see protocol.py for the WS envelope):
  GET    /health            readiness + gpu + kv counters (the supervisor polls this)
  POST   /load              load/replace the model (409 while a session is live)
  POST   /session           start the (single) realtime session → session_id + KV budget
  DELETE /session/{sid}     stop it
  POST   /chat/stream       offline chat SSE (images arrive pre-resolved as base64;
                            videos arrive as CAS handles, resolved to blob paths
                            worker-side via the shared DATA_DIR)
  WS     /session/{sid}/io  frames/prompts in, coalesced output + 1 Hz status out

The worker sees exactly one GPU (`CUDA_VISIBLE_DEVICES` pins it) so every
device reference here is cuda:0.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ..config import Settings, get_settings
from ..gpu.kv_budget import KvBudget, compute_kv_budget, kv_bytes_per_token
from ..logging_conf import configure_logging, get_logger
from ..device_compat import (is_available, get_device_properties,
                                  mem_get_info, memory_allocated, memory_reserved,
                                  empty_cache)
from ..schemas import ChatRequest
from . import protocol as wp

log = get_logger(__name__)

OUT_COALESCE_S = 0.03      # merge output chunks landing within this window
OUT_POLL_S = 0.25          # poll_output blocking wait for the first chunk
STATUS_INTERVAL_S = 1.0    # status + KV watchdog tick
KV_WARN_RATIO = 0.85


class WorkerState:
    def __init__(self, settings: Settings, worker_id: int, fake: bool):
        self.settings = settings
        self.worker_id = worker_id
        self.fake = fake
        self.adapter: Any = None
        self.kv: Optional[KvBudget] = None
        self.baseline_alloc: int = 0     # bytes allocated after load+warmup
        self.session: Any = None         # live VlmRealtimeSession (≤1)
        self.session_ended_reason: Optional[str] = None
        self.lock = asyncio.Lock()       # session create/stop serialization

    # ---- gpu introspection (worker owns CUDA; fake mode reports nothing) ----

    def gpu_snapshot(self) -> Dict[str, Any]:
        if self.fake:
            return {"name": "fake", "cc": [0, 0], "mem_total_mib": 0, "mem_used_mib": 0}

        if not is_available():
            return {"name": "cpu", "cc": [0, 0], "mem_total_mib": 0, "mem_used_mib": 0}
        props = get_device_properties(0)
        free, total = mem_get_info(0)
        return {
            "name": props.name,
            "cc": [props.major, props.minor],
            "mem_total_mib": total // 2**20,
            "mem_used_mib": (total - free) // 2**20,
            "mem_allocated_gb": round(memory_allocated(0) / 2**30, 2),
            "mem_reserved_gb": round(memory_reserved(0) / 2**30, 2),
        }

    def kv_used_tokens(self) -> int:
        if self.fake or self.kv is None or self.session is None:
            return 0

        if not is_available():
            return 0
        grown = max(0, memory_allocated(0) - self.baseline_alloc)
        return int(grown // max(1, self.kv.bytes_per_token))

    def kv_payload(self) -> Optional[Dict[str, Any]]:
        if self.kv is None:
            return None
        used = self.kv_used_tokens()
        left = max(0, self.kv.budget_tokens - used)
        return {
            **self.kv.as_dict(),
            "used_tokens": used,
            "est_seconds_left": round(left / max(1e-6, self.kv.tokens_per_second), 1),
        }

    def health_payload(self) -> Dict[str, Any]:
        status = self.adapter.status() if self.adapter is not None else {}
        session_active = bool(self.session is not None
                              and self.session.status().get("active"))
        return {
            "ready": bool(status.get("loaded")),
            "worker_id": self.worker_id,
            "fake": self.fake,
            "gpu": self.gpu_snapshot(),
            "kv": self.kv_payload(),
            "session": {
                "active": session_active,
                "session_id": getattr(self.session, "session_id", None) if session_active else None,
            },
            **status,  # loaded / model_path / hf_mode / attn_impl / modes
        }


# --------------------------- load / warmup ---------------------------


def _warmup_and_verify_attn(adapter: Any) -> None:
    """Tiny text-only forward to smoke the attention kernels.

    A wrong-arch flash-attn build (sm_90-only cubins on a 4090/Blackwell)
    survives `from_pretrained` and dies HERE with "no kernel image ..." —
    flip every config's `_attn_implementation` to sdpa in place and retry.
    """
    import torch

    model = adapter.model
    device = adapter.device

    def tiny_forward() -> None:
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
        n = ids.shape[1]
        positions = (torch.arange(n, dtype=torch.long, device=device)
                     .view(1, 1, n).expand(3, 1, n).contiguous())
        with torch.no_grad():
            model(input_ids=ids,
                  attention_mask=torch.ones(1, n, dtype=torch.long, device=device),
                  position_ids=positions,
                  cache_position=torch.arange(n, dtype=torch.long, device=device),
                  use_cache=False)

    try:
        tiny_forward()
        log.info("attention warmup OK (attn=%s)", adapter._attn_impl)
        return
    except RuntimeError as exc:
        log.warning("attention warmup failed under %s (%s) — flipping to sdpa",
                    adapter._attn_impl, exc)
    for cfg in adapter._attn_configs():
        cfg._attn_implementation = "sdpa"
    adapter._attn_impl = "sdpa"
    tiny_forward()  # raises if even sdpa can't run — genuinely broken box
    log.info("attention warmup OK after sdpa flip")


def _load_model(state: WorkerState, model_path: str, hf_mode: str,
                attn_impl: Optional[str]) -> None:
    """Blocking load + warmup + KV budget (run via to_thread)."""
    import torch

    state.adapter.load(model_path, 0, hf_mode, attn_impl_override=attn_impl)
    _warmup_and_verify_attn(state.adapter)
    empty_cache()
    state.baseline_alloc = memory_allocated(0)
    free, total = mem_get_info(0)
    try:
        bpt = kv_bytes_per_token(state.adapter.model_config)
    except ValueError as exc:
        log.warning("KV budget disabled — %s", exc)
        state.kv = None
        return
    state.kv = compute_kv_budget(free, bpt, state.settings,
                                 gpu_total_mib=total // 2**20)
    log.info("KV budget: %s", state.kv.as_dict())


# --------------------------- app factory ---------------------------


def create_worker_app(worker_id: int = 0) -> FastAPI:
    settings = get_settings()
    fake = settings.vlm_worker_fake
    state = WorkerState(settings, worker_id, fake)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        if fake:
            from .fake import FakeWorkerVlmAdapter

            state.adapter = FakeWorkerVlmAdapter(settings)
            log.info("worker %d up (FAKE mode)", worker_id)
        else:
            from ..adapters.vlm.moss_vl_hf.adapter import HfMossVlAdapter

            state.adapter = HfMossVlAdapter(settings)
            if settings.model_path and settings.autoload_vlm:
                attn = os.environ.get("VLM_WORKER_ATTN") or None
                await asyncio.to_thread(
                    _load_model, state, settings.model_path, settings.hf_mode, attn)
            log.info("worker %d ready (loaded=%s)", worker_id, state.adapter.is_loaded())
        yield
        session = state.session
        if session is not None:
            await asyncio.to_thread(session.stop, 5.0)

    app = FastAPI(title=f"MOSS-VL worker {worker_id}", lifespan=lifespan)

    # ---- REST ----

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return state.health_payload()

    @app.post("/load")
    async def load(body: Dict[str, Any]) -> Dict[str, Any]:
        if state.session is not None and state.session.status().get("active"):
            raise HTTPException(status_code=409, detail="a realtime session is live")
        if fake:
            state.adapter.load(body.get("model_path", ""), 0, body.get("hf_mode", ""))
        else:
            try:
                await asyncio.to_thread(
                    _load_model, state, body["model_path"],
                    body.get("hf_mode", settings.hf_mode), body.get("attn_impl"))
            except Exception as exc:  # noqa: BLE001
                log.exception("worker load failed: %s", exc)
                raise HTTPException(status_code=500, detail=str(exc))
        return {"ok": True, **state.health_payload()}

    @app.post("/session", status_code=201)
    async def create_session(body: Dict[str, Any]) -> Dict[str, Any]:
        async with state.lock:
            if state.session is not None and state.session.status().get("active"):
                raise HTTPException(status_code=409, detail="a realtime session is live")
            try:
                session = await asyncio.to_thread(
                    lambda: state.adapter.start_realtime_session(**body))
            except RuntimeError as exc:
                code = 409 if "already running" in str(exc) else 500
                raise HTTPException(status_code=code, detail=str(exc))
            state.session = session
            state.session_ended_reason = None
        kv = state.kv
        return {
            "session_id": session.session_id,
            "kv_budget_tokens": kv.budget_tokens if kv else None,
            "est_max_seconds": round(kv.est_max_seconds, 1) if kv else None,
            "kv_floor_met": kv.floor_met if kv else None,
        }

    @app.delete("/session/{sid}")
    async def delete_session(sid: str) -> Dict[str, Any]:
        async with state.lock:
            session = state.session
            if session is None or session.session_id != sid:
                raise HTTPException(status_code=404, detail=f"session not found: {sid}")
            state.session = None
        status = await asyncio.to_thread(session.stop, 10.0)
        return {"ok": True, "status": status}

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        # mirrors routers/chat.py SSE shapes so the pool can re-emit verbatim
        async def event_gen():
            yield _sse({"type": "generation_start"})
            try:
                async for delta in state.adapter.generate_stream(req):
                    yield _sse({"type": "generation_delta", "delta": delta})
                yield _sse({"type": "generation_end"})
            except Exception as exc:  # noqa: BLE001
                log.exception("worker chat stream failed: %s", exc)
                yield _sse({"type": "generation_error", "message": str(exc)})

        return StreamingResponse(event_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache, no-transform"})

    # ---- session WS ----

    @app.websocket("/session/{sid}/io")
    async def session_io(ws: WebSocket, sid: str) -> None:
        session = state.session
        if session is None or session.session_id != sid:
            await ws.close(code=4404)
            return
        await ws.accept()
        send_lock = asyncio.Lock()

        async def send_json(payload: Dict[str, Any]) -> None:
            async with send_lock:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))

        async def output_pump() -> None:
            """poll_output → coalesced {"t":"out"} pushes; ends with {"t":"ended"}."""
            while True:
                batch = await asyncio.to_thread(session.poll_output, OUT_POLL_S, 128)
                if batch.chunks:
                    # brief coalesce: catch chunks emitted while we ship this one
                    await asyncio.sleep(OUT_COALESCE_S)
                    more = await asyncio.to_thread(session.poll_output, 0.0, 128)
                    if more.chunks:
                        batch.chunks.extend(more.chunks)
                        batch.chunk_events.extend(more.chunk_events)
                        batch = type(batch)(active=more.active, chunks=batch.chunks,
                                            chunk_events=batch.chunk_events, status=more.status)
                    await send_json({"t": wp.T_OUT, "active": batch.active,
                                     "chunks": batch.chunks,
                                     "chunk_events": batch.chunk_events,
                                     "status": batch.status})
                if not batch.active:
                    reason = state.session_ended_reason or "stopped"
                    await send_json({"t": wp.T_ENDED, "reason": reason})
                    return

        async def status_loop() -> None:
            """1 Hz status + KV watchdog (warn at 85 %, hard-stop at exhaustion)."""
            warned = False
            while True:
                await asyncio.sleep(STATUS_INTERVAL_S)
                kv = state.kv_payload()
                await send_json({"t": wp.T_STATUS, "status": session.status(),
                                 "kv": kv, "gpu": state.gpu_snapshot()})
                if kv is None or state.kv is None or not state.kv.budget_tokens:
                    continue
                used_ratio = kv["used_tokens"] / state.kv.budget_tokens
                # OOM backstop: trip on real free VRAM regardless of token math
                free_low = False
                if not state.fake:

                    if is_available():
                        free, _ = mem_get_info(0)
                        free_low = free < state.settings.kv_safety_margin_mib * 2**20
                if used_ratio >= KV_WARN_RATIO and not warned:
                    warned = True
                    await send_json({"t": wp.T_KV_WARNING, **kv})
                if (used_ratio >= 1.0 or free_low) and state.kv.enforce_hard:
                    log.warning("KV exhausted (used=%.0f%% free_low=%s) — stopping session %s",
                                used_ratio * 100, free_low, sid)
                    state.session_ended_reason = "kv_exhausted"
                    await asyncio.to_thread(session.stop, 5.0)
                    return

        async def inbound_loop() -> None:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                raw = message.get("bytes")
                if raw is not None:
                    header, payload = wp.unpack_msg(raw)
                    t = header.get("t")
                    # a bad input (corrupt frame, session racing its stop) must
                    # not kill this loop — that tears down the io WS and briefly
                    # quarantines the replica; log, skip, keep pumping
                    try:
                        if t == wp.T_FRAME:
                            await asyncio.to_thread(
                                session.put_frame, payload, header.get("ts"),
                                header.get("size", len(payload)))
                        elif t == wp.T_PROMPT_FRAME:
                            await asyncio.to_thread(
                                session.put_prompt_frame, str(header.get("text") or ""),
                                payload, header.get("ts"), header.get("size", len(payload)),
                                bool(header.get("drop_pending", True)))
                    except Exception as exc:  # noqa: BLE001
                        log.warning("session %s: %s rejected: %s", sid, t, exc)
                    continue
                text = message.get("text")
                if text is None:
                    continue
                event = json.loads(text)
                t = event.get("t")
                if t == wp.T_PROMPT:
                    await asyncio.to_thread(session.put_prompt, str(event.get("text") or ""))
                elif t == wp.T_TURN_END:
                    await asyncio.to_thread(session.request_turn_end)
                elif t == wp.T_STOP:
                    state.session_ended_reason = state.session_ended_reason or "stopped"
                    await asyncio.to_thread(session.stop, 5.0)
                elif t == wp.T_PING:
                    await send_json({"t": wp.T_PONG})

        tasks = [asyncio.create_task(coro, name=f"worker-{name}-{sid[:8]}")
                 for coro, name in ((inbound_loop(), "in"), (output_pump(), "out"),
                                    (status_loop(), "status"))]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    log.warning("worker io task failed for %s: %s", sid, exc)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    app.state.worker = state
    return app


def _sse(obj: Any) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
