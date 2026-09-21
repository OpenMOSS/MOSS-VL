"""Gateway plane REST control surface (GATEWAY_PLAN.md §2-P1).

Paths are absolute (no router prefix):

  POST   /v1/realtime/sessions            create session → 201 + one-shot ws_token
  GET    /v1/realtime/sessions/{sid}      status snapshot (phase/counters/turn_id)
  POST   /v1/realtime/sessions/{sid}/reset  same-replica reconnect + fresh ws_token
  DELETE /v1/realtime/sessions/{sid}      release session + replica slot
  GET    /v1/realtime/health              pool水位: instances/active/capacity/replicas
  GET    /v1/realtime/metrics             P4 gauges/counters + pool.status() summary
  GET    /v1/models                        proxy the first READY replica verbatim

Error shape: FastAPI HTTPException with a structured detail dict, e.g.
404 {"detail": {"code": "session_not_found"}} /
503 {"detail": {"code": "session_capacity_exceeded"}}.
"""
from __future__ import annotations

import asyncio
import time

import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..logging_conf import get_logger
from .deps import get_gateway_pool, get_gateway_registry, get_gateway_tokens
from .metrics import END_CLIENT_DELETE
from .pool import GatewayCapacityError, GatewayUnavailableError

log = get_logger(__name__)
router = APIRouter(tags=["gateway"])


def _not_found(session_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail={
        "code": "session_not_found", "message": f"no such session: {session_id}"})


def _capacity(exc: GatewayCapacityError) -> HTTPException:
    return HTTPException(status_code=503, detail={
        "code": "session_capacity_exceeded", "message": str(exc)})


def _unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail={
        "code": "no_available_replica", "message": str(exc)})


@router.post("/v1/realtime/sessions", status_code=201)
async def create_session(request: Request):
    registry = get_gateway_registry(request)
    try:
        session = await registry.create()
    except GatewayCapacityError as exc:
        registry.metrics.inc_error("session_capacity_exceeded")
        raise _capacity(exc)
    except GatewayUnavailableError as exc:
        registry.metrics.inc_error("no_available_replica")
        raise _unavailable(exc)
    try:
        return session.create_payload(session.mint_token())
    except KeyError:
        raise _not_found(session.session_id)


@router.get("/v1/realtime/sessions/{sid}")
async def get_session(request: Request, sid: str):
    registry = get_gateway_registry(request)
    session = registry.get(sid)
    if session is None:
        registry.metrics.inc_error("session_not_found")
        raise _not_found(sid)
    return session.snapshot()


@router.post("/v1/realtime/sessions/{sid}/reset")
async def reset_session(request: Request, sid: str):
    registry = get_gateway_registry(request)
    session = registry.get(sid)
    if session is None:
        registry.metrics.inc_error("session_not_found")
        raise _not_found(sid)
    try:
        return await session.reset()
    except KeyError:
        registry.metrics.inc_error("session_not_found")
        raise _not_found(sid)
    except GatewayCapacityError as exc:
        registry.metrics.inc_error("session_capacity_exceeded")
        raise _capacity(exc)
    except GatewayUnavailableError as exc:
        registry.metrics.inc_error("no_available_replica")
        raise _unavailable(exc)


@router.delete("/v1/realtime/sessions/{sid}")
async def delete_session(request: Request, sid: str):
    registry = get_gateway_registry(request)
    session = registry.get(sid)
    if session is None:
        registry.metrics.inc_error("session_not_found")
        raise _not_found(sid)
    await session.adestroy("deleted", end_reason=END_CLIENT_DELETE)
    return {"session_id": sid, "deleted": True}


@router.get("/v1/realtime/health")
async def gateway_health(request: Request):
    return get_gateway_pool(request).status()


@router.get("/v1/realtime/metrics")
async def gateway_metrics(request: Request):
    """P4 scrape endpoint: all gauges/counters (errors split by code) plus a
    pool.status() summary. Replica gauges are refreshed from the pool here, at
    read time, so they can never drift from slot bookkeeping."""
    registry = get_gateway_registry(request)
    pool = get_gateway_pool(request)
    registry.metrics.update_pool_gauges(pool)
    snap = registry.metrics.snapshot()
    snap["pool"] = pool.status()
    return snap


@router.get("/v1/models")
async def list_models(request: Request):
    pool = get_gateway_pool(request)
    deadline = time.monotonic() + max(0.01, pool.s.gateway_create_timeout_s)
    bad_json = False
    for index, url in pool.routable_replicas():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        def fetch():
            with requests.get(f"{url}/v1/models", timeout=min(
                max(1.0, pool.s.sglang_omni_connect_timeout_s), remaining / 2
            )) as response:
                response.raise_for_status() if response.status_code >= 500 else None
                return response.status_code, response.json()
        try:
            status, body = await asyncio.to_thread(fetch)
        except requests.RequestException:
            pool.mark_unhealthy(index)
            continue
        except ValueError:
            bad_json = True
            pool.mark_unhealthy(index)
            continue
        return JSONResponse(status_code=status, content=body)
    if bad_json:
        raise HTTPException(status_code=502, detail={"code": "bad_upstream", "message": "replicas returned invalid model metadata"})
    raise _unavailable(GatewayUnavailableError("no reachable model metadata endpoint"))
