"""Optional v2 application sharing the native session capacity and model."""

from __future__ import annotations

import asyncio
import hmac
import os
from contextlib import asynccontextmanager, contextmanager, suppress
from functools import partial

import uvicorn
from fastapi import FastAPI, WebSocket

from .session import V2Session
from .usage import empty_usage


async def create_v2_app(manager, *, model_version=None):
    info = await manager.client.admin("realtime_v2_info", timeout_s=20)
    sources = [
        result
        for result in info.get("results", [])
        if result.get("success") and result.get("data", {}).get("realtime_v2")
    ]
    if len(sources) != 1:
        raise RuntimeError("v2 requires exactly one accounting-capable MOSS-VL stage")
    stage = sources[0]["stage"]
    limit = sources[0]["data"]["context_limit"]
    if type(limit) is not int or limit <= 0:
        raise RuntimeError("backend did not advertise a valid context limit")
    factory = partial(
        V2Session,
        context_limit=limit,
        accounting_stage=stage,
        model_version=model_version,
    )
    active = set()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.gather(
            *(session.teardown() for session in tuple(active)), return_exceptions=True
        )
        await asyncio.gather(
            *(manager.close(session.session_id) for session in tuple(active)),
            return_exceptions=True,
        )

    app = FastAPI(title="MOSS-VL API v2", lifespan=lifespan)
    key = os.environ.get("VL_API_V2_API_KEY")

    @app.get("/health")
    async def health():
        return manager.client.health()

    @app.websocket("/v1/video/realtime")
    async def realtime(ws: WebSocket):
        if key and not hmac.compare_digest(
            ws.headers.get("authorization", ""), "Bearer " + key
        ):
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            session = manager.open(ws, session_factory=factory)
        except RuntimeError as exc:
            await ws.send_json(
                {
                    "type": "error",
                    "code": "session_capacity_exceeded",
                    "message": str(exc),
                }
            )
            await ws.send_json(
                {"type": "session.done", "reason": "error", "usage": empty_usage()}
            )
            await ws.close(code=1013)
            return
        active.add(session)
        try:
            await session.run()
        finally:
            try:
                await session.teardown()
            finally:
                active.discard(session)
                await manager.close(session.session_id)

    return app


class SecondaryServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        # The existing primary server remains the single signal owner.
        yield


async def serve_pair(primary, secondary, failure):
    servers = [
        asyncio.create_task(primary.serve()),
        asyncio.create_task(secondary.serve()),
    ]
    failure_task = asyncio.create_task(failure)
    try:
        done, _ = await asyncio.wait(
            [*servers, failure_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            await task
    finally:
        primary.should_exit = secondary.should_exit = True
        with suppress(TimeoutError):
            await asyncio.wait_for(asyncio.gather(*servers, return_exceptions=True), 60)
        for task in [*servers, failure_task]:
            task.cancel()
        await asyncio.gather(*servers, failure_task, return_exceptions=True)
