"""FastAPI application factory + lifespan.

Startup (multi-GPU aware): probe the GPU topology → compute the placement plan
(VLM workers / ASR device / TTS sidecars) → spawn the TTS sidecar pool → build
the runtime → spawn + health-gate the VLM workers (VLM_DEPLOY=workers) → start
voice. All heavy work runs off the event loop.

NOTE: frontend serving is intentionally NOT wired here yet — this backend exposes
only the API surface (the SPA/dist mount is deferred to the frontend-integration
milestone). Run:

    <repo>/.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .deps import build_runtime, set_runtime
from .gpu.placement import format_plan, plan_placement
from .gpu.supervisor import SglangSidecarSupervisor, TtsSidecarPool, VlmWorkerSupervisor
from .gpu.topology import probe_topology
from .logging_conf import configure_logging, get_logger
from .routers import chat, history, media, ops, session_ws, sessions, speech

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # settings first: get_settings() merges .env.deploy into os.environ, and
    # configure_logging reads LOG_LEVEL/MOSS_LOG_FILE from there
    settings = get_settings()
    configure_logging()
    log.info("Starting MOSS-Realtime backend (deploy=%s, capture_mode=%s)",
             settings.vlm_deploy, settings.capture_mode)

    # ---- gateway plane (server/gateway/): REST control + WSS passthrough in
    # front of the sglang-omni pool; only when replicas are configured ----
    gateway_registry = None
    gateway_pool = None
    if settings.sglang_omni_urls.strip():
        from .gateway.pool import GatewayPool
        from .gateway.session import GatewayRegistry
        from .gateway.tokens import TokenIssuer

        gateway_pool = GatewayPool(settings)
        gateway_pool.start_prober()
        gateway_registry = GatewayRegistry(
            settings, gateway_pool, TokenIssuer(settings.gateway_ws_token_ttl_s))
        app.state.gateway_pool = gateway_pool
        app.state.gateway_registry = gateway_registry
        app.state.gateway_tokens = gateway_registry.tokens
        log.info("Gateway plane up: %d sglang-omni replica(s)", gateway_pool.capacity)

    # ---- GPU topology → placement plan (drives VLM/ASR/TTS process layout) ----
    topology = await asyncio.to_thread(probe_topology)
    plan = plan_placement(topology, settings)
    log.info("%s", format_plan(plan))

    # sidecars first: the voice runtime probes their /health once at init
    tts_pool = TtsSidecarPool(settings, plan)
    await asyncio.to_thread(tts_pool.spawn_all)
    app.state.tts_pool = tts_pool

    runtime = build_runtime(plan)
    runtime.tts_pool = tts_pool
    set_runtime(runtime)

    # size the default executor for N sessions' worth of to_thread hops
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=max(32, 8 * max(1, plan.capacity)),
                           thread_name_prefix="gw"))

    # ---- VLM workers + offline sglang sidecars (spawned CONCURRENTLY: the
    # sglang first boot — weights + flashinfer JIT — overlaps the staggered
    # worker gate; disjoint GPUs and ports make this safe) ----
    supervisor = None
    sglang_supervisor = None
    spawn_jobs = []
    if hasattr(runtime.vlm, "set_replica_health"):  # VlmReplicaPool
        supervisor = VlmWorkerSupervisor(settings, plan, runtime.vlm)
        runtime.vlm_supervisor = supervisor
        spawn_jobs.append(asyncio.to_thread(supervisor.spawn_all))  # spawn + health-gate
    vlm_offline = getattr(runtime, "vlm_offline", None)  # tests inject partial Runtimes
    if vlm_offline is not None:
        sglang_supervisor = SglangSidecarSupervisor(settings, plan, vlm_offline)
        runtime.sglang_supervisor = sglang_supervisor
        spawn_jobs.append(asyncio.to_thread(sglang_supervisor.spawn_all))
    if spawn_jobs:
        await asyncio.gather(*spawn_jobs)
    if supervisor is not None:
        supervisor.start_monitor()
    if sglang_supervisor is not None:
        sglang_supervisor.start_monitor()

    # heavy startup work off the event loop
    await asyncio.to_thread(runtime.open_persistence)
    await asyncio.to_thread(runtime.start_voice)
    await asyncio.to_thread(runtime.maybe_load_vlm)
    ops.start_loop_lag_gauge(app)
    log.info("Backend ready (vlm capacity=%d, offline sglang=%d).",
             getattr(runtime.vlm, "capacity", 1),
             getattr(vlm_offline, "capacity", 0) if vlm_offline else 0)
    try:
        yield
    finally:
        log.info("Shutting down backend.")
        if gateway_registry is not None:
            try:
                await gateway_registry.aclose()
            except Exception as exc:  # noqa: BLE001
                log.warning("gateway registry shutdown failed: %s", exc)
            gateway_pool.close()
        try:
            await runtime.session_manager.aclose()
        except Exception as exc:  # noqa: BLE001
            log.warning("session manager shutdown failed: %s", exc)
        finally:
            set_runtime(None)
            await ops.stop_loop_lag_gauge(app)
            if supervisor is not None:
                await supervisor.stop_monitor()
                await asyncio.to_thread(supervisor.stop_all)
            if sglang_supervisor is not None:
                await sglang_supervisor.stop_monitor()
                await asyncio.to_thread(sglang_supervisor.stop_all)
            try:
                # after aclose(): the manager's finalize jobs must reach the index
                await asyncio.to_thread(runtime.close_persistence)
            except Exception as exc:  # noqa: BLE001
                log.warning("persistence shutdown failed: %s", exc)
            await asyncio.to_thread(tts_pool.stop_all)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="MOSS-Realtime Backend", version="0.1.0", lifespan=lifespan)
    origins = [o.strip() for o in settings.cors_origins.split(",")] if settings.cors_origins else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(ops.router)
    # single-socket session plane (backend_overhaul.md §3)
    app.include_router(sessions.router)
    app.include_router(session_ws.router)
    # offline chat stays separate from the realtime socket by design (§1)
    app.include_router(chat.router)
    # one-shot ASR/TTS for the chat page (dictation + read-aloud)
    app.include_router(speech.router)
    # durable history + CAS media store (server/persistence/)
    app.include_router(history.router)
    app.include_router(media.router)
    # gateway plane (server/gateway/): REST control + WSS passthrough for
    # external clients speaking the sglang-omni data-plane protocol verbatim;
    # only mounted when replicas are configured (GATEWAY_PLAN.md §2-P1)
    if settings.sglang_omni_urls.strip():
        from .gateway import rest as gateway_rest
        from .gateway import ws as gateway_ws
        app.include_router(gateway_rest.router)
        app.include_router(gateway_ws.router)
    return app


app = create_app()
