"""Gateway plane REST tests (GATEWAY_PLAN.md §2-P1 acceptance).

Run:  <repo>/.venv/bin/python -m pytest server/tests/test_gateway_rest.py -q

Boots a bare FastAPI app with ONLY the gateway routers (pool/registry/tokens
injected into app.state — deps.py) behind a real in-process uvicorn, fronted by
a FakeSglangOmniServer. Covers the full REST lifecycle, 404/503 error shapes,
the health aggregate, and the /v1/models proxy. The WS-side harness lives here
too (start_gateway/http/wait_until) and is reused by test_gateway_ws.py.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

import uvicorn
from fastapi import FastAPI

from server.config import Settings
from server.gateway import rest as gateway_rest
from server.gateway import ws as gateway_ws
from server.gateway.pool import READY, GatewayPool
from server.gateway.session import GatewayRegistry
from server.gateway.tokens import TokenIssuer
from server.tests.test_sglang_omni_adapter import FakeSglangOmniServer


class GatewayFakeOmni(FakeSglangOmniServer):
    """FakeSglangOmniServer + the OpenAI-style GET /v1/models endpoint."""

    async def _process_request(self, connection: Any, request: Any) -> Any:
        if request.path == "/v1/models":
            return connection.respond(200, json.dumps(
                {"object": "list",
                 "data": [{"id": "fake-moss-vl", "object": "model",
                           "created": 1, "owned_by": "fake"}]}))
        return await super()._process_request(connection, request)


class Rig:
    def __init__(self, server, task, host, pool, tokens, registry):
        self.server = server
        self.task = task
        self.host = host
        self.base = f"http://{host}"
        self.ws_base = f"ws://{host}"
        self.pool = pool
        self.tokens = tokens
        self.registry = registry

    async def stop(self) -> None:
        self.server.should_exit = True
        await asyncio.wait_for(self.task, timeout=10)
        await self.registry.aclose()
        self.pool.close()


async def start_gateway(fake: GatewayFakeOmni, *, ws_max_size: int = 64 * 1024 * 1024,
                        **overrides: Any) -> Rig:
    kwargs = dict(
        sglang_omni_urls=fake.url,
        sglang_omni_connect_timeout_s=5.0,
        sglang_omni_health_interval_s=600.0,  # prober effectively off in tests
    )
    kwargs.update(overrides)
    settings = Settings(**kwargs)
    pool = GatewayPool(settings)
    await asyncio.to_thread(pool.probe_all)
    tokens = TokenIssuer(settings.gateway_ws_token_ttl_s)
    registry = GatewayRegistry(settings, pool, tokens)
    app = FastAPI(title="gateway-plane-test")
    app.include_router(gateway_rest.router)
    app.include_router(gateway_ws.router)
    app.state.gateway_pool = pool
    app.state.gateway_registry = registry
    app.state.gateway_tokens = tokens

    config = uvicorn.Config(app, host="127.0.0.1", port=0,
                            ws_max_size=ws_max_size,
                            log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
        assert not task.done(), "uvicorn failed to start"
    port = server.servers[0].sockets[0].getsockname()[1]
    return Rig(server, task, f"127.0.0.1:{port}", pool, tokens, registry)


def http(method: str, url: str, body: Optional[dict] = None) -> Tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


async def rest(method: str, url: str, body: Optional[dict] = None) -> Tuple[int, dict]:
    return await asyncio.to_thread(http, method, url, body)


async def wait_until(predicate, timeout: float = 6.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


# ---------------------------------------------------------------- lifecycle


async def _test_rest_lifecycle() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        # create → 201 with the token bundle
        code, body = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body)
        sid = body["session_id"]
        assert body["ws_token"] and body["ws_url"] == "/v1/realtime"
        assert body["expires_in"] == 60

        # status snapshot: fields straight from the cached session.created
        code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
        assert code == 200, (code, snap)
        assert snap["session_id"] == sid and snap["status"] == "created"
        assert snap["frames_accepted"] == 0 and snap["turn_id"] == 0
        assert snap["model"] == "fake-moss-vl" and snap["request_id"] == "req-1"
        assert snap["created_at"] > 0

        # reset → same shape, fresh token, counters zeroed, same session_id
        code, reset = await rest("POST", f"{rig.base}/v1/realtime/sessions/{sid}/reset")
        assert code == 200, (code, reset)
        assert reset["session_id"] == sid and reset["ws_token"] != body["ws_token"]
        assert reset["ws_url"] == "/v1/realtime"
        code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{sid}")
        assert code == 200 and snap["status"] == "created"

        # delete → 200; afterwards every session route 404s with the coded detail
        code, _ = await rest("DELETE", f"{rig.base}/v1/realtime/sessions/{sid}")
        assert code == 200
        for method, url in (("GET", f"{rig.base}/v1/realtime/sessions/{sid}"),
                            ("DELETE", f"{rig.base}/v1/realtime/sessions/{sid}"),
                            ("POST", f"{rig.base}/v1/realtime/sessions/{sid}/reset")):
            code, err = await rest(method, url)
            assert code == 404 and err["detail"]["code"] == "session_not_found", (method, code, err)
        print("REST lifecycle (create→status→reset→delete, 404 shape): OK")
    finally:
        await rig.stop()
        fake.close()


def test_rest_lifecycle() -> None:
    asyncio.run(_test_rest_lifecycle())


# ---------------------------------------------------------------- capacity


async def _test_rest_capacity() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        code, body = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body)
        sid = body["session_id"]
        # slot=1 per replica: the second create must 503 with the coded detail
        code, err = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 503 and err["detail"]["code"] == "session_capacity_exceeded", (code, err)
        # freeing the slot makes room again
        code, _ = await rest("DELETE", f"{rig.base}/v1/realtime/sessions/{sid}")
        assert code == 200
        code, body2 = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body2)
        print("REST capacity (busy 503 → release → create): OK")
    finally:
        await rig.stop()
        fake.close()


def test_rest_capacity() -> None:
    asyncio.run(_test_rest_capacity())


async def _test_rest_capacity_rejected_by_omni() -> None:
    fake = GatewayFakeOmni(reject_capacity=True).start()
    rig = await start_gateway(fake)
    try:
        code, err = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 503 and err["detail"]["code"] == "session_capacity_exceeded", (code, err)
        # capacity rejection = the instance is full, not wedged: BUSY, not DOWN
        assert rig.pool.replicas[0].state == "BUSY"
        print("REST capacity (omni reject → 503 + replica BUSY/full): OK")
    finally:
        await rig.stop()
        fake.close()


def test_rest_capacity_rejected_by_omni() -> None:
    asyncio.run(_test_rest_capacity_rejected_by_omni())


async def _test_rest_multislot_same_replica() -> None:
    """slots=2: two sessions share one replica, the third 503s; freeing a slot
    makes room again."""
    fake = GatewayFakeOmni(max_sessions=2).start()
    rig = await start_gateway(fake, sglang_omni_sessions_per_replica=2)
    try:
        code, body1 = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body1)
        assert rig.pool.replicas[0].state == READY  # one slot still free
        code, body2 = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body2)
        assert rig.pool.replicas[0].state == "BUSY" and rig.pool.busy == 2

        code, err = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 503 and err["detail"]["code"] == "session_capacity_exceeded", (code, err)

        code, _ = await rest("DELETE", f"{rig.base}/v1/realtime/sessions/{body1['session_id']}")
        assert code == 200
        assert rig.pool.replicas[0].state == READY and rig.pool.busy == 1
        code, body3 = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body3)
        print("REST multislot (same replica ×2 → 503 → release → create): OK")
    finally:
        await rig.stop()
        fake.close()


def test_rest_multislot_same_replica() -> None:
    asyncio.run(_test_rest_multislot_same_replica())


async def _test_rest_capacity_reject_skips_to_next_replica() -> None:
    """A capacity-rejecting replica is marked full and skipped; the session
    lands on the next READY replica."""
    wedged = GatewayFakeOmni(reject_capacity=True).start()
    good = GatewayFakeOmni().start()
    rig = None
    try:
        rig = await start_gateway(good, sglang_omni_urls=f"{wedged.url},{good.url}")
        code, body = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body)
        assert rig.pool.replicas[0].state == "BUSY"  # full, not quarantined
        assert rig.pool.replicas[1].used == 1
        code, snap = await rest("GET", f"{rig.base}/v1/realtime/sessions/{body['session_id']}")
        assert code == 200 and snap["replica"] == good.url, (code, snap)
        print("REST capacity reject → next READY replica: OK")
    finally:
        if rig is not None:
            await rig.stop()
        wedged.close()
        good.close()


def test_rest_capacity_reject_skips_to_next_replica() -> None:
    asyncio.run(_test_rest_capacity_reject_skips_to_next_replica())


async def _test_rest_full_mark_cleared_by_prober() -> None:
    """The full mark from an omni capacity rejection is cleared by the prober
    once the instance is healthy and has room again (no permanent wedge)."""
    fake = GatewayFakeOmni(reject_capacity=True).start()
    rig = await start_gateway(fake, sglang_omni_health_interval_s=0.2)
    rig.pool.start_prober()
    try:
        code, err = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 503, (code, err)
        assert rig.pool.replicas[0].state == "BUSY"  # full mark, not DOWN

        fake.reject_capacity = False  # omni has room again
        assert await wait_until(lambda: rig.pool.replicas[0].state == READY, timeout=8.0), \
            "prober must clear the stale full mark"
        code, body = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201, (code, body)
        print("REST full mark cleared by prober (503 → recover → 201): OK")
    finally:
        await rig.stop()
        fake.close()


def test_rest_full_mark_cleared_by_prober() -> None:
    asyncio.run(_test_rest_full_mark_cleared_by_prober())


# ---------------------------------------------------------------- health + models


async def _test_health_and_models() -> None:
    fake = GatewayFakeOmni().start()
    rig = await start_gateway(fake)
    try:
        code, health = await rest("GET", f"{rig.base}/v1/realtime/health")
        assert code == 200, (code, health)
        assert health["instances"] == 1 and health["capacity"] == 1
        assert health["active_sessions"] == 0
        assert health["replicas"] == [{"url": fake.url, "state": READY}]

        code, body = await rest("POST", f"{rig.base}/v1/realtime/sessions")
        assert code == 201
        code, health = await rest("GET", f"{rig.base}/v1/realtime/health")
        assert health["active_sessions"] == 1
        assert health["replicas"][0]["state"] == "BUSY"

        code, models = await rest("GET", f"{rig.base}/v1/models")
        assert code == 200 and models["data"][0]["id"] == "fake-moss-vl", (code, models)
        print("health aggregate + /v1/models proxy: OK")
    finally:
        await rig.stop()
        fake.close()


def test_health_and_models() -> None:
    asyncio.run(_test_health_and_models())


def main() -> int:
    asyncio.run(_test_rest_lifecycle())
    asyncio.run(_test_rest_capacity())
    asyncio.run(_test_rest_capacity_rejected_by_omni())
    asyncio.run(_test_rest_multislot_same_replica())
    asyncio.run(_test_rest_capacity_reject_skips_to_next_replica())
    asyncio.run(_test_rest_full_mark_cleared_by_prober())
    asyncio.run(_test_health_and_models())
    print("\nGATEWAY REST TESTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
