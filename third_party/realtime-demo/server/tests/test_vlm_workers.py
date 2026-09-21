"""Multi-GPU workers integration: real gateway + N FAKE VLM worker processes.

Run:  <repo>/.venv/bin/python -m server.tests.test_vlm_workers

Boots the REAL app lifespan (topology probe → placement → supervisor spawns 4
`python -m server.vlm_worker` subprocesses in VLM_WORKER_FAKE mode) and drives
it over HTTP/WS: 4 concurrent sessions stream scripted captions through the
proxy/pool plane, the 5th create 409s, a DELETE frees a replica for a new
create, offline chat SSE relays through a worker, and /api/status surfaces the
pool aggregate. No GPU, no model, no board FS needed.
"""
from __future__ import annotations

# env BEFORE any server import — Settings() is env-driven and cached
import os  # noqa: E402

os.environ.update({
    "VLM_DEPLOY": "workers",
    "VLM_WORKER_FAKE": "1",
    "VLM_WORKER_GPUS": "0,0,0,0",           # 4 fake workers on "gpu 0"
    "VLM_WORKER_BASE_PORT": "19410",
    "VLM_SPAWN_STAGGER_S": "0",
    "VLM_HEALTH_TIMEOUT_S": "60",
    "ASR_ENABLED": "0",
    "TTS_ENABLED": "0",
    "TTS_SPAWN": "0",
    "HISTORY_ENABLED": "0",
    "MEDIA_ENABLED": "0",
    "AUTOLOAD_VLM": "0",
    "MODEL_PATH": "",
    "SESSION_GRACE_SECONDS": "5",
})

import asyncio  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from typing import Optional, Tuple  # noqa: E402

import uvicorn  # noqa: E402
import websockets  # noqa: E402

N_WORKERS = 4


def http(method: str, url: str, body: Optional[dict] = None) -> Tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


async def start_server():
    from server import app as app_module

    config = uvicorn.Config(app_module.app, host="127.0.0.1", port=0,
                            log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
        assert not task.done(), "uvicorn failed to start"
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, f"127.0.0.1:{port}"


async def drive_text_turn(host: str, sid: str, ws_url: str) -> str:
    """Attach, send a typed turn, collect the response text, detach."""
    text = ""
    async with websockets.connect(f"ws://{host}{ws_url}") as ws:
        created = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert created["type"] == "session.created" and created["session_id"] == sid
        await ws.send(json.dumps({"type": "text.input", "text": "描述一下"}))
        deadline = asyncio.get_event_loop().time() + 15
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            assert remaining > 0, f"timeout waiting response.done (text so far: {text!r})"
            msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
            if isinstance(msg, (bytes, bytearray)):
                continue
            ev = json.loads(msg)
            if ev["type"] == "response.text.delta":
                text += ev["delta"]
            elif ev["type"] == "response.done":
                return text


async def main_async() -> None:
    server, server_task, host = await start_server()
    base = f"http://{host}"
    try:
        # ---- pool is up: capacity N, all replicas ready ----
        code, status = await asyncio.to_thread(http, "GET", f"{base}/api/status")
        assert code == 200, (code, status)
        vlm = status["vlm"]
        assert vlm["loaded"] is True and vlm["capacity"] == N_WORKERS, vlm
        assert vlm["busy"] == 0 and len(vlm["replicas"]) == N_WORKERS
        assert status["placement"]["workers"][0]["fake"] is True
        print(f"pool up: capacity={vlm['capacity']}: OK")

        # ---- N concurrent sessions; N+1 → 409 all-replicas-busy ----
        sessions = []
        for _ in range(N_WORKERS):
            code, body = await asyncio.to_thread(http, "POST", f"{base}/api/sessions",
                                                 {"config": {"capture_mode": "ptt"}})
            assert code == 201, (code, body)
            sessions.append(body)
        code, body = await asyncio.to_thread(http, "POST", f"{base}/api/sessions", {})
        assert code == 409, (code, body)
        assert "replicas busy" in str(body.get("detail")), body
        code, status = await asyncio.to_thread(http, "GET", f"{base}/api/status")
        assert status["vlm"]["busy"] == N_WORKERS
        print(f"{N_WORKERS} sessions + 409 on {N_WORKERS + 1}th: OK")

        # ---- all sessions stream scripted output concurrently over WS ----
        texts = await asyncio.gather(*[
            drive_text_turn(host, s["session_id"], s["ws_url"]) for s in sessions])
        assert all("fake VLM worker" in t for t in texts), texts
        print(f"{N_WORKERS} concurrent WS turns: OK")

        # ---- DELETE frees a replica; a new create succeeds ----
        code, _ = await asyncio.to_thread(
            http, "DELETE", f"{base}/api/sessions/{sessions[0]['session_id']}")
        assert code == 200
        await asyncio.sleep(0.5)
        code, body = await asyncio.to_thread(http, "POST", f"{base}/api/sessions", {})
        assert code == 201, (code, body)
        code, _ = await asyncio.to_thread(
            http, "DELETE", f"{base}/api/sessions/{body['session_id']}")
        assert code == 200
        print("release + re-acquire replica: OK")

        # ---- offline chat SSE relays through a worker ----
        req = urllib.request.Request(
            f"{base}/api/chat/stream",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        deltas = []

        def read_sse() -> None:
            with urllib.request.urlopen(req, timeout=20) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if line.startswith("data:"):
                        ev = json.loads(line[5:])
                        if ev["type"] == "generation_delta":
                            deltas.append(ev["delta"])
                        elif ev["type"] in ("generation_end", "generation_error"):
                            assert ev["type"] == "generation_end", ev
                            return

        await asyncio.to_thread(read_sse)
        assert "".join(deltas) == "fake offline chat reply", deltas
        print("offline chat via worker SSE: OK")

        # ---- worker crash: session errors cleanly; supervisor respawns ----
        import signal
        import subprocess

        code, body = await asyncio.to_thread(http, "POST", f"{base}/api/sessions", {})
        assert code == 201, (code, body)
        crash_sid, crash_ws = body["session_id"], body["ws_url"]
        pid = int(subprocess.run(
            ["pgrep", "-f", r"server\.vlm_worker --port 19410"],
            capture_output=True, text=True).stdout.split()[0])
        async with websockets.connect(f"ws://{host}{crash_ws}") as ws:
            created = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            assert created["type"] == "session.created"
            os.kill(pid, signal.SIGKILL)  # hard-crash the worker mid-session
            deadline = asyncio.get_event_loop().time() + 20
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                assert remaining > 0, "no vlm_stopped error after worker crash"
                ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
                if ev["type"] == "error" and ev.get("code") == "vlm_stopped":
                    break
        code, _ = await asyncio.to_thread(http, "DELETE", f"{base}/api/sessions/{crash_sid}")
        assert code == 200
        print("worker crash → clean session error: OK")

        # supervisor respawns the worker; replica returns to READY ≤ 60 s
        deadline = asyncio.get_event_loop().time() + 60
        while True:
            assert asyncio.get_event_loop().time() < deadline, "replica did not recover"
            code, status = await asyncio.to_thread(http, "GET", f"{base}/api/status")
            states = [r["state"] for r in status["vlm"]["replicas"]]
            if states.count("ready") == N_WORKERS:
                break
            await asyncio.sleep(2.0)
        code, body = await asyncio.to_thread(http, "POST", f"{base}/api/sessions", {})
        assert code == 201, (code, body)  # the respawned replica is usable
        await asyncio.to_thread(http, "DELETE", f"{base}/api/sessions/{body['session_id']}")
        print("worker respawn + replica recovery: OK")
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=30)


def main() -> int:
    asyncio.run(main_async())
    print("\nVLM WORKERS INTEGRATION OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
