"""Integration smoke test for native DP realtime serving (tp=1, dp=2).

Launches a real two-replica server via examples/run_moss_vl_realtime_server.py
as a subprocess and drives the WebSocket protocol end to end:

* the launcher waits for *both* replica warmups before listening;
* two sessions land on different replicas (aggregate capacity 2);
* a third session is rejected with session_capacity_exceeded / close 1013;
* aborting one session frees its replica; the other session is unaffected.

Requires a local MOSS-VL checkpoint and two idle GPUs
(sets and honors :envvar:`MOSS_VL_REALTIME_TEST_MODEL`). Skipped cleanly
otherwise, e.g. on GPU-less CI hosts.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCHER = _REPO_ROOT / "examples" / "run_moss_vl_realtime_server.py"
_MODEL = os.environ.get("MOSS_VL_REALTIME_TEST_MODEL", "")


def _two_gpus_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except Exception:  # noqa: BLE001 - probing torch init, any failure means "unusable"
        return False


pytestmark = pytest.mark.skipif(
    not _MODEL or not _two_gpus_available(),
    reason="needs MOSS_VL_REALTIME_TEST_MODEL and two idle CUDA GPUs",
)

_STARTUP_TIMEOUT_S = 900.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _WsSession:
    """Minimal native realtime client: created -> configured -> ready/abort."""

    def __init__(self, websocket) -> None:
        self.websocket = websocket

    async def handshake(self) -> None:
        message = json.loads(await self.websocket.recv())
        assert message["type"] == "session.created"
        await self.websocket.send(json.dumps({"type": "session.configure"}))
        message = json.loads(await self.websocket.recv())
        assert message["type"] == "session.configured"
        message = json.loads(await self.websocket.recv())
        assert message["type"] == "session.ready"
        self.session_id = message["session_id"]

    async def abort(self) -> None:
        await self.websocket.send(json.dumps({"type": "session.abort"}))
        while True:
            message = json.loads(await self.websocket.recv())
            if message["type"] == "session.done":
                return


async def _run_dp_smoke(port: int) -> None:
    import websockets

    url = f"ws://127.0.0.1:{port}/v1/video/realtime"

    async with websockets.connect(url, max_size=None) as ws1, websockets.connect(
        url, max_size=None
    ) as ws2:
        session1 = _WsSession(ws1)
        session2 = _WsSession(ws2)
        await session1.handshake()
        await session2.handshake()

        # max_running_requests=1 per replica, two replicas -> the third
        # session must be rejected while both replicas are full.
        import websockets as _ws

        async with _ws.connect(url, max_size=None) as ws3:
            message = json.loads(await ws3.recv())
            assert message.get("code") == "session_capacity_exceeded"
            with pytest.raises(_ws.exceptions.ConnectionClosed) as rejection:
                await ws3.recv()
        assert rejection.value.code == 1013

        # Releasing one replica admits a replacement session; the pinned
        # session is untouched.
        await session1.abort()
        async with websockets.connect(url, max_size=None) as ws3:
            session3 = _WsSession(ws3)
            await session3.handshake()
            assert session3.session_id != session2.session_id

        await session2.abort()


def test_dp2_realtime_smoke(tmp_path: Path) -> None:
    """Two-replica realtime server: balanced pinning, 1013 at capacity."""
    port = _free_port()
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(_LAUNCHER),
                "--model-path",
                _MODEL,
                "--dp-size",
                "2",
                "--gpus",
                "0,1",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--max-running-requests",
                "1",
                "--context-length",
                "131072",
                "--mem-fraction-static",
                "0.40",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_listen(port, proc, log_path)
            asyncio.run(asyncio.wait_for(_run_dp_smoke(port), timeout=300.0))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def _wait_for_listen(port: int, proc: subprocess.Popen, log_path: Path) -> None:
    import time
    import urllib.request

    deadline = time.monotonic() + _STARTUP_TIMEOUT_S
    while True:
        if proc.poll() is not None:
            raise AssertionError(
                f"server exited early (rc={proc.returncode}); log:\n"
                + log_path.read_text()[-8000:]
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except Exception:  # noqa: BLE001, S110 - poll until the server binds
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"server did not listen within {_STARTUP_TIMEOUT_S}s; log:\n"
                + log_path.read_text()[-8000:]
            )
        time.sleep(1.0)
