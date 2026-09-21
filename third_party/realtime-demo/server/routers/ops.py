"""Ops endpoints: health + consolidated status.

Model loading lives in routers/sessions.py (`POST /api/models/load`) so it can
guard against live sessions holding the GPU replicas.

`/api/status` additionally surfaces the VLM replica pool aggregate
(capacity/busy/replicas — inside the `vlm` blob), the placement plan summary,
and an event-loop lag gauge (the WSS-smoothness acceptance metric: p99 < 50 ms
at full replica load).
"""
from __future__ import annotations

import asyncio
import os
import socket
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from fastapi import APIRouter, Depends, FastAPI

from ..config import REPO_ROOT
from ..deps import Runtime, get_runtime
from ..logging_conf import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["ops"])


def _git_rev() -> Optional[str]:
    """Current commit (short), via plain file reads — no subprocess, no git dep."""
    git = os.path.join(REPO_ROOT, ".git")
    try:
        with open(os.path.join(git, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head[:12]
        ref = head.split(None, 1)[1]
        loose = os.path.join(git, *ref.split("/"))
        if os.path.exists(loose):
            with open(loose, encoding="utf-8") as f:
                return f.read().strip()[:12]
        with open(os.path.join(git, "packed-refs"), encoding="utf-8") as f:
            for line in f:
                if line.strip().endswith(" " + ref):
                    return line.split()[0][:12]
    except OSError:
        pass
    return None


# Which-stack-am-I-on discriminator: several stacks can serve off this shared
# repo at once (stale pods keep old gateways alive AND share data/, so their
# journals interleave — see `demo.sh doctor`). Stamped once at import and shown
# by /api/health + /api/status; a gateway answering WITHOUT `boot` is a
# pre-stamp build, i.e. almost certainly a stale stray.
BOOT_INFO: Dict[str, Any] = {
    "ts": time.time(),
    "pid": os.getpid(),
    "host": socket.gethostname(),
    "git": _git_rev(),
}

_LAG_TICK_S = 0.5
_LAG_WINDOW = 240  # ~2 minutes of samples


class LoopLagGauge:
    """Measures event-loop scheduling delay: sleep(t) waking late by X ⇒ every
    coroutine (WS sends included) waited ~X for the loop."""

    def __init__(self) -> None:
        self.samples: Deque[float] = deque(maxlen=_LAG_WINDOW)
        self._task: Optional[asyncio.Task] = None

    async def _run(self) -> None:
        while True:
            before = time.monotonic()
            await asyncio.sleep(_LAG_TICK_S)
            self.samples.append(max(0.0, time.monotonic() - before - _LAG_TICK_S))

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run(), name="loop-lag-gauge")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def snapshot(self) -> Optional[Dict[str, float]]:
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
        return {
            "p50_ms": round(pick(0.50) * 1000.0, 2),
            "p99_ms": round(pick(0.99) * 1000.0, 2),
            "max_ms": round(ordered[-1] * 1000.0, 2),
            "samples": float(len(ordered)),
        }


_loop_lag_gauge: Optional[LoopLagGauge] = None


def start_loop_lag_gauge(app: FastAPI) -> None:
    global _loop_lag_gauge
    gauge = LoopLagGauge()
    gauge.start()
    app.state.loop_lag = gauge
    _loop_lag_gauge = gauge


async def stop_loop_lag_gauge(app: FastAPI) -> None:
    global _loop_lag_gauge
    gauge = getattr(app.state, "loop_lag", None) or _loop_lag_gauge
    _loop_lag_gauge = None
    if gauge is not None:
        await gauge.stop()


def _placement_summary(rt: Runtime) -> Optional[Dict[str, Any]]:
    plan = getattr(rt, "plan", None)  # tests build partial Runtimes
    if plan is None:
        return None
    return {
        "gpus": [
            {"index": g.index, "name": g.name, "mem_total_mib": g.mem_total_mib,
             "compute_cap": list(g.compute_cap)}
            for g in plan.gpus],
        "workers": [
            {"worker_id": w.worker_id, "gpu": w.gpu_index, "port": w.port,
             "attn_impl": w.attn_impl, "fake": w.fake}
            for w in plan.workers],
        "tts_sidecars": [
            {"sidecar_id": t.sidecar_id, "gpu": t.gpu_index, "port": t.port}
            for t in plan.tts],
        "offline": [
            {"replica_id": o.replica_id, "gpu": o.gpu_index, "port": o.port}
            for o in getattr(plan, "offline", ())],
        "asr_device": plan.asr_device,
        "warnings": list(plan.warnings),
    }


@router.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "boot": BOOT_INFO}


@router.get("/status")
async def status(rt: Runtime = Depends(get_runtime)) -> Dict[str, Any]:
    gauge = _loop_lag_gauge
    memory = getattr(rt, 'memory_store', None)
    writer = getattr(rt, 'memory_writer', None)
    return {
        "ok": True,
        "boot": BOOT_INFO,
        "vlm": rt.vlm.status(),
        # offline chat plane; None = no dedicated plane (chat rides rt.vlm),
        # loaded:false = plane exists but down (chat is FALLING BACK right now)
        "vlm_offline": rt.vlm_offline.status() if getattr(rt, "vlm_offline", None) else None,
        "voice": rt.voice_status(),
        "history": rt.history.status() if getattr(rt, 'history', None) is not None else None,
        "memory": await asyncio.to_thread(memory.resource_status) if memory is not None else None,
        "memory_writer": writer.status() if writer is not None else None,
        "capture_mode": rt.settings.capture_mode,
        "sessions": rt.session_manager.list_snapshots(),
        "placement": _placement_summary(rt),
        "loop_lag": gauge.snapshot() if gauge is not None else None,
    }
