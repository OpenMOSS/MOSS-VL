"""Process supervision for VLM workers, TTS sidecars and offline sglang.

Generalizes the server/sidecars.py pattern — spawn / adopt-if-healthy /
health-gate / killpg — to N processes from a PlacementPlan:

- `VlmWorkerSupervisor.spawn_all()` (blocking, run via to_thread) starts one
  `python -m server.vlm_worker` per WorkerSpec, staggered (N parallel bf16
  loads thrash the shared FS), then health-gates them all in parallel.
- `start_monitor()` runs a 10 s asyncio tick: refresh each worker's /health
  into the replica pool (the ONLY place worker health I/O happens once the
  app is up — `pool.status()` reads are cached and loop-safe), and respawn
  crashed workers with bounded backoff.
- `TtsSidecarPool` spawns/stops the plan's TTS sidecars the same way.
- `SglangSidecarSupervisor` runs one `sglang.launch_server` (fnlp-vision fork,
  .venv-sglang) per offline placement GPU: identity-checked adoption (shared
  boxes run OTHER sglang servers — never adopt/kill a foreign model), gate on
  /health_generate (a real decode: flashinfer JIT + warmup done before UP),
  then the same monitor/respawn loop.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..logging_conf import get_logger
from ..device_compat import set_visible_device
from ..sidecars import spawn_tts_sidecar, stop_tts_sidecar
from .placement import OfflineSpec, PlacementPlan, TtsSpec, WorkerSpec

log = get_logger(__name__)

MONITOR_INTERVAL_S = 10.0
RESPAWN_MAX_ATTEMPTS = 3
RESPAWN_BACKOFF_S = 10.0  # doubled per attempt
HEALTH_STRIKES = 2        # consecutive failed polls before DOWN


def _with_ffmpeg_libs(env: Dict[str, str], settings: Settings) -> Dict[str, str]:
    """Prepend the vendored FFmpeg .so closure to LD_LIBRARY_PATH.

    torchcodec (used by the checkpoints' video remote code) dlopens
    libavutil.so.5x at import; boxes routinely lack the shared libs even when
    an ffmpeg BINARY is installed. run_backend.sh exports this for the gateway,
    but children must not depend on HOW the gateway was launched (manual
    uvicorn, tests, adopted orphans) — so every supervisor applies it too.
    """
    ff = settings.ffmpeg_libs
    if ff and os.path.isdir(ff):
        current = env.get("LD_LIBRARY_PATH", "")
        if ff not in current.split(":"):
            env["LD_LIBRARY_PATH"] = f"{ff}:{current}" if current else ff
    return env


class _WorkerProc:
    def __init__(self, spec: WorkerSpec):
        self.spec = spec
        self.proc: Optional[subprocess.Popen] = None  # None = adopted / not spawned
        self.respawn_attempts = 0
        self.next_respawn_at = 0.0
        self.health_strikes = 0


class VlmWorkerSupervisor:
    def __init__(self, settings: Settings, plan: PlacementPlan, pool: Any):
        self.s = settings
        self.plan = plan
        self.pool = pool  # VlmReplicaPool — set_replica_health(i, health|None)
        self.workers = [_WorkerProc(spec) for spec in plan.workers]
        self._monitor_task: Optional[asyncio.Task] = None
        self._stopping = False

    # ------------------------------------------------------------ spawn / gate

    def _worker_env(self, spec: WorkerSpec) -> Dict[str, str]:
        env = _with_ffmpeg_libs(dict(os.environ), self.s)
        set_visible_device(env, spec.gpu_index)
        env["VLM_WORKER_ATTN"] = spec.attn_impl
        env["VLM_WORKER_FAKE"] = "1" if spec.fake else "0"
        # workers autoload (mirrors the gateway's old maybe_load_vlm condition);
        # the gateway's maybe_load_vlm no-ops in workers mode instead
        env["AUTOLOAD_VLM"] = "1" if self._expect_autoload() else "0"
        # rotating python log, one file per worker process (logging_conf.py);
        # uvicorn's banner/access lines land there too (log_config=None +
        # the uvicorn takeover in logging_conf)
        env["MOSS_LOG_FILE"] = self._log_path(spec)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        return env

    def _log_path(self, spec: WorkerSpec) -> str:
        # the worker's rotating log under logs/handler/backend/workers/ — its
        # ONLY log artifact: raw stdout goes to /dev/null at spawn
        return os.path.join(self.s.vlm_worker_log_dir, f"worker_{spec.worker_id}.log")

    def _spawn_one(self, w: _WorkerProc) -> None:
        spec = w.spec
        proxy = self.pool.replicas[spec.worker_id].proxy
        if proxy.fetch_health(timeout=2.0) is not None:
            log.info("Adopting already-healthy VLM worker %d at %s", spec.worker_id, spec.base_url)
            return
        cmd = [sys.executable, "-m", "server.vlm_worker",
               "--port", str(spec.port), "--worker-id", str(spec.worker_id)]
        log.info("Spawning VLM worker %d: %s (gpu=%d attn=%s log=%s)",
                 spec.worker_id, " ".join(cmd), spec.gpu_index, spec.attn_impl,
                 self._log_path(spec))
        # stdout/stderr are DISCARDED by design: everything python-level goes
        # through the worker's rotating MOSS_LOG_FILE (configured at import,
        # before anything heavy). The one loss is native stderr (CUDA aborts,
        # glibc) — accepted; the monitor still logs the exit code.
        w.proc = subprocess.Popen(
            cmd, env=self._worker_env(spec),
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group (killpg teardown)
        )

    def spawn_all(self) -> None:
        """Spawn every worker (staggered), then health-gate in parallel. Blocking."""
        spawned_any = False
        for w in self.workers:
            if spawned_any and not w.spec.fake and self.s.vlm_spawn_stagger_s > 0:
                time.sleep(self.s.vlm_spawn_stagger_s)
            before = w.proc
            self._spawn_one(w)
            spawned_any = spawned_any or (w.proc is not before)
        self._health_gate()

    def _expect_autoload(self) -> bool:
        return bool(self.s.autoload_vlm and self.s.model_path)

    def _health_gate(self) -> None:
        deadline = time.monotonic() + self.s.vlm_health_timeout_s
        # without autoload, "up" (health answers) is the gate — the model loads
        # later via POST /api/models/load, exactly like the old inproc flow
        need_loaded = self._expect_autoload()

        def gate(w: _WorkerProc) -> None:
            proxy = self.pool.replicas[w.spec.worker_id].proxy
            while time.monotonic() < deadline and not self._stopping:
                if w.proc is not None and w.proc.poll() is not None:
                    log.error("VLM worker %d exited during startup (rc=%s) — see %s",
                              w.spec.worker_id, w.proc.returncode, self._log_path(w.spec))
                    self.pool.set_replica_health(w.spec.worker_id, None)
                    return
                health = proxy.fetch_health(timeout=3.0)
                if health is not None:
                    self.pool.set_replica_health(w.spec.worker_id, health)
                    if health.get("loaded") or not need_loaded:
                        log.info("VLM worker %d up (loaded=%s attn=%s)",
                                 w.spec.worker_id, health.get("loaded"),
                                 health.get("attn_impl"))
                        return
                time.sleep(2.0)
            log.error("VLM worker %d not ready within %.0fs",
                      w.spec.worker_id, self.s.vlm_health_timeout_s)
            self.pool.set_replica_health(w.spec.worker_id, None)

        with ThreadPoolExecutor(max_workers=max(1, len(self.workers))) as pool:
            list(pool.map(gate, self.workers))

    # ------------------------------------------------------------ monitor / respawn

    def start_monitor(self) -> None:
        self._monitor_task = asyncio.get_running_loop().create_task(
            self._monitor_loop(), name="vlm-worker-monitor")

    async def stop_monitor(self) -> None:
        task, self._monitor_task = self._monitor_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_S)
            for w in self.workers:
                try:
                    await asyncio.to_thread(self._monitor_one, w)
                except Exception as exc:  # noqa: BLE001
                    log.warning("worker %d monitor tick failed: %s", w.spec.worker_id, exc)

    def _monitor_one(self, w: _WorkerProc) -> None:
        proxy = self.pool.replicas[w.spec.worker_id].proxy
        health = proxy.fetch_health(timeout=3.0)
        if health is not None:
            w.health_strikes = 0
            w.respawn_attempts = 0
            self.pool.set_replica_health(w.spec.worker_id, health)
            return
        w.health_strikes += 1
        if w.health_strikes < HEALTH_STRIKES and (w.proc is None or w.proc.poll() is None):
            return  # one blip — give it another tick before declaring DOWN
        self.pool.set_replica_health(w.spec.worker_id, None)
        self._maybe_respawn(w)

    def _maybe_respawn(self, w: _WorkerProc) -> None:
        if self._stopping:
            return
        crashed = w.proc is not None and w.proc.poll() is not None
        if w.proc is not None and not crashed:
            return  # process alive but unhealthy — let it recover / next strike
        if w.respawn_attempts >= RESPAWN_MAX_ATTEMPTS:
            if w.respawn_attempts == RESPAWN_MAX_ATTEMPTS:
                w.respawn_attempts += 1  # log once
                log.error("VLM worker %d exceeded %d respawns — staying DOWN",
                          w.spec.worker_id, RESPAWN_MAX_ATTEMPTS)
            return
        now = time.monotonic()
        if now < w.next_respawn_at:
            return
        w.respawn_attempts += 1
        w.next_respawn_at = now + RESPAWN_BACKOFF_S * (2 ** (w.respawn_attempts - 1))
        log.warning("Respawning VLM worker %d (attempt %d/%d)",
                    w.spec.worker_id, w.respawn_attempts, RESPAWN_MAX_ATTEMPTS)
        _kill_proc(w.proc)
        w.proc = None
        self._spawn_one(w)

    # ------------------------------------------------------------ teardown

    def stop_all(self) -> None:
        """Blocking teardown of every OWNED worker (adopted ones are left alone)."""
        self._stopping = True
        for w in self.workers:
            _kill_proc(w.proc)
            w.proc = None


class TtsSidecarPool:
    """Spawn/stop the plan's TTS sidecars (adopt-don't-own, killpg teardown)."""

    def __init__(self, settings: Settings, plan: PlacementPlan):
        self.s = settings
        self.specs: List[TtsSpec] = list(plan.tts)
        self.procs: List[Optional[subprocess.Popen]] = [None] * len(self.specs)

    @property
    def base_urls(self) -> List[str]:
        return [spec.base_url for spec in self.specs]

    def _log_path(self, spec: TtsSpec) -> str:
        if spec.sidecar_id == 0:
            return self.s.tts_sidecar_log  # keep today's primary log path
        root, ext = os.path.splitext(self.s.tts_sidecar_log)
        return f"{root}_{spec.sidecar_id}{ext or '.log'}"

    def spawn_all(self) -> None:
        """Blocking (run via to_thread); sidecars spawn+gate IN PARALLEL.

        Safe: each spec has its own port, GPU and log file, and spawn is
        adopt-don't-own per port. Serial gating cost ~2.3 min for 4 pytorch
        sidecars on the 8-GPU box; parallel is the slowest single spec.
        """
        if len(self.specs) == 1:
            self.procs[0] = spawn_tts_sidecar(
                self.s, base_url=self.specs[0].base_url,
                gpu_index=self.specs[0].gpu_index,
                log_path=self._log_path(self.specs[0]))
            return
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(self.specs)) as pool:
            futures = [
                pool.submit(spawn_tts_sidecar, self.s, base_url=spec.base_url,
                            gpu_index=spec.gpu_index, log_path=self._log_path(spec))
                for spec in self.specs
            ]
            for i, fut in enumerate(futures):
                try:
                    self.procs[i] = fut.result()
                except Exception as exc:  # noqa: BLE001 — TTS degrades, never kills the api
                    log.exception("TTS sidecar %d spawn failed: %s", i, exc)
                    self.procs[i] = None

    def stop_all(self) -> None:
        for i, proc in enumerate(self.procs):
            stop_tts_sidecar(proc)
            self.procs[i] = None


class _SglangProc:
    def __init__(self, spec: OfflineSpec):
        self.spec = spec
        self.proc: Optional[subprocess.Popen] = None  # None = adopted / not spawned
        self.respawn_attempts = 0
        self.next_respawn_at = 0.0
        self.health_strikes = 0
        self.foreign = False  # port answers with a DIFFERENT model — hands off
        self.gated = False    # passed a real /health_generate decode since (re)spawn


class SglangSidecarSupervisor:
    """Spawn/adopt/gate/monitor the offline sglang sidecars (one per OfflineSpec).

    Mirrors VlmWorkerSupervisor with two twists: adoption is IDENTITY-CHECKED
    (`/get_model_info` must echo OFFLINE_MODEL_PATH — a foreign server on our
    port is left alone and the replica stays DOWN), and the startup gate polls
    `/health_generate` (a real decode) so flashinfer JIT/warmup never leaks
    into the first user request. Gate failure degrades — chat falls back to
    the online pool — it never raises into the lifespan.
    """

    def __init__(self, settings: Settings, plan: PlacementPlan, pool: Any):
        self.s = settings
        self.plan = plan
        self.pool = pool  # SglangOfflinePool — set_replica_health(i, health|None)
        self.sidecars = [_SglangProc(spec) for spec in plan.offline]
        self._monitor_task: Optional[asyncio.Task] = None
        self._stopping = False
        if settings.sglang_tp_size > 1:
            log.warning("SGLANG_TP_SIZE=%d: auto-placement assumes tp=1 per "
                        "offline GPU — multi-GPU tensor parallel needs a manual "
                        "layout (OFFLINE_GPU_COUNT + SGLANG_EXTRA_ARGS)",
                        settings.sglang_tp_size)

    # ------------------------------------------------------------ spawn / gate

    def _log_path(self, spec: OfflineSpec) -> str:
        return os.path.join(self.s.sglang_log_dir, f"sglang_{spec.replica_id}.log")

    def _cmd(self, spec: OfflineSpec) -> List[str]:
        cmd = [self.s.sglang_python, "-m", "sglang.launch_server",
               "--model-path", self.s.offline_model_path,
               "--host", "127.0.0.1", "--port", str(spec.port),
               "--trust-remote-code", "--enable-multimodal",
               "--dtype", "bfloat16",
               "--tp-size", str(self.s.sglang_tp_size),
               "--mem-fraction-static", str(self.s.sglang_mem_fraction),
               "--disable-fast-image-processor"]
        # NO --attention-backend knob: the fork force-selects flashinfer prefill
        # for MossVL and asserts on anything else (srt/server_args.py)
        if self.s.sglang_extra_args:
            cmd += self.s.sglang_extra_args.split()
        return cmd

    def _env(self, spec: OfflineSpec) -> Dict[str, str]:
        env = _with_ffmpeg_libs(dict(os.environ), self.s)
        set_visible_device(env, spec.gpu_index)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        # flashinfer JIT toolchain: archs without prebuilt cubins (sm_120 —
        # RTX PRO 6000 Blackwell) compile kernels at engine load, which needs
        # `ninja` (installed in the sglang venv's bin — never on the gateway's
        # PATH) and `nvcc` (CUDA_HOME/bin — the toolkit often isn't on PATH on
        # desktop boxes either). Prepend both so the JIT works no matter how
        # the gateway itself was launched.
        extra = [os.path.dirname(self.s.sglang_python)]
        cuda_home = env.get("CUDA_HOME") or "/usr/local/cuda"
        if os.path.isfile(os.path.join(cuda_home, "bin", "nvcc")):
            extra.append(os.path.join(cuda_home, "bin"))
            env.setdefault("CUDA_HOME", cuda_home)
        env["PATH"] = ":".join(extra + [env["PATH"]]) if env.get("PATH") else ":".join(extra)
        return env

    def _identity(self, w: _SglangProc) -> Optional[bool]:
        """None = port silent; True = OUR model answers; False = foreign server."""
        replica = self.pool.replicas[w.spec.replica_id]
        info = replica.fetch_model_info(timeout=3.0)
        if info is None:
            return None
        served = str(info.get("model_path") or "")
        ours = os.path.realpath(self.s.offline_model_path)
        return bool(served) and os.path.realpath(served) == ours

    def _spawn_one(self, w: _SglangProc) -> None:
        spec = w.spec
        identity = self._identity(w)
        if identity is True:
            log.info("Adopting already-healthy offline sglang %d at %s",
                     spec.replica_id, spec.base_url)
            return
        if identity is False:
            if not w.foreign:
                w.foreign = True
                log.error("port %d answers with a FOREIGN model — offline replica "
                          "%d stays DOWN (not adopting, not killing; repoint "
                          "SGLANG_BASE_PORT)", spec.port, spec.replica_id)
            return
        w.foreign = False
        if not os.path.exists(self.s.sglang_python):
            log.warning("offline sglang python missing (%s) — replica %d not "
                        "spawned; build .venv-sglang (scripts/build_venv_sglang.sh) "
                        "or set OFFLINE_PROVIDER=none", self.s.sglang_python,
                        spec.replica_id)
            return
        cmd = self._cmd(spec)
        log_path = self._log_path(spec)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log.info("Spawning offline sglang %d: %s (gpu=%d log=%s)",
                 spec.replica_id, " ".join(cmd), spec.gpu_index, log_path)
        # sglang logs to stdio only — pipe through the rotating tee (same
        # pipeline as the pytorch TTS sidecar; killpg reaches server AND tee)
        from ..sidecars import _ROTATING_TEE

        tee = [sys.executable, _ROTATING_TEE, "--quiet", log_path]
        spawn_cmd = ["bash", "-c",
                     'set -o pipefail; exec "$0" "$@" 2>&1 | exec ' + shlex.join(tee),
                     *cmd]
        w.proc = subprocess.Popen(
            spawn_cmd, env=self._env(spec),
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def spawn_all(self) -> None:
        """Spawn every sidecar (staggered — parallel 22 GB weight reads thrash
        the shared FS), then gate in parallel. Blocking; run via to_thread."""
        spawned_any = False
        for w in self.sidecars:
            if spawned_any and self.s.vlm_spawn_stagger_s > 0:
                time.sleep(self.s.vlm_spawn_stagger_s)
            before = w.proc
            self._spawn_one(w)
            spawned_any = spawned_any or (w.proc is not before)
        self._health_gate()

    def _health_gate(self) -> None:
        deadline = time.monotonic() + self.s.sglang_health_timeout_s

        def gate(w: _SglangProc) -> None:
            replica = self.pool.replicas[w.spec.replica_id]
            if w.foreign:
                self.pool.set_replica_health(w.spec.replica_id, None)
                return
            if w.proc is None and self._identity(w) is not True:
                # nothing spawned (no venv) and nothing adopted — degrade quietly
                self.pool.set_replica_health(w.spec.replica_id, None)
                return
            while time.monotonic() < deadline and not self._stopping:
                if w.proc is not None and w.proc.poll() is not None:
                    log.error("offline sglang %d exited during startup (rc=%s) — see %s",
                              w.spec.replica_id, w.proc.returncode, self._log_path(w.spec))
                    self.pool.set_replica_health(w.spec.replica_id, None)
                    return
                health = replica.fetch_health_generate(timeout=30.0)
                if health is not None:
                    w.gated = True
                    self.pool.set_replica_health(w.spec.replica_id, health)
                    log.info("offline sglang %d up at %s", w.spec.replica_id,
                             w.spec.base_url)
                    return
                time.sleep(3.0)
            log.error("offline sglang %d not ready within %.0fs — offline chat "
                      "falls back to the online pool (see %s)",
                      w.spec.replica_id, self.s.sglang_health_timeout_s,
                      self._log_path(w.spec))
            self.pool.set_replica_health(w.spec.replica_id, None)

        if not self.sidecars:
            return
        with ThreadPoolExecutor(max_workers=len(self.sidecars)) as pool:
            list(pool.map(gate, self.sidecars))

    # ------------------------------------------------------------ monitor / respawn

    def start_monitor(self) -> None:
        self._monitor_task = asyncio.get_running_loop().create_task(
            self._monitor_loop(), name="sglang-sidecar-monitor")

    async def stop_monitor(self) -> None:
        task, self._monitor_task = self._monitor_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_S)
            for w in self.sidecars:
                try:
                    await asyncio.to_thread(self._monitor_one, w)
                except Exception as exc:  # noqa: BLE001
                    log.warning("offline sglang %d monitor tick failed: %s",
                                w.spec.replica_id, exc)

    def _monitor_one(self, w: _SglangProc) -> None:
        replica = self.pool.replicas[w.spec.replica_id]
        # steady-state liveness is the cheap /health; an ungated replica (fresh
        # respawn) must pass a REAL decode first — /health alone answers while
        # weights/JIT are still warming and would mark UP prematurely
        if w.gated:
            health = replica.fetch_health(timeout=3.0)
        else:
            health = replica.fetch_health_generate(timeout=8.0)
        if health is not None:
            if w.foreign:
                return  # alive, but not ours — never mark UP
            w.gated = True
            w.health_strikes = 0
            w.respawn_attempts = 0
            self.pool.set_replica_health(w.spec.replica_id, health)
            return
        w.health_strikes += 1
        if w.health_strikes < HEALTH_STRIKES and (w.proc is None or w.proc.poll() is None):
            return  # one blip — next tick decides
        self.pool.set_replica_health(w.spec.replica_id, None)
        self._maybe_respawn(w)

    def _maybe_respawn(self, w: _SglangProc) -> None:
        if self._stopping:
            return
        crashed = w.proc is not None and w.proc.poll() is not None
        if w.proc is not None and not crashed:
            return  # alive but unhealthy — let it recover / next strike
        if w.respawn_attempts >= RESPAWN_MAX_ATTEMPTS:
            if w.respawn_attempts == RESPAWN_MAX_ATTEMPTS:
                w.respawn_attempts += 1  # log once
                log.error("offline sglang %d exceeded %d respawns — staying DOWN",
                          w.spec.replica_id, RESPAWN_MAX_ATTEMPTS)
            return
        now = time.monotonic()
        if now < w.next_respawn_at:
            return
        w.respawn_attempts += 1
        w.next_respawn_at = now + RESPAWN_BACKOFF_S * (2 ** (w.respawn_attempts - 1))
        log.warning("Respawning offline sglang %d (attempt %d/%d)",
                    w.spec.replica_id, w.respawn_attempts, RESPAWN_MAX_ATTEMPTS)
        _kill_proc(w.proc)
        w.proc = None
        w.gated = False  # the monitor re-gates on /health_generate before UP
        self._spawn_one(w)

    # ------------------------------------------------------------ teardown

    def stop_all(self) -> None:
        """Blocking teardown of every OWNED sidecar (adopted/foreign untouched)."""
        self._stopping = True
        for w in self.sidecars:
            _kill_proc(w.proc)
            w.proc = None


def _kill_proc(proc: Optional[subprocess.Popen], grace_s: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
