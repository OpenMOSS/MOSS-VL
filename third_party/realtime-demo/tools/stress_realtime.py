"""MOSS-VL-Realtime 实时会话压测工具（sglang-omni 数据面 / WS）。

回答评审文档 §7 的八维度压测要求。并发定义：同一时间持续发视频帧并由
服务端保持推理状态的活跃会话数；对外只报满足时延与稳定性目标的「稳定并发」。

依赖：标准库 + websockets（仓库 .venv 已装 17.1）；cv2/numpy 仅视频文件
抽帧时需要，jpeg 目录帧源不需要。

用法示例：

  # direct：直连 omni 实例（后端裸性能）
  .venv/bin/python tools/stress_realtime.py \
      --mode direct --url http://127.0.0.1:30000 \
      --sessions 4 --duration 60 --fps 2 --video /path/to/video.mp4 \
      --prompt-interval 10 --gpu-sampling --out reports/stress_direct

  # gateway：走网关平面 POST /v1/realtime/sessions → WS /v1/realtime?ws_token=
  .venv/bin/python tools/stress_realtime.py \
      --mode gateway --url http://127.0.0.1:8080 \
      --ramp 1,2,4,8 --duration 60 --fps 2 --video /path/to/frames_dir \
      --prompt-interval 10 --gpu-sampling --out reports/stress_gateway

协议要点（与 server/adapters/vlm/moss_vl_sglang_omni/ 核实一致）：
  connect → session.created → session.configure(extra=forbid) →
  session.configured + session.ready → 两段式帧上行
  (input.frame 元数据 → input.frame.ready → 裸二进制 → input.frame.accepted) →
  input.prompt（与帧共用从 0 递增的 dense seq_no）→ 结束用 session.abort。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None

ACK_TIMEOUT_S = 30.0          # waiting for frame.ready / *.accepted must not hang forever
HANDSHAKE_TIMEOUT_S = 30.0
DRAIN_AFTER_ABORT_S = 3.0
PROMPT_ARM_TIMEOUT_S = 2.0    # fallback if input.*.processed (with the new turn_id) never arrives
MAX_INBOUND_BYTES = 64 * 1024 * 1024

# NOTE: the task brief said input.prompt carries `text`; every in-repo source
# (adapter session.py, MIGRATION_PLAN.md, gateway tests) uses `prompt`, and
# configure is extra=forbid — so `prompt` it is. Kept as a constant to flip fast.
PROMPT_TEXT_FIELD = "prompt"

DEFAULT_PROMPTS = [
    "现在画面里有什么？",
    "描述一下刚才发生了什么。",
    "画面里有几个人？他们在做什么？",
    "刚才的画面和之前比有什么变化？",
]

CONFIGURE_FIELDS = ("type", "prompt", "system_prompt", "max_new_tokens",
                    "temperature", "top_p", "input_queue_capacity")


# ------------------------------------------------------------------ config


@dataclass
class StressConfig:
    mode: str = "direct"                 # direct | gateway
    url: str = "http://127.0.0.1:30000"
    sessions: int = 1
    duration: float = 60.0
    fps: float = 2.0
    video: str = ""                      # video file or jpeg directory
    prompt_interval: float = 10.0
    prompts_file: str = ""
    ramp: str = ""                       # e.g. "1,2,4"
    ramp_rest_s: float = 30.0
    max_frame_bytes: int = 512 * 1024
    out: str = "stress_report"
    gpu_sampling: bool = False
    gpu_interval_s: float = 5.0
    # session configure knobs
    session_prompt: str = "你是一个实时视频理解助手，请持续观察画面并回答问题。"
    system_prompt: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.8
    input_queue_capacity: int = 4
    # stable-concurrency targets
    target_ttft_p95_s: float = 3.0
    target_success_rate: float = 0.99
    # hardware annotation (auto-detected via nvidia-smi when possible)
    gpu_name: str = ""
    gpu_count: int = 0
    instances: int = 1
    connect_timeout_s: float = 15.0


def parse_args(argv: Optional[List[str]] = None) -> StressConfig:
    p = argparse.ArgumentParser(description="MOSS-VL-Realtime realtime-session stress tool")
    p.add_argument("--mode", choices=["direct", "gateway"], default="direct")
    p.add_argument("--url", default="http://127.0.0.1:30000",
                   help="direct: omni base URL; gateway: gateway base URL")
    p.add_argument("--sessions", type=int, default=1, help="concurrent sessions (single level)")
    p.add_argument("--duration", type=float, default=60.0, help="seconds per session per level")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--video", required=True, help="video file (needs cv2) or jpeg directory")
    p.add_argument("--prompt-interval", type=float, default=10.0)
    p.add_argument("--prompts-file", default="", help="one prompt per line; overrides defaults")
    p.add_argument("--ramp", default="", help='concurrency sweep, e.g. "1,2,4"; overrides --sessions')
    p.add_argument("--ramp-rest", type=float, default=30.0, help="rest seconds between ramp levels")
    p.add_argument("--max-frame-bytes", type=int, default=512 * 1024)
    p.add_argument("--out", default="stress_report", help="report file prefix (<prefix>.json/.md)")
    p.add_argument("--gpu-sampling", action="store_true", help="sample nvidia-smi in background")
    p.add_argument("--gpu-interval", type=float, default=5.0)
    p.add_argument("--session-prompt", default=StressConfig.session_prompt)
    p.add_argument("--system-prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--input-queue-capacity", type=int, default=4)
    p.add_argument("--target-ttft-p95", type=float, default=3.0,
                   help="stable-concurrency target: first-response P95 seconds")
    p.add_argument("--target-success-rate", type=float, default=0.99)
    p.add_argument("--gpu-name", default="", help="override hardware annotation")
    p.add_argument("--gpu-count", type=int, default=0, help="override hardware annotation")
    p.add_argument("--instances", type=int, default=1, help="omni instance count (annotation)")
    p.add_argument("--connect-timeout", type=float, default=15.0)
    ns = p.parse_args(argv)
    cfg = StressConfig(
        mode=ns.mode, url=ns.url, sessions=ns.sessions, duration=ns.duration,
        fps=ns.fps, video=ns.video, prompt_interval=ns.prompt_interval,
        prompts_file=ns.prompts_file, ramp=ns.ramp, ramp_rest_s=ns.ramp_rest,
        max_frame_bytes=ns.max_frame_bytes, out=ns.out,
        gpu_sampling=ns.gpu_sampling, gpu_interval_s=ns.gpu_interval,
        session_prompt=ns.session_prompt, system_prompt=ns.system_prompt,
        max_new_tokens=ns.max_new_tokens, temperature=ns.temperature, top_p=ns.top_p,
        input_queue_capacity=ns.input_queue_capacity,
        target_ttft_p95_s=ns.target_ttft_p95,
        target_success_rate=ns.target_success_rate,
        gpu_name=ns.gpu_name, gpu_count=ns.gpu_count, instances=ns.instances,
        connect_timeout_s=ns.connect_timeout)
    if cfg.sessions < 1:
        p.error("--sessions must be >= 1")
    if cfg.fps <= 0:
        p.error("--fps must be > 0")
    return cfg


def load_prompts(cfg: StressConfig) -> List[str]:
    if cfg.prompts_file:
        lines = [ln.strip() for ln in Path(cfg.prompts_file).read_text(encoding="utf-8").splitlines()]
        prompts = [ln for ln in lines if ln]
        if prompts:
            return prompts
    return list(DEFAULT_PROMPTS)


def ramp_levels(cfg: StressConfig) -> List[int]:
    if not cfg.ramp.strip():
        return [cfg.sessions]
    levels = []
    for part in cfg.ramp.split(","):
        part = part.strip()
        if part:
            levels.append(int(part))
    if not levels:
        raise ValueError("--ramp parsed to an empty level list")
    return levels


# ------------------------------------------------------------------ frame source


class FrameCursor:
    """Per-session (or shared) frame iterator; next() returns JPEG bytes."""

    async def next(self) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError


class JpegDirSource:
    def __init__(self, path: Path):
        files = sorted(p for p in path.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg"))
        if not files:
            raise ValueError(f"jpeg directory has no .jpg/.jpeg files: {path}")
        self._frames = [p.read_bytes() for p in files]
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self._probe_dims()

    def _probe_dims(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            img = cv2.imdecode(np.frombuffer(self._frames[0], dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is not None:
                self.height, self.width = int(img.shape[0]), int(img.shape[1])
        except ImportError:
            pass  # dims stay unknown; not required for the load itself

    def describe(self) -> Dict[str, Any]:
        dims = f"{self.width}x{self.height}" if self.width else "unknown"
        return {"kind": "jpeg_dir", "frame_count": len(self._frames), "resolution": dims}

    def new_cursor(self, offset: int = 0) -> FrameCursor:
        frames = self._frames

        class _Cursor(FrameCursor):
            def __init__(self) -> None:
                self._i = offset % len(frames)

            async def next(self) -> bytes:
                raw = frames[self._i % len(frames)]
                self._i += 1
                return raw

        return _Cursor()


class VideoFileSource:
    """cv2-decoded video; ONE shared capture, sessions pull under a lock."""

    def __init__(self, path: Path):
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "视频文件抽帧需要 opencv-python（cv2）+ numpy；请 "
                "`.venv/bin/pip install opencv-python-headless`，或改用 --video 指向 "
                "jpeg 目录（无需 cv2）。") from exc
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cv2 cannot open video file: {path}")
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._lock = asyncio.Lock()

    def describe(self) -> Dict[str, Any]:
        return {"kind": "video_file", "resolution": f"{self.width}x{self.height}",
                "source_fps": round(self.source_fps, 3)}

    def new_cursor(self, offset: int = 0) -> FrameCursor:
        del offset  # shared stream: every session pulls the next decoded frame
        cap, cv2, lock = self._cap, self._cv2, self._lock

        class _Cursor(FrameCursor):
            async def next(self) -> bytes:
                async with lock:
                    ok, frame = cap.read()
                    if not ok:  # EOF → loop the clip
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("cv2 failed to decode any frame from the video")
                    ok, buf = cv2.imencode(".jpg", frame)
                    if not ok:
                        raise RuntimeError("cv2.imencode(.jpg) failed")
                    return buf.tobytes()

        return _Cursor()

    def close(self) -> None:
        self._cap.release()


def build_frame_source(cfg: StressConfig) -> Any:
    path = Path(cfg.video)
    if path.is_dir():
        return JpegDirSource(path)
    if path.is_file():
        return VideoFileSource(path)
    raise ValueError(f"--video path does not exist: {path}")


# ------------------------------------------------------------------ GPU sampling


class GpuSampler(threading.Thread):
    """nvidia-smi poller; silently disabled (with a note) when unavailable."""

    def __init__(self, interval_s: float = 5.0):
        super().__init__(name="gpu-sampler", daemon=True)
        self.interval_s = max(1.0, interval_s)
        self.samples: List[Dict[str, Any]] = []
        self.gpu_name = ""
        self.gpu_count = 0
        self.note = ""
        # must NOT be named `_stop`: that shadows threading.Thread._stop()
        # and join() then crashes with "'Event' object is not callable"
        self._stop_evt = threading.Event()

    def run(self) -> None:
        exe = shutil.which("nvidia-smi")
        if exe is None:
            self.note = "nvidia-smi not found on this host; GPU sampling skipped"
            return
        try:
            out = subprocess.run(
                [exe, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout
            names = [ln.strip() for ln in out.splitlines() if ln.strip()]
            self.gpu_count = len(names)
            self.gpu_name = names[0] if names else ""
        except Exception as exc:  # noqa: BLE001
            self.note = f"nvidia-smi probe failed: {exc}; GPU sampling skipped"
            return
        while not self._stop_evt.is_set():
            try:
                out = subprocess.run(
                    [exe, "--query-gpu=index,utilization.gpu,memory.used",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10).stdout
                for ln in out.splitlines():
                    parts = [p.strip() for p in ln.split(",")]
                    if len(parts) != 3:
                        continue
                    self.samples.append({
                        "t": round(time.time(), 3),
                        "index": int(parts[0]),
                        "util_pct": float(parts[1].rstrip("%")),
                        "mem_mib": float(parts[2].rstrip(" MiB")),
                    })
            except Exception as exc:  # noqa: BLE001
                self.note = f"nvidia-smi sampling error: {exc}"
            self._stop_evt.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_evt.set()


# ------------------------------------------------------------------ per-session metrics


@dataclass
class SessionResult:
    index: int
    session_id: str = ""
    ok: bool = False                    # ran the full duration without fatal error
    capacity_rejected: bool = False
    abnormal_disconnect: bool = False
    finished_reason: str = ""
    errors: List[str] = field(default_factory=list)
    error_events: int = 0
    invalid_requests: int = 0
    frames_sent: int = 0                # metadata on the wire
    frames_accepted: int = 0
    frames_processed: int = 0
    frames_oversized_dropped: int = 0
    frame_bytes: int = 0
    prompts_sent: int = 0
    deltas: int = 0
    text_chars: int = 0
    silence_events: int = 0
    interrupted_events: int = 0
    turns_completed: int = 0
    turns_unanswered: int = 0
    ttft_first_s: Optional[float] = None
    turn_ttfts_s: List[float] = field(default_factory=list)
    turn_latencies_s: List[float] = field(default_factory=list)
    backlog_peak: int = 0
    ready_latency_s: Optional[float] = None
    actual_duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class _Waiter:
    """One in-flight input; recv task signals, sender waits (two-phase send)."""

    def __init__(self, seq_no: int, kind: str, prompt: bool = False):
        self.seq_no = seq_no
        self.kind = kind                # "frame" | "prompt"
        self.prompt = prompt            # input carries a prompt (pure or frame-carried)
        self.ready = asyncio.Event()
        self.accepted = asyncio.Event()
        self.error: Optional[str] = None


class _OpenTurn:
    """One question→answer round, armed by the prompt's processed event.

    MOSS-VL-Realtime speaks SPONTANEOUSLY: response.text.delta streams keep
    arriving on the current turn_id with no prompt at all. So a prompt opens an
    UNARMED turn; only the prompt's input.*.processed event (which carries
    `interrupted_turn_id` = old and `turn_id` = NEW) arms the measurement, and
    only deltas/silence/interrupted carrying that NEW turn_id are attributed.
    """

    def __init__(self, t_send: float, prompt_seq: int):
        self.t_send = t_send
        self.prompt_seq = prompt_seq
        self.armed_turn_id: Optional[int] = None
        self.arm_any = False            # timeout fallback: attribute any delta
        self.first_delta_at: Optional[float] = None
        self.last_delta_at: Optional[float] = None
        self.deltas = 0


# ------------------------------------------------------------------ one stress session


class StressSession:
    def __init__(self, cfg: StressConfig, index: int, cursor: FrameCursor,
                 prompts: List[str], stop: asyncio.Event):
        self.cfg = cfg
        self.result = SessionResult(index=index)
        self._cursor = cursor
        self._prompts = prompts
        self._stop = stop
        self._ws: Any = None
        self._waiters: Dict[int, _Waiter] = {}
        self._seq_counter = 0
        self._last_ts = 0.0
        self._t_ready = 0.0
        self._ended = False             # server signalled done / transport died
        self._open_turn: Optional[_OpenTurn] = None
        # prompt seq_no → new turn_id, from input.*.processed events that arrive
        # before the sender task gets to create the open turn (accepted and
        # processed can be dispatched back-to-back)
        self._processed_turn_ids: Dict[int, Optional[int]] = {}

    # ------------------------------------------------------------ url / entry

    def _direct_ws_url(self) -> str:
        url = self.cfg.url.strip().rstrip("/")
        if url.startswith("https://"):
            url = "wss://" + url[len("https://"):]
        elif url.startswith("http://"):
            url = "ws://" + url[len("http://"):]
        elif not url.startswith(("ws://", "wss://")):
            url = "ws://" + url
        return url + "/v1/video/realtime"

    async def _gateway_ws_url(self) -> str:
        """POST /v1/realtime/sessions → {session_id, ws_token, ws_url}."""
        base = self.cfg.url.strip().rstrip("/")
        endpoint = base + "/v1/realtime/sessions"

        def post() -> Dict[str, Any]:
            req = urllib.request.Request(
                endpoint, data=b"{}", method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.cfg.connect_timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            payload = await asyncio.to_thread(post)
        except urllib.error.HTTPError as exc:
            code = ""
            try:
                code = str(json.loads(exc.read().decode("utf-8")).get("detail", {}).get("code") or "")
            except Exception:  # noqa: BLE001
                pass
            if exc.code == 503 or code == "session_capacity_exceeded":
                raise _CapacityRejected(f"gateway refused session: HTTP {exc.code} {code}")
            raise
        self.result.session_id = str(payload.get("session_id") or "")
        token = str(payload.get("ws_token") or "")
        ws_path = str(payload.get("ws_url") or "/v1/realtime")
        if ws_path.startswith(("ws://", "wss://")):
            # gateway returned an absolute WS URL (not our gateway's default
            # relative "/v1/realtime") — use it verbatim
            return f"{ws_path}?ws_token={token}"
        ws_base = base
        if ws_base.startswith("https://"):
            ws_base = "wss://" + ws_base[len("https://"):]
        elif ws_base.startswith("http://"):
            ws_base = "ws://" + ws_base[len("http://"):]
        return f"{ws_base}{ws_path}?ws_token={token}"

    async def run(self) -> SessionResult:
        res = self.result
        t0 = time.monotonic()
        try:
            ws_url = (self._direct_ws_url() if self.cfg.mode == "direct"
                      else await self._gateway_ws_url())
            async with websockets.connect(
                    ws_url, open_timeout=self.cfg.connect_timeout_s,
                    max_size=MAX_INBOUND_BYTES, close_timeout=2.0) as ws:
                self._ws = ws
                await self._handshake(ws)
                res.ready_latency_s = time.monotonic() - t0
                recv_task = asyncio.create_task(self._receiver(ws))
                try:
                    await self._sender(ws)
                finally:
                    recv_task.cancel()
                    await asyncio.gather(recv_task, return_exceptions=True)
                await self._teardown(ws)
            res.ok = not res.abnormal_disconnect and not res.finished_reason.startswith("fatal")
            if not res.finished_reason:
                res.finished_reason = "duration_elapsed"
        except _CapacityRejected as exc:
            res.capacity_rejected = True
            res.finished_reason = "capacity_rejected"
            res.errors.append(str(exc))
        except asyncio.CancelledError:
            res.finished_reason = res.finished_reason or "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 — one session must not kill the level
            res.finished_reason = f"fatal:{type(exc).__name__}"
            res.errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            self._close_open_turn()
            res.actual_duration_s = time.monotonic() - t0
        return res

    # ------------------------------------------------------------ handshake

    async def _handshake(self, ws: Any) -> None:
        first = json.loads(await asyncio.wait_for(ws.recv(), HANDSHAKE_TIMEOUT_S))
        if first.get("type") == "error":
            if first.get("code") == "session_capacity_exceeded":
                raise _CapacityRejected(str(first.get("message") or "capacity exceeded"))
            raise RuntimeError(f"server rejected session: {first}")
        if first.get("type") != "session.created":
            raise RuntimeError(f"expected session.created, got: {first}")
        self.result.session_id = self.result.session_id or str(first.get("session_id") or "")
        configure = {
            "type": "session.configure",
            "prompt": self.cfg.session_prompt,
            "system_prompt": self.cfg.system_prompt or None,
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "input_queue_capacity": self.cfg.input_queue_capacity,
        }
        assert set(configure) == set(CONFIGURE_FIELDS)
        await ws.send(json.dumps(configure, ensure_ascii=False))
        configured = ready = False
        deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
        while not (configured and ready):
            msg = json.loads(await asyncio.wait_for(
                ws.recv(), max(0.1, deadline - time.monotonic())))
            mtype = msg.get("type")
            if mtype == "session.configured":
                configured = True
            elif mtype == "session.ready":
                ready = True
            elif mtype == "error":
                raise RuntimeError(f"configure failed: {msg}")
            # any other early event is handled after the receiver task starts;
            # during handshake we only care about the handshake itself
        self._t_ready = time.monotonic()

    # ------------------------------------------------------------ sender

    async def _sender(self, ws: Any) -> None:
        cfg = self.cfg
        frame_period = 1.0 / cfg.fps
        deadline = time.monotonic() + cfg.duration
        next_frame_at = time.monotonic()
        # guarantee at least one prompt even when interval >= duration
        first_prompt_at = time.monotonic() + min(
            cfg.prompt_interval, max(0.5, cfg.duration / 3.0))
        next_prompt_at = first_prompt_at
        while time.monotonic() < deadline and not self._stop.is_set() and not self._ended:
            now = time.monotonic()
            turn = self._open_turn
            if (turn is not None and turn.armed_turn_id is None and not turn.arm_any
                    and now - turn.t_send > PROMPT_ARM_TIMEOUT_S):
                # abnormal path: the processed event carrying the new turn_id
                # never arrived — attribute any incoming delta rather than
                # leave the turn unmeasured forever
                turn.arm_any = True
            if now >= next_prompt_at:
                await self._send_prompt(ws)
                next_prompt_at = now + cfg.prompt_interval
                continue
            if now >= next_frame_at:
                await self._send_frame(ws)
                next_frame_at += frame_period
                if next_frame_at < now:  # fell behind: resync, don't burst
                    next_frame_at = now + frame_period
                continue
            await asyncio.sleep(min(next_frame_at, next_prompt_at, deadline) - now)

    def _next_seq(self) -> int:
        seq = self._seq_counter
        self._seq_counter += 1
        return seq

    def _next_ts(self) -> float:
        # seconds-level float, 0.1s granularity, monotonic non-decreasing
        ts = round(max(0.0, time.monotonic() - self._t_ready), 1)
        ts = max(ts, self._last_ts)
        self._last_ts = ts
        return ts

    async def _wait_ack(self, waiter: _Waiter, event: asyncio.Event, label: str) -> None:
        try:
            await asyncio.wait_for(event.wait(), ACK_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._ended = True
            raise TimeoutError(f"timed out waiting for {label} (seq_no={waiter.seq_no})")
        if waiter.error:
            raise RuntimeError(waiter.error)

    async def _send_frame(self, ws: Any) -> None:
        res = self.result
        raw = await self._cursor.next()
        if len(raw) > self.cfg.max_frame_bytes:
            res.frames_oversized_dropped += 1
            return
        waiter = _Waiter(self._next_seq(), "frame")
        self._waiters[waiter.seq_no] = waiter
        try:
            await ws.send(json.dumps({
                "type": "input.frame",
                "seq_no": waiter.seq_no,
                "timestamp": self._next_ts(),
                "final": False,
                "mime_type": "image/jpeg",
            }))
            res.frames_sent += 1
            await self._wait_ack(waiter, waiter.ready, "input.frame.ready")
            await ws.send(raw)          # raw binary JPEG
            res.frame_bytes += len(raw)
            await self._wait_ack(waiter, waiter.accepted, "input.frame.accepted")
            res.frames_accepted += 1
            self._note_backlog()
        except RuntimeError as exc:
            if "invalid_request" in str(exc):
                return  # counted in _dispatch; server rejected one event, session survives
            raise

    async def _send_prompt(self, ws: Any) -> None:
        res = self.result
        text = self._prompts[res.prompts_sent % len(self._prompts)]
        waiter = _Waiter(self._next_seq(), "prompt", prompt=True)
        self._waiters[waiter.seq_no] = waiter
        try:
            await ws.send(json.dumps({
                "type": "input.prompt",
                "seq_no": waiter.seq_no,
                PROMPT_TEXT_FIELD: text,
                "final": False,
            }, ensure_ascii=False))
            res.prompts_sent += 1
            await self._wait_ack(waiter, waiter.accepted, "input.prompt.accepted")
            # a new prompt opens a new turn: close the previous one
            self._close_open_turn()
            turn = _OpenTurn(t_send=time.monotonic(), prompt_seq=waiter.seq_no)
            self._open_turn = turn
            # the processed event may have beaten us here (dispatched right
            # after accepted); if so, arm immediately from its new turn_id
            if waiter.seq_no in self._processed_turn_ids:
                self._arm_turn(turn, self._processed_turn_ids.pop(waiter.seq_no))
        except RuntimeError as exc:
            if "invalid_request" in str(exc):
                return  # counted in _dispatch
            raise

    async def _teardown(self, ws: Any) -> None:
        try:
            await ws.send(json.dumps({"type": "session.abort"}))
        except Exception:  # noqa: BLE001 — socket may already be gone
            return
        # give the server a moment for response.done → session.done → close
        try:
            await asyncio.wait_for(ws.recv(), DRAIN_AFTER_ABORT_S)
        except Exception:  # noqa: BLE001 — timeout/close both fine here
            pass

    # ------------------------------------------------------------ receiver

    async def _receiver(self, ws: Any) -> None:
        res = self.result
        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    res.errors.append("unexpected binary event from server")
                    continue
                self._dispatch(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — ConnectionClosed included
            if not self._ended:
                res.abnormal_disconnect = True
                res.finished_reason = res.finished_reason or f"ws_closed:{type(exc).__name__}"
            self._ended = True

    def _dispatch(self, msg: Dict[str, Any]) -> None:
        res = self.result
        mtype = str(msg.get("type") or "")
        seq_no = msg.get("seq_no")
        waiter = self._waiters.get(seq_no) if isinstance(seq_no, int) else None

        if mtype == "input.frame.ready":
            if waiter is not None:
                waiter.ready.set()
            return
        if mtype in ("input.frame.accepted", "input.prompt.accepted"):
            pending = msg.get("pending_events")
            if isinstance(pending, int):
                res.backlog_peak = max(res.backlog_peak, pending)
            if waiter is not None:
                waiter.accepted.set()
            return
        if mtype in ("input.frame.processed", "input.prompt.processed"):
            if waiter is not None:
                if waiter.kind == "frame":
                    res.frames_processed += 1
                self._waiters.pop(waiter.seq_no, None)
                if waiter.prompt:
                    self._on_prompt_processed(waiter, msg)
            self._note_backlog()
            return
        if mtype == "response.text.delta":
            text = str(msg.get("delta") or "")
            if not text:
                return
            now = time.monotonic()
            res.deltas += 1
            res.text_chars += len(text)
            turn = self._open_turn
            if turn is not None and self._turn_matches(turn, msg):
                # only deltas on the ARMED turn_id count toward this prompt's
                # latency; spontaneous-speech deltas (any other turn_id) only
                # feed the global deltas/text_chars counters above
                if turn.first_delta_at is None:
                    turn.first_delta_at = now
                    ttft = now - turn.t_send
                    res.turn_ttfts_s.append(ttft)
                    if res.ttft_first_s is None:
                        res.ttft_first_s = ttft
                turn.last_delta_at = now
                turn.deltas += 1
            return
        if mtype in ("response.turn.silence", "response.turn.interrupted"):
            if mtype == "response.turn.silence":
                res.silence_events += 1
            else:
                res.interrupted_events += 1
            turn = self._open_turn
            if turn is not None and self._turn_matches(turn, msg):
                # the armed turn went idle / was barged in → round over;
                # a spontaneous turn's silence/interrupted (other turn_id)
                # must NOT close the measurement
                self._close_open_turn()
            return
        if mtype in ("response.done", "session.done"):
            self._close_open_turn()
            self._ended = True
            return
        if mtype == "error":
            res.error_events += 1
            code = str(msg.get("code") or "")
            text = f"error[{code}]: {msg.get('message')}"
            if len(res.errors) < 20:
                res.errors.append(text)
            if code == "invalid_request":
                # single event rejected, session survives: fail its waiter,
                # roll the dense seq counter back over the rejected seq_no
                res.invalid_requests += 1
                if waiter is not None:
                    waiter.error = text
                    waiter.ready.set()
                    waiter.accepted.set()
                    self._waiters.pop(waiter.seq_no, None)
                    if waiter.seq_no == self._seq_counter - 1:
                        self._seq_counter = waiter.seq_no
                return
            self._ended = True  # input_submission_failed / response_failed: fatal
            res.finished_reason = res.finished_reason or f"fatal:error[{code}]"
            return
        # unknown event types are ignored (forward-compatible)

    def _note_backlog(self) -> None:
        # fallback when the server does not report pending_events: local
        # in-flight count (metadata sent, not yet processed)
        self.result.backlog_peak = max(self.result.backlog_peak, len(self._waiters))

    # ------------------------------------------------------------ turn attribution

    def _on_prompt_processed(self, waiter: _Waiter, msg: Dict[str, Any]) -> None:
        """Arm the open turn from the prompt's processed event.

        Field names per sglang-omni video_realtime.py: `interrupted_turn_id`
        is the old (barged-in) turn, `turn_id` is the NEW turn the answer will
        stream on. A missing turn_id arms the accept-anything fallback.
        """
        new_turn = msg.get("turn_id")
        if not isinstance(new_turn, int):
            alt = msg.get("next_turn_id")
            new_turn = alt if isinstance(alt, int) else None
        turn = self._open_turn
        if turn is not None and turn.prompt_seq == waiter.seq_no:
            self._arm_turn(turn, new_turn)
        else:
            # processed beat the sender task (or the turn was superseded by a
            # newer prompt): stash for pickup at open-turn creation
            self._processed_turn_ids[waiter.seq_no] = new_turn

    def _arm_turn(self, turn: _OpenTurn, new_turn: Optional[int]) -> None:
        if new_turn is None:
            turn.arm_any = True
        else:
            turn.armed_turn_id = new_turn

    @staticmethod
    def _turn_matches(turn: _OpenTurn, msg: Dict[str, Any]) -> bool:
        """Does this delta/silence/interrupted belong to the open turn?"""
        if turn.arm_any:
            return True
        if turn.armed_turn_id is None:
            return False                # not armed yet: spontaneous speech
        tid = msg.get("turn_id")
        if not isinstance(tid, int):
            return True                 # server omitted turn_id: be lenient
        return tid == turn.armed_turn_id

    def _close_open_turn(self) -> None:
        turn = self._open_turn
        if turn is None:
            return
        self._open_turn = None
        res = self.result
        if turn.deltas > 0 and turn.last_delta_at is not None:
            res.turns_completed += 1
            res.turn_latencies_s.append(turn.last_delta_at - turn.t_send)
        else:
            res.turns_unanswered += 1


class _CapacityRejected(RuntimeError):
    """Server refused the session at accept time (error + close 1013 / HTTP 503)."""


# ------------------------------------------------------------------ aggregation


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def pctiles(values: List[float]) -> Dict[str, Optional[float]]:
    return {"p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99)}


@dataclass
class LevelResult:
    level: int
    sessions: List[SessionResult] = field(default_factory=list)
    wall_time_s: float = 0.0
    interrupted: bool = False

    def aggregate(self, cfg: StressConfig) -> Dict[str, Any]:
        rs = self.sessions
        attempted = len(rs)
        completed = sum(1 for r in rs if r.ok)
        cap_rej = sum(1 for r in rs if r.capacity_rejected)
        abnormal = sum(1 for r in rs if r.abnormal_disconnect)
        ttfts = [r.ttft_first_s for r in rs if r.ttft_first_s is not None]
        turn_ttfts = [v for r in rs for v in r.turn_ttfts_s]
        turn_lats = [v for r in rs for v in r.turn_latencies_s]
        accepted = sum(r.frames_accepted for r in rs)
        wall = max(self.wall_time_s, 1e-9)
        success_rate = completed / attempted if attempted else 0.0
        agg: Dict[str, Any] = {
            "level": self.level,
            "attempted": attempted,
            "completed": completed,
            "capacity_rejected": cap_rej,
            "abnormal_disconnects": abnormal,
            "success_rate": round(success_rate, 4),
            "abnormal_disconnect_rate": round(abnormal / attempted, 4) if attempted else 0.0,
            "ttft_first_s": pctiles([float(v) for v in ttfts]),
            "turn_ttft_s": pctiles(turn_ttfts),
            "turn_latency_s": pctiles(turn_lats),
            "frames_accepted": accepted,
            "frame_throughput_fps": round(accepted / wall, 2),
            "frame_bytes_total": sum(r.frame_bytes for r in rs),
            "avg_frame_bytes": round(sum(r.frame_bytes for r in rs) / accepted, 1) if accepted else 0,
            "frames_oversized_dropped": sum(r.frames_oversized_dropped for r in rs),
            "prompts_sent": sum(r.prompts_sent for r in rs),
            "deltas": sum(r.deltas for r in rs),
            "text_chars": sum(r.text_chars for r in rs),
            "turns_completed": sum(r.turns_completed for r in rs),
            "turns_unanswered": sum(r.turns_unanswered for r in rs),
            "silence_events": sum(r.silence_events for r in rs),
            "interrupted_events": sum(r.interrupted_events for r in rs),
            "error_events": sum(r.error_events for r in rs),
            "invalid_requests": sum(r.invalid_requests for r in rs),
            "backlog_peak": max((r.backlog_peak for r in rs), default=0),
            "wall_time_s": round(wall, 2),
            "interrupted": self.interrupted,
        }
        p95 = agg["ttft_first_s"]["p95"]
        agg["meets_targets"] = bool(
            success_rate >= cfg.target_success_rate
            and p95 is not None and p95 < cfg.target_ttft_p95_s)
        return agg


async def run_level(cfg: StressConfig, level: int,
                    stop: asyncio.Event) -> LevelResult:
    source = build_frame_source(cfg)
    prompts = load_prompts(cfg)
    result = LevelResult(level=level)
    t0 = time.monotonic()
    try:
        tasks = [
            asyncio.create_task(
                StressSession(cfg, i, source.new_cursor(offset=i), prompts, stop).run())
            for i in range(level)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, item in enumerate(results):
            if isinstance(item, SessionResult):
                result.sessions.append(item)
            else:  # gather-level exception escaping StressSession (shouldn't happen)
                res = SessionResult(index=i, finished_reason=f"fatal:{type(item).__name__}")
                res.errors.append(str(item))
                result.sessions.append(res)
    finally:
        result.wall_time_s = time.monotonic() - t0
        close = getattr(source, "close", None)
        if callable(close):
            close()
    result.interrupted = stop.is_set()
    return result


# ------------------------------------------------------------------ reports


def _fmt(v: Optional[float], nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def write_reports(cfg: StressConfig, levels: List[LevelResult],
                  aggs: List[Dict[str, Any]], sampler: Optional[GpuSampler],
                  frame_desc: Dict[str, Any], stable: Optional[int],
                  interrupted: bool) -> None:
    prefix = Path(cfg.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    gpu_samples = sampler.samples if sampler else []
    gpu_note = sampler.note if sampler and sampler.note else ""
    gpu_name = cfg.gpu_name or (sampler.gpu_name if sampler else "")
    gpu_count = cfg.gpu_count or (sampler.gpu_count if sampler else 0)

    payload = {
        "tool": "stress_realtime",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "interrupted": interrupted,
        "conditions": {
            "mode": cfg.mode, "url": cfg.url, "levels": [l.level for l in levels],
            "duration_s": cfg.duration, "fps": cfg.fps,
            "prompt_interval_s": cfg.prompt_interval,
            "max_frame_bytes": cfg.max_frame_bytes,
            "session_prompt": cfg.session_prompt,
            "system_prompt": cfg.system_prompt or None,
            "max_new_tokens": cfg.max_new_tokens, "temperature": cfg.temperature,
            "top_p": cfg.top_p, "input_queue_capacity": cfg.input_queue_capacity,
            "video": cfg.video, "video_source": frame_desc,
            "targets": {"ttft_first_p95_s": cfg.target_ttft_p95_s,
                        "success_rate": cfg.target_success_rate},
        },
        "hardware": {"gpu_name": gpu_name or "unknown",
                     "gpu_count": gpu_count, "instances": cfg.instances},
        "stable_concurrency": stable,
        "levels": [
            {**agg, "sessions": [r.to_dict() for r in lvl.sessions]}
            for agg, lvl in zip(aggs, levels)
        ],
        "gpu_sampling": {"note": gpu_note, "samples": gpu_samples},
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    prefix.with_suffix(".md").write_text(
        render_markdown(cfg, levels, aggs, gpu_samples, gpu_note,
                        gpu_name, gpu_count, frame_desc, stable, interrupted),
        encoding="utf-8")


def render_markdown(cfg: StressConfig, levels: List[LevelResult],
                    aggs: List[Dict[str, Any]], gpu_samples: List[Dict[str, Any]],
                    gpu_note: str, gpu_name: str, gpu_count: int,
                    frame_desc: Dict[str, Any], stable: Optional[int],
                    interrupted: bool) -> str:
    L: List[str] = []
    L.append("# MOSS-VL-Realtime 实时会话压测报告")
    L.append("")
    L.append(f"- 模式：`{cfg.mode}`  目标：`{cfg.url}`")
    L.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"
             + ("  **（Ctrl-C 中断，结果为部分数据）**" if interrupted else ""))
    L.append(f"- 达标条件：首次响应 P95 < {cfg.target_ttft_p95_s}s 且 会话成功率 ≥ "
             f"{cfg.target_success_rate:.0%}")
    L.append("")

    # 1 硬件
    L.append("## 1. 硬件")
    L.append("")
    L.append(f"- GPU：{gpu_name or '未采集（无 nvidia-smi，可用 --gpu-name 标注）'}"
             f" × {gpu_count or '?'}")
    L.append(f"- sglang-omni 实例数：{cfg.instances}（--instances 标注）")
    L.append("")

    # 2 视频输入
    L.append("## 2. 视频输入")
    L.append("")
    avg_bytes = max((a["avg_frame_bytes"] for a in aggs), default=0)
    L.append(f"- 来源：`{cfg.video}`（{frame_desc.get('kind', '?')}），"
             f"分辨率 {frame_desc.get('resolution', 'unknown')}，"
             + (f"源帧率 {frame_desc.get('source_fps')}fps，" if frame_desc.get("source_fps") else "")
             + "编码 image/jpeg")
    L.append(f"- 目标推帧：{cfg.fps} fps/会话；实测平均帧大小 "
             f"{avg_bytes / 1024:.1f} KiB（上限 {cfg.max_frame_bytes} 字节）")
    L.append("")

    # 3 会话负载
    L.append("## 3. 会话负载")
    L.append("")
    total_turns = sum(a["turns_completed"] for a in aggs)
    total_chars = sum(a["text_chars"] for a in aggs)
    L.append(f"- 每会话时长 {cfg.duration}s；提问间隔 {cfg.prompt_interval}s"
             f"（每会话提问约 {max(1, int(cfg.duration / max(cfg.prompt_interval, 0.1)))} 次）")
    L.append(f"- max_new_tokens={cfg.max_new_tokens}，temperature={cfg.temperature}，"
             f"top_p={cfg.top_p}，input_queue_capacity={cfg.input_queue_capacity}")
    if total_turns:
        L.append(f"- 实测平均每轮输出 {total_chars / total_turns:.0f} 字符（{total_turns} 轮）")
    L.append("")

    # 4 并发
    L.append("## 4. 并发")
    L.append("")
    L.append("并发定义：同一时间持续发视频帧并由服务端保持推理状态的活跃会话数。")
    L.append("")
    L.append("| 并发 | 成功率 | 异常断开 | 首次响应P95(s) | 轮时延P95(s) | 达标 |")
    L.append("|---:|---:|---:|---:|---:|:---:|")
    for a in aggs:
        L.append(f"| {a['level']} | {a['success_rate']:.1%} "
                 f"| {a['abnormal_disconnects']} "
                 f"| {_fmt(a['ttft_first_s']['p95'])} "
                 f"| {_fmt(a['turn_latency_s']['p95'])} "
                 f"| {'✅' if a['meets_targets'] else '❌'} |")
    L.append("")
    L.append(f"**稳定并发 = {stable if stable is not None else '无（最低并发级即不达标）'}**"
             "（满足时延与稳定性目标的最大并发级）")
    L.append("")

    # 5 时延
    L.append("## 5. 时延")
    L.append("")
    L.append("| 并发 | 首次响应 P50/P95/P99 (s) | 轮回答 P50/P95/P99 (s) |")
    L.append("|---:|---|---|")
    for a in aggs:
        t, tl = a["ttft_first_s"], a["turn_latency_s"]
        L.append(f"| {a['level']} | {_fmt(t['p50'])} / {_fmt(t['p95'])} / {_fmt(t['p99'])} "
                 f"| {_fmt(tl['p50'])} / {_fmt(tl['p95'])} / {_fmt(tl['p99'])} |")
    L.append("")

    # 6 稳定性
    L.append("## 6. 稳定性")
    L.append("")
    L.append("| 并发 | 成功率 | 异常断开率 | error事件 | invalid_request | 背压积压峰值 | 超限丢帧 |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|")
    for a in aggs:
        L.append(f"| {a['level']} | {a['success_rate']:.1%} "
                 f"| {a['abnormal_disconnect_rate']:.1%} | {a['error_events']} "
                 f"| {a['invalid_requests']} | {a['backlog_peak']} "
                 f"| {a['frames_oversized_dropped']} |")
    L.append("")

    # 7 资源
    L.append("## 7. 资源")
    L.append("")
    if gpu_samples:
        per_gpu: Dict[int, List[Dict[str, Any]]] = {}
        for s in gpu_samples:
            per_gpu.setdefault(s["index"], []).append(s)
        L.append("| GPU | util% min/avg/max | 显存 MiB min/avg/max | 采样数 |")
        L.append("|---:|---|---|---:|")
        for idx in sorted(per_gpu):
            ss = per_gpu[idx]
            utils = [s["util_pct"] for s in ss]
            mems = [s["mem_mib"] for s in ss]
            L.append(f"| {idx} | {min(utils):.0f}/{sum(utils)/len(utils):.0f}/{max(utils):.0f} "
                     f"| {min(mems):.0f}/{sum(mems)/len(mems):.0f}/{max(mems):.0f} "
                     f"| {len(ss)} |")
        L.append("")
        L.append(f"（每 {cfg.gpu_interval_s}s 采样一次；完整曲线见 JSON `gpu_sampling.samples`）")
    else:
        L.append(f"未采集。{gpu_note or '未开启 --gpu-sampling。'}")
    L.append("")

    # 8 容量策略
    L.append("## 8. 容量策略")
    L.append("")
    total_cap_rej = sum(a["capacity_rejected"] for a in aggs)
    L.append(f"- 观察到的容量拒绝：{total_cap_rej} 次"
             + ("（error[session_capacity_exceeded] + close 1013 / 网关 HTTP 503）"
                if total_cap_rej else "（未触发，容量上限高于测试并发）"))
    L.append(f"- 服务端降级行为：invalid_request 拒绝 "
             f"{sum(a['invalid_requests'] for a in aggs)} 次（单事件拒绝，会话存活，"
             "seq_no 回滚复用）；背压表现为 input.frame.ready 延迟，"
             "积压峰值见稳定性表")
    L.append(f"- 结论：对外只报稳定并发 **{stable if stable is not None else '无'}**"
             f"（目标：首次响应 P95 < {cfg.target_ttft_p95_s}s，成功率 ≥ "
             f"{cfg.target_success_rate:.0%}）")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ entry


def evaluate_stable_concurrency(aggs: List[Dict[str, Any]]) -> Optional[int]:
    stable: Optional[int] = None
    for a in aggs:
        if a["meets_targets"]:
            stable = a["level"]
    return stable


def run(cfg: StressConfig) -> int:
    if websockets is None:
        print("ERROR: the `websockets` package is required "
              "(repo .venv already has it)", file=sys.stderr)
        return 2
    levels = ramp_levels(cfg)
    # probe the frame source once up front for the report's 视频输入 section
    probe = build_frame_source(cfg)
    frame_desc = probe.describe()
    if callable(getattr(probe, "close", None)):
        probe.close()

    sampler: Optional[GpuSampler] = None
    if cfg.gpu_sampling:
        sampler = GpuSampler(cfg.gpu_interval_s)
        sampler.start()

    results: List[LevelResult] = []
    interrupted = False

    def flush(interrupted_flag: bool) -> Optional[int]:
        """Rewrite <out>.json/.md from everything collected so far. Called
        after EVERY ramp level so a crash in teardown never loses results;
        defensive: a report-writing failure must not kill the run itself."""
        aggs = [r.aggregate(cfg) for r in results]
        stable = evaluate_stable_concurrency(aggs)
        try:
            write_reports(cfg, results, aggs, sampler, frame_desc, stable,
                          interrupted_flag)
        except Exception as exc:  # noqa: BLE001
            print(f"[stress] WARNING: report flush failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return stable

    async def drive() -> None:
        nonlocal interrupted
        stop = asyncio.Event()
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, stop.set)
                except (NotImplementedError, RuntimeError):
                    pass
        except RuntimeError:
            pass
        for i, level in enumerate(levels):
            if stop.is_set():
                interrupted = True
                break
            print(f"[stress] level {level} session(s), {cfg.duration}s each ...",
                  flush=True)
            res = await run_level(cfg, level, stop)
            results.append(res)
            agg = res.aggregate(cfg)
            print(f"[stress] level {level}: success={agg['success_rate']:.1%} "
                  f"ttft_p95={_fmt(agg['ttft_first_s']['p95'])}s "
                  f"frames={agg['frames_accepted']} "
                  f"cap_rejected={agg['capacity_rejected']}", flush=True)
            flush(stop.is_set())  # checkpoint: cumulative report after each level
            if stop.is_set():
                interrupted = True
                break
            if i < len(levels) - 1:
                print(f"[stress] resting {cfg.ramp_rest_s}s before next level ...",
                      flush=True)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=cfg.ramp_rest_s)
                except asyncio.TimeoutError:
                    pass

    try:
        asyncio.run(drive())
    except KeyboardInterrupt:
        interrupted = True

    if sampler is not None:
        sampler.stop()
        sampler.join(timeout=3.0)

    stable = flush(interrupted)
    print(f"[stress] stable concurrency: {stable}; reports: {cfg.out}.json / {cfg.out}.md",
          flush=True)
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
