#!/usr/bin/env python3
"""GPU probe for the MOSS-VL realtime vision-KV two-level sliding window.

Boots ``run_moss_vl_realtime_server.py`` as a subprocess, streams N synthetic
solid-color frames at a fixed real-time pace with fixed video timestamps, and
prints an observable timeline from both sides:

- client side: frame accepted / processed events with elapsed seconds;
- server side: ``Realtime frame window ...`` INFO log lines carrying
  ``encoder_length`` before/after, surviving frame count, and the KV pool's
  remaining slots (``pool_free_slots``) on every evict/pool round.

Run it twice (``--window off`` / ``--window on``) and diff the two outputs to
see the bounded vs unbounded encoder region. ``--window both`` runs off first,
then on, and prints a side-by-side summary.

Example (see the bottom of this file, `RUN COMMAND`, for the canonical form):

    python examples/probe_frame_window.py \
        --model-path /path/to/MOSS-VL-Realtime-sglang-checkpoint \
        --frames 24 --frame-interval 1.0 --ts-step 1.0 \
        --raw-window 8 --pool-window 24 --pool-ratio 4 --window both
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import websockets

WINDOW_LOG_PREFIX = "Realtime frame window"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8231)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=1.0,
        help="Real-time pace in seconds per frame (keep within 0.5-2.0).",
    )
    parser.add_argument(
        "--ts-step",
        type=float,
        default=1.0,
        help="Video-time step (seconds) carried by each synthetic frame.",
    )
    parser.add_argument("--raw-window", type=float, default=8.0)
    parser.add_argument("--pool-window", type=float, default=24.0)
    parser.add_argument("--pool-ratio", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--window",
        choices=("off", "on", "both"),
        default="both",
        help="off/on run once; both runs off then on and prints a diff summary.",
    )
    parser.add_argument("--server-timeout", type=float, default=600.0)
    parser.add_argument("--disable-decode-cuda-graph", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if not 0.5 <= args.frame_interval <= 2.0:
        parser.error("--frame-interval must stay within 0.5-2.0 seconds")
    if args.ts_step <= 0:
        parser.error("--ts-step must be positive")
    return args


def make_synthetic_frames(count: int, directory: Path) -> list[Path]:
    """Deterministic solid-color frames; color drifts so pooling sees variety."""
    from PIL import Image

    paths = []
    for index in range(count):
        r = (37 * index) % 256
        g = (91 * index + 40) % 256
        b = (53 * index + 17) % 256
        path = directory / f"frame_{index:04d}.png"
        Image.new("RGB", (64, 64), color=(r, g, b)).save(path)
        paths.append(path)
    return paths


async def _recv_until(websocket, expected: str, received: list[dict]):
    while True:
        raw = await websocket.recv()
        if not isinstance(raw, str):
            raise TypeError("server returned an unexpected binary message")
        event = json.loads(raw)
        received.append(event)
        if event.get("type") == "error":
            raise RuntimeError(event.get("message", "probe server error"))
        if event.get("type") == expected:
            return event


async def _observe(websocket, delay: float, received: list[dict]) -> None:
    deadline = asyncio.get_running_loop().time() + delay
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        except TimeoutError:
            return
        event = json.loads(raw)
        received.append(event)
        if event.get("type") == "error":
            raise RuntimeError(event.get("message", "probe server error"))
        if event.get("type") == "session.done":
            raise RuntimeError("session ended before all frames were sent")


async def stream_frames(args: argparse.Namespace, frames: list[Path]) -> list[dict]:
    url = f"ws://{args.host}:{args.port}/v1/video/realtime"
    received: list[dict] = []
    started_at = time.monotonic()
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as websocket:
        await _recv_until(websocket, "session.created", received)
        await websocket.send(
            json.dumps(
                {
                    "type": "session.configure",
                    "prompt": "Describe relevant changes briefly.",
                    "max_new_tokens": args.max_new_tokens,
                    "max_tokens_per_turn": 86400.0,
                    "input_queue_capacity": 4,
                }
            )
        )
        await _recv_until(websocket, "session.configured", received)
        await _recv_until(websocket, "session.ready", received)
        next_deadline = asyncio.get_running_loop().time()
        for index, path in enumerate(frames):
            delay = next_deadline - asyncio.get_running_loop().time()
            if delay > 0:
                await _observe(websocket, delay, received)
            final = index == len(frames) - 1
            await websocket.send(
                json.dumps(
                    {
                        "type": "input.frame",
                        "seq_no": index,
                        "timestamp": float(index) * args.ts_step,
                        "final": final,
                        "mime_type": "image/png",
                    }
                )
            )
            await _recv_until(websocket, "input.frame.ready", received)
            await websocket.send(path.read_bytes())
            await _recv_until(websocket, "input.frame.accepted", received)
            next_deadline += args.frame_interval
        await _recv_until(websocket, "session.done", received)
    for event in received:
        event["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
    return received


def start_server(args: argparse.Namespace, *, window_on: bool):
    env = dict(os.environ)
    if window_on:
        env.update(
            {
                "REALTIME_FRAME_WINDOW_ENABLED": "1",
                "REALTIME_FRAME_WINDOW_RAW_S": str(args.raw_window),
                "REALTIME_FRAME_POOL_WINDOW_S": str(args.pool_window),
                "REALTIME_FRAME_POOL_RATIO": str(args.pool_ratio),
            }
        )
    else:
        for name in (
            "REALTIME_FRAME_WINDOW_ENABLED",
            "REALTIME_FRAME_WINDOW_RAW_S",
            "REALTIME_FRAME_POOL_WINDOW_S",
            "REALTIME_FRAME_POOL_RATIO",
        ):
            env.pop(name, None)
    example = Path(__file__).with_name("run_moss_vl_realtime_server.py")
    command = [
        sys.executable,
        str(example),
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu",
        str(args.gpu),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.disable_decode_cuda_graph:
        command.append("--disable-decode-cuda-graph")
    log_file = open(  # noqa: SIM115 - kept alive for the server subprocess
        tempfile.mktemp(prefix="probe_frame_window_", suffix=".log"), "w+"
    )
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return process, log_file


def wait_for_server(args: argparse.Namespace, process: subprocess.Popen) -> None:
    url = f"ws://{args.host}:{args.port}/v1/video/realtime"
    deadline = time.monotonic() + args.server_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with code {process.returncode}")
        try:

            async def _try() -> None:
                async with websockets.connect(url):
                    return

            asyncio.run(_try())
            return
        except OSError:
            time.sleep(2.0)
    raise TimeoutError("server did not open the realtime endpoint in time")


WINDOW_LINE = re.compile(
    r"evicted_virtual=(?P<evict>\d+) pooled_raw=(?P<pooled>\d+) "
    r"produced_virtual=(?P<produced>\d+) dropped_raw=(?P<dropped>\d+) "
    r"encoder_length=(?P<before>\d+)->(?P<after>\d+) "
    r"surviving_frames=(?P<surviving>\d+) pool_free_slots=(?P<free>\S+)"
)


def server_window_events(log_path: str) -> list[tuple[float, dict]]:
    events = []
    with open(log_path) as handle:
        for line in handle:
            if WINDOW_LOG_PREFIX not in line:
                continue
            match = WINDOW_LINE.search(line)
            if match is None:
                events.append((None, {"raw": line.strip()}))
                continue
            data = match.groupdict()
            events.append(
                (
                    None,
                    {
                        "evicted_virtual": int(data["evict"]),
                        "pooled_raw": int(data["pooled"]),
                        "produced_virtual": int(data["produced"]),
                        "dropped_raw": int(data["dropped"]),
                        "encoder_length": f"{data['before']}->{data['after']}",
                        "surviving_frames": int(data["surviving"]),
                        "pool_free_slots": data["free"],
                    },
                )
            )
    return events


def run_once(args: argparse.Namespace, frames: list[Path], *, window_on: bool) -> dict:
    mode = "on" if window_on else "off"
    print(f"\n===== probe run: frame window {mode} =====", flush=True)
    process, log_file = start_server(args, window_on=window_on)
    try:
        wait_for_server(args, process)
        started = time.monotonic()
        received = asyncio.run(stream_frames(args, frames))
        elapsed = time.monotonic() - started
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    window_events = server_window_events(log_file.name)
    frame_processed = [
        e
        for e in received
        if e.get("type") == "stream"
        and e.get("data", {}).get("event") == "input.frame.processed"
    ]
    print(
        f"streamed {len(frames)} frames in {elapsed:.1f}s "
        f"({len(frame_processed)} processed acks)",
        flush=True,
    )
    if window_events:
        print("server frame-window timeline:", flush=True)
        for _, data in window_events:
            print(f"  {data}", flush=True)
    else:
        print(
            "server frame-window timeline: <none> (feature off or no eviction)",
            flush=True,
        )
    return {
        "mode": mode,
        "received": received,
        "window_events": [data for _, data in window_events],
        "log_path": log_file.name,
        "elapsed_s": round(elapsed, 2),
    }


def print_diff(off: dict, on: dict) -> None:
    print("\n===== off/on comparison =====", flush=True)
    print(f"off: window events={len(off['window_events'])} log={off['log_path']}")
    print(f"on:  window events={len(on['window_events'])} log={on['log_path']}")
    if on["window_events"]:
        last = on["window_events"][-1]
        print(
            "on: final encoder_length "
            f"{last.get('encoder_length')}, surviving frames "
            f"{last.get('surviving_frames')}, pool free slots "
            f"{last.get('pool_free_slots')} — bounded by the raw window; "
            "off grows the encoder region for every frame.",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="probe_frame_window_frames_") as tmp:
        frames = make_synthetic_frames(args.frames, Path(tmp))
        runs = []
        modes = ["off", "on"] if args.window == "both" else [args.window]
        for mode in modes:
            runs.append(run_once(args, frames, window_on=mode == "on"))
        if len(runs) == 2:
            print_diff(runs[0], runs[1])


if __name__ == "__main__":
    main()

# RUN COMMAND
# ---------------------------------------------------------------------------
# Single run with the window enabled (small windows so eviction is visible
# within ~30 frames at 1s pace):
#
#   python examples/probe_frame_window.py \
#       --model-path /path/to/MOSS-VL-Realtime-sglang-checkpoint \
#       --frames 30 --frame-interval 1.0 --ts-step 1.0 \
#       --raw-window 8 --pool-window 24 --pool-ratio 4 --window on
#
# Off/on comparison run (baseline first, then the windowed server):
#
#   python examples/probe_frame_window.py \
#       --model-path /path/to/MOSS-VL-Realtime-sglang-checkpoint \
#       --window both
# ---------------------------------------------------------------------------
