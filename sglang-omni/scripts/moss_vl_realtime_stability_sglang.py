#!/usr/bin/env python3
"""P11 one-minute real-time stability harness for MOSS-VL realtime (SGLang).

Unlike the P10.2 perf harness (ack-paced, warm-path latencies), this script
replays frames on a wall-clock schedule like a real 1 FPS (or faster) stream:

  - frames loop over the case's frame events with fresh seq_no / monotonic
    timestamps for --duration-s seconds at --fps;
  - flow control is the server's own: input.frame.ready is awaited before the
    payload, so its wait time is the backpressure signal;
  - a non-final probe prompt is sent at --probe-at-s to verify the model
    produces visible speech mid-stream ("能正常说话");
  - GPU memory is sampled locally every --mem-sample-s seconds;
  - optional --disconnect-at-s drops the WebSocket abruptly mid-stream, then a
    fresh probe session proves the server is still healthy and talking.

Output JSON: per-phase timings, backpressure (ready-wait) stats, visible text
with arrival times, memory samples, and any error events.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import websockets

PROBE_PROMPT = "What is happening in the video right now? Answer briefly."
FINAL_PROMPT = "Describe everything that happened in the video, in detail."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/video/realtime")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=65.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--probe-at-s", type=float, default=30.0)
    parser.add_argument("--probe-window-s", type=float, default=15.0)
    parser.add_argument("--final-window-s", type=float, default=20.0)
    parser.add_argument("--disconnect-at-s", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens-per-turn", type=float, default=86400.0)
    parser.add_argument("--input-queue-capacity", type=int, default=4)
    parser.add_argument("--mem-sample-s", type=float, default=2.0)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


class Session:
    """One WebSocket session with a background reader timestamping events."""

    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self.events: list[dict[str, Any]] = []
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        async for raw in self.websocket:
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            event["_arrived"] = time.perf_counter()
            self.events.append(event)

    def find_all(self, event_type: str, after: int = 0) -> list[dict[str, Any]]:
        return [
            event for event in self.events[after:] if event.get("type") == event_type
        ]

    async def wait_for(
        self, event_type: str, *, timeout: float, after: int = 0, **match: Any
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            for event in self.events[after:]:
                if event.get("type") == "error":
                    raise RuntimeError(event.get("message", "video realtime error"))
                if event.get("type") != event_type:
                    continue
                if all(event.get(k) == v for k, v in match.items()):
                    return event
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {event_type} {match}")
            await asyncio.sleep(0.005)

    async def send_json(self, payload: dict[str, Any]) -> float:
        await self.websocket.send(json.dumps(payload))
        return time.perf_counter()

    async def push_frame(
        self,
        seq_no: int,
        timestamp: float,
        payload: bytes,
        mime_type: str,
        timeout: float,
    ) -> dict[str, float]:
        """Send one frame; the ready wait is the server backpressure signal."""
        cursor = len(self.events)
        sent = await self.send_json(
            {
                "type": "input.frame",
                "seq_no": seq_no,
                "timestamp": timestamp,
                "mime_type": mime_type,
            }
        )
        ready = await self.wait_for(
            "input.frame.ready", timeout=timeout, after=cursor, seq_no=seq_no
        )
        await self.websocket.send(payload)
        return {
            "sent": sent,
            "ready_wait": ready["_arrived"] - sent,
            "acked": time.perf_counter(),
        }

    async def close(self) -> None:
        try:
            await self.websocket.close()
        finally:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)


async def connect_session(args: argparse.Namespace) -> Session:
    websocket = await websockets.connect(args.url, max_size=64 * 1024 * 1024)
    session = Session(websocket)
    await session.wait_for("session.created", timeout=args.timeout)
    return session


async def configure(
    session: Session, args: argparse.Namespace, case: dict[str, Any]
) -> None:
    cursor = len(session.events)
    await session.send_json(
        {
            "type": "session.configure",
            "prompt": case["initial_prompt"],
            "system_prompt": case.get("system_prompt"),
            "max_new_tokens": args.max_new_tokens,
            "max_tokens_per_turn": args.max_tokens_per_turn,
            "input_queue_capacity": args.input_queue_capacity,
        }
    )
    await session.wait_for("session.ready", timeout=args.timeout, after=cursor)


async def collect_visible(
    session: Session, *, cursor: int, window_s: float
) -> list[dict[str, Any]]:
    """Collect response.text.delta events for a fixed wall-clock window."""
    deadline = time.monotonic() + window_s
    while time.monotonic() < deadline:
        if session.find_all("error", after=cursor):
            message = session.find_all("error", after=cursor)[0].get("message")
            raise RuntimeError(message or "video realtime error")
        await asyncio.sleep(0.05)
    deltas = session.find_all("response.text.delta", after=cursor)
    return [{"t": d["_arrived"], "delta": str(d.get("delta", ""))} for d in deltas]


def sample_memory(args: argparse.Namespace, samples: list[dict[str, Any]]) -> None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={args.gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        mib = int(out.stdout.strip())
    except (OSError, ValueError):
        return
    samples.append({"t": time.perf_counter(), "mib": mib})


async def memory_sampler(
    args: argparse.Namespace, samples: list[dict[str, Any]], stop: asyncio.Event
) -> None:
    while not stop.is_set():
        sample_memory(args, samples)
        try:
            await asyncio.wait_for(stop.wait(), timeout=args.mem_sample_s)
        except asyncio.TimeoutError:
            pass


def load_frames(case: dict[str, Any]) -> list[dict[str, Any]]:
    mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    frames = []
    for event in case["events"]:
        if event.get("type") != "frame":
            continue
        path = Path(event["frame_path"])
        frames.append(
            {
                "payload": path.read_bytes(),
                "mime_type": mime_types[path.suffix.lower()],
            }
        )
    if not frames:
        raise ValueError("case has no frame events")
    return frames


async def run_stream(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    frames = load_frames(case)
    mem_samples: list[dict[str, Any]] = []
    mem_stop = asyncio.Event()
    sampler = asyncio.create_task(memory_sampler(args, mem_samples, mem_stop))
    session = await connect_session(args)
    disconnected = False
    try:
        await configure(session, args, case)
        t0 = time.perf_counter()
        probe_at = t0 + args.probe_at_s
        probe_sent_at: float | None = None
        probe_task: asyncio.Task | None = None
        probe_cursor = 0
        frame_marks: list[dict[str, float]] = []
        frame_count = int(args.duration_s * args.fps)
        next_seq_no = 0  # events require consecutive seq_no starting at 0
        for index in range(frame_count):
            schedule_t = t0 + index / args.fps
            await asyncio.sleep(max(0.0, schedule_t - time.perf_counter()))
            if probe_task is None and time.perf_counter() >= probe_at:
                probe_cursor = len(session.events)
                await session.send_json(
                    {
                        "type": "input.prompt",
                        "seq_no": next_seq_no,
                        "prompt": PROBE_PROMPT,
                        "final": False,
                    }
                )
                next_seq_no += 1
                probe_sent_at = time.perf_counter() - t0
                # Collect concurrently: frames must keep flowing, since a
                # parked request only wakes on new events.
                probe_task = asyncio.create_task(
                    collect_visible(
                        session, cursor=probe_cursor, window_s=args.probe_window_s
                    )
                )
            frame = frames[index % len(frames)]
            marks = await session.push_frame(
                seq_no=next_seq_no,
                timestamp=index / args.fps,
                payload=frame["payload"],
                mime_type=frame["mime_type"],
                timeout=args.timeout,
            )
            next_seq_no += 1
            marks["schedule_lag"] = marks["sent"] - schedule_t
            frame_marks.append(marks)
            if (
                args.disconnect_at_s
                and time.perf_counter() - t0 >= args.disconnect_at_s
            ):
                # Abrupt client drop: no final event, no session.abort.
                await session.websocket.close()
                disconnected = True
                break

        probe_done: dict[str, Any] | None = None
        if probe_task is not None:
            deltas = await probe_task
            probe_done = {
                "at_s": probe_sent_at,
                "deltas": deltas,
                "text": "".join(d["delta"] for d in deltas),
            }
        final_result: dict[str, Any] | None = None
        if not disconnected:
            cursor = len(session.events)
            await session.send_json(
                {
                    "type": "input.prompt",
                    "seq_no": next_seq_no,
                    "prompt": FINAL_PROMPT,
                    "final": True,
                }
            )
            deltas = await collect_visible(
                session, cursor=cursor, window_s=args.final_window_s
            )
            final_result = {
                "deltas": deltas,
                "text": "".join(d["delta"] for d in deltas),
            }
        elapsed = time.perf_counter() - t0
        processed = session.find_all("input.frame.processed")
        errors = session.find_all("error")
        return {
            "elapsed_s": elapsed,
            "frames_sent": len(frame_marks),
            "frames_processed": len(processed),
            "disconnected_at_s": args.disconnect_at_s if disconnected else None,
            "ready_wait_seconds": summarize(
                [mark["ready_wait"] for mark in frame_marks]
            ),
            "schedule_lag_seconds": summarize(
                [mark["schedule_lag"] for mark in frame_marks]
            ),
            "probe": probe_done,
            "final": final_result,
            "errors": [
                {k: v for k, v in event.items() if k != "_arrived"} for event in errors
            ],
            "memory_mib": mem_samples,
        }
    finally:
        mem_stop.set()
        await sampler
        await session.close()


async def run_reconnect_probe(
    args: argparse.Namespace, case: dict[str, Any]
) -> dict[str, Any]:
    """After an abrupt disconnect, prove the server still serves and talks."""
    frames = load_frames(case)
    await asyncio.sleep(3.0)
    session = await connect_session(args)
    try:
        await configure(session, args, case)
        for index in range(3):
            frame = frames[index]
            await session.push_frame(
                seq_no=index,
                timestamp=float(index),
                payload=frame["payload"],
                mime_type=frame["mime_type"],
                timeout=args.timeout,
            )
        cursor = len(session.events)
        await session.send_json(
            {
                "type": "input.prompt",
                "seq_no": 3,
                "prompt": PROBE_PROMPT,
                "final": True,
            }
        )
        deltas = await collect_visible(
            session, cursor=cursor, window_s=args.final_window_s
        )
        return {
            "text": "".join(d["delta"] for d in deltas),
            "delta_count": len(deltas),
            "errors": session.find_all("error"),
        }
    finally:
        await session.close()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_tokens_per_turn <= 0:
        raise ValueError("--max-tokens-per-turn must be positive")
    case = load_case(args.manifest, args.case_id)
    result: dict[str, Any] = {
        "backend": "sglang-omni",
        "url": args.url,
        "case_id": args.case_id,
        "config": {
            "duration_s": args.duration_s,
            "fps": args.fps,
            "probe_at_s": args.probe_at_s,
            "disconnect_at_s": args.disconnect_at_s,
            "input_queue_capacity": args.input_queue_capacity,
            "max_tokens_per_turn": args.max_tokens_per_turn,
        },
    }
    result["stream"] = await run_stream(args, case)
    if result["stream"]["disconnected_at_s"]:
        result["reconnect_probe"] = await run_reconnect_probe(args, case)
    return result


def load_case(path: Path, case_id: str) -> dict[str, Any]:
    matches = [
        record
        for line in path.read_text().splitlines()
        if line
        for record in [json.loads(line)]
        if record.get("case_id") == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"case_id {case_id!r} was not found exactly once")
    return matches[0]


def main() -> None:
    args = parse_args()
    try:
        result = asyncio.run(run(args))
    except BaseException as exc:
        result = {
            "backend": "sglang-omni",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    stream = result.get("stream", {})
    probe = (stream.get("probe") or {}).get("text", "")
    final = (stream.get("final") or {}).get("text", "")
    print(
        f"frames: sent={stream.get('frames_sent')} "
        f"processed={stream.get('frames_processed')} "
        f"elapsed={stream.get('elapsed_s', 0):.1f}s"
    )
    print(
        f"ready_wait p50/p95: "
        f"{stream.get('ready_wait_seconds', {}).get('p50', 0) * 1000:.1f}/"
        f"{stream.get('ready_wait_seconds', {}).get('p95', 0) * 1000:.1f} ms"
    )
    print(f"errors: {stream.get('errors')}")
    print(f"probe text: {probe[:200]}")
    print(f"final text: {final[:200]}")
    reconnect = result.get("reconnect_probe")
    if reconnect:
        print(f"reconnect text: {reconnect.get('text', '')[:200]}")
    mem = stream.get("memory_mib") or []
    if mem:
        values = [sample["mib"] for sample in mem]
        print(
            f"memory MiB: first={values[0]} last={values[-1]} "
            f"max={max(values)} delta={values[-1] - values[0]}"
        )


if __name__ == "__main__":
    main()
