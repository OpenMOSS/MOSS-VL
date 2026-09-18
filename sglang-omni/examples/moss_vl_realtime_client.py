#!/usr/bin/env python3
"""Send timestamped binary image frames to /v1/video/realtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/video/realtime")
    parser.add_argument("--frame", action="append")
    parser.add_argument("--timestamp", action="append", type=float)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--prompt", default="Describe relevant changes.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-tokens-per-turn", type=float, default=86400.0)
    parser.add_argument("--initial-delay", type=float, default=0.0)
    parser.add_argument("--frame-interval", type=float)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--input-queue-capacity", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_manifest_case(path: Path, case_id: str | None) -> dict[str, Any]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not cases:
        raise ValueError(f"manifest is empty: {path}")
    if case_id is None:
        if len(cases) != 1:
            raise ValueError(
                "--case-id is required when the manifest has multiple cases"
            )
        return cases[0]
    matches = [case for case in cases if case.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError(f"case_id {case_id!r} was not found exactly once")
    return matches[0]


def resolve_frame_interval(
    args: argparse.Namespace, case: dict[str, Any] | None
) -> float:
    if args.fps is not None and args.frame_interval is not None:
        raise ValueError("--fps and --frame-interval cannot be used together")
    if args.fps is not None:
        if args.fps <= 0:
            raise ValueError("--fps must be positive")
        return 1.0 / args.fps
    if args.frame_interval is not None:
        return float(args.frame_interval)
    if case is not None:
        if case.get("fps") is not None:
            fps = float(case["fps"])
            if fps <= 0:
                raise ValueError("manifest fps must be positive")
            return 1.0 / fps
        if case.get("frame_interval_seconds") is not None:
            return float(case["frame_interval_seconds"])
    return 1.0


def resolve_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    if args.manifest is not None:
        if args.frame or args.timestamp:
            raise ValueError("--manifest cannot be combined with --frame/--timestamp")
        case = load_manifest_case(args.manifest, args.case_id)
        events = list(case.get("events") or ())
        if not events:
            raise ValueError("manifest case has no input events")
        interval = resolve_frame_interval(args, case)
        config = {
            "prompt": case["initial_prompt"],
            "system_prompt": case.get("system_prompt"),
        }
        return config, events, interval

    frames = args.frame or []
    timestamps = args.timestamp or []
    if not frames or not timestamps:
        raise ValueError("provide --manifest or matching --frame/--timestamp values")
    if len(frames) != len(timestamps):
        raise ValueError("--frame and --timestamp counts must match")
    events = [
        {
            "type": "frame",
            "seq_no": seq_no,
            "timestamp": timestamp,
            "frame_path": frame,
            "final": seq_no == len(frames) - 1,
        }
        for seq_no, (frame, timestamp) in enumerate(
            zip(frames, timestamps, strict=True)
        )
    ]
    interval = resolve_frame_interval(args, None)
    return {"prompt": args.prompt, "system_prompt": None}, events, interval


async def receive_until(
    websocket: Any,
    expected_type: str,
    *,
    received: list[dict[str, Any]] | None = None,
    started_at: float | None = None,
) -> dict[str, Any]:
    while True:
        raw = await websocket.recv()
        if not isinstance(raw, str):
            raise TypeError("server returned an unexpected binary message")
        event = json.loads(raw)
        if received is not None:
            recorded = dict(event)
            if started_at is not None:
                recorded["elapsed_seconds"] = time.monotonic() - started_at
            received.append(recorded)
        event_type = event.get("type")
        if event_type == "response.text.delta":
            print(event.get("delta", ""), end="", flush=True)
        elif event_type == "response.done":
            print(
                f"\n[response.done: {event.get('finish_reason')}]",
                flush=True,
            )
        elif event_type == "error":
            raise RuntimeError(event.get("message", "video realtime error"))
        if event_type == expected_type:
            return event


async def observe_during_delay(
    websocket: Any,
    delay: float,
    *,
    received: list[dict[str, Any]],
    started_at: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + delay
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        except TimeoutError:
            return
        if not isinstance(raw, str):
            raise TypeError("server returned an unexpected binary message")
        event = json.loads(raw)
        recorded = dict(event)
        recorded["elapsed_seconds"] = time.monotonic() - started_at
        received.append(recorded)
        event_type = event.get("type")
        if event_type == "response.text.delta":
            print(event.get("delta", ""), end="", flush=True)
        elif event_type == "error":
            raise RuntimeError(event.get("message", "video realtime error"))
        elif event_type == "session.done":
            raise RuntimeError("realtime request ended before the first frame")


async def run(args: argparse.Namespace) -> None:
    config, events, frame_interval = resolve_inputs(args)
    if frame_interval < 0:
        raise ValueError("--frame-interval must be non-negative")
    if not 1 <= args.input_queue_capacity <= 256:
        raise ValueError("--input-queue-capacity must be between 1 and 256")
    if args.max_tokens_per_turn <= 0:
        raise ValueError("--max-tokens-per-turn must be positive")
    frame_payloads: dict[int, tuple[str, bytes]] = {}
    for event in events:
        if event.get("type") != "frame":
            continue
        path = Path(event["frame_path"])
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(path.suffix.lower())
        if mime_type is None:
            raise ValueError(f"unsupported frame extension: {path.suffix}")
        frame_payloads[int(event["seq_no"])] = (mime_type, path.read_bytes())
    received: list[dict[str, Any]] = []
    started_at = time.monotonic()
    receive_kwargs = {"received": received, "started_at": started_at}
    async with websockets.connect(args.url, max_size=64 * 1024 * 1024) as websocket:
        created = await receive_until(websocket, "session.created", **receive_kwargs)
        print(
            f"session={created['session_id']} request={created['request_id']}",
            flush=True,
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "session.configure",
                    "prompt": config["prompt"],
                    "system_prompt": config["system_prompt"],
                    "max_new_tokens": args.max_new_tokens,
                    "max_tokens_per_turn": args.max_tokens_per_turn,
                    "input_queue_capacity": args.input_queue_capacity,
                }
            )
        )
        await receive_until(websocket, "session.configured", **receive_kwargs)
        ready = await receive_until(websocket, "session.ready", **receive_kwargs)
        stream_started_at = time.monotonic()
        print(
            f"ready session={ready['session_id']} after "
            f"{stream_started_at - started_at:.3f}s",
            flush=True,
        )
        if args.initial_delay < 0:
            raise ValueError("--initial-delay must be non-negative")
        if args.initial_delay:
            await observe_during_delay(
                websocket,
                args.initial_delay,
                received=received,
                started_at=started_at,
            )

        schedule_start = asyncio.get_running_loop().time()
        next_event_deadline = schedule_start
        for event_index, event in enumerate(events):
            event_type = event.get("type")
            seq_no = int(event["seq_no"])
            final = bool(event.get("final", event_index == len(events) - 1))
            # Honor the manifest's per-event timestamp when present so prompts
            # spaced after the last frame keep their intended gaps; otherwise
            # fall back to the fixed per-frame cadence.
            event_timestamp = event.get("timestamp")
            if event_timestamp is not None:
                next_event_deadline = schedule_start + float(event_timestamp)
            delay = next_event_deadline - asyncio.get_running_loop().time()
            if delay > 0:
                await observe_during_delay(
                    websocket,
                    delay,
                    received=received,
                    started_at=started_at,
                )
            if event_type == "prompt":
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input.prompt",
                            "seq_no": seq_no,
                            "prompt": event["prompt"],
                            "final": final,
                        }
                    )
                )
                await receive_until(
                    websocket, "input.prompt.accepted", **receive_kwargs
                )
                continue
            if event_type != "frame":
                raise ValueError(f"unsupported manifest event type: {event_type!r}")
            timestamp = float(event["timestamp"])
            mime_type, frame_bytes = frame_payloads[seq_no]
            await websocket.send(
                json.dumps(
                    {
                        "type": "input.frame",
                        "seq_no": seq_no,
                        "timestamp": timestamp,
                        "prompt": event.get("prompt"),
                        "final": final,
                        "mime_type": mime_type,
                    }
                )
            )
            await receive_until(websocket, "input.frame.ready", **receive_kwargs)
            await websocket.send(frame_bytes)
            await receive_until(websocket, "input.frame.accepted", **receive_kwargs)
            next_event_deadline += frame_interval

        await receive_until(websocket, "session.done", **receive_kwargs)
        print()
    if args.output is not None:
        result = {
            "backend": "sglang-omni",
            "case_id": args.case_id,
            "frame_interval_seconds": frame_interval,
            "fps": 1.0 / frame_interval if frame_interval else None,
            "input_queue_capacity": args.input_queue_capacity,
            "elapsed_seconds": time.monotonic() - started_at,
            "session_ready_seconds": stream_started_at - started_at,
            "stream_elapsed_seconds": time.monotonic() - stream_started_at,
            "normalized_text": "".join(event.get("delta", "") for event in received),
            "events": received,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
