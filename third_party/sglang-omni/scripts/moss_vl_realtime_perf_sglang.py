#!/usr/bin/env python3
"""SGLang-Omni same-caliber perf baseline for MOSS-VL realtime (P10.2).

Client counterpart of scripts/moss_vl_realtime_perf_transformers.py. Drives a
running video realtime server over the binary-frame WebSocket protocol and
measures:
  - initial text prefill latency (configure -> session.ready, R sessions),
  - single-frame incremental extend latency (binary sent -> processed ack),
  - fixed-count decode TPOT / tokens/s from per-token delta arrivals.

TTFT is intentionally not measured: the realtime model emits runs of
<|silence|> before answering, and the silence-run length (a model-semantics
property that diverges between backends) dominates any TTFT number.

Frames are pushed without artificial FPS sleeps: the harness waits for each
processed ack, so measurements are warm-path latencies, not replay schedules.
The measurement session is configured with benchmark_ignore_eos=True so the
final prompt decodes exactly --decode-tokens visible steps without an early
EOS stop. Start the server with --enable-benchmark-mode; production-mode
servers reject this benchmark-only session option.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import websockets

DECODE_PROBE_PROMPT = (
    "Describe everything that happened in the video so far, in detail."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/video/realtime")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-reps", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--measure-frames", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--input-queue-capacity", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


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
        "min": ordered[0],
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

    def _find(self, predicate: Any, after: int) -> dict[str, Any] | None:
        for index in range(after, len(self.events)):
            if predicate(self.events[index]):
                return self.events[index]
        return None

    async def wait_for(
        self, event_type: str, *, timeout: float, after: int = 0, **match: Any
    ) -> dict[str, Any]:
        def predicate(event: dict[str, Any]) -> bool:
            if event.get("type") == "error":
                return True
            if event.get("type") != event_type:
                return False
            return all(event.get(key) == value for key, value in match.items())

        deadline = time.monotonic() + timeout
        while True:
            found = self._find(predicate, after)
            if found is not None:
                if found.get("type") == "error":
                    raise RuntimeError(found.get("message", "video realtime error"))
                return found
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {event_type} {match}")
            await asyncio.sleep(0.005)

    async def send_json(self, payload: dict[str, Any]) -> float:
        await self.websocket.send(json.dumps(payload))
        return time.perf_counter()

    async def send_frame(
        self,
        event: dict[str, Any],
        payload: bytes,
        mime_type: str,
        *,
        final: bool,
        timeout: float,
    ) -> dict[str, float]:
        seq_no = int(event["seq_no"])
        cursor = len(self.events)
        await self.send_json(
            {
                "type": "input.frame",
                "seq_no": seq_no,
                "timestamp": float(event["timestamp"]),
                "prompt": event.get("prompt"),
                "final": final,
                "mime_type": mime_type,
            }
        )
        await self.wait_for(
            "input.frame.ready", timeout=timeout, after=cursor, seq_no=seq_no
        )
        await self.websocket.send(payload)
        bytes_sent = time.perf_counter()
        await self.wait_for(
            "input.frame.accepted", timeout=timeout, after=cursor, seq_no=seq_no
        )
        processed = await self.wait_for(
            "input.frame.processed", timeout=timeout, after=cursor, seq_no=seq_no
        )
        return {"bytes_sent": bytes_sent, "processed": processed["_arrived"]}

    async def close(self) -> None:
        try:
            await self.websocket.close()
        finally:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)


async def connect_session(args: argparse.Namespace) -> Session:
    deadline = time.monotonic() + 30.0
    while True:
        websocket = await websockets.connect(args.url, max_size=64 * 1024 * 1024)
        session = Session(websocket)
        try:
            await session.wait_for("session.created", timeout=args.timeout)
            return session
        except RuntimeError:
            # A previous session may still be tearing down server-side.
            await session.close()
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(0.5)


async def measure_prefill(
    args: argparse.Namespace, case: dict[str, Any]
) -> list[float]:
    samples: list[float] = []
    for _ in range(args.prefill_reps):
        session = await connect_session(args)
        try:
            cursor = len(session.events)
            sent = await session.send_json(
                {
                    "type": "session.configure",
                    "prompt": case["initial_prompt"],
                    "system_prompt": case.get("system_prompt"),
                    "max_new_tokens": args.max_new_tokens,
                    "input_queue_capacity": args.input_queue_capacity,
                }
            )
            await session.wait_for(
                "session.configured", timeout=args.timeout, after=cursor
            )
            ready = await session.wait_for(
                "session.ready", timeout=args.timeout, after=cursor
            )
            samples.append(ready["_arrived"] - sent)
        finally:
            await session.close()
    return samples


async def measure_session(
    args: argparse.Namespace, case: dict[str, Any]
) -> dict[str, Any]:
    frame_events = [event for event in case["events"] if event.get("type") == "frame"]
    needed = args.warmup_frames + args.measure_frames
    if len(frame_events) < needed:
        raise ValueError(f"case has {len(frame_events)} frame events, need {needed}")
    mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

    session = await connect_session(args)
    try:
        cursor = len(session.events)
        await session.send_json(
            {
                "type": "session.configure",
                "prompt": case["initial_prompt"],
                "system_prompt": case.get("system_prompt"),
                "max_new_tokens": args.max_new_tokens,
                "input_queue_capacity": args.input_queue_capacity,
                "benchmark_ignore_eos": True,
            }
        )
        # Prefill latency for this session is not reported here; the dedicated
        # fresh-session repetitions in measure_prefill carry that metric.
        await session.wait_for("session.ready", timeout=args.timeout, after=cursor)

        async def push_frame(event: dict[str, Any], *, final: bool) -> dict[str, float]:
            path = Path(event["frame_path"])
            mime_type = mime_types.get(path.suffix.lower())
            if mime_type is None:
                raise ValueError(f"unsupported frame extension: {path.suffix}")
            return await session.send_frame(
                event,
                path.read_bytes(),
                mime_type,
                final=final,
                timeout=args.timeout,
            )

        for event in frame_events[: args.warmup_frames]:
            await push_frame(event, final=False)

        extend_seconds: list[float] = []
        for event in frame_events[args.warmup_frames : needed]:
            marks = await push_frame(event, final=False)
            extend_seconds.append(marks["processed"] - marks["bytes_sent"])

        cursor = len(session.events)
        final_seq_no = max(int(event["seq_no"]) for event in frame_events[:needed]) + 1
        prompt_sent = await session.send_json(
            {
                "type": "input.prompt",
                "seq_no": final_seq_no,
                "prompt": DECODE_PROBE_PROMPT,
                "final": True,
            }
        )
        processed = await session.wait_for(
            "input.prompt.processed",
            timeout=args.timeout,
            after=cursor,
            seq_no=final_seq_no,
        )

        # Collect decode-tokens visible deltas after the final prompt. Delta
        # arrivals are per-token stream messages; special tokens that decode
        # to empty text are invisible, so validate gap density afterwards.
        cursor = session.events.index(processed) + 1
        delta_times: list[float] = []
        delta_texts: list[str] = []
        deadline = time.monotonic() + args.timeout
        while len(delta_times) < args.decode_tokens:
            found = session._find(
                lambda event: event.get("type") == "response.text.delta",
                cursor,
            )
            if found is not None:
                cursor = session.events.index(found) + 1
                delta_times.append(found["_arrived"])
                delta_texts.append(str(found.get("delta", "")))
                continue
            if session._find(
                lambda event: event.get("type") == "response.done", cursor
            ):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out collecting decode deltas")
            await asyncio.sleep(0.002)

        intervals = [
            later - earlier for earlier, later in itertools.pairwise(delta_times)
        ]
        tpot = summarize(intervals)
        gap_flag = bool(tpot.get("max") and tpot["max"] > 5.0 * tpot["p50"])
        return {
            "frame_extend_seconds": {
                "samples": extend_seconds,
                **summarize(extend_seconds),
            },
            "final_prompt_sent_to_processed_seconds": (
                processed["_arrived"] - prompt_sent
            ),
            "decode_deltas_collected": len(delta_times),
            "decode_text": "".join(delta_texts),
            "decode_tpot_seconds": {"samples": intervals, **tpot},
            "decode_tokens_per_second": (
                1.0 / tpot["mean"] if tpot.get("mean") else None
            ),
            "decode_gap_warning": gap_flag,
        }
    finally:
        await session.close()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    case = load_case(args.manifest, args.case_id)
    prefill_samples = await measure_prefill(args, case)
    measured = await measure_session(args, case)
    result = {
        "backend": "sglang-omni",
        "url": args.url,
        "case_id": args.case_id,
        "config": {
            "prefill_reps": args.prefill_reps,
            "warmup_frames": args.warmup_frames,
            "measure_frames": args.measure_frames,
            "decode_tokens": args.decode_tokens,
            "max_new_tokens": args.max_new_tokens,
            "input_queue_capacity": args.input_queue_capacity,
        },
        "initial_prefill_seconds": {
            "samples": prefill_samples,
            **summarize(prefill_samples),
        },
        **measured,
    }
    return result


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
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
