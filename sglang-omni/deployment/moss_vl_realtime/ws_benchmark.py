"""Realtime WebSocket multi-session benchmark (server-mode, DP-aware).

Boots a real ``examples/run_moss_vl_realtime_server.py`` server process with
the shared deployment helpers (``common.server_command`` / ``common.Child`` /
``common.environment``) and, for each session count in ``--sessions``, runs
``--repeats`` measured rounds of N concurrent WebSocket lanes against
``/v1/video/realtime``. Every lane replays the standard LATENCY_CASE workload
(12 frames + 2 prompts) at ``--fps`` with a per-session ``--token-rate``
output pace, exactly like the public multi-session test — but over the real
serving path (Coordinator + entry-stage replicas), unlike the in-process
driver in ``evaluation.py``.

Use ``--dp-size N`` for a native data-parallel boot: the script mutates
``common.CONFIG`` in memory (it never writes ``config.json``), the resolved
server command ships ``--dp-size N --gpus 0..N-1`` on device-local ids, and
the child process sees exactly the ``--gpus`` GPU selection through
``CUDA_VISIBLE_DEVICES`` (the same narrowing rule as ``entry.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common
from common import (
    CONFIG,
    Child,
    free_port,
    gpu_inventory,
    select_gpus,
    validate_model,
    write_json,
)
from concurrency_benchmark import WARMUP_FRAMES, workload
from semantic_checks import load_suite, stats

PROTOCOL = "ws_realtime_dp_v1"
LATENCY_CASE = "cd067_sbpro_L2_stream_000122"
# Same silence-policy override as concurrency_benchmark.measure(): answer
# generation in every lane starts exactly at the scheduled prompts.
INITIAL_PROMPT_OVERRIDE = "Watch the video. Stay silent until a question is asked."
# Frames travel as binary WebSocket messages; match the server transport cap
# plus headroom, like examples/moss_vl_realtime_client.py.
WS_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deployment/moss_vl_realtime/ws_benchmark.py", description=__doc__
    )
    parser.add_argument("model_path", type=Path, help="Local model directory")
    parser.add_argument("--cases-dir", type=Path, default=HERE / "cases")
    parser.add_argument(
        "--gpus",
        help="Server GPU indices/UUIDs; must select exactly --dp-size GPUs",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Entry-stage data-parallel replicas (native DP) for the run; "
        "passed to the server via an in-memory CONFIG override",
    )
    parser.add_argument(
        "--max-sessions-per-replica",
        type=int,
        default=None,
        help="Per-replica session cap (server --max-running-requests); the "
        "aggregate WebSocket admission cap is this times --dp-size. Default: "
        "config.json max_sessions, raised as needed to fit --sessions",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Concurrent session counts to measure",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Measured rounds per session count"
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Input frame rate")
    parser.add_argument(
        "--token-rate",
        type=float,
        default=10.0,
        help="Per-session max output tokens per second; 86400 is unthrottled",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Per-session generated-token cap",
    )
    parser.add_argument(
        "--host", default=CONFIG["host"], help="Server bind host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Server port; 0 picks a free port",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results",
        help="Result directory; the JSON lands as ws_benchmark_dp<N>.json",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for the server to start listening",
    )
    parser.add_argument(
        "--drain-seconds",
        type=float,
        default=15.0,
        help="Extra seconds a lane may answer after its final input",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=300.0,
        help="Per-wait deadline for protocol handshakes inside a session",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the resolved plan; do not boot",
    )
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.dp_size < 1:
        parser.error("--dp-size must be at least 1")
    if args.max_sessions_per_replica is None:
        # Like evaluation.py, the per-replica pool covers the largest planned
        # session count; replicas partition sessions, so divide by dp_size.
        args.max_sessions_per_replica = max(
            CONFIG["max_sessions"],
            -(-max(args.sessions) // args.dp_size),
        )
    if args.max_sessions_per_replica < 1:
        parser.error("--max-sessions-per-replica must be at least 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if any(count < 1 for count in args.sessions):
        parser.error("--sessions counts must be positive")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    if not math.isfinite(args.token_rate) or args.token_rate <= 0:
        parser.error("--token-rate must be positive and finite")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    total_capacity = args.dp_size * args.max_sessions_per_replica
    if max(args.sessions) > total_capacity:
        parser.error(
            f"--sessions {max(args.sessions)} exceeds the total session "
            f"capacity {total_capacity} (dp_size={args.dp_size} x "
            f"max_sessions_per_replica={args.max_sessions_per_replica})"
        )
    if args.drain_seconds <= 0:
        parser.error("--drain-seconds must be positive")
    if args.wait_timeout <= 0:
        parser.error("--wait-timeout must be positive")
    if args.port and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")


def configure_config(args: argparse.Namespace) -> None:
    """Apply the run's controls to the shared deployment CONFIG in memory.

    ``common.server_command`` reads ``CONFIG`` when it formats the server CLI,
    so the subprocess inherits these values without touching config.json.
    """
    CONFIG["dp_size"] = args.dp_size
    CONFIG["max_sessions"] = args.max_sessions_per_replica


def resolve_server_gpus(args: argparse.Namespace) -> list[dict]:
    """Pick server GPUs the same way entry.py does; expect --dp-size of them."""
    gpus = select_gpus(
        args.gpus, gpu_inventory(), os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if len(gpus) != args.dp_size:
        raise ValueError(
            f"--gpus must select exactly --dp-size={args.dp_size} GPU(s), "
            f"the replicas of this run need one each; got {len(gpus)}"
        )
    return gpus


def resolve_plan(args: argparse.Namespace, model: Path) -> dict:
    """Resolve the server command and per-round session plan.

    The CONFIG overrides take effect only while the command is formatted and
    are rolled back immediately after, so importing test drivers can call this
    without leaking state into other deployment tests.
    """
    saved_config = dict(CONFIG)
    try:
        configure_config(args)
        port = args.port or free_port(args.host)
        command = common.server_command(model, port, host=args.host)
    finally:
        CONFIG.clear()
        CONFIG.update(saved_config)
    case = _load_case(args.cases_dir)
    events = workload(case, args.fps)
    return {
        "model_path": str(model),
        "host": args.host,
        "port": port,
        "command": command,
        "dp_size": args.dp_size,
        "max_sessions_per_replica": args.max_sessions_per_replica,
        "case_id": case["case_id"],
        "events_per_session": len(events),
        "fps": args.fps,
        "token_rate": args.token_rate,
        "sessions": list(args.sessions),
        "repeats": args.repeats,
    }


def _load_case(cases_dir: Path) -> dict:
    case = next(
        (c for c in load_suite(cases_dir) if c["case_id"] == LATENCY_CASE), None
    )
    if case is None:
        raise ValueError(f"case {LATENCY_CASE!r} not found under {cases_dir}")
    return case


def wait_for_listen(
    *,
    host: str,
    port: int,
    child: Child,
    log_path: Path,
    timeout_s: float,
) -> None:
    """Block until /health answers 200; fail fast on an early-died server."""
    deadline = time.monotonic() + timeout_s
    while True:
        if child.process.poll() is not None:
            raise RuntimeError(
                f"server exited early (rc={child.process.returncode}); "
                f"log tail:\n{log_path.read_text()[-8000:]}"
            )
        try:
            with urllib_request.urlopen(
                f"http://{host}:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"server did not listen within {timeout_s}s; "
                f"log tail:\n{log_path.read_text()[-8000:]}"
            )
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# One WebSocket lane. Protocol choreography mirrors the example client
# (examples/moss_vl_realtime_client.py): session.created -> session.configure
# -> session.configured/session.ready -> [input.frame.ready -> binary frame ->
# input.frame.accepted] per event -> session.done after the final input.
# ---------------------------------------------------------------------------


def _load_frame_payloads(events: list[dict]) -> dict[int, tuple[str, bytes]]:
    payloads: dict[int, tuple[str, bytes]] = {}
    for event in events:
        if event["type"] != "frame":
            continue
        path = Path(event["frame_path"])
        mime_type = _MIME_TYPES.get(path.suffix.lower())
        if mime_type is None:
            raise ValueError(f"unsupported frame extension: {path.suffix}")
        payloads[int(event["seq_no"])] = (mime_type, path.read_bytes())
    return payloads


def _lane_metrics(
    *,
    lane_index: int,
    epoch: float,
    events: list[dict],
    sent: list[dict],
    received: list[dict],
) -> tuple[dict, list[float]]:
    """Derive per-lane timing metrics from the wall-clock transcript.

    Returns the lane result plus the raw token-arrival gaps (kept out of the
    lane result so rounds stay compact; ``_summarize`` pools them).
    """
    sent_by_seq = {record["seq_no"]: record for record in sent}
    text_events: list[dict] = []
    processed_by_seq: dict[int, float] = {}
    usage_final: dict | None = None
    for entry in received:
        payload = entry["payload"]
        payload_type = payload.get("type")
        if payload_type == "response.text.delta":
            text_events.append(
                {"arrived_at": entry["arrived_at"], "turn_id": payload.get("turn_id")}
            )
        elif payload_type in ("input.frame.processed", "input.prompt.processed"):
            processed_by_seq[int(payload["seq_no"])] = entry["arrived_at"]
        elif payload_type == "session.usage":
            usage_final = payload

    first_text_at = min(
        (event["arrived_at"] for event in text_events), default=None
    )
    last_text_at = max(
        (event["arrived_at"] for event in text_events), default=None
    )
    decode_span_s = (
        last_text_at - first_text_at
        if first_text_at is not None and last_text_at is not None
        else 0.0
    )
    # Prefer the scheduler's sampled decoder count (session.usage, requested
    # via include_usage) over counting transport deltas.
    tokens = (
        int(usage_final["decoder_tokens"])
        if usage_final and usage_final.get("decoder_tokens") is not None
        else len(text_events)
    )

    # Pair the k-th prompt with the k-th answer turn (distinct turn_id in
    # arrival order); prompts that produced no visible text get ttft=None.
    distinct_turns: list = []
    for event in text_events:
        if event["turn_id"] not in distinct_turns:
            distinct_turns.append(event["turn_id"])
    questions: list[dict] = []
    frame_delays: list[float] = []
    question_index = 0
    for event in events:
        record = sent_by_seq[int(event["seq_no"])]
        processed_at = processed_by_seq.get(record["seq_no"])
        if event["type"] == "frame":
            if event["frame_index"] >= WARMUP_FRAMES and processed_at is not None:
                frame_delays.append(processed_at - record["sent_at"])
            continue
        turn_key = (
            distinct_turns[question_index]
            if question_index < len(distinct_turns)
            else None
        )
        question_index += 1
        first_answer_at = min(
            (
                item["arrived_at"]
                for item in text_events
                if turn_key is not None and item["turn_id"] == turn_key
            ),
            default=None,
        )
        questions.append(
            {
                "seq_no": record["seq_no"],
                "turn_id": turn_key,
                "sent_at": record["sent_at"],
                "processed_at": processed_at,
                "first_text_at": first_answer_at,
                "ttft_s": (
                    first_answer_at - record["sent_at"]
                    if first_answer_at is not None
                    else None
                ),
            }
        )

    raw_gaps: list[float] = []
    for turn_key in distinct_turns:
        arrivals = [
            event["arrived_at"]
            for event in text_events
            if event["turn_id"] == turn_key
        ]
        raw_gaps.extend(
            later - earlier for earlier, later in itertools.pairwise(arrivals)
        )

    lane = {
        "lane": lane_index,
        "ttft_s": (first_text_at - epoch) if first_text_at is not None else None,
        "questions": questions,
        "generated_tokens": tokens,
        "text_delta_events": len(text_events),
        "elapsed_s": (
            max((entry["arrived_at"] for entry in received), default=epoch) - epoch
        ),
        "decode_span_s": decode_span_s,
        "tokens_per_second": tokens / decode_span_s if decode_span_s > 0 else None,
        "token_gaps": stats(raw_gaps),
        "frame_delay": stats(frame_delays),
        "send_lag": stats(
            [record["sent_at"] - record["planned_at"] for record in sent]
        ),
        "errors": [
            f"missing processed event: seq {record['seq_no']}"
            for record in sent
            if record["seq_no"] not in processed_by_seq
        ],
    }
    return lane, raw_gaps


async def _run_lane(
    *,
    url: str,
    lane_index: int,
    stagger_s: float,
    events: list[dict],
    case: dict,
    frame_payloads: dict[int, tuple[str, bytes]],
    args: argparse.Namespace,
) -> tuple[dict, list[float]]:
    import websockets

    received: list[dict] = []
    sent: list[dict] = []

    async with websockets.connect(url, max_size=WS_MAX_PAYLOAD_BYTES) as websocket:
        queue: asyncio.Queue = asyncio.Queue()
        stop_marker = object()

        async def reader() -> None:
            async for raw in websocket:
                arrived = time.perf_counter()
                payload = json.loads(raw) if isinstance(raw, str) else raw
                await queue.put((payload, arrived))
            await queue.put((stop_marker, time.perf_counter()))

        reader_task = asyncio.create_task(reader())

        async def wait_for(type_name: str, timeout: float) -> tuple[dict, float]:
            deadline = time.perf_counter() + timeout
            while True:
                payload, arrived = await asyncio.wait_for(
                    queue.get(), max(0.0, deadline - time.perf_counter())
                )
                if payload is stop_marker:
                    raise ConnectionError("server closed the websocket")
                received.append({"payload": payload, "arrived_at": arrived})
                if payload.get("type") == "error":
                    raise RuntimeError(
                        f"server error: {payload.get('code')}: "
                        f"{payload.get('message')}"
                    )
                if payload.get("type") == type_name:
                    return payload, arrived

        epoch = 0.0
        try:
            await wait_for("session.created", args.wait_timeout)
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.configure",
                        "prompt": INITIAL_PROMPT_OVERRIDE,
                        "system_prompt": case["system_prompt"],
                        "max_new_tokens": args.max_new_tokens,
                        # The per-turn soft cap is the session pacing knob;
                        # 86400 matches the public test's unthrottled setting.
                        "max_tokens_per_turn": args.token_rate,
                        "include_usage": True,
                    }
                )
            )
            await wait_for("session.configured", args.wait_timeout)
            await wait_for("session.ready", args.wait_timeout)

            epoch = time.perf_counter() + stagger_s
            for lane_event in events:
                due = epoch + lane_event["offset"]
                await asyncio.sleep(max(0.0, due - time.perf_counter()))
                sent_at = time.perf_counter()
                seq_no = int(lane_event["seq_no"])
                sent.append({"seq_no": seq_no, "planned_at": due, "sent_at": sent_at})
                if lane_event["type"] == "prompt":
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "input.prompt",
                                "seq_no": seq_no,
                                "prompt": lane_event["prompt"],
                                "final": bool(lane_event["final"]),
                            }
                        )
                    )
                    await wait_for("input.prompt.accepted", args.wait_timeout)
                    continue
                if lane_event["type"] != "frame":
                    raise ValueError(
                        f"unsupported workload event type: {lane_event['type']!r}"
                    )
                mime_type, frame_bytes = frame_payloads[seq_no]
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input.frame",
                            "seq_no": seq_no,
                            "timestamp": float(lane_event["timestamp"]),
                            "final": bool(lane_event["final"]),
                            "mime_type": mime_type,
                        }
                    )
                )
                await wait_for("input.frame.ready", args.wait_timeout)
                await websocket.send(frame_bytes)
                await wait_for("input.frame.accepted", args.wait_timeout)

            done_deadline = epoch + events[-1]["offset"] + args.drain_seconds
            await wait_for(
                "session.done", max(0.0, done_deadline - time.perf_counter())
            )
        finally:
            # Release the replica slot even when the lane failed mid-stream.
            try:
                await websocket.send(json.dumps({"type": "session.abort"}))
            except Exception as exc:  # noqa: BLE001 - transport already down
                logger.debug("lane abort notification skipped: %s", exc)
            reader_task.cancel()
            await asyncio.gather(reader_task, return_exceptions=True)

    return _lane_metrics(
        lane_index=lane_index,
        epoch=epoch,
        events=events,
        sent=sent,
        received=received,
    )


async def _run_round(
    *,
    url: str,
    count: int,
    events: list[dict],
    case: dict,
    frame_payloads: dict[int, tuple[str, bytes]],
    args: argparse.Namespace,
) -> tuple[list[dict], list[list[float]]]:
    """One measured round: N lanes with the public-harness send rhythm."""
    outcomes = await asyncio.gather(
        *(
            _run_lane(
                url=url,
                lane_index=index,
                # Same per-lane stagger as concurrency_benchmark.measure().
                stagger_s=index * 0.05 / args.fps,
                events=events,
                case=case,
                frame_payloads=frame_payloads,
                args=args,
            )
            for index in range(count)
        )
    )
    lanes = [lane for lane, _ in outcomes]
    gaps = [lane_gaps for _, lane_gaps in outcomes]
    errors = [error for lane in lanes for error in lane["errors"]]
    if errors:
        raise RuntimeError(f"lane errors in this round: {errors}")
    return lanes, gaps


def _summarize(trials: list[dict], dp_size: int) -> list[dict]:
    """Per-session-count aggregates across rounds and lanes."""
    rows = []
    for count in sorted({trial["sessions"] for trial in trials}):
        selected = [trial for trial in trials if trial["sessions"] == count]
        lanes = [lane for trial in selected for lane in trial["lanes"]]
        gaps = [sample for trial in selected for sample in trial["lane_gaps"]]
        raw_gaps = [sample for gap_list in gaps for sample in gap_list]
        lane_tps = [lane["tokens_per_second"] for lane in lanes]
        lane_tps = [value for value in lane_tps if value]
        lane_ttft = [lane["ttft_s"] for lane in lanes if lane["ttft_s"] is not None]
        rows.append(
            {
                "label": f"dp{dp_size}",
                "sessions": count,
                "rounds": len(selected),
                "lane_tokens_per_second_mean": (
                    sum(lane_tps) / len(lane_tps) if lane_tps else None
                ),
                "lane_tokens_per_second_min": min(lane_tps) if lane_tps else None,
                "ttft_s_mean": (
                    sum(lane_ttft) / len(lane_ttft) if lane_ttft else None
                ),
                "token_gap_stats": stats(raw_gaps),
                "dp_size": dp_size,
            }
        )
    return rows


def _print_summary(rows: list[dict]) -> None:
    header = (
        f"{'label':>6} {'sessions':>8} {'rounds':>6} "
        f"{'tps_mean':>9} {'tps_min':>8} {'ttft_s':>7} {'gap_p95_ms':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label']:>6} {row['sessions']:>8} {row['rounds']:>6} "
            f"{_fmt(row['lane_tokens_per_second_mean']):>9} "
            f"{_fmt(row['lane_tokens_per_second_min']):>8} "
            f"{_fmt(row['ttft_s_mean']):>7} "
            f"{_fmt(row['token_gap_stats'].get('p95'), scale=1000):>11}"
        )


def _fmt(value, *, scale: float = 1.0) -> str:
    return "-" if value is None else f"{value * scale:.3f}"


def run(args: argparse.Namespace) -> int:
    model = validate_model(args.model_path)
    gpus = resolve_server_gpus(args)
    plan = resolve_plan(args, model)
    case = _load_case(args.cases_dir)
    events = workload(case, args.fps)
    frame_payloads = _load_frame_payloads(events)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"ws_benchmark_dp{args.dp_size}.server.log"
    env = common.environment(",".join(gpu["uuid"] for gpu in gpus))
    child = Child(plan["command"], env, log_path)
    try:
        wait_for_listen(
            host=args.host,
            port=plan["port"],
            child=child,
            log_path=log_path,
            timeout_s=args.startup_timeout,
        )
        url = f"ws://{args.host}:{plan['port']}/v1/video/realtime"
        trials = []
        for count in sorted(set(args.sessions)):
            for repeat in range(args.repeats * args.sessions.count(count)):
                lanes, lane_gaps = asyncio.run(
                    _run_round(
                        url=url,
                        count=count,
                        events=events,
                        case=case,
                        frame_payloads=frame_payloads,
                        args=args,
                    )
                )
                trials.append(
                    {
                        "sessions": count,
                        "repeat": repeat,
                        "lanes": lanes,
                        "lane_gaps": lane_gaps,
                    }
                )
                tps = [
                    lane["tokens_per_second"]
                    for lane in lanes
                    if lane["tokens_per_second"]
                ]
                print(
                    f"[dp{args.dp_size}] sessions={count} round={repeat}: "
                    f"lane_tps_mean={_fmt(sum(tps) / len(tps) if tps else None)}",
                    flush=True,
                )
    finally:
        child.stop()

    summary = _summarize(trials, args.dp_size)
    result = {
        "schema": PROTOCOL,
        "model_path": plan["model_path"],
        "command": plan["command"],
        "plan": {k: v for k, v in plan.items() if k not in ("command", "model_path")},
        "trials": trials,
        "summary": summary,
    }
    out_path = output_dir / f"ws_benchmark_dp{args.dp_size}.json"
    write_json(out_path, result)
    _print_summary(summary)
    print(f"Wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        plan = resolve_plan(args, Path(args.model_path))
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    try:
        return run(args)
    except Exception:  # noqa: BLE001 - CLI wrapper reports and exits non-zero
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
