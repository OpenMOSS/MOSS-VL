#!/usr/bin/env python3
"""Replay benchmark manifest cases with one loaded Transformers model."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--frame-interval", type=float)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--final-wait", type=float, default=2.0)
    return parser.parse_args()


def load_cases(path: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    cases = []
    for case_id in case_ids:
        matches = [case for case in records if case.get("case_id") == case_id]
        if len(matches) != 1:
            raise ValueError(f"case_id {case_id!r} was not found exactly once")
        cases.append(matches[0])
    return cases


def resolve_frame_interval(args: argparse.Namespace, case: dict[str, Any]) -> float:
    if args.fps is not None and args.frame_interval is not None:
        raise ValueError("--fps and --frame-interval cannot be used together")
    if args.fps is not None:
        if args.fps <= 0:
            raise ValueError("--fps must be positive")
        return 1.0 / args.fps
    if args.frame_interval is not None:
        return float(args.frame_interval)
    if case.get("fps") is not None:
        fps = float(case["fps"])
        if fps <= 0:
            raise ValueError("manifest fps must be positive")
        return 1.0 / fps
    if case.get("frame_interval_seconds") is not None:
        return float(case["frame_interval_seconds"])
    return 1.0


def drain_outputs(
    session: Any, outputs: list[dict[str, Any]], started_at: float
) -> None:
    while True:
        chunk = session.poll_output(timeout=0.0)
        if chunk is None:
            return
        outputs.append(
            {
                "elapsed_seconds": time.monotonic() - started_at,
                "text": chunk,
            }
        )
        print(chunk, end="", flush=True)


def wait_and_drain(
    session: Any,
    outputs: list[dict[str, Any]],
    started_at: float,
    duration: float,
) -> None:
    deadline = time.monotonic() + duration
    while True:
        drain_outputs(session, outputs, started_at)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.05, remaining))


def run_case(
    model: Any, processor: Any, case: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    frame_interval = resolve_frame_interval(args, case)
    if frame_interval < 0:
        raise ValueError("--frame-interval must be non-negative")

    frames_by_seq: dict[int, Image.Image] = {}
    for event in case["events"]:
        if event["type"] != "frame":
            continue
        with Image.open(event["frame_path"]) as image:
            frames_by_seq[int(event["seq_no"])] = image.convert("RGB")

    session = model.create_realtime_session(
        processor,
        initial_prompt=case["initial_prompt"],
        system_prompt=case.get("system_prompt"),
        do_sample=False,
        use_cache=True,
    )
    outputs: list[dict[str, Any]] = []
    input_events: list[dict[str, Any]] = []
    started_at = time.monotonic()
    session.start()
    try:
        next_event_deadline = time.monotonic()
        for event in case["events"]:
            wait_and_drain(
                session,
                outputs,
                started_at,
                max(0.0, next_event_deadline - time.monotonic()),
            )
            if event["type"] == "prompt":
                session.push_prompt(event["prompt"])
                input_events.append(
                    {
                        "seq_no": event["seq_no"],
                        "type": "prompt",
                        "dropped": False,
                        "elapsed_seconds": time.monotonic() - started_at,
                    }
                )
            elif event["type"] == "frame":
                frame = frames_by_seq[int(event["seq_no"])]
                prompt = event.get("prompt")
                if prompt is None:
                    dropped = session.push_frame(
                        frame, timestamp=float(event["timestamp"])
                    )
                else:
                    dropped = session.push_prompt_frame(
                        prompt,
                        frame,
                        timestamp=float(event["timestamp"]),
                    )
                input_events.append(
                    {
                        "seq_no": event["seq_no"],
                        "type": "frame",
                        "dropped": dropped,
                        "elapsed_seconds": time.monotonic() - started_at,
                    }
                )
                if dropped:
                    raise RuntimeError(f"frame event {event['seq_no']} was dropped")
                next_event_deadline += frame_interval
            else:
                raise ValueError(f"unsupported event type: {event['type']!r}")
        wait_and_drain(session, outputs, started_at, args.final_wait)
    finally:
        session.close(timeout=30.0)
    drain_outputs(session, outputs, started_at)

    raw_text = "".join(item["text"] for item in outputs)
    normalized_text = raw_text
    for token in sorted(processor.tokenizer.all_special_tokens, key=len, reverse=True):
        normalized_text = normalized_text.replace(token, "")
    result = {
        "backend": "transformers",
        "case_id": case["case_id"],
        "frame_interval_seconds": frame_interval,
        "fps": 1.0 / frame_interval if frame_interval else None,
        "elapsed_seconds": time.monotonic() - started_at,
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "input_events": input_events,
        "outputs": outputs,
    }
    print()
    return result


def main() -> None:
    args = parse_args()
    if args.final_wait < 0:
        raise ValueError("--final-wait must be non-negative")
    cases = load_cases(args.manifest, args.case_id)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation=args.attn_implementation,
    ).eval()
    results = [run_case(model, processor, case, args) for case in cases]
    payload: dict[str, Any]
    if len(results) == 1:
        payload = results[0]
    else:
        payload = {
            "backend": "transformers",
            "model_path": args.model_path,
            "results": results,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        output_path = parse_args().output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "backend": "transformers",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise
