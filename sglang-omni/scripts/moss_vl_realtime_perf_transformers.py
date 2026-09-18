#!/usr/bin/env python3
"""Transformers same-caliber perf baseline for MOSS-VL realtime (P10.2).

Measures, with one loaded Transformers 5.12.1 model and manifest pixels:
  - initial text prefill latency (independent repetitions),
  - single-frame incremental extend latency (vision encode + prefill),
  - fixed-count decode TPOT / tokens/s.

TTFT is intentionally not measured: the realtime model emits runs of
<|silence|> before answering, and the silence-run length (a model-semantics
property that diverges between backends) dominates any TTFT number.

The SGLang counterpart is scripts/moss_vl_realtime_perf_sglang.py. Both sides
use the same model checkpoint, the same manifest case (identical PNG pixels,
prompts and timestamps), greedy decoding, warmup, and the same decode token
count, so the numbers are directly comparable.

Event flow mirrors production: each extend coalesces the one pending sampled
token left by the previous step, and sampling after an extend leaves a new
pending token for the next extend.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from sglang_omni.models.moss_vl_realtime.model_step import MossVLRealtimeStepper

DECODE_PROBE_PROMPT = (
    "Describe everything that happened in the video so far, in detail."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--prefill-reps", type=int, default=5)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--measure-frames", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=64)
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


def sync() -> None:
    torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.prefill_reps < 1:
        raise ValueError("--prefill-reps must be positive")
    if args.measure_frames < 1:
        raise ValueError("--measure-frames must be positive")
    if args.decode_tokens < 2:
        raise ValueError("--decode-tokens must be >= 2")

    case = load_case(args.manifest, args.case_id)
    frame_events = [event for event in case["events"] if event.get("type") == "frame"]
    if len(frame_events) < args.warmup_frames + args.measure_frames:
        raise ValueError(
            f"case has {len(frame_events)} frame events, need "
            f"{args.warmup_frames + args.measure_frames}"
        )

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
    stepper = MossVLRealtimeStepper(model, processor)

    encoded = processor.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": case["system_prompt"]},
            {"role": "user", "content": case["initial_prompt"]},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    prompt_tokens = int(input_ids.shape[1])

    frames: dict[int, tuple[Any, float]] = {}
    for event in frame_events:
        with Image.open(event["frame_path"]) as image:
            frames[int(event["seq_no"])] = (
                image.convert("RGB"),
                float(event["timestamp"]),
            )

    def extend_event(event: dict[str, Any]) -> float:
        frame, timestamp = frames[int(event["seq_no"])]
        sync()
        started = time.perf_counter()
        stepper.apply_event_and_extend(
            state,
            prompt=event.get("prompt"),
            frames=[(frame, timestamp)],
        )
        sync()
        return time.perf_counter() - started

    def sample_pending() -> float:
        sync()
        started = time.perf_counter()
        stepper.sample_next_token(state)
        sync()
        return time.perf_counter() - started

    # Metric 1: initial text prefill latency over independent repetitions.
    prefill_seconds: list[float] = []
    state = None
    for _ in range(args.prefill_reps):
        sync()
        started = time.perf_counter()
        state = stepper.initial_prefill(input_ids.clone(), attention_mask.clone())
        sync()
        prefill_seconds.append(time.perf_counter() - started)
    assert state is not None

    # Warmup frames (untimed): extend coalesces the pending token; sampling
    # leaves a fresh pending token for the next extend, as in production.
    for event in frame_events[: args.warmup_frames]:
        extend_event(event)
        sample_pending()

    # Metric 2: per-frame incremental extend latency (the pending token stays
    # pending for the next extend, as in production). TTFT is intentionally
    # not measured: the model emits runs of <|silence|> before answering, and
    # the silence-run length dominates any TTFT number on both backends.
    extend_seconds: list[float] = []
    vision_tokens_per_frame: list[int] = []
    for event in frame_events[
        args.warmup_frames : args.warmup_frames + args.measure_frames
    ]:
        visible_before = state.visible_vision_length
        extend_seconds.append(extend_event(event))
        sample_pending()
        vision_tokens_per_frame.append(state.visible_vision_length - visible_before)

    # Metric 3: fixed-count decode after a final prompt-only event.
    sync()
    started = time.perf_counter()
    stepper.apply_event_and_extend(state, prompt=DECODE_PROBE_PROMPT, frames=[])
    sync()
    prompt_extend_seconds = time.perf_counter() - started

    token_seconds: list[float] = []
    for _ in range(args.decode_tokens):
        sync()
        started = time.perf_counter()
        stepper.decode_one_step(state)
        sync()
        token_seconds.append(time.perf_counter() - started)
    tpot = summarize(token_seconds)

    result = {
        "backend": "transformers",
        "model_path": args.model_path,
        "case_id": args.case_id,
        "attn_implementation": args.attn_implementation,
        "config": {
            "prefill_reps": args.prefill_reps,
            "warmup_frames": args.warmup_frames,
            "measure_frames": args.measure_frames,
            "decode_tokens": args.decode_tokens,
        },
        "prompt_tokens": prompt_tokens,
        "vision_tokens_per_frame": vision_tokens_per_frame,
        "initial_prefill_seconds": {
            "samples": prefill_seconds,
            **summarize(prefill_seconds),
        },
        "frame_extend_seconds": {
            "samples": extend_seconds,
            **summarize(extend_seconds),
        },
        "final_prompt_extend_seconds": prompt_extend_seconds,
        "decode_tpot_seconds": {"samples": token_seconds, **tpot},
        "decode_tokens_per_second": (1.0 / tpot["mean"] if tpot.get("mean") else None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, indent=2, ensure_ascii=False))


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
