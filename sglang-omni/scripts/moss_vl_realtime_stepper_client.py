#!/usr/bin/env python3
"""Replay a manifest through explicit Transformers model steps."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from sglang_omni.models.moss_vl_realtime.model_step import MossVLRealtimeStepper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-tokens-per-event", type=int, default=256)
    return parser.parse_args()


def load_case(path: Path, case_id: str) -> dict:
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
    case = load_case(args.manifest, args.case_id)
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
    state = stepper.initial_prefill(input_ids, torch.ones_like(input_ids))
    silence_token_id = processor.tokenizer.convert_tokens_to_ids("<|silence|>")
    generated: list[int] = []

    def drive_until_silence() -> None:
        for _ in range(args.max_tokens_per_event):
            token = stepper.sample_next_token(state)
            token_id = int(token.item())
            generated.append(token_id)
            if token_id == silence_token_id:
                return
            stepper.commit_pending_tokens(state)
        raise RuntimeError(
            "model did not emit silence within the per-event token limit"
        )

    drive_until_silence()
    for event in case["events"]:
        frames = []
        if event["type"] == "frame":
            with Image.open(event["frame_path"]) as image:
                frame = image.convert("RGB")
            frames.append((frame, float(event["timestamp"])))
        stepper.apply_event_and_extend(
            state,
            prompt=event.get("prompt"),
            frames=frames,
        )
        drive_until_silence()

    raw_text = processor.tokenizer.decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    normalized_text = processor.tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    result = {
        "backend": "transformers-stepper",
        "case_id": args.case_id,
        "token_ids": generated,
        "raw_text": raw_text,
        "normalized_text": normalized_text,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(repr(normalized_text), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        output_path = parse_args().output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "backend": "transformers-stepper",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise
