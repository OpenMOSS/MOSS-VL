#!/usr/bin/env python3
"""Fingerprint one manifest event after MOSS-VL preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor


def tensor_hash(value) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--seq-no", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case = next(
        record
        for line in args.manifest.read_text().splitlines()
        if line
        for record in [json.loads(line)]
        if record.get("case_id") == args.case_id
    )
    event = next(item for item in case["events"] if item["seq_no"] == args.seq_no)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompt_text = ""
    if event.get("prompt") is not None:
        prompt_text = (
            "<|im_end|>\n<|im_start|>user\n"
            f"{event['prompt']}"
            "<|im_end|>\n<|im_start|>assistant\n"
            "<|silence|>"
        )
    timestamp = float(event["timestamp"])
    text = (
        prompt_text
        + "<|vision_start|><|time_start|>"
        + f"{timestamp:.1f} seconds"
        + "<|time_end|><|image|><|vision_end|>"
    )
    with Image.open(event["frame_path"]) as image:
        frame = image.convert("RGB")
    processed = processor(
        text=text,
        images=[frame],
        add_special_tokens=False,
        return_tensors="pt",
    )
    result = {
        "input_ids": processed["input_ids"].flatten().tolist(),
        "input_ids_hash": tensor_hash(processed["input_ids"]),
        "pixel_values_shape": list(processed["pixel_values"].shape),
        "pixel_values_hash": tensor_hash(processed["pixel_values"]),
        "grid_thw": processed["grid_thw"].tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
