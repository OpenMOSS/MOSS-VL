#!/usr/bin/env python3
"""Trace next-token logits for a training-aligned packed manifest prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--through-seq-no", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="eager")
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


def tensor_summary(value: torch.Tensor) -> dict:
    tensor = value.detach().float().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "max_abs": float(tensor.abs().max()),
    }


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def frame_text(timestamp: float) -> str:
    return (
        "<|vision_start|><|time_start|>"
        f"{timestamp:.1f} seconds"
        "<|time_end|><|image|><|vision_end|>"
    )


def prompt_text(prompt: str, *, followed_by_frame: bool) -> str:
    text = (
        "<|im_end|>\n<|im_start|>user\n"
        f"{prompt}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    return text + ("<|silence|>" if followed_by_frame else "")


def main() -> None:
    args = parse_args()
    case = load_case(args.manifest, args.case_id)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    prefix = processor.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": case["system_prompt"]},
            {"role": "user", "content": case["initial_prompt"]},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    prefix += case["initial_expected"]
    frames = []
    for event in case["events"]:
        if event["seq_no"] > args.through_seq_no:
            break
        is_frame = event["type"] == "frame"
        if event.get("prompt") is not None:
            prefix += prompt_text(event["prompt"], followed_by_frame=is_frame)
        if is_frame:
            prefix += frame_text(float(event["timestamp"]))
            with Image.open(event["frame_path"]) as image:
                frames.append(image.convert("RGB"))
        if event["seq_no"] < args.through_seq_no:
            prefix += event["expected_after"]

    if frames:
        processed = processor(
            text=prefix,
            images=frames,
            add_special_tokens=False,
            return_tensors="pt",
        )
    else:
        processed = processor.tokenizer(
            prefix,
            add_special_tokens=False,
            return_tensors="pt",
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation=args.attn_implementation,
    ).eval()
    traces = {}
    hooks = []

    def capture(name):
        def hook(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is not None:
                traces[name] = tensor_summary(tensor)

        return hook

    def capture_rope_inputs(_module, inputs):
        if len(inputs) >= 2 and isinstance(inputs[1], torch.Tensor):
            traces["rope_position_ids"] = tensor_summary(inputs[1])

    language_model = model.model.language_model
    hooks.append(
        language_model.embed_tokens.register_forward_hook(capture("embedding"))
    )
    hooks.append(
        language_model.rotary_emb.register_forward_pre_hook(capture_rope_inputs)
    )
    hooks.append(language_model.rotary_emb.register_forward_hook(capture("rope_cos")))
    layer_indices = sorted({0, 1, len(language_model.layers) - 1})
    for layer_index in layer_indices:
        hooks.append(
            language_model.layers[layer_index].register_forward_hook(
                capture(f"text_layer_{layer_index}")
            )
        )
    hooks.append(language_model.norm.register_forward_hook(capture("final_norm")))
    model_inputs = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value
        for key, value in processed.items()
    }
    with torch.inference_mode():
        outputs = model(
            **model_inputs,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    for hook in hooks:
        hook.remove()
    scores = outputs.logits[0, -1].float()
    values, indices = torch.topk(scores, k=10)
    top_tokens = [
        {
            "token_id": int(token_id),
            "token": processor.tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "logit": float(logit),
        }
        for logit, token_id in zip(values, indices, strict=True)
    ]
    result = {
        "case_id": args.case_id,
        "through_seq_no": args.through_seq_no,
        "sequence_length": int(processed["input_ids"].shape[1]),
        "grid_thw": (processed["grid_thw"].tolist() if "grid_thw" in processed else []),
        "top_tokens": top_tokens,
        "traces": traces,
        "parameter_traces": {
            "embed_tokens": tensor_summary(language_model.embed_tokens.weight),
            "rope_inv_freq": tensor_summary(language_model.rotary_emb.inv_freq),
            "layer_0_q_proj": tensor_summary(
                language_model.layers[0].self_attn.q_proj.weight
            ),
            "final_norm": tensor_summary(language_model.norm.weight),
            "lm_head": tensor_summary(model.lm_head.weight),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
