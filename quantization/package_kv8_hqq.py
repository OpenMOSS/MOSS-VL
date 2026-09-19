#!/usr/bin/env python3
"""Package a MOSS-VL checkpoint with Transformers HQQ INT8 KV cache (KV8).

Copies a quantized (or BF16) checkpoint via hard links, patches
modeling_moss_vl.py so cross-attention reads also see quantized cache
history, and writes an HQQ INT8 KV-cache cache_config into
generation_config.json.

Usage:
    python package_kv8_hqq.py \
        --source /path/to/MOSS-VL-FP8-Dynamic \
        --output /path/to/MOSS-VL-FP8-Dynamic-KV8-HQQ \
        --weight-quantization fp8-dynamic
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


CROSS_ATTN_ORIGINAL = """        elif cache_position[0] != 0:
            key_states, value_states = (
                past_key_values.layers[self.layer_idx].keys,
                past_key_values.layers[self.layer_idx].values,
            ) 
"""

CROSS_ATTN_PATCHED = """        elif cache_position[0] != 0:
            # QuantizedCache stores older entries outside the BF16 residual tail.
            cache_layer = past_key_values.layers[self.layer_idx]
            if hasattr(cache_layer, "_quantized_keys"):
                quantized_keys = cache_layer._dequantize(cache_layer._quantized_keys)
                quantized_values = cache_layer._dequantize(cache_layer._quantized_values)
                if cache_layer.keys.numel():
                    key_states = torch.cat((quantized_keys, cache_layer.keys), dim=-2)
                    value_states = torch.cat((quantized_values, cache_layer.values), dim=-2)
                else:
                    key_states, value_states = quantized_keys, quantized_values
            else:
                key_states, value_states = cache_layer.keys, cache_layer.values
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weight-quantization",
        default="unknown",
        help="Label recorded in the report, e.g. fp8-dynamic or nf4-keep-ends.",
    )
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--residual-length", type=int, default=128)
    return parser.parse_args()


def hardlink_or_copy(source: str, destination: str) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_checkpoint(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(source, output, copy_function=hardlink_or_copy)


def detach_hardlink(path: Path) -> None:
    data = path.read_bytes()
    path.unlink()
    path.write_bytes(data)


def patch_modeling(output: Path) -> None:
    path = output / "modeling_moss_vl.py"
    detach_hardlink(path)
    text = path.read_text(encoding="utf-8")
    count = text.count(CROSS_ATTN_ORIGINAL)
    if count != 1:
        raise RuntimeError(
            "Failed to patch cross-attention cache reading in "
            f"{path}: expected exactly one matching block, found {count}. "
            "The modeling file may have changed - update CROSS_ATTN_ORIGINAL."
        )
    path.write_text(
        text.replace(CROSS_ATTN_ORIGINAL, CROSS_ATTN_PATCHED),
        encoding="utf-8",
    )


def patch_generation_config(
    output: Path,
    q_group_size: int,
    residual_length: int,
) -> None:
    path = output / "generation_config.json"
    detach_hardlink(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["cache_implementation"] = "quantized"
    config["cache_config"] = {
        "backend": "hqq",
        "nbits": 8,
        "axis_key": 0,
        "axis_value": 0,
        "q_group_size": q_group_size,
        "residual_length": residual_length,
    }
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_report(
    source: Path,
    output: Path,
    weight_quantization: str,
    q_group_size: int,
    residual_length: int,
) -> None:
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_weight_checkpoint": str(source),
        "weight_quantization": weight_quantization,
        "transformers_kv_cache": {
            "backend": "hqq",
            "nbits": 8,
            "axis_key": 0,
            "axis_value": 0,
            "q_group_size": q_group_size,
            "residual_length": residual_length,
        },
        "sglang_kv_cache": {
            "runtime_argument": "--kv-cache-dtype fp8_e4m3",
            "note": "SGLang FP8 E4M3 and Transformers HQQ INT8 are separate native KV8 implementations.",
        },
        "storage": "Model shards are hard-linked to the source checkpoint when supported.",
        "compatibility_patch": (
            "Transformers cross-attention materializes quantized cache history "
            "one layer at a time."
        ),
    }
    (output / "mossvl_kv8_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)

    copy_checkpoint(source, output)
    patch_modeling(output)
    patch_generation_config(
        output,
        q_group_size=args.q_group_size,
        residual_length=args.residual_length,
    )
    write_report(
        source,
        output,
        weight_quantization=args.weight_quantization,
        q_group_size=args.q_group_size,
        residual_length=args.residual_length,
    )
    print(output)


if __name__ == "__main__":
    main()
