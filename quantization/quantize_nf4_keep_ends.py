#!/usr/bin/env python3
"""BitsAndBytes NF4 weight-only quantization for MOSS-VL, keeping the first/last
language layers, the vision tower, cross-attention and lm_head in BF16.

Usage:
    python quantize_nf4_keep_ends.py \
        --source /path/to/MOSS-VL-checkpoint \
        --output /path/to/MOSS-VL-NF4-KeepEnds4 \
        --keep-end-layers 4

The language-layer count is read from --source/config.json
(text_config.num_hidden_layers); override with --num-layers when the config
does not carry it.

Requires a CUDA GPU and: torch, transformers, bitsandbytes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig


BASE_SKIP_MODULES = ["model.visual", "cross_attn", "lm_head"]
# Files that save_pretrained regenerates; everything else is copied verbatim.
GENERATED_FILES = {"config.json", "model.safetensors.index.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-end-layers", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-reload", action="store_true")
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def resolve_num_layers(source: Path, num_layers_arg: int | None) -> int:
    if num_layers_arg is not None:
        return num_layers_arg
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    num_layers = config.get("text_config", {}).get("num_hidden_layers")
    if num_layers is None:
        raise RuntimeError(
            "Cannot derive text_config.num_hidden_layers from "
            f"{config_path}; pass --num-layers explicitly."
        )
    return num_layers


def kept_layer_indices(keep_end_layers: int, num_layers: int) -> list[int]:
    if keep_end_layers < 1 or keep_end_layers * 2 >= num_layers:
        raise ValueError(
            f"--keep-end-layers must be in [1, {num_layers // 2 - 1}] "
            f"for {num_layers} language layers"
        )
    return list(range(keep_end_layers)) + list(
        range(num_layers - keep_end_layers, num_layers)
    )


def build_skip_modules(kept: list[int]) -> list[str]:
    skip_modules = list(BASE_SKIP_MODULES)
    for layer_idx in kept:
        skip_modules.append(f"model.language_model.layers.{layer_idx}")
    # SGLang-style naming (extra "model." level) for engine-side compatibility;
    # harmless under Transformers name resolution.
    for layer_idx in kept:
        skip_modules.append(f"model.language_model.model.layers.{layer_idx}")
    return skip_modules


def quantization_config(skip_modules: list[str]) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules,
    )


def cross_layer_indices(source: Path) -> list[int]:
    """Cross-attention layers hold no self_attn; their Linears live under
    `.cross_attn.*` (skipped) and `.mlp.*` (quantized), i.e. 3 per layer."""
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return [
        int(layer)
        for layer in config.get("text_config", {}).get("cross_attention_layers", [])
    ]


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def copy_support_files(source: Path, output: Path) -> None:
    for src in source.iterdir():
        if not src.is_file():
            continue
        if src.name in GENERATED_FILES:
            continue
        if src.name.startswith("model-") and src.suffix == ".safetensors":
            continue
        dst = output / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    src_assets = source / "assets"
    dst_assets = output / "assets"
    if src_assets.is_dir() and not dst_assets.exists():
        shutil.copytree(src_assets, dst_assets)


def quantized_modules(model: torch.nn.Module) -> tuple[str, list[str]]:
    import bitsandbytes as bnb

    four_bit = [
        name
        for name, module in model.named_modules()
        if isinstance(module, bnb.nn.Linear4bit)
    ]
    eight_bit = [
        name
        for name, module in model.named_modules()
        if isinstance(module, bnb.nn.Linear8bitLt)
    ]
    if four_bit and eight_bit:
        raise RuntimeError("Unexpected mixed bnb 4-bit and 8-bit modules")
    if four_bit:
        return "4bit", four_bit
    if eight_bit:
        return "8bit", eight_bit
    raise RuntimeError("No bitsandbytes quantized Linear modules found")


def validate_scope(
    names: list[str], kept: list[int], num_layers: int, cross_layers: list[int]
) -> None:
    forbidden = [
        name
        for name in names
        if name.startswith("model.visual")
        or ".cross_attn." in name
        or name == "lm_head"
    ]
    if forbidden:
        raise RuntimeError(f"Sensitive modules were quantized: {forbidden[:10]}")
    bad = [
        name
        for name in names
        if any(
            name.startswith(f"model.language_model.layers.{idx}.") for idx in kept
        )
    ]
    mid = [i for i in range(num_layers) if i not in kept]
    mid_self = [i for i in mid if i not in cross_layers]
    mid_cross = [i for i in mid if i in cross_layers]
    expected = 7 * len(mid_self) + 3 * len(mid_cross)
    if bad or len(names) != expected:
        raise RuntimeError(
            f"Keep-ends scope mismatch: count={len(names)} "
            f"expected={expected} (7 x {len(mid_self)} self-attention layers "
            f"+ 3 x {len(mid_cross)} cross-attention layers) "
            f"bad={bad[:10]}"
        )


def write_report(output: Path, report: dict) -> None:
    (output / "quantization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "QUANTIZATION.md").write_text(
        "# MOSS-VL quantized checkpoint\n\n"
        f"- Variant: `{report['variant']}`\n"
        f"- Method: `{report['method']}`\n"
        "- Compute dtype: `bfloat16`\n"
        f"- Quantized Linear modules: `{report['quantized_linear_count']}`\n"
        "- Vision tower, cross-attention, embeddings, lm_head and norms remain BF16.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if not source.is_dir():
        raise FileNotFoundError(source)

    num_layers = resolve_num_layers(source, args.num_layers)
    cross_layers = cross_layer_indices(source)
    kept = kept_layer_indices(args.keep_end_layers, num_layers)
    skip_modules = build_skip_modules(kept)
    method = f"bitsandbytes NF4 W4A16 keep-first-last-{args.keep_end_layers}"

    prepare_output(output, args.overwrite)
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        source,
        trust_remote_code=True,
        device_map={"": 0},
        dtype=torch.bfloat16,
        quantization_config=quantization_config(skip_modules),
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    bit_kind, names = quantized_modules(model)
    validate_scope(names, kept, num_layers, cross_layers)

    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
    model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(output)
    copy_support_files(source, output)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "output": str(output),
        "variant": "w4_keep_ends",
        "method": method,
        "compute_dtype": "bfloat16",
        "num_language_layers": num_layers,
        "keep_end_layers": args.keep_end_layers,
        "kept_layers_bf16": kept,
        "skip_modules": skip_modules,
        "quantized_kind": bit_kind,
        "quantized_linear_count": len(names),
        "quantized_module_samples": names[:20],
        "peak_gpu_memory_gib_during_conversion": round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        ),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "bitsandbytes": package_version("bitsandbytes"),
        },
        "reload_verified": False,
    }
    write_report(output, report)

    if args.verify_reload:
        del processor
        del model
        torch.cuda.empty_cache()
        reloaded = AutoModelForCausalLM.from_pretrained(
            output,
            trust_remote_code=True,
            device_map={"": 0},
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=args.attn_implementation,
        )
        reload_kind, reload_names = quantized_modules(reloaded)
        validate_scope(reload_names, kept, num_layers, cross_layers)
        report["reload_verified"] = True
        report["reload_quantized_kind"] = reload_kind
        report["reload_quantized_linear_count"] = len(reload_names)
        report["peak_gpu_memory_gib_after_reload"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        )
        write_report(output, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
