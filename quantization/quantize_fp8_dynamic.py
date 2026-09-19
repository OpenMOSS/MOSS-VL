#!/usr/bin/env python3
"""One-shot FP8-Dynamic quantization for MOSS-VL language-model self-attention layers.

Quantizes q/k/v/o and gate/up/down projections of every language layer that
is NOT a cross-attention layer, using llmcompressor + compressed-tensors.
Vision tower, cross-attention layers, embeddings and lm_head stay BF16.

Usage:
    python quantize_fp8_dynamic.py \
        --source /path/to/MOSS-VL-checkpoint \
        --output /path/to/MOSS-VL-FP8-Dynamic

The language-layer count and cross-attention layer list are read from
--source/config.json (text_config.num_hidden_layers /
text_config.cross_attention_layers). Override with --num-layers and
--cross-layers when the config does not carry them.

Requires a CUDA GPU and: torch, transformers, llmcompressor.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument(
        "--cross-layers",
        default=None,
        help="Comma-separated cross-attention layer indices, e.g. 2,6,10",
    )
    return parser.parse_args()


def resolve_layers(
    source: Path, num_layers_arg: int | None, cross_layers_arg: str | None
) -> tuple[int, list[int]]:
    if num_layers_arg is not None and cross_layers_arg is not None:
        num_layers = num_layers_arg
        cross_layers = [int(item) for item in cross_layers_arg.split(",") if item.strip()]
    else:
        config_path = source / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config file: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text_config = config.get("text_config", {})
        if num_layers_arg is not None:
            num_layers = num_layers_arg
        else:
            num_layers = text_config.get("num_hidden_layers")
            if num_layers is None:
                raise RuntimeError(
                    "Cannot derive text_config.num_hidden_layers from "
                    f"{config_path}; pass --num-layers explicitly."
                )
        if cross_layers_arg is not None:
            cross_layers = [
                int(item) for item in cross_layers_arg.split(",") if item.strip()
            ]
        else:
            cross_layers = text_config.get("cross_attention_layers")
            if cross_layers is None:
                raise RuntimeError(
                    "Cannot derive text_config.cross_attention_layers from "
                    f"{config_path}; pass --cross-layers explicitly."
                )
            cross_layers = [int(layer) for layer in cross_layers]
    cross_layers = sorted(set(cross_layers))
    bad = [layer for layer in cross_layers if layer < 0 or layer >= num_layers]
    if bad:
        raise ValueError(
            f"cross-attention layers out of range [0, {num_layers}): {bad}"
        )
    return num_layers, cross_layers


def copy_support_files(source_dir: Path, output_dir: Path) -> None:
    for source in source_dir.iterdir():
        if not source.is_file():
            continue
        if source.name == "config.json":
            continue
        if source.name == "model.safetensors.index.json":
            continue
        if source.name.startswith("model-") and source.suffix == ".safetensors":
            continue
        destination = output_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    src_assets = source_dir / "assets"
    dst_assets = output_dir / "assets"
    if src_assets.is_dir() and not dst_assets.exists():
        shutil.copytree(src_assets, dst_assets)


def add_sglang_quantization_metadata(
    output_dir: Path,
    sglang_target_regex: str,
    sglang_ignore: list[str],
) -> None:
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant_config = config["quantization_config"]
    quant_config["packed_modules_mapping"] = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    targets = quant_config["config_groups"]["group_0"]["targets"]
    if sglang_target_regex not in targets:
        targets.append(sglang_target_regex)
    for ignored_layer in sglang_ignore:
        if ignored_layer not in quant_config["ignore"]:
            quant_config["ignore"].append(ignored_layer)
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    num_layers, cross_layers = resolve_layers(
        source, args.num_layers, args.cross_layers
    )
    self_layers = tuple(
        index for index in range(num_layers) if index not in cross_layers
    )
    if not self_layers:
        raise RuntimeError("No self-attention language layers left to quantize")
    self_layer_group = "|".join(str(index) for index in self_layers)
    hf_target_regex = (
        r"re:^model\.language_model\.layers\."
        rf"({self_layer_group})\."
        r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(gate_proj|up_proj|down_proj))$"
    )
    sglang_target_regex = (
        r"re:^model\.language_model\.model\.layers\."
        rf"({self_layer_group})\."
        r"(self_attn\.(q_proj|k_proj|v_proj|qkv_proj|o_proj)|"
        r"mlp\.(gate_proj|up_proj|gate_up_proj|down_proj))$"
    )
    sglang_ignore = [
        r"re:^model\.visual\.",
        (
            r"re:^model\.language_model\.model\.layers\."
            rf"({'|'.join(str(index) for index in cross_layers)})\."
        ),
        "model.language_model.lm_head",
    ]

    started_at = time.time()
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    target_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("model.language_model.layers.")
        and int(name.split(".")[3]) in self_layers
        and (".self_attn." in name or ".mlp." in name)
    ]
    expected_targets = 7 * len(self_layers)
    if len(target_names) != expected_targets:
        raise RuntimeError(
            f"Expected {expected_targets} FP8 targets "
            f"(7 x {len(self_layers)} self-attention layers), "
            f"found {len(target_names)}"
        )

    quantized_model = oneshot(
        model=model,
        recipe=QuantizationModifier(
            targets=[hf_target_regex],
            scheme="FP8_DYNAMIC",
        ),
        output_dir=None,
        trust_remote_code_model=True,
        precision="bfloat16",
        save_compressed=True,
    )
    quantized_model.save_pretrained(
        output,
        save_compressed=True,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(output)
    copy_support_files(source, output)
    add_sglang_quantization_metadata(output, sglang_target_regex, sglang_ignore)

    report = {
        "source_model": str(source),
        "output_dir": str(output),
        "format": "compressed-tensors",
        "scheme": "FP8_DYNAMIC",
        "weights": "FP8 channel-wise static",
        "input_activations": "FP8 per-token dynamic",
        "num_language_layers": num_layers,
        "quantized_linear_count": len(target_names),
        "quantized_layers": list(self_layers),
        "excluded_cross_attention_layers": list(cross_layers),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "torch": torch.__version__,
    }
    (output / "mossvl_fp8_dynamic_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
