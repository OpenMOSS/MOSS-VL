# Quantizing MOSS-VL

[English](README.md) | [中文](README_zh.md)

This directory documents how to quantize MOSS-VL and provides the scripts used to produce the released quantized checkpoints. All recipes are post-training quantization: they convert a finished checkpoint directly, with no calibration data and no retraining, so they apply to any MOSS-VL-format checkpoint, including your own fine-tuned one.

Pre-quantized checkpoints are available on Hugging Face and ModelScope:

| Checkpoint | Weight scheme | Engines |
|-|-|-|
| MOSS-VL-Instruct-0708-FP8 | FP8-Dynamic W8A8 | Transformers, SGLang |
| MOSS-VL-Instruct-0708-NF4 | NF4 W4 Keep-4 | Transformers |
| MOSS-VL-Realtime-FP8 | FP8-Dynamic W8A8 | Transformers, SGLang |
| MOSS-VL-Realtime-NF4 | NF4 W4 Keep-4 + HQQ KV8 | Transformers |

## Directory Structure

```
quantization/
├── quantize_fp8_dynamic.py     # FP8-Dynamic one-shot conversion (llmcompressor)
├── quantize_nf4_keep_ends.py   # BitsAndBytes NF4 conversion, keeping the first/last layers in BF16
├── package_kv8_hqq.py          # Package a checkpoint with HQQ INT8 KV cache for Transformers
└── requirements.txt            # Quantization extras on top of requirements.txt
```

## Environment

The quantization stack is the repository's `requirements.txt` plus [`quantization/requirements.txt`](requirements.txt), pinned to the versions that produced the released checkpoints:

```bash
pip install -i https://pypi.org/simple --no-build-isolation -r requirements.txt
pip install -i https://pypi.org/simple -r quantization/requirements.txt
# FP8 conversion only; see quantization/requirements.txt for why --no-deps is needed
pip install -i https://pypi.org/simple --no-deps llmcompressor==0.10.0
```

## Quantize a checkpoint

Run one weight-conversion script, then optionally package HQQ INT8 KV cache:

```bash
# FP8-Dynamic — recommended, one checkpoint runs on both Transformers and SGLang
python quantization/quantize_fp8_dynamic.py --source /path/to/your-checkpoint --output ./my-model-fp8

# or NF4 Keep-4 — lowest memory, Transformers only
python quantization/quantize_nf4_keep_ends.py --source /path/to/your-checkpoint --output ./my-model-nf4 --verify-reload

# optional: runtime HQQ INT8 KV cache for Transformers
python quantization/package_kv8_hqq.py --source ./my-model-fp8 --output ./my-model-fp8-kv8
```

Both weight scripts read `text_config.num_hidden_layers` and `text_config.cross_attention_layers` from the checkpoint's `config.json` to decide the quantization scope automatically, and abort with an explicit error if the scope looks wrong. Use `--num-layers`, `--cross-layers`, or `--keep-end-layers` to override.

Inference works exactly as with an unquantized checkpoint: see [`inference/offline/`](../inference/offline/README.md) for offline image and video generation and [`inference/realtime/`](../inference/realtime/README.md) for streaming.

## FP8-Dynamic

This is the recipe behind `MOSS-VL-Instruct-0708-FP8` and `MOSS-VL-Realtime-FP8`. Weights are quantized to `float8_e4m3fn` channel-wise static, and input activations to the quantized kernels are FP8 per-token dynamic, through compressed-tensors `FP8_DYNAMIC`. The scheme targets only the language model:

- quantized: the `q_proj/k_proj/v_proj/o_proj` and `gate_proj/up_proj/down_proj` Linears of every language-model layer without cross-attention — MOSS-VL-0708 has 48 language layers with cross-attention at indices `2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46`, so 36 layers and 252 Linears are quantized
- kept in BF16: the vision tower and merger, all cross-attention layers, embeddings, norms, and `lm_head`

The conversion is a one-shot [LLM Compressor](https://github.com/vllm-project/llm-compressor) call:

```python
import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/path/to/your-checkpoint"
CROSS_LAYERS = (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46)
SELF_LAYERS = tuple(i for i in range(48) if i not in CROSS_LAYERS)
GROUP = "|".join(str(i) for i in SELF_LAYERS)
TARGET_REGEX = (
    r"re:^model\.language_model\.layers\."
    rf"({GROUP})\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))$"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
)
quantized = oneshot(
    model=model,
    recipe=QuantizationModifier(targets=[TARGET_REGEX], scheme="FP8_DYNAMIC"),
    output_dir=None,
    trust_remote_code_model=True,
    precision="bfloat16",
    save_compressed=True,
)
quantized.save_pretrained("/path/to/output", save_compressed=True, safe_serialization=True)
AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True).save_pretrained("/path/to/output")
# then copy over the remaining support files: processor configs, modeling code, chat template, ...
```

For SGLang compatibility, `quantize_fp8_dynamic.py` additionally writes the fused-module mapping and SGLang-side target/ignore regexes into the saved `config.json`:

```json
"packed_modules_mapping": {
  "qkv_proj": ["q_proj", "k_proj", "v_proj"],
  "gate_up_proj": ["gate_proj", "up_proj"]
}
```

and adds to `ignore`: `re:^model\.visual\.`, the cross-attention layers under `model.language_model.model.layers`, and `model.language_model.lm_head`.

## NF4 Keep-4

This is the recipe behind the NF4 checkpoints: [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) NF4 weight-only 4-bit quantization with double quantization and BF16 compute. It is Transformers-only and gives the lowest memory footprint of the two recipes.

- quantized: self-attention and MLP Linears of the 40 middle language layers, 240 Linears in total
- kept in BF16: the first 4 and last 4 language layers, all cross-attention layers, vision tower, embeddings, norms, and `lm_head` — keeping the endpoint layers full-precision is noticeably more stable than quantizing every layer

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

keep = 4
skip_modules = ["model.visual", "cross_attn", "lm_head"] + [
    f"model.language_model.layers.{i}"
    for i in list(range(keep)) + list(range(48 - keep, 48))
]
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/your-checkpoint",
    trust_remote_code=True,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules,
    ),
    low_cpu_mem_usage=True,
)
model.save_pretrained("/path/to/output", safe_serialization=True)
```

## Runtime KV cache quantization

Weight quantization alone is often not enough for long videos: the KV cache keeps growing with the stream. KV quantization is a generation-time setting and applies unchanged to any fine-tuned checkpoint.

**Transformers.** `package_kv8_hqq.py` writes an HQQ INT8 KV-cache config into `generation_config.json`:

```json
"cache_implementation": "quantized",
"cache_config": {
  "backend": "hqq", "nbits": 8,
  "axis_key": 0, "axis_value": 0,
  "q_group_size": 64, "residual_length": 128
}
```

This requires the `hqq` package. One caveat: with Transformers' `QuantizedCache`, older entries live in `_quantized_keys/_quantized_values` while only a short BF16 tail sits in `keys/values`, so MOSS-VL's cross-attention cache read path is patched to dequantize the history and concatenate it with the residual tail. The script applies this patch to `modeling_moss_vl.py` automatically, and the released KV8 checkpoints contain it as a reference.

**SGLang.** Use the engine's own KV dtype instead of HQQ, for example `--kv-cache-dtype fp8_e4m3` with the FP8 checkpoint.
