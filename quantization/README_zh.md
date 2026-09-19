# MOSS-VL 量化

[English](README.md) | [中文](README_zh.md)

本目录说明如何量化 MOSS-VL，并提供了产出官方量化模型的转换脚本。所有方案都是训练后量化：直接对已完成的 checkpoint 做转换，不需要校准数据，也不需要重新训练，因此对任何 MOSS-VL 格式的 checkpoint 都适用，包括你自己微调后的版本。

我们已在 Hugging Face 和 ModelScope 发布预量化模型：

| 模型 | 权重量化方案 | 推理引擎 |
|-|-|-|
| MOSS-VL-Instruct-0708-FP8 | FP8-Dynamic W8A8 | Transformers、SGLang |
| MOSS-VL-Instruct-0708-NF4 | NF4 W4 Keep-4 | Transformers |
| MOSS-VL-Realtime-FP8 | FP8-Dynamic W8A8 | Transformers、SGLang |
| MOSS-VL-Realtime-NF4 | NF4 W4 Keep-4 + HQQ KV8 | Transformers |

## 目录结构

```
quantization/
├── quantize_fp8_dynamic.py     # FP8-Dynamic 一次性转换（llmcompressor）
├── quantize_nf4_keep_ends.py   # BitsAndBytes NF4 转换，首尾若干层保留 BF16
├── package_kv8_hqq.py          # 为 checkpoint 打包 HQQ INT8 KV Cache（Transformers 侧）
└── requirements.txt            # 在仓库 requirements.txt 之上的量化依赖
```

## 环境

量化环境为仓库的 `requirements.txt` 加上 [`quantization/requirements.txt`](requirements.txt)，版本固定为产出已发布模型的那一组：

```bash
pip install -i https://pypi.org/simple --no-build-isolation -r requirements.txt
pip install -i https://pypi.org/simple -r quantization/requirements.txt
# 仅 FP8 转换需要；为什么用 --no-deps 见 quantization/requirements.txt 注释
pip install -i https://pypi.org/simple --no-deps llmcompressor==0.10.0
```

## 量化一个 checkpoint

先跑一个权重转换脚本，再按需打包 HQQ INT8 KV Cache：

```bash
# FP8-Dynamic —— 推荐，一份 checkpoint 同时兼容 Transformers 和 SGLang
python quantization/quantize_fp8_dynamic.py --source /path/to/your-checkpoint --output ./my-model-fp8

# 或 NF4 Keep-4 —— 显存最低，仅 Transformers
python quantization/quantize_nf4_keep_ends.py --source /path/to/your-checkpoint --output ./my-model-nf4 --verify-reload

# 可选：为 Transformers 打包运行时 HQQ INT8 KV Cache
python quantization/package_kv8_hqq.py --source ./my-model-fp8 --output ./my-model-fp8-kv8
```

两个权重转换脚本会从 checkpoint 的 `config.json` 自动读取 `text_config.num_hidden_layers` 和 `text_config.cross_attention_layers` 来决定量化范围，范围异常时会直接报错；需要覆盖时用 `--num-layers`、`--cross-layers` 或 `--keep-end-layers`。

量化后的推理与普通 checkpoint 完全一样：离线图片和视频生成见 [`inference/offline/`](../inference/offline/README.md)，流式推理见 [`inference/realtime/`](../inference/realtime/README.md)。

## FP8-Dynamic

这是 `MOSS-VL-Instruct-0708-FP8` 和 `MOSS-VL-Realtime-FP8` 使用的方案：权重以 `float8_e4m3fn` channel-wise static 量化，被量化算子接收的输入激活使用 FP8 per-token dynamic 量化，通过 compressed-tensors `FP8_DYNAMIC` 实现。方案只作用于语言模型：

- 量化范围：没有 cross-attention 的语言层的 `q_proj/k_proj/v_proj/o_proj` 与 `gate_proj/up_proj/down_proj` Linear——MOSS-VL-0708 共 48 层语言模型，cross-attention 位于第 `2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46` 层，因此剩余 36 层共 252 个 Linear 被量化
- 保留 BF16：视觉塔与视觉 merger、全部 cross-attention 层、embedding、norm、`lm_head`

转换只需一次 [LLM Compressor](https://github.com/vllm-project/llm-compressor) 调用：

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
# 最后把其余支撑文件拷贝过来：processor 配置、modeling 代码、chat template 等
```

为了兼容 SGLang，`quantize_fp8_dynamic.py` 还会在保存出的 `config.json` 的 `quantization_config` 中补充融合模块映射和 SGLang 侧的目标/忽略正则：

```json
"packed_modules_mapping": {
  "qkv_proj": ["q_proj", "k_proj", "v_proj"],
  "gate_up_proj": ["gate_proj", "up_proj"]
}
```

并在 `ignore` 中追加：`re:^model\.visual\.`、`model.language_model.model.layers` 下的 cross-attention 层、以及 `model.language_model.lm_head`。

## NF4 Keep-4

这是 NF4 模型使用的方案：[BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) NF4 4-bit weight-only 量化，double quantization，BF16 计算。该方案仅支持 Transformers，是两个方案中显存最低的。

- 量化范围：中间 40 个语言层的 self-attention 与 MLP Linear，共 240 个
- 保留 BF16：首 4 层、末 4 层，全部 cross-attention 层，视觉塔，embedding，norm，`lm_head`——首尾层保持全精度比逐层全部量化明显更稳定

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

## KV Cache 运行时量化

仅量化权重对长视频往往不够：KV Cache 会随流持续增长。KV 量化是生成时的配置，对任何微调后的 checkpoint 都同样适用。

**Transformers.** `package_kv8_hqq.py` 会向 `generation_config.json` 写入 HQQ INT8 KV Cache 配置：

```json
"cache_implementation": "quantized",
"cache_config": {
  "backend": "hqq", "nbits": 8,
  "axis_key": 0, "axis_value": 0,
  "q_group_size": 64, "residual_length": 128
}
```

需要安装 `hqq` 包。注意一点：Transformers 的 `QuantizedCache` 把较早的 KV 存在 `_quantized_keys/_quantized_values` 中，`keys/values` 只保留一段 BF16 residual 尾部，因此 MOSS-VL cross-attention 的 cache 读取路径需要打补丁：先反量化历史 KV、再与 residual 尾部拼接。脚本会自动对 `modeling_moss_vl.py` 应用该补丁，已发布的 KV8 checkpoint 里也有该实现的参考。

**SGLang.** 不使用 HQQ，改用引擎自身的 KV dtype，例如对 FP8 checkpoint 使用 `--kv-cache-dtype fp8_e4m3`。
