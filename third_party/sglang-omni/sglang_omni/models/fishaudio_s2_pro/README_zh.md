# FishAudio S2 语音合成

[English](./README.md) | **简体中文**

## 概述

SGLang-Omni 集成 FishAudio S2 的 Dual-AR 主干，支持语音克隆与流式音频。上游单张 H200、batch size 1 的参考结果为 RTF 0.34、63.3 tokens/s。

此工作由 SGLang-Omni 与 [FishAudio](https://fish.audio) 团队合作完成。参与者：Jingwen Gu、Yitong Guan、Xiaole Guo、Shidong Li、Shuai Shi、Junrong Lin、Fan Yin、Leng Yue、Shenggui Li、Chenyang Zhao。

## 背景

S2 的自回归主干预测离散音频 token，再由 codec 还原波形，因此同样需要高效的 KV 管理与服务调度。S2 使用 Dual-AR 结构，支持自然语言标签控制韵律和情绪。上游发布资料报告训练数据超过 1,000 万小时、覆盖约 100 种语言，并使用 GRPO 对齐；Audio Turing Test 后验均值为 0.515，EmergentTTS-Eval 相对 gpt-4o-mini-tts 胜率为 81.88%，Seed-TTS Eval WER 在其对比模型中最低。模型训练与榜单详情应参阅对应发布资料。

集成时需要处理 VQ 码本嵌入、每个 Slow-AR 步骤后的多个 Fast-AR 步骤，以及码本结构约束，同时保留前缀缓存。

## 架构

```text
Text -> Preprocessing (CPU) -> SGLang AR Engine (GPU) -> DAC Vocoder (GPU) -> Audio
```

1. 预处理：将文本编码为 Qwen3 风格提示词。语音克隆时先用 DAC 将参考音频编码为 VQ code，再作为 system message 前缀。
2. Dual-AR 生成：Slow-AR 每步预测 semantic token；四层 Fast-AR 根据隐藏状态生成其余 9 个残差码本 token。在指定位置注入 VQ 嵌入，共享 SGLang KV 管理。启动时直接创建 Fast-AR，只加载 `audio_decoder.*`，不创建无用的 HF Slow-AR 包装器。
3. 声码器：DAC 将累积的码本索引解码为波形。

## 使用

启动与请求见[本地 S2-Pro Cookbook](../../../docs/cookbook/fishaudio_s2_pro.md) 和 [TTS API](../../../docs/basic_usage/tts.md)。

## 优化

- Paged KV cache 管理 Slow-AR 的缓存与并发。
- Radix 缓存共享系统提示和参考音频前缀；上游参考 TTFT 约 18 ms、首音频约 140 ms。
- Slow-AR/Fast-AR decode CUDA Graph 在 SM89、SM120 的 batch 1、2、4，关闭 `torch.compile` 时经过验证。SM89/SM100/SM120 默认不编译 Fast-AR，但保留 decode Graph；SM90 保持原有 compile 默认值。Prefill Graph、更大 batch 及 Graph 与 compile 联用需要单独验证。
- 注意力后端按架构选择。

### 注意力策略

| 架构 | Slow-AR | Fast-AR KV |
| --- | --- | --- |
| SM89 | FlashInfer | FlashInfer |
| SM90 | FA3 | FA3 |
| SM100 | FlashInfer | FlashInfer |
| SM120 | FlashInfer | FlashInfer |

显式 `attention_backend` 仅覆盖 Slow-AR。Fast-AR 调用独立 KV kernel，遵循自身策略。不要在 SM89/SM100/SM120 上强制 FA3；不支持的架构在启动时明确报错。

CUDA Graph 设计参考：[Dual-AR CUDA Graph](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/torch/cuda-graph/readme-2-en.md)。

## 后续方向

验证 `torch.compile` 与 CUDA Graph 联用时的正确性、显存和数值行为；将目前逐请求执行的 Fast-AR 码本解码批量化，以提高大 batch 的利用率。

## 工程说明

S2 的训练 RoPE 频率使用 BF16。SGLang 默认 FP32 `cos_sin_cache` 曾导致 logit 偏移和异常长音频序列；适配采用：

```python
def _truncate_rope_to_bf16(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "cos_sin_cache"):
            module.cos_sin_cache.data = module.cos_sin_cache.data.to(torch.bfloat16).to(
                torch.float32
            )
```

开发中曾观察到 FlashInfer 与训练 FlashAttention 路径的提前 EOS 差异，但并非受控质量对比，只作为历史调试背景。
