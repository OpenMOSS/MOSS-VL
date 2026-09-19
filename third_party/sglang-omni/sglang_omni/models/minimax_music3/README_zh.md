# MiniMax Music 3

[English](./README.md) | **简体中文**

文本生成音乐模型：Qwen3 主干生成八码本 RVQ 帧，再经过 flow-matching DIT 与 DAC 解码器，输出 32 kHz 立体声。

## 启动

在独立环境中安装本仓库。DIT 使用 `sglang.multimodal_gen`，依赖在 [pyproject.toml](../../../pyproject.toml) 中声明。不要让不兼容的 `flashinfer-cubin` 覆盖固定的 FlashInfer 版本。

```bash
# Choose one layout.
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000
```

只暴露一张 GPU 时两个阶段共卡；两张及以上时 DIT/DAV 放在第二张卡。声学阶段均使用 FP32。默认启用主干与 RVQ 深度解码 CUDA Graph、DIT/DAV 编译及批量 seeded sampling。

## 请求

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Verse]\nHello from SGLang Omni",
    "instructions": "A bright piano pop song with a warm female vocal at 100 BPM",
    "seed": 7,
    "max_new_tokens": 250
  }' \
  --output song.wav
```

`input` 是歌词，`instructions` 是描述。结构标签独占一行，同一行标签后的文本会被规范化步骤丢弃。`max_new_tokens` 限制音频帧数，每秒 25 帧、最多 9,000 帧；模型可以提前结束。

完整参数、拒绝项与其他示例见 [Cookbook](../../../docs/cookbook/minimax_music3.md)。

## 测试

```bash
python -m pytest tests/unit_test/minimax_music3 -q
```

这些测试验证模型和请求接口。声学质量与 GPU 吞吐需另行运行模型；单元测试结果不是音频质量指标。
