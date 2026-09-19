# Audar-TTS-V1 Turbo

[English](./README.md) | **简体中文**

安装可选的 GGUF 与 codec 依赖：

```bash
pip install -e '.[audar-tts]'
```

需要 CUDA 版 llama.cpp 时，在安装 SGLang-Omni 前，按目标 CUDA 环境的构建参数安装 `llama-cpp-python`。

Turbo 模型仓库包含 GGUF 权重，没有 Transformers `config.json`，因此启动时显式指定配置：

```bash
sgl-omni serve --config examples/configs/audar_tts_turbo.yaml \
  --allowed-local-media-path /path/to/references
```

发送一段 5-15 秒参考音频及其转录：

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audarai/Audar-TTS-V1-Turbo",
    "input": "مرحبا، أهلا وسهلا بكم.",
    "ref_audio": "file:///path/to/references/voice.wav",
    "ref_text": "النص المطابق للمقطع المرجعي.",
    "response_format": "wav"
  }' \
  --output audar.wav
```

后端从 `input` 推断输出语言。可选的 `language` 字段被接受为元数据，但模型不使用它。

## 上游验证记录

上游 PR [#1090](https://github.com/sgl-project/sglang-omni/pull/1090) 基于 [#1096](https://github.com/sgl-project/sglang-omni/pull/1096)，将集成代码从 797 行减少到 619 行（不计测试和文档，减少 22.3%）。50 组修改前后的 PCM WAV 输出逐字节一致。阿拉伯语 ASR 指标为 5.43% WER、1.46% CER、88.75 BLEU 和 95.57 chrF++；配对 H100 运行的性能相当。
