# Playground

[English](./README.md) | **简体中文**

本目录提供 Qwen3-Omni、S2 Pro 与 Higgs Audio v3 的浏览器示例。MOSS-VL 实时视频应用见独立的 [Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo)。

| 子目录 | 模型与功能 | UI |
| --- | --- | --- |
| `qwen-omni/` | Qwen3-Omni：文本、音频、图像、视频对话 | HTML / CSS / JS |
| `s2pro/` | S2 Pro：语音克隆，流式与非流式 TTS | Gradio |
| `higgs/` | Higgs Audio v3：多语言 TTS、情绪/风格/音效/韵律控制 | HTML / CSS / JS |

各 `start.sh` 启动后端，等待 `/health`，再以前台方式启动 UI；`Ctrl-C` 停止两者。也可以单独启动后端，再将 UI 指向它。先安装对应模型和 UI 所需依赖。

## Qwen3-Omni

```bash
./playground/qwen-omni/start.sh \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
```

分别启动后端与 UI：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8000
SGLANG_OMNI_API_BASE=http://localhost:8000 \
  python playground/qwen-omni/app.py --port 7860
```

浏览器访问 <http://localhost:7860>，后端为 <http://localhost:8000>。`--port` 指定后端端口，`--playground-port` 指定 UI 端口。

## S2 Pro TTS

```bash
./playground/s2pro/start.sh \
  --model-path fishaudio/s2-pro
```

分别启动：

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
python -m playground.s2pro.app --api-base http://localhost:8000 --port 7899
```

访问 <http://localhost:7899>。`Non-Streaming` 在生成后返回 WAV，`Streaming` 增量播放 `/v1/audio/speech` 的 PCM 数据。端口参数为 `--port` 和 `--gradio-port`。

## Higgs Audio v3 TTS

```bash
./playground/higgs/start.sh \
  --model-path bosonai/higgs-tts-3-4b
```

分别启动：

```bash
sgl-omni serve \
  --model-path bosonai/higgs-tts-3-4b \
  --port 8000
python -m playground.higgs.app --api-base http://localhost:8000 --port 7860
```

访问 <http://localhost:7860>，支持流式/非流式切换、麦克风/文件/URL 参考音频，以及在光标处插入 `<|category:name|>` 的控制标签。端口参数为 `--port` 和 `--playground-port`。

## SSH 转发

从本机转发远端后端和 UI 端口：

```bash
ssh -L 8000:localhost:8000 -L 7860:localhost:7860 user@host
```
