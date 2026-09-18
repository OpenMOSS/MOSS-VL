# 基准测试

[English](./README.md) | **简体中文**

本目录是通用音频与多模态基准。MOSS-VL 实时精度和时延使用[三项交付测试](../deployment/moss_vl_realtime/README_zh.md)。

## 目录

| 目录 | 内容 |
| --- | --- |
| [tasks](./tasks/) | 任务执行 |
| [metrics](./metrics/) | 性能与精度指标 |
| [dataset](./dataset/) | 数据加载与下载工具 |
| [benchmarker](./benchmarker/) | Runner、数据结构与工具 |
| [eval](./eval/) | CLI 入口 |
| [tts_serving](./tts_serving/README_zh.md) | TTS 接口与负载测试 |

缓存与生成结果属于本地运行产物，不是源码。

## 入口

| 入口 | 用途 |
| --- | --- |
| [benchmark_tts_seedtts.py](./eval/benchmark_tts_seedtts.py) | TTS 速度、WER 与语音克隆 |
| [benchmark_tts_serving.py](./eval/benchmark_tts_serving.py) | TTS HTTP、流式、WebSocket、音色与 batch 接口 |
| [benchmark_omni_seedtts.py](./eval/benchmark_omni_seedtts.py) | Omni 语音输出速度与 WER |
| [benchmark_omni_mmsu.py](./eval/benchmark_omni_mmsu.py) | MMSU 音频理解 |
| [benchmark_omni_mmau.py](./eval/benchmark_omni_mmau.py) | MMAU 音频理解 |
| [benchmark_omni_mmar.py](./eval/benchmark_omni_mmar.py) | MMAR 音频推理 |
| [benchmark_omni_mmmu.py](./eval/benchmark_omni_mmmu.py) | MMMU 图像理解 |
| [benchmark_omni_videomme.py](./eval/benchmark_omni_videomme.py) | Video-MME 视频理解 |
| [benchmark_omni_videoamme.py](./eval/benchmark_omni_videoamme.py) | Video-AMME 视频与音频理解 |
| [benchmark_asr_seedtts.py](./eval/benchmark_asr_seedtts.py) | ASR 并发、WER、覆盖率与 RTFx |

使用 `python -m benchmarks.eval.<entry> --help` 查看数据集、服务、并发与输出参数。服务端需匹配目标模态；纯文本、语音输出和转录是不同的负载。

## TTS 示例

先在 8000 端口启动兼容的 S2-Pro 服务，再从仓库根目录执行：

```bash
python -m benchmarks.dataset.prepare --dataset seedtts
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model fishaudio/s2-pro --port 8000 \
  --output-dir results/s2pro_en --lang en --max-samples 50 --concurrency 8
```

模型启动命令见 [TTS 指南](../docs/basic_usage/tts.md)。不要让多个模型服务占用相同端口或重叠使用 GPU。

## 生成与评估

将生成和转录分开，避免 GPU 竞争。第二条命令复用已保存音频，分配 ASR 设备前先停止生成服务。

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --stream --model fishaudio/s2-pro --port 8000 \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir results/s2pro_en --lang en --max-samples 50 --concurrency 8

python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only --model fishaudio/s2-pro \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir results/s2pro_en --lang en --device cuda:0

python -m benchmarks.eval.benchmark_tts_seedtts \
  --utmos-only --output-dir results/s2pro_en --device cuda:0
python -m benchmarks.eval.benchmark_tts_seedtts \
  --similarity-only --output-dir results/s2pro_en --device cuda:0
```

TTS 的 `--concurrency` 与 `--max-concurrency` 等价。参考音频默认使用 `ref_audio`/`ref_text`；Higgs TTS 和 MOSS-TTS 使用 `--ref-format references`。MOSS-TTS 支持 `--token-count auto`；不克隆音色时可用 `--no-ref-audio` 和模型支持的 `--voice`。

Omni SeedTTS 也支持分阶段运行，其转录阶段使用 `--port` 指定的 ASR 服务。复用 TTS 命令前先阅读对应模块帮助。

## ASR 并发

先启动对应 ASR 服务，再运行：

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --max-samples 20 --concurrencies 2 --repeats 1 --stream
```

报告包含 WER、评估覆盖率、RTF、RTFx（每秒墙钟时间成功处理的输入音频秒数）和路由均衡情况。流式模式额外记录文本 TTFT 与 chunk 间隔。

## 离线音频质量

UTMOS 使用 `balacoon/utmos` 预测自然度；说话人相似度通过 WavLM 比较生成语音和参考音色。两者读取已保存音频，不需要 TTS 服务。

| 指标 | 输出 | 默认缓存 / 覆盖变量 |
| --- | --- | --- |
| UTMOS | `utmos_results.json`：均值、中位数、P5/P95、评估/跳过计数与逐样例分数 | `~/.cache/sglang-omni/utmos` / `UTMOS_CACHE_DIR` |
| 说话人相似度 | `similarity_results.json`：余弦相似度乘以 100 的均值、计数与逐样例分数 | `~/.cache/sglang-omni/speaker_sim` / `SEEDTTS_SIM_CACHE_DIR` |

```bash
python -m benchmarks.metrics.utmos --warm-cache
python -m benchmarks.metrics.speaker_similarity_assets --warm-cache
```

报告质量与时延时一并保留模型、数据集、硬件和生成配置。接口测试通过不等于语义或音频质量得分。
