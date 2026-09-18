# TTS 服务基准

[English](./README.md) | **简体中文**

测试 OpenAI 兼容 TTS 服务在负载下的行为：语音生成、原始 PCM 流、WebSocket、批量合成、非法请求和有状态音色管理。SeedTTS 质量评估是[另一项基准](../README_zh.md)。

## 运行

单独启动目标服务，再从仓库根目录执行：

```bash
python -m benchmarks.eval.benchmark_tts_serving \
  --spec benchmarks/tts_serving/examples/stress.json \
  --out results/tts_serving/stress
```

在 [stress.json](./examples/stress.json) 中设置 `base_url`、`model_name` 和可选的 `auth.api_key_env`。内置配置面向 Higgs TTS，使用 `docs/_static/audio` 中的参考音频；目标服务应允许访问该本地目录。参考路径属于服务端文件系统，而不是压测客户端。

直接运行需要 FFmpeg 解码压缩音频；使用语料的配置可能下载固定的 SeedTTS 元数据。

## 配置

| 字段 | 用途 |
| --- | --- |
| `base_url` / `model_name` | 目标服务与请求模型 ID |
| `test_type` / `run_id` | 产物标签；类型为 `engine`、`e2e`、`external` |
| `seed` | 确定性场景顺序与到达时间，默认 0 |
| `auth.api_key_env` | 存储 bearer token 的环境变量名 |
| `params.profile` | `stress` |
| `params.enabled_endpoints` | `speech`、`speech_stream`、`voices`、`batch`、`websocket` |
| `params.load_stages` | 分阶段负载计划 |
| `params.total_requests` / `max_concurrency` | 未指定阶段时的请求数与并发 |
| `params.timeout_s` | 单请求超时 |
| `params.speaker_max_uploaded` | 服务端预期音色容量 |
| `params.voice_cache_pressure_voice_count` / `voice_speaker_cap_count` | 音色缓存压力与容量测试预算 |
| `params.file_ref_audio` / `file_ref_text` | 服务端可读取的参考音频及文本 |

完整结构与确定性场景矩阵见 [spec.py](./spec.py) 和 [scenarios.py](./scenarios.py)。

## 接口检查

- 语音：响应格式、参考音频、速度边界、SDK 兼容性、解码后非零音频与流式输出。
- 错误：非法 HTTP 请求返回包含 `message`、`type`、`param`、`code` 的 JSON `error`。缺失音色资源返回 404；删除不存在音色返回 `success: false` 及非空 `error` 对象或字符串。
- Batch：1-32 项、逐项参数、逐项结果与超限拒绝。
- 音色：上传、列表、元数据、覆盖、删除、复用、竞争与缓存压力；删除创建的音色并验证清理。
- WebSocket：配置、增量文本、二进制音频、事件顺序、断连以及非法或缺少配置的请求。

缓存压力需要可观测的 `cache_stats`：entries、memory_bytes、max_bytes、eviction_count、hit_count、miss_count、delete_invalidation_counter。操作应改变对应计数；负载达到声明预算时应发生淘汰。启用但缺失的接口明确失败。

## 负载

内置配置包含 `mixed-production`、`voice-cache-pressure`、`voice-speaker-cap`。混合阶段运行 300 秒，组合 REST、流式 REST、WebSocket、batch-32 与长 prefill，每 15 秒启动一组六类同时到达请求，中间穿插背景流量和必要 API 覆盖。覆盖流量与工作负载分位数分别统计。

容量测试先读取已有音色，再补到配置上限，并要求溢出请求失败。缓存测试验证实际状态变化，不仅检查请求成功。

## 结果

| 产物 | 内容 |
| --- | --- |
| `results.json` | 通过状态、覆盖率、负载有效性、时延与缺失接口 |
| `manifest.json` | 配置/场景哈希与产物元数据 |
| `raw/*.jsonl` | 逐场景记录 |
| `logs/harness.log` | 负载阶段日志 |

**同时检查退出码和 `overall.passed`。** 退出 0 只表示工具完成并写出结果，服务仍可能不通过；非零表示基础设施或运行错误，能写报告时记录 `harness_status="error"`。

解释性能前检查 `overall.coverage_contract_valid`、`overall.load_generation_valid`、`overall.mixed_arrival_valid`、`metrics.by_stage_and_workload`、`unsupported_contracts` 与 `coverage_failures`。

## 独立容器

这是压测客户端镜像，不是推理后端：

```bash
docker build -f benchmarks/tts_serving/Dockerfile \
  -t sglang-omni-tts-serving-benchmark .
mkdir -p results/tts_serving/stress
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/benchmarks/tts_serving/examples/stress.json:/etc/benchmark/spec.json:ro" \
  -v "$PWD/results/tts_serving/stress:/var/benchmark/out" \
  sglang-omni-tts-serving-benchmark
```

输入为 `/etc/benchmark/spec.json`，输出为 `/var/benchmark/out`。目标地址应在容器内可达。镜像内含 FFmpeg；参考音频仍须由目标服务读取。
