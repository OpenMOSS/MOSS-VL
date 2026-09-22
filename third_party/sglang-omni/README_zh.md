# SGLang-Omni Realtime for MOSS-VL

[English](./README.md) | **简体中文**

基于 [SGLang-Omni](https://github.com/sgl-project/sglang-omni) 开发的 MOSS-VL 实时视频理解后端，使用 [SGLang](https://github.com/sgl-project/sglang) 推理。本仓库是 SGLang-Omni 的特化分支，以上游提交 `573ce796`（2026-08-15）为基准，专注于 MOSS-VL 实时视频理解的推理服务。客户端通过 WebSocket 持续发送视频帧与问题，接收文本、静默和输入处理事件。

[安装指南](./docs/get_started/installation_zh.md) | [启动与测试](./deployment/moss_vl_realtime/README_zh.md) | [WebSocket 协议](./docs/cookbook/moss_vl_realtime.md) | [Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo)

- 增量视觉特征与 KV，支持 JPEG、PNG、WebP 帧。
- 持续会话、新问题打断与静默后唤醒。
- 动态多会话调度、单卡、TP 多卡以及原生数据并行（`--dp-size`）多副本推理。
- Decode CUDA Graph、视觉 KV 滑窗与有界输入队列。

> **所需 checkpoint**：本后端只加载 SGLang 格式的实时 checkpoint [`OpenMOSS-Team/MOSS-VL-Realtime-SGLANG`](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG)，请下载后将 `MODEL_PATH` 指向本地目录； Transformers 格式的 `MOSS-VL-Realtime` checkpoint **不能**直接用于本后端。

浏览器交互、ASR/TTS、文本 memory 和 REST 网关由独立的 [Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo) 提供。

## 安装

Ascend 集成请使用独立的 [NPU 部署与验收指南](./deployment/npu/README_zh.md)。以下安装步骤面向 CUDA。

按[安装指南](./docs/get_started/installation_zh.md)创建 Python 3.12 环境并安装哈希依赖锁。需要完整应用时，直接使用 [Demo 安装器](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/blob/main/README_zh.md)，不要重复安装后端。

配套仓库与运行依赖见 [Demo README](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/blob/main/README_zh.md#兼容性与更新)。安装跟随当前仓库，不固定源码提交；排错时记录实际安装版本。请在目标硬件验证显存配置与 JIT。

## 启动服务

可选的 [VL API v2 入口](./vl_api_adapter/README_zh.md) 在独立端口提供逐段回答结算；现有客户端保留原生入口与语义。

完成安装后，在仓库根目录执行：

```bash
source .venv/bin/activate
export MODEL_PATH="$HOME/models/MOSS-VL-Realtime-SGLANG"
export CUDA_HOME="$(python deployment/repro/cuda_toolkit.py)"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
python deployment/moss_vl_realtime/check_env.py "$MODEL_PATH"
bash deployment/moss_vl_realtime/start.sh "$MODEL_PATH"
```

启动器自动选择空闲 GPU，加载模型并预热视觉处理。用 `--gpus 0` 指定设备、`--port 18510` 更换端口、`--dry-run` 查看配置。

| 默认配置 | 值 |
| --- | --- |
| 服务地址 | `http://127.0.0.1:18500` |
| 会话容量 / context | 4 / 131072 |
| 静态显存比例 | 0.5 |
| 视觉 KV 窗口 | 开启，60 秒 |
| Pooling / async decode | 关闭 |

这些值来自 [config.json](./deployment/moss_vl_realtime/config.json)，适用于 `start.sh`。底层 Python launcher 和 Demo 托管部署有各自默认值。

```bash
curl --fail http://127.0.0.1:18500/health
curl --fail http://127.0.0.1:18500/v1/models
```

服务模型 ID 为 `moss-vl-realtime`，实际 checkpoint 由模型路径决定。对外部署需配置鉴权、TLS 和访问控制。TP 与高级参数见 [Cookbook](./docs/cookbook/moss_vl_realtime.md#start-the-server)。

## 调用示例

以下使用仓库内置的两张视频帧：

```bash
python examples/moss_vl_realtime_client.py \
  --url ws://127.0.0.1:18500/v1/video/realtime \
  --prompt "Describe the visible scene." \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0000.png --timestamp 0.0 \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0001.png --timestamp 1.0
```

客户端按时间戳回放，并将最后一帧标为 `final`。持续摄像头输入使用 Demo 或自定义客户端：

```text
session.created -> session.configure -> session.configured / session.ready
input.frame -> input.frame.ready -> binary frame -> input.frame.accepted
input.frame.processed -> response.text.delta / response.turn.silence
```

帧与 `input.prompt` 共用递增的 `seq_no`。发送图像前等待 `input.frame.ready`。`session.abort` 结束整个会话，即使输入处理正在等待容量也可处理。有效配置的默认等待期限为 180 秒，不包含模型 prefill。

尚未发送画面时也可通过 `input.prompt` 提问。服务端会预先补齐训练格式中的 assistant 开头，
避免将这个开头误当成静默结束信号；模型实际生成的静默仍然有效，包括用户明确要求保持安静的情况。

`max_tokens_per_turn` 是每路 tokens/s 的软目标，不是回答长度；`max_new_tokens` 是每次输入后重新锚定的生成余量。字段、错误码和用量事件见[协议](./docs/cookbook/moss_vl_realtime.md#websocket-protocol)。

## 配合 Demo

仅手动或历史部署在 Demo 的 `.env.deploy` 中配置：

```dotenv
VLM_DEPLOY=sglang_omni
SGLANG_OMNI_URLS=http://127.0.0.1:18500
SGLANG_OMNI_SESSIONS_PER_REPLICA=4
SGLANG_OMNI_CONTEXT_LENGTH=131072
MODEL_PATH=/absolute/path/to/MOSS-VL-Realtime-SGLANG
```

URL、会话容量与 context 应和后端一致，模型路径须在 Demo 主机可读。Demo 使用独立环境与端口，详见其 [README](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/blob/main/README_zh.md)。

推荐的 `bootstrap.sh` / `scripts/repro/run.py` 不读取 `.env.deploy`，由自身管理后端和 API
（默认 18500/18501，网页 18502），请使用该入口的命令行配置，不要按本节重复启动后端。
可选 VL API v2 使用独立监听端口，不替换 Demo 使用的原生接口。

视觉滑窗回收旧帧 KV，但历史位置和文本上下文仍会增长。跨 context 的长会话由 Demo memory rollover 管理，见[容量规划](./docs/cookbook/moss_vl_realtime_capacity.md)。

## 测试与开发

三项公开测试验证精度对齐、单路性能和实时多路性能。命令、参考表格与折线图见[测试指南](./deployment/moss_vl_realtime/README_zh.md)。

```bash
bash deployment/moss_vl_realtime/test_accuracy.sh "$MODEL_PATH"
bash deployment/moss_vl_realtime/test_latency.sh "$MODEL_PATH"
bash deployment/moss_vl_realtime/test_concurrency.sh "$MODEL_PATH"
```

不加载权重的回归测试：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_video_realtime*.py
```

| 路径 | 内容 |
| --- | --- |
| [deployment/moss_vl_realtime](./deployment/moss_vl_realtime/) | 环境检查、启动器、测试与样例 |
| [models/moss_vl_realtime](./sglang_omni/models/moss_vl_realtime/) | 模型接入、增量输入、调度与 KV 管理 |
| [serve/video_realtime.py](./sglang_omni/serve/video_realtime.py) | WebSocket、背压与会话生命周期 |
| [Realtime Cookbook](./docs/cookbook/moss_vl_realtime.md) | 高级部署与协议参考 |

### 恢复文本历史

`GET /v1/video/realtime/capabilities` 通过 `prefill_messages: true` 声明支持。
客户端可在 `session.configure` 中传入 `prefill_messages`，以 `{role, content}`
保留文本历史的角色（`system`、`user` 或 `assistant`）。最多 64 条消息，
文本总计最多 131072 字符，同时受模型上下文长度限制。服务自动补齐 assistant
起始标记。该接口重建文本上下文，不恢复旧 KV 或历史图片、视频张量。
连接旧后端时，客户端应先探测支持情况再发送此字段。

## 来源与许可证

本项目基于 [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni)（上游提交 `573ce796`，2026-08-15），保留其框架、Git 历史和 [Apache License 2.0](./LICENSE)。此前开发使用 [CloudRipple/sglang-omni](https://github.com/CloudRipple/sglang-omni)，当前版本在 [fnlp-vision/sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime) 维护。

在该基准之上，本仓库围绕 MOSS-VL 实时服务开展开发：

- MOSS-VL 实时模型与服务栈（`sglang_omni/models/`、`deployment/moss_vl_realtime/`），涵盖流式会话、视觉 KV 滑窗与多副本推理；
- Ascend NPU 支持（`patches/npu/`、`deployment/npu/`）；
- 面向多实例部署的路由适配层（`vl_legacy_adapter/`、`vl_api_adapter/`）；
- 对共享运行时（`sglang_omni/pipeline/`、`sglang_omni/scheduling/`、`sglang_omni/serve/`、`sglang_omni/config/`）的适配修改，服务于实时会话、KV 生命周期与平台放置。

问题反馈请提交至[本仓库 Issues](https://github.com/fnlp-vision/sglang-omni-realtime/issues)。感谢 SGLang-Omni、SGLang 与 MOSS-VL 团队。
