# VL API v2

[English](./README.md) | **简体中文**

MOSS-VL Realtime 的 WebSocket 协议适配层，提供逐段回答、用量结算和会话终态事件。与原生 v1 共用推理服务，在独立端口监听。

[API 参考](./API_zh.md) | [验证结果](./VALIDATION_zh.md) | [环境安装](../docs/get_started/installation_zh.md)

## 启动

先按环境安装文档准备后端和本地模型。以下命令均在仓库根目录执行：

```bash
source .venv/bin/activate
export MODEL_PATH=/path/to/MOSS-VL-Realtime-SGLANG
export CUDA_HOME="$(python deployment/repro/cuda_toolkit.py)"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

bash vl_api_adapter/start.sh "$MODEL_PATH"
```

默认自动选择一张空闲 GPU。可追加 `--gpus 0` 指定设备，或 `--dry-run` 检查启动配置。

| 配置 | 默认值 |
| --- | --- |
| v2 WebSocket | `ws://127.0.0.1:18610/v1/video/realtime` |
| 原生 v1 端口 | 18500 |
| 总会话容量 | 4，v1/v2 共享 |

| 参数 | 用途 |
| --- | --- |
| `--host`、`--port` | 监听地址、原生 v1 端口 |
| `VL_API_V2_PORT` | v2 端口，默认 18610 |
| `VL_API_V2_API_KEY` | v2 鉴权密钥；客户端发送 `Authorization: Bearer <key>` |
| `VL_API_V2_MODEL_VERSION` | 可选部署版本标签 |

未配置密钥时仅用于可信网络；公网访问需配置 TLS 和访问控制。TP 或自定义部署见 [Cookbook](../docs/cookbook/moss_vl_realtime.md#start-the-server)，为 Python 启动器追加 `--vl-api-v2-port`。

## 调用

```bash
python vl_api_adapter/client.py \
  --url ws://127.0.0.1:18610/v1/video/realtime \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0000.png \
  --prompt "请描述画面。"
```

`--frame`、`--prompt` 可重复。启用鉴权时，客户端读取同名环境变量 `VL_API_V2_API_KEY`。

- `response.done` 结束一段回答，连接继续保留；`session.done` 结束会话。
- 用 `response_id` 关联回答。同一 `turn_id` 可以包含多段回答。
- `include_usage` 仅控制观测推送，不影响结算用量。

完整字段、计量规则和异常断连处理见 [API 参考](./API_zh.md)。现有 v1/Demo/legacy 客户端继续使用原入口；切换到 v2 前必须调整回答结束处理。

## 测试

```bash
python -m pytest vl_api_adapter/tests -q
```

原版精度测试及 1/2/4 路协议对照结果见[验证结果](./VALIDATION_zh.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `adapter/` | 协议实现，随后端 Python 包安装 |
| `tests/` | CPU 协议回归测试 |
| `start.sh`、`client.py` | 启动入口、示例客户端 |
| `API*.md`、`VALIDATION*.md` | 接口参考、验证记录 |
