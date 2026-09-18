# VL 旧协议适配层

[English](./README.md) | **简体中文**

将旧版 VL WebSocket 消息转换为 sglang-omni 原生视频实时协议。每个客户端连接使用独立的上游会话，模型权重常驻后端。

## 启动

Ascend 请使用 [NPU 部署指南](../deployment/npu/README_zh.md)，启动器会在原生后端旁启动同一个适配层。

按照[安装指南](../docs/get_started/installation_zh.md)准备后端环境。在仓库根目录分别打开终端运行：

```bash
bash deployment/moss_vl_realtime/start.sh /path/to/MOSS-VL-Realtime-SGLANG --port 18500
```

```bash
cd vl_legacy_adapter
OMNI_WS_URL=ws://127.0.0.1:18500/v1/video/realtime ../../.venv-main/bin/python -m adapter
```

旧客户端连接 `ws://<adapter-host>:18600/v1/realtime`。适配层使用后端环境已有的 `websockets`，冒烟客户端还使用 Pillow。不提供鉴权，请仅部署在可信网络或有鉴权的网关之后。

## 协议

```text
start -> ready -> frame 元信息 + JPEG 二进制（批量发送）
      -> 每帧接受后返回 frame_ack -> output 增量文字 -> output <|im_end|>
      -> stop / 断开连接
```

发送帧时不要逐帧等待 ACK。适配层通过 300 ms 静默窗口推断末帧，因此末帧 ACK 包含这段等待。已经收到元信息的帧会等待二进制完成或接收超时，不参与静默结束判断。开始提交末帧后拒绝新增输入。收到结束标记后发送 `stop` 并关闭连接，下一轮重新连接。

| 旧字段或事件 | 原生协议映射 |
| --- | --- |
| `start.prompt` | 附加到末帧 `input.frame.prompt`，不设置 `session.configure.prompt` |
| `frame_queue_size` | `input_queue_capacity`，限制在 1-256 |
| `max_new_tokens` | `max_new_tokens`，至少为 1 |
| `max_tokens_per_second` | `max_tokens_per_turn`，正数 token 速率 |
| `temperature` / `top_p` | 限制在 0-2 / (0, 1]；`do_sample=false` 强制 temperature 为 0 |
| `top_k` / `repetition_penalty` | 不透传；原生 configure 不支持 |
| `frame.timestamp` | 非负、非递减时间戳 |
| `frame_ack` | 上游 `input.frame.accepted` 后返回，不代表处理完成 |
| `output.text` | 增量 `response.text.delta` |
| `<|im_end|>` | 会话结束或满足末帧静默规则时，由适配层生成 |
| `stop` / 断连 | 中止并关闭上游会话 |

非法数值参数、启动失败、帧二进制接收超时会返回 `error` 并关闭本轮。没有可见文字的轮次返回错误，不单独返回结束标记。容量不足的错误包含 `realtime session is already active`；被拒绝连接的网络操作不持有容量锁。适配层槽位释放不等待上游清理，因此获准进入新一轮不保证后端容量立即可用。

## 配置

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `LISTEN_HOST` / `LISTEN_PORT` | `0.0.0.0` / `18600` | 适配层监听地址 |
| `LISTEN_PATH` | `/v1/realtime` | 允许网关路径前缀，忽略查询参数 |
| `OMNI_WS_URL` | `ws://127.0.0.1:18500/v1/video/realtime` | 后端地址 |
| `MAX_INFLIGHT` | `1` | 适配层连接上限；0 表示由后端控制容量 |
| `FRAME_BUFFER_CAP` | `8` | 适配层队列中等待处理的完整帧数；单张 JPEG 上限 10 MiB |
| `START_TIMEOUT_S` | `10` | 建连后等待 start 的上限 |
| `SETUP_TIMEOUT_S` | `10` | 从 start 到 ready 的整体上限 |
| `READY_TIMEOUT_S` / `ACK_TIMEOUT_S` | `10` / `10` | 上游就绪 / 每次帧握手 ACK 的等待上限 |
| `FRAME_RECEIVE_TIMEOUT_S` | `10` | 元信息到完整二进制的接收上限，包括首帧 |
| `FINALIZE_QUIET_MS` | `300` | 没有待接收二进制时的输入静默窗口 |
| `SILENCE_FINALIZE_MS` | `1000` | 末帧提交后的静默收尾阈值 |

这些是分别计时的上限，不保证整轮在 10 秒内完成。下一帧元信息到达前的间隔超过静默窗口时，本轮输入可能已经结束；较慢的发送端应调大窗口。静默收尾是一种启发式规则，不是模型显式的回答结束信号。

## 测试

在 `vl_legacy_adapter/` 下运行 CPU 回归测试或连接 GPU 后端的冒烟测试：

```bash
../../.venv-main/bin/python -m pytest tests/test_lifecycle.py -q
../../.venv-main/bin/python tests/smoke_client.py --url ws://127.0.0.1:18600/v1/realtime \
  --testdata ../deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122
```

回归测试覆盖延迟或缺失帧、输入关闭、启动失败与取消、容量释放和参数映射。冒烟测试覆盖批量输入、连续轮次、损坏 JPEG、busy 拒绝和断连恢复；可用 `--tests vl01,vl02` 选择部分用例。冒烟输出不等于语义对齐或外部完整契约验收结论。

冒烟客户端在 `ready` 前遇到 busy 时最多重新连接三次，按 0.5/1/1.5 秒退避，并共用本轮处理期限。重试历史与最终错误分开记录。VL-02 禁用重试，严格检查连续轮次之间的容量释放。测试客户端保持直连，不使用环境代理。

### 时延参考

2026-09-10 已预热 GPU 后端的冒烟测量（当前修订版本），单位为秒。数据为观测值，不是时延保证。

| 用例 | Ready | 首 ACK | 末 ACK | 首段文字 | 结束标记 |
| --- | ---: | ---: | ---: | ---: | ---: |
| VL-01，4 帧 | 0.065 | 0.077 | 0.389 | 0.602 | 1.423 |
| VL-01，8 帧 | 0.054 | 0.078 | 0.394 | 0.468 | 0.932 |
| VL-02，第二轮 | 0.065 | 0.077 | 0.388 | 0.468 | 2.078 |
| VL-03，恢复轮 | 0.057 | 0.069 | 0.381 | 0.454 | 0.756 |

每个新会话仍会执行初始 prefill。接口适配目标为 `model-api-protocol-asr-tts-vl.md` 第 5 章。该外部文档和 `vision.go` 调用端未随仓库分发，最终对接验收需要使用实际调用端。
