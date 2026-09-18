# MOSS-VL Realtime WebSocket API v2

[English](./API.md) | **简体中文**

适用于 MOSS-VL Realtime 的 `vl-api-v2` 接口。[部署与调用](./README_zh.md) | [验证结果](./VALIDATION_zh.md)

**目录**：[连接](#1-传输与连接) · [生命周期](#2-会话状态机) · [客户端事件](#3-客户端事件) · [服务端事件](#4-服务端事件) · [计量](#5-用量与计量) · [限制](#6-限制与容量) · [错误](#7-终止与错误) · [时序](#8-完整时序与客户端约定)

## 1. 传输与连接

| 项目 | 定义 |
| --- | --- |
| 默认地址 | `ws://127.0.0.1:18610/v1/video/realtime` |
| 文本消息 | UTF-8 JSON，每条消息一个控制事件对象 |
| 二进制消息 | 完整图像文件字节 |
| 会话 | 每个被接纳的连接对应一个会话，无需额外创建请求 |
| 协议识别 | `protocol_version` 与 `capabilities`，不以 URL 路径区分 |

原生 v1 默认端口为 18500，与 v2 共享模型和容量。接纳失败见 §7.4。

所有控制事件**必须**含 `type` 字段。服务端对客户端控制事件采取严格校验（`extra="forbid"`），**多传字段会导致整条事件被拒**。字段类型必须正确，不接受 NaN/Infinity；整数和布尔值不能用字符串代替。服务端可增加输出字段，客户端应忽略不认识的输出字段。

部署设置 `VL_API_V2_API_KEY` 时，握手必须携带 `Authorization: Bearer <key>`。未设置时没有应用层鉴权，应仅用于可信网络或有鉴权的网关之后。公网 TLS/WSS 由部署入口配置。`GET /health` 提供运行状态，不是会话事件或计量接口。

本接口不提供 REST 会话创建、`task_id` 分配、reset 或跨连接记忆合并；这些由调用方负责。

## 2. 会话状态机

| 阶段 | 事件 / 行为 |
| --- | --- |
| 配置 | `session.created` → 客户端 `session.configure` → `session.configured` |
| 就绪 | 收到 `session.ready` 后可提交输入 |
| 推理 | 输入确认、文本和静默事件；`response.done` 仅结算一段回答 |
| 结束 | `session.done` 后关闭连接；异常例外见 §7.2 |

收到 `session.ready` 后才可推帧或提问。文字与观测事件可穿插于输入确认之间，客户端必须持续接收所有事件，不能只等待单一 ACK。每条输入保证 accepted 在对应 processed 之前。

**回答结束不等于会话结束**：有可见文字后，模型发出的 silence 结束该段回答，并触发一次 `response.done`。只有重复静默、没有新文字时，不生成空回答或重复结算。网络一段时间没有消息不是回答边界。

任何有序且最终计量成功的终止均以唯一的 `session.done` 收尾；其后不再发送文本、用量或 ACK。无法获得可信最终用量或传输失效时的例外见 §7.2。

## 3. 客户端事件

### 3.1 `session.configure`

连接后**必须**发送一次，在 `session.created.configure_timeout_s` 内。

```json
{
  "type": "session.configure",
  "prompt": "请持续观察画面，回答我的问题。",
  "max_new_tokens": 128,
  "max_tokens_per_turn": 10,
  "include_usage": true
}
```

| 字段 | 类型 | 必填 | 默认值 / 含义 |
| --- | --- | --- | --- |
| `type` | string | 是 | `session.configure` |
| `prompt` | string | 否 | `""`；初始用户提示词，不是系统提示词 |
| `system_prompt` | string/null | 否 | `null`；省略、null 或空字符串使用后端默认系统提示词，非空字符串作为独立系统提示词 |
| `prefill_messages` | array/null | 否 | `null`；可选文本历史，格式见下文 |
| `max_new_tokens` | int | 否 | 4096；正整数，生成额度，见下文 |
| `max_tokens_per_turn` | number | 否 | 86400；正数，实为生成速率上限（token/s），非总量，也不是最低速度保证 |
| `temperature` | number | 否 | 0；范围 [0, 2] |
| `top_p` | number | 否 | 1；范围 (0, 1] |
| `input_queue_capacity` | int | 否 | 4；范围 [1, 256]，含义见 §6 |
| `include_usage` | bool | 否 | false；仅控制可选观测通道 `session.usage` |
| `benchmark_ignore_eos` | bool | 否 | false；仅限服务端显式允许的基准模式，普通部署传 true 会被拒绝 |

`prefill_messages` 是 1–64 条消息的数组，每条仅含 `role`（`system`、`user` 或 `assistant`）和字符串 `content`，总内容不超过 131072 个字符。提供后替代初始 `prompt` 构造历史；若首条不是 system，补入生效系统提示词；若末条不是 user，补入空用户消息。历史 assistant 消息按模型格式补充起始 silence 标记。实际预填充位置计入文本输入用量，仍受上下文容量限制。

`include_usage` **不影响** `response.done.usage` / `session.done.usage`。用量为协议固有部分，无论该字段取值如何都下发。

**生成额度**：初始输入处理后，以及每次 extend 处理新帧或问题后，剩余生成额度设为 `max_new_tokens`，同时受上下文容量约束。设为 128 时，每次 extend 后最多再生成 128 个 token，包括 silence 等控制 token。

该值不是单段或整场的总输出上限；持续输入可使累计输出超过它。`response.done` 不补充额度。模型可提前静默，或在额度耗尽/满足停止条件时结束请求。

### 3.2 `input.frame`

图像帧的第一阶段：先发元数据，等 `input.frame.ready`，再发二进制。

```json
{
  "type": "input.frame",
  "seq_no": 0,
  "timestamp": 0,
  "mime_type": "image/jpeg"
}
```

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `type` | string | 是 | `input.frame` |
| `seq_no` | int | 是 | 非负整数，见 §3.4 |
| `timestamp` | number | 是 | 视频时间戳，秒；非负、有限，不得小于之前接纳帧的时间戳 |
| `mime_type` | string | 是 | `image/jpeg`、`image/png` 或 `image/webp`，必须与实际编码一致 |
| `prompt` | string/null | 否 | 默认 null；与帧一起输入的新问题。空白字符串按无问题处理 |
| `final` | bool | 否 | 默认 false；true 表示这是最后一次输入 |

**不接受表外字段。** 传入 `size_bytes` 等会导致整帧被拒。

**握手规则**：客户端收到 `input.frame.ready` 后**必须**紧接着发送对应的二进制帧，且在此之前**不得**发送该帧的二进制。二进制消息直接承载完整图片文件字节，不是 JSON、base64 或 multipart。

同一连接仅允许一个待接收二进制的帧握手。客户端应等待该帧 accepted 后再发送下一条输入；未完成握手时发送另一帧元数据或问题会被拒。ready 后默认 10 秒内必须交付二进制。图像为空、超限、损坏或 MIME 不匹配会被拒，见 §7.3。

### 3.3 `input.prompt`

```json
{
  "type": "input.prompt",
  "seq_no": 1,
  "prompt": "画面里有什么？",
  "final": false
}
```

| 字段 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `type` | string | 是 | `input.prompt` |
| `seq_no` | int | 是 | 见 §3.4 |
| `prompt` | string | 是 | 提问文本，不得为空或全空白。字段名是 `prompt`，不是 `text` |
| `final` | bool | 否 | 默认 false；true 表示这是最后一次输入 |

问题沿用最近帧的时间戳，不接收单独的 `timestamp`。非 final 问题可以触发回答，也可以由模型选择静默等待后续画面，不保证每次提问立即产生文字。

`final=true` 被接纳后不得继续输入。模型按原有生成/停止条件完成剩余推理后结束会话；不是在第一段 silence 时立即关闭。客户端等待 `session.done`，或主动 abort。

### 3.4 序号规则

`input.frame` 与 `input.prompt` **共用同一条 `seq_no` 递增序列**，从 0 开始。成功接纳的输入**不得**复用序号，**不得**跳号。

只有明确被拒且未 accepted 的输入可以修正后复用原序号；拒绝不会推进预期序号。跳号被拒且会话保持存活。若连接断开、无法判断是否接纳，不得假设重发具有幂等性，应新建会话。

### 3.5 `session.abort`

```json
{
  "type": "session.abort"
}
```

请求结束会话。不占用 `seq_no`，不接受其他字段，不等待帧握手或输入额度。正常取得最终计量后，服务端以 `session.done`（`reason: "aborted"`）响应。已有致命错误时保留其错误原因，不改写为 aborted。停止与最终计量有有界等待，不保证零延迟，见 §7.2。

## 4. 服务端事件

### 4.1 `session.created`

```json
{
  "type": "session.created",
  "session_id": "video_sess_...",
  "request_id": "video_req_...",
  "model": "moss-vl-realtime",
  "turn_id": 0,
  "configure_timeout_s": 180,
  "protocol_version": "vl-api-v2",
  "model_version": null,
  "model_version_source": "unknown",
  "capabilities": [
    "session.usage",
    "response.done.usage",
    "session.done.usage",
    "error.seq_no",
    "input.frame.rejected",
    "response.done.per_response",
    "response.id",
    "usage.text_input_output"
  ]
}
```

`session_id`、`request_id` 是后端标识，客户端应视为不透明字符串，不等同于网关的 task/trace ID。当前接口不返回 `task_id`。

`model_version` 是部署设置 `VL_API_V2_MODEL_VERSION` 提供的版本标签；设置时 source 为 `deployment`，缺省为 null/`unknown`。它不是协议版本，也不是由服务端自动推导的权重校验和。

客户端必须检查能力通告。按段记账需要 `response.done.per_response` 与 `response.id`，不能仅凭 `session.usage` 或模型名称判断。配置期限从后端会话创建时开始，到有效 configure 被接纳时结束，不包含模型预填充时间。

### 4.2 `session.configured`

```json
{
  "type": "session.configured",
  "session_id": "video_sess_...",
  "request_id": "video_req_...",
  "max_frame_bytes": 33554432,
  "input_queue_capacity": 4,
  "max_tokens_per_turn": 10,
  "parked_request_timeout_s": 3600,
  "context_limit": 131072
}
```

返回生效的队列、生成速率、帧字节上限、parked 超时和实际后端上下文容量。示例采用仓库启动配置，不应硬编码为所有部署的固定值。

### 4.3 `session.ready`

```json
{
  "type": "session.ready",
  "session_id": "video_sess_...",
  "request_id": "video_req_...",
  "turn_id": 0
}
```

表示模型会话已就绪，可以开始推帧。

### 4.4 输入确认与拒绝

| 事件 | 含义 | 字段（除 type） |
| --- | --- | --- |
| `input.frame.ready` | 等待该帧二进制，背压信号 | `seq_no` |
| `input.frame.accepted` | 图像通过接收校验并提交后端 | `seq_no`、`timestamp`、`final`、`pending_events`、`interrupts_current_turn` |
| `input.prompt.accepted` | 问题已提交后端 | `seq_no`、`final`、`pending_events`、`interrupts_current_turn` |
| `input.frame.processed` | 模型已处理该帧输入，不只是收到字节 | `seq_no`、`timestamp`、`final`、`pending_events`；可含 `turn_id`、`interrupted_turn_id` |
| `input.prompt.processed` | 模型已处理问题 | 同 processed 字段 |
| `input.frame.rejected` | 该次帧握手被拒 | `seq_no`、`reason` |

`pending_events` 是发送确认时尚未完成的输入数量，不是全局会话数。accepted 不是模型处理完成或回答完成保证。`interrupts_current_turn` 表示输入携带新问题；实际轮次变更以 processed/interrupted 事件为准。

可恢复帧拒绝同时发送 `error` 与 `input.frame.rejected`，客户端必须结束该次握手等待，不再等它的 ready/processed。缺失或不可用的序号不能虚构，rejected 的 `seq_no` 可为 null。致命会话错误由终态统一结束所有等待。

### 4.5 `response.text.delta`

```json
{
  "type": "response.text.delta",
  "delta": "画面中是",
  "turn_id": 1,
  "response_id": "response_...",
  "response_seq": 1
}
```

`delta` 是增量文本；客户端按接收顺序拼接。首次非空白可见文字创建回答 ID。此前的纯空白增量可携带 null 的 `response_id`/`response_seq`，不得据此建立已结算回答。

### 4.6 `response.turn.silence`

```json
{
  "type": "response.turn.silence",
  "turn_id": 1,
  "seq_no": 0,
  "timestamp": 0,
  "silence_seq": 1
}
```

模型选择进入静默。`seq_no`/`timestamp` 在尚无关联输入时可为 null，`silence_seq` 为后端静默序号，不是回答序号。

若此前有尚未结束的可见回答，先发送 silence，再发送该段 `response.done`。**只有静默、没有新文字时不产生 `response.done`。** 其消耗的 token 计入下一条 `response.done` 的增量或最终残差。

### 4.7 `response.turn.interrupted`

```json
{
  "type": "response.turn.interrupted",
  "turn_id": 1,
  "next_turn_id": 2,
  "seq_no": 1,
  "response_id": "response_..."
}
```

新问题推进轮次并打断旧轮。`response_id` 标识被打断的活跃段，没有活跃段时为 null。被打断的段落**不补发正常 `response.done`**。一场持续被打断的会话可能没有任何 response.done，因此最终用量必须以 session.done 兜底。

### 4.8 `response.done`

一段回答结束，不是会话终点。

```json
{
  "type": "response.done",
  "response_id": "response_...",
  "response_seq": 1,
  "turn_id": 1,
  "finish_reason": "stop",
  "boundary": "silence",
  "usage": {
    "vision_tokens": 145,
    "text_input_tokens": 78,
    "text_output_tokens": 10,
    "text_tokens": 88,
    "total_tokens": 233,
    "cumulative": {
      "vision_tokens": 145,
      "text_input_tokens": 78,
      "text_output_tokens": 10,
      "text_tokens": 88,
      "total_tokens": 233
    }
  }
}
```

模型 silence 结束的段使用 `boundary=silence`、`finish_reason=stop`。若后端自然结束时仍有活跃段，使用 `boundary=request_end`，透传模型结束原因（正常停止为 `stop`，额度耗尽为 `length`），随后发送 session.done。已在 silence 结算的段不重复结算。abort/故障不伪造正常回答结束。

### 4.9 `session.usage`

可选运行时观测通道，`include_usage=true` 时最多 1 Hz，不保证周期心跳或发送最后一份快照。它不是计量数据源。

```json
{
  "type": "session.usage",
  "encoder_tokens": 3536,
  "decoder_tokens": 329,
  "encoder_kv_tokens": 1547,
  "token_space_used": 3865,
  "context_limit": 131072,
  "context_remaining": 127207
}
```

| 字段 | 含义 |
| --- | --- |
| `encoder_tokens` | 历史视觉位置数 |
| `decoder_tokens` | 后端当前观测到的文本位置数 |
| `encoder_kv_tokens` | 当前驻留视觉 KV 位置数，可能因滑窗减少 |
| `token_space_used` | 历史上下文位置使用量，不是驻留显存或驻留 KV 总量 |
| `context_limit` | 实际配置的上下文容量 |
| `context_remaining` | `max(0, context_limit - token_space_used)` |

快照可能落后于结算水位，不得用它替代 response/session done 的准确计量。剩余容量也不保证下一条任意大小输入一定可接纳。

### 4.10 `session.done`

```json
{
  "type": "session.done",
  "session_id": "video_sess_...",
  "request_id": "video_req_...",
  "reason": "completed",
  "aborted": false,
  "usage": {
    "vision_tokens": 145,
    "text_input_tokens": 78,
    "text_output_tokens": 10,
    "text_tokens": 88,
    "total_tokens": 233
  }
}
```

`usage` 为整场会话的**累计终值**，口径与 `response.done.usage.cumulative` 完全一致。它包含最后一次 response.done 后的消耗。

| reason | 含义 |
| --- | --- |
| `completed` | 后端按正常停止/长度条件结束，通常是 final 输入后的结束 |
| `aborted` | 客户端请求中止 |
| `context_exhausted` | 上下文用尽 |
| `error` | 超时或服务端故障，此前已发送 error |

`aborted` 仅在 reason 为 aborted 时为 true。容量拒绝发生在会话创建之前，终态的精简格式见 §7.4。

### 4.11 `error`

```json
{
  "type": "error",
  "code": "invalid_request",
  "seq_no": 12,
  "message": "expected seq_no 1, received 12"
}
```

`code` 和 `message` 必填。错误由某条输入触发且其整数序号可识别时携带 `seq_no`；连接级错误、配置错误或缺失/不可解析序号时省略。不以字符串匹配 message 驱动客户端逻辑。

### 4.12 标识与轮次

- `turn_id` 由后端分配，初始为 0。模型处理新问题时推进；帧附带非空问题也会推进。单纯推帧、静默或同轮继续发言不推进。accepted 不表示模型已经推进轮次。
- 同一 `turn_id` 可以有多段回答。客户端不得把 turn_id 当作 response.done 的唯一键，也不能假定每个 turn 都有 done。
- `response_id` 在每段首次可见文字时生成。客户端以会话标识加 response_id 关联文本、打断与结算。
- `response_seq` 在会话内从 1 递增；被打断的段也占号，因此相邻 done 的序号可以跳号，不等同于必须连续接纳的输入 seq_no。
- 新连接产生新的会话、请求和回答标识；跨会话去重、记忆和用量聚合由调用方负责。

## 5. 用量与计量

### 5.1 Token 定义

| 字段 | 含义 |
| --- | --- |
| `vision_tokens` | 已提交的视觉 encoder 逻辑位置，包括视觉分隔位置；按实际输入计算，不是固定每帧 145 |
| `text_input_tokens` | 初始用户/系统提示词、问题、帧相关文本及实际模板等已提交输入位置 |
| `text_output_tokens` | 已确认生成的 token，包括文字、silence、EOS 和控制 token |
| `text_tokens` | text_input_tokens + text_output_tokens |
| `total_tokens` | vision_tokens + text_tokens |

计量按实际模型逻辑位置，不是只对用户字符串 tokenize，也不是图片字节、FLOPs 或驻留显存。生成 token 在后续步骤作为输入使用时不重复计入文本输入；失败未提交输入和丢弃的前瞻步骤不计。TP 不按 rank 数重复计量。

### 5.2 本段增量

```text
本段增量 = 本次 cumulative - 上次 response.done 的 cumulative
所有增量之和 = 最后一次 response.done 的 cumulative
```

对五个计量字段分别计算，首次结算减数为零。初始模板、回答间的帧、静默及被打断段的消耗归入下一次结算；末次回答后的消耗归入最终残差。增量和累计值同时返回，客户端应检查等式以发现丢失或重复事件。

### 5.3 累计与最终残差

`cumulative` 与 `session.done.usage` 采用**历史累计**口径：已提交的模型位置永久计入，视觉 KV 淘汰不扣减历史用量。

```text
最终残差 = session.done.usage - 已收到的 response.done 增量之和
```

消费方不得把 session.done 总值再与所有增量相加，否则会重复计费。没有任何 response.done 时，整场消耗都在最终总值中。reset/rollover 若通过新建连接实现，新连接单独起账，不自动承接旧账本。

所有示例数字仅说明字段和算术，不代表固定单帧开销或最低/最高计费值。

## 6. 限制与容量

| 项 | 当前默认 / 来源 |
| --- | --- |
| configure 期限 | 180 秒；以 session.created 通告为准 |
| 单帧上限 | 当前后端 32 MiB；以 session.configured.max_frame_bytes 为准，部署入口可能另有限制 |
| input_queue_capacity | 默认 4；每个会话已预留、尚未 processed 的输入额度，包括待收二进制的帧 |
| 二进制交付期限 | ready 后默认 10 秒；图像校验另外使用默认 10 秒处理期限 |
| parked 超时 | 底层默认 300 秒，仓库启动配置 3600 秒；以 session.configured 通告为准 |
| context_limit | 实际后端配置；仓库启动配置 131072 |
| 总会话容量 | 仓库启动配置 4；v1/v2 共用，不是各有 4 个 |
| 视觉滑窗 | 仓库启动配置保留最近 60 秒原始视觉 KV；不保证回收历史上下文位置 |

`input_queue_capacity` 不是并发帧握手数，也不是请求处理 batch 大小。额度满时新输入会等待可用额度；客户端必须持续读事件，遵守 ready/accepted 背压，避免大量堆积消息。

parked 超时针对模型进入静默并等待新输入的空闲请求，不是整场会话时长或 WebSocket 心跳期限。视觉滑窗不等同于文本记忆压缩；本接口不会自动创建 memory/ASR/TTS 或执行应用层 rollover。

视觉位置数取决于输入分辨率和实际预处理。不得用固定 145 token/帧推导所有视频的最大时长；应结合观测和实际容量规划输入。

## 7. 终止与错误

### 7.1 错误码

| code | 含义 | 会话是否继续 |
| --- | --- | --- |
| `invalid_request` | 客户端事件不合法，包括字段、序号或可恢复图像校验错误 | 继续 |
| `context_exhausted` | 上下文用尽，无法继续接纳输入 | 终止 |
| `session_timeout` | 配置、二进制接收/校验或 parked 空闲超时 | 终止 |
| `response_failed` | 服务端故障、输入队列溢出或最终计量失败等 | 终止 |
| `session_capacity_exceeded` | 连接接纳时没有空闲会话槽位 | 不建立模型会话 |

客户端**必须**把未知 `code` 按「终止」处理，以便将来新增错误码时无需同步改造。

### 7.2 终止保证与例外

正常完成或有序中止时，后端先确认在途步骤并冻结用量，再发送唯一 `session.done`。致命错误按 `error -> session.done -> close` 收尾；`session_timeout` 对应 `reason=error`。

最终计量确认默认最多等待 20 秒，网络发送和清理另有有界等待；20 秒不是整个关闭流程的总 SLA。客户端应立即停止推流，但必须继续接收终态。

> **最终用量不可用**：后端无法确认最终快照时，尽力发送 `response_failed` 并以 1011 关闭，不发送 `session.done`。断网或进程崩溃也可能没有终态。客户端必须将此类会话标记为“最终用量未知”，不得按零用量或完整结算处理。

### 7.3 可恢复拒绝

客户端事件不合法时，服务端回 error 并保持连接。可识别序号时带 seq_no；被拒帧另回 input.frame.rejected，使客户端结束握手等待。

拒绝另一条元数据不会清除仍有效的待接收帧。当前帧二进制被可恢复地拒绝时，释放其输入额度，可用同一序号重新发送元数据。重复二进制、输入堆积等致命情况按 §7.1 结束会话。

### 7.4 接纳失败

容量不足时，WebSocket 接受后直接返回如下事件并以 1013 关闭；无 session.created，尚无模型消耗，也未分配会话标识：

```json
{
  "type": "error",
  "code": "session_capacity_exceeded",
  "message": "video realtime service has no free session slot (capacity 4)"
}
```

```json
{
  "type": "session.done",
  "reason": "error",
  "usage": {
    "vision_tokens": 0,
    "text_input_tokens": 0,
    "text_output_tokens": 0,
    "text_tokens": 0,
    "total_tokens": 0
  }
}
```

鉴权失败在 WebSocket 接受之前拒绝，通常表现为握手 HTTP 403，不保证能收到 WebSocket error/session.done 事件。

## 8. 完整时序与客户端约定

```text
client                                      server
  |-------- WebSocket Upgrade -------------->|
  |<------- session.created -----------------| capabilities / limits
  |-------- session.configure -------------->|
  |<------- session.configured --------------| context_limit
  |<------- session.ready -------------------|
  |-------- input.frame (seq 0) ------------>|
  |<------- input.frame.ready ---------------|
  |-------- binary image ------------------->|
  |<------- input.frame.accepted ------------|
  |<------- input.frame.processed -----------|
  |<------- response.turn.silence -----------| no text: no done
  |-------- input.prompt (seq 1) ----------->|
  |<------- input.prompt.accepted -----------|
  |<------- interrupted / processed ---------| turn advances
  |<------- response.text.delta x N ---------| response A
  |<------- response.turn.silence -----------|
  |<------- response.done -------------------| settle A; stay connected
  |-------- input.frame (seq 2) ... -------->|
  |<------- response.text.delta x N ---------| response B, same turn
  |<------- response.turn.silence -----------|
  |<------- response.done -------------------| settle B
  |-------- input.prompt (seq 3, final) ---->|
  |<------- accepted / processed / output ---|
  |<------- response.done -------------------| only if a segment completes
  |<------- session.done --------------------| frozen total, residual included
  |<------- WebSocket Close 1000 ------------|
```

时序中的回答取决于模型，输入可以只产生静默；观测事件省略。客户端须按以下规则消费：持续读取所有事件；按 response_id 关联回答；收到 response.done 后保持连接；按累计值核对增量与最终残差；任何终态或异常断连均结束 pending 握手。旧 v1/Demo 客户端不得在不调整 done 处理逻辑的情况下改连此入口。
