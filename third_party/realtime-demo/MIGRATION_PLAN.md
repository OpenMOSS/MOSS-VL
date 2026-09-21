# 迁移方案：demo × sglang-omni realtime × pi_agent memory

把 board（`/inspire/hdd/project/video-understanding/public/personal/train/board/`）里验证过的 realtime 推理后端（sglang-omni）和新版 memory 管理（pi_agent decide/compact + 本地 4B + prefetch）整体迁移进本 demo 仓库。**只改 realtime 链路；offline chat 代码不动**（本部署形态下无 offline sglang 面，chat 页将不可用，属预期）。语音链路（SenseVoice ASR / TTS）沿用 demo 自带 API 路径，不改动。

## 0. 部署拓扑

> **现状备注（2026-09 修订）**：下方是最初方案的拓扑，落地形态已变更——**所有服务都在 GPU 节点本地拉起**（`start_demo.sh` 一键启动，详见 `docs/ops_runbook.md`），CPU 节点只作浏览器入口转发（经 rtunnel 回连 127.0.0.1:2222，`-R` 推 web :20941、`-D` 推 MiniMax 出口 :17890），不再依赖 CPU→GPU 的 10008 `-L` 转发。实际端口：gateway :8100（`.env.deploy` 的 `PORT=8100`）、pi_agent :38082、4B :38090、omni 实例 18500+i。

```
CPU 节点（本仓库）                          GPU 节点（127.0.0.1:10008 反向隧道）
├─ demo gateway  .venv  :8000              ├─ sglang-omni server × N（每卡 TP1，18500+i）
├─ React 前端    vite preview :20941       ├─ pi_agent        :38082（decide/compact）
└─ SenseVoice ASR / TTS（in-process，CPU） └─ Qwen3-4B sglang :38090（decide/compact 后端）
        │                                            ▲
        └── SSH 本地转发（ssh -fN -L ... -p 10008）──┘
```

- 副本形态：8 卡都可部署，**默认 TP1 × 8 实例**，启动脚本参数化 GPU 列表与 TP_SIZE（预留多 TP）；gateway 侧 pool 按 URL 列表逐个建 replica，与 TP 无关。
- 前置核实（执行第 0 步）：通读 sglang-omni 最新源码（`/inspire/qb-ilm/.../MOSS-VL-Realtime-sglang/sglang-omni-main/`，上游近期大改），确认 `/v1/video/realtime` 握手序列、`session.configure` 字段全集、健康/管理端点、是否暴露会话 token 统计。所有适配以最新协议为准。

## 1. 改动一：新 VLM adapter（realtime ↔ sglang-omni）

新建 `server/adapters/vlm/moss_vl_sglang_omni/`：

### 1.1 `client.py` — WS 客户端（移植 board `backend/sglang_realtime_worker.py`）

- 握手：`session.created` → 发 `session.configure`（prompt/system_prompt/max_new_tokens/max_tokens_per_turn/temperature/top_p/input_queue_capacity）→ 等 `session.ready`。
- 上行：`input.frame{seq_no,timestamp,prompt?,mime_type}`（等 `input.frame.ready` 后发二进制 JPEG 体）、`input.prompt{seq_no,prompt}`、`session.abort`。
- 下行事件：`input.[frame|prompt].accepted/processed`（credit 回收）、`response.created`、`response.text.delta{delta,turn_id}`、`response.turn.interrupted`、`response.turn.silence`、`response.done`、`session.done`、`error`。
- 线程模型照 board：一个 web 线程包同步收发（daemon 收线程 + 有界发送队列），不污染 gateway 的 asyncio loop。

### 1.2 `session.py` — `SglangOmniSession` 实现 `VlmRealtimeSession`（`server/adapters/base.py:87-103`）

关键设计：**把 sglang-omni 的结构化事件翻译回 demo orchestrator 已有的控制 token 文本流**，使 `server/session/orchestrator.py` 的 drain loop / 控制 token 解析（orchestrator.py:47-58）**零改动**，rollover 的 idle 触发点（收到 `<|silence|>`，orchestrator.py:815）自动保留：

| sglang-omni 事件 | 映射成的 chunk |
|---|---|
| `response.created` / 首个 delta | `<|round_start|>` + delta |
| `response.text.delta` | 原文 |
| `response.turn.silence` | `<|silence|>` |
| `response.turn.interrupted` | `<|round_end|>`+`<|round_start|>` |
| `response.done` | `<|round_end|>` |
| `session.done` / 连接断 | `batch.active=False`（走现有 `_mark_vlm_dead`） |

接口逐一映射：

- `put_frame(jpeg, ts, size)` → **credit 门**：在途输入数（等待 `input.*.processed` 回执的 seq_no 集合）≥ `SGLANG_OMNI_INPUT_QUEUE_CAPACITY`(默认 4) 时，Condition 最多等 `SGLANG_OMNI_INPUT_DROP_WAIT_SECONDS`(0.5s)，超时丢帧（`frames_dropped+1`，元数据未发出、协议安全）。**纯帧才过 credit 门；带 prompt 的帧和纯 prompt 永不丢**（board 已验证：容量小时 prompt 会被慢帧堵死）。
- `put_prompt(text)` → `input.prompt`；`put_prompt_frame(prompt, jpeg, ts, drop_pending)` → `input.frame` 带 prompt 字段（drop_pending 语义由服务端处理，不在 adapter 内丢 prompt）。
- `request_turn_end()` → `session.abort`（barge-in 软打断；下行 `response.turn.interrupted`）。
- `poll_output(timeout, max_items)` → 从入站队列聚合成 `OutputBatch(active, chunks, chunk_events, status)`。
- `status()` → 透出 frames_received/dropped/consumed、outputs_emitted、active、以及 **`text_tokens`**：上游事件若带 token 统计直接用；否则 session 内做 TokenMirror 式镜像（venv 内有 transformers 则用 ckpt tokenizer 精确计数，挂掉退化为 字符数//2）。
- `stop(timeout)` → `session.abort` + 关 WS + 等 `session.done`。
- 时间戳单调性：`ts = max(requested_ts, last_ts)`。

### 1.3 `pool.py` — `SglangOmniPool`

- 外观对齐 `VlmReplicaPool`（`online_pool.py`）：`capacity` / `busy` / `is_loaded` / `status` / `start_realtime_session(**params)` / `load()`(远端探活)。
- replica = 每个 `SGLANG_OMNI_URLS` 条目一个；选副本 = 锁内取最低序号 READY → 置 BUSY → 再建会话，失败回 READY；无空位**复用** `online_pool.NoFreeReplica`（`routers/sessions.py:19,127` 的 409+Retry-After 契约不动）。
- `start_realtime_session(**params)` 接受 `routers/sessions.py:35-52` 的全量 kwargs（HF 专属字段 min_pixels/video_fps 等直接忽略），把 prompt/system_prompt/采样参数映射进 `session.configure`；**rollover reseat 传入的 `prefill_messages`（JSON 字符串）在 adapter 内转成 `(system_prompt, prompt)`**：system 消息 → system_prompt，tail user/assistant 消息 → 渲染成 prompt 文本块（demo 的 `decode_prefill_messages` 校验逻辑复用）。
- **不暴露 `set_replica_health`**：`app.py:68` 的 hasattr 判定自动跳过本地 `VlmWorkerSupervisor`，不 spawn 任何本地 worker 进程。
- `generate_stream` 抛 NotSupported（offline chat 由 `routers/chat.py` 的 offline 面服务，本部署无 offline 面 → chat 页返回不可用，代码不改）。
- 自愈：`session_capacity_exceeded` → 等 5s 重试一次，仍失败标该 replica DOWN；DOWN 副本后台探活恢复；WS 断 → `active=False` → 走 orchestrator 现有 vlm_dead 路径（emit `error[vlm_stopped]`）。

### 1.4 接线点

- `server/adapters/registry.py:78-86` `build_vlm` 加分支：`VLM_DEPLOY=sglang_omni` → `SglangOmniPool(settings)`（不读 plan.workers）。
- `server/config.py` 新增字段（见 §3）。
- 前端**零改动**（协议、事件、`memory.recalled` 全是 demo 原生）。

## 2. 改动二：memory 接入 pi_agent + 新版策略

保留 demo 的 `server/memory/` 骨架（store/writer/retrieval/inject/rollover，它比我们多精确 token 计数和 reseat 不换 session_id——alias 映射那套不需要带）。替换/增强四个点：

1. **`server/memory/pi_client.py`（新）**：从 board `backend/memory/pi_client.py` 移植——stdlib HTTP `POST {MEMORY_PI_URL}/decide`（`{conversation_id, recent_turns, pending_user_text}` → `{retrieve, query, reason}`，超时 8s）、`POST /compact`（`{conversation_id, journal}` → `{summary, pins}`，超时 120s）；退避重试 3 次、4xx 硬失败、warning 60s 节流、`reachable()` 探活缓存。
2. **decide 门**：新增 `MEMORY_DECISION_MODE = vector | llm | hybrid`（默认 hybrid：本地向量 raw 分先预筛，过 `MEMORY_RETRIEVAL_PREFILTER_SCORE` 才调 /decide；回顾性问句"刚才/之前/记得"用放宽门）。接在 `MemorySession.recall_for_turn` 之前；`recent_turns` 取自 `MemoryStore.recent`。
3. **compact provider**：`RolloverManager._summarize`（`memory/rollover.py:292`）加 `memory_summary_provider == "pi"` 分支：journal 序列化后 `POST /compact`，`summary` 进摘要段、`pins` 进置顶段（对齐 board 的 system+RECALL_FORMAT_NOTE+pins+摘要 结构）。原 `"offline"` 分支原样保留。
4. **rollover 预取（异步提前量）**：新增 `MEMORY_ROLLOVER_PREFETCH_RATIO=0.6`——tokens 越过 `idle_tokens × ratio` 时后台线程预跑 /compact 缓存结果；`build_prefix` 时优先取缓存（进行中最多 join 30s），失败/为空照旧降级 verbatim tail。触发点挂进现有 `should_rollover` 评估路径（orchestrator 的 1Hz status tick + idle 点）。
5. **入库卫生**（对齐 board 规则）：
   - writer 加 utterance 去重：每 conversation 记 `(role, 白空格归一化文本)` 集合，重复直接丢；
   - `note_assistant_turn` 增加 `commit` 参数：被 barge-in/`<|eot_id|>` 打断的回合 `commit=False`——留在 recent_turns 供 compact 上下文，但不写长期记忆。

## 3. 改动三：config / 环境 / 部署

### 3.1 config.py 新增字段

```
VLM_DEPLOY 增加取值 sglang_omni
SGLANG_OMNI_URLS="http://127.0.0.1:18500,..."   # 每 URL 一个 replica
SGLANG_OMNI_INPUT_QUEUE_CAPACITY=4
SGLANG_OMNI_INPUT_DROP_WAIT_SECONDS=0.5
SGLANG_OMNI_CONNECT_TIMEOUT_S=10
SGLANG_OMNI_HEALTH_INTERVAL_S=10
MEMORY_PI_URL=http://127.0.0.1:38082
MEMORY_PI_DECIDE_TIMEOUT_S=8 / MEMORY_PI_COMPACT_TIMEOUT_S=120
MEMORY_DECISION_MODE=hybrid
MEMORY_ROLLOVER_PREFETCH_RATIO=0.6
MEMORY_SUMMARY_PROVIDER 增加 "pi"
```

改完跑 `scripts/dev/check_env.py --write` 重新生成 `env_manifest.sh` 与 `.env.deploy.example`（不跑则 demo.sh 的 tmux window 拿不到新变量）。

### 3.2 CPU 节点环境

- `.venv`：`python3 -m venv .venv && pip install -r requirements.txt`（**无代理直连**；torch 装 CPU wheel 省盘，funasr/SenseVoice CPU 可跑；如遇包拉取问题再逐个换源处理）。前端：`npm install && npm run build`。
- ~~SSH 转发（CPU 节点上）~~【已废弃】原方案的 CPU→GPU `-L` 转发（`ssh -fN -L ... -p 10008`）挂在 rtunnel 上、rtunnel 一断就死，已废弃。现行形态：所有服务在 GPU 节点本地拉起（`start_demo.sh`），只经 rtunnel 回连（127.0.0.1:2222）向 CPU 推 `-R 20941` 入口转发；pi_agent :38082、4B :38090、omni 18500+i 均为 GPU 节点本地端口。
- `.env.deploy` 样例：`VLM_DEPLOY=sglang_omni` / `SGLANG_OMNI_URLS=...` / `OFFLINE_PROVIDER=none` / `MEMORY_ENABLED=1` / `MEMORY_SUMMARY_PROVIDER=pi` / `DEMO_SKIP_GPU=1`（CPU 节点跳过 GPU 探测）。

### 3.3 GPU 节点启动脚本（新建 `scripts/gpu/`）

- `start_sglang_omni.sh`：参数化 `GPUS` 与 `TP_SIZE`（默认 8 卡 TP1 → 8 实例，端口 18500+i；TP=2 → 4 实例 ……），ssh 到 GPU 节点逐实例拉起 `examples/run_moss_vl_realtime_server.py`，带 `REALTIME_FRAME_WINDOW_*` 等 env（帧窗口由服务端内化）。
- `start_pi_agent.sh`：拉起 pi_agent + Qwen3-4B（`DECIDE_LOCAL=1 / COMPACT_LOCAL=1 / AIGW_*` 指向 38090，env 组复用 board `start.sh:127-222` 的成熟配置）。
- **不改 sglang-omni 上游仓库**；pi_agent 是我们自己的可改。

## 4. 验证计划

1. **单测（CPU 节点，全 fake，无 GPU）**：`server/tests/fakes.py` 加 `FakeSglangOmniServer`（asyncio WS 服务端说 omni 协议）+ 新增测试：credit 丢帧/事件→控制 token 映射/打断/reseat（prefill_messages→configure 映射）/rollover 预取/decide 三门模式/compact pi provider/utterance 去重/打断不入库。风格跟随现有 script 式测试。回归：现有 `server/tests/` 全套保持绿。
2. **GPU 联调**：先单实例（1 卡）跑通 `scripts/e2e_session.py`，再铺 8 实例。
3. **浏览器验收清单**：initial prompt 立即可见；长聊帧不累积、不卡死（status 里 frames_dropped 有计数、无无限增长）；rollover 后跨会话记忆召回（数字考题）；barge-in 即时打断；字幕/TTS 正常。

## 5. 执行顺序

0. 读 sglang-omni 最新源码，锁定协议字段（若有出入，以源码为准修订 adapter 细节）
1. CPU 环境（.venv / 前端 build / SSH 转发）
2. adapter + pool + registry + config（§1、§3.1）
3. memory 四件套（§2）
4. 单测 + 回归（§4.1）
5. GPU 联调 + 浏览器验收（§4.2、§4.3）
6. 把本方案落一份 `MIGRATION_PLAN.md` 到仓库根（plan mode 限制下当前无法直接写入）

## 6. 明确不做

- 不改 offline chat 任何代码（本部署下 chat 页不可用，属预期）。
- 不改 ASR/TTS 链路（沿用 demo 自带 API 路径，CPU 本地跑）。
- 不改前端。
- 不改 sglang-omni 上游仓库代码。
- 不带 board 的 session_id alias 映射（demo 的 reseat 不换 session_id，天然无感）。
