# 部署运维与历史环境参考

新安装以 [README](../README_zh.md) 的 `bootstrap.sh` / `scripts/repro/run.py` 为准：
后端/API/网页为 18500/18501/18502，pi-agent/4B/TTS 为 18503/18504/18505。
健康检查：`curl --fail http://127.0.0.1:18501/api/status`。
该入口不读取 `.env.deploy`，不要求内部 SSH 转发或 MiniMax 凭据。

下文是 `start_demo.sh` 的特定历史环境预案，包含 8100、20941、38082、38090 及内部转发。
这些不是推荐安装的默认值，不可与推荐命令混用。8000 属于另行配置的手动部署示例。
端口差异不是 HTTP 请求协议变更。可选 VL API v2 是另一个监听端口，不能替换 Demo 的 native 后端地址。

> 面向运维/值班。依据：`GATEWAY_PLAN.md` §0 决策表与 P8 节、`start_demo.sh` / `stop_demo.sh` /
> `deploy.conf`、`server/gateway/` 包、`docs/gateway_alerting.md`（告警规则，本文档引用，不重复定义）。
> 使用历史环境前需按实际配置核对端口和路径；未确认项保留「待确认」。

## 1. 部署拓扑

```
客户端（浏览器 / 平台侧调用方）
  │
  ▼
平台网关 —— 鉴权 / 并发额度 / 限流 / 白名单（平台职责，本层不实现，见 GATEWAY_PLAN §0）
  │
  ▼
本适配网关（GPU 节点，uvicorn 单进程，:8100，.env.deploy 的 PORT=8100）
  │   · 透传平面：REST /v1/realtime/sessions 族、WSS /v1/realtime?ws_token=、
  │     GET /v1/realtime/health、GET /v1/realtime/metrics、GET /v1/models
  │   · demo 平面：/api/*（session_ws 翻译协议链路，与透传平面同进程并存）
  │
  ├─► sglang-omni 推理实例池（内网明文，:18500 起，端口 = OMNI_PORT_BASE + 序号，
  │     一卡一实例 TP1 或 OMNI_TP_SIZE 多卡一实例；仅内网暴露）
  ├─► 4B decide/compact 后端（Qwen3-4B sglang，:38090，memory 子系统共享）
  ├─► pi_agent（:38082，memory decide/compact 编排）
  └─► MiniMax 云 TTS 出口（GPU 本地 SOCKS5 :17890，经 rtunnel 回连 127.0.0.1:2222 出网）

CPU 节点转发入口（不承担任何服务进程）：
  CPU 127.0.0.1:20941 → GPU :20941（web，vite preview；nat2 /proxy/20941/ 浏览器入口）
```

要点：

- gateway 与 omni 实例同 GPU 节点；**一期单副本**（GATEWAY_PLAN §0），gateway 重启 = 会话全断。
- 薄网关只做会话管理、路由、透传、计量、保活；客户鉴权/额度/限流/白名单由平台负责。故障 memory 接力仅属于 Demo 平面。最新限制与配置见 [网关契约](gateway_contract.md)。
- `start_demo.sh` / `stop_demo.sh` **只能在 GPU 节点执行**（脚本自检 `nvidia-smi`，CPU 节点直接拒绝）。

## 2. 启动 / 停止 / 状态

### 2.1 一键启动（GPU 节点）

```bash
cd <repo>
./start_demo.sh
```

五步顺序（MiniMax 出口先于 gateway，保证 minimax lane 启动探测就绪）：

1. **sglang-omni 推理实例**：`OMNI_GPUS` 每张卡（÷`OMNI_TP_SIZE`）一个实例，:18500 起；
   逐实例探 `/health`，全部 healthy 则跳过；并把实例 URL 自动同步进 `.env.deploy` 的 `SGLANG_OMNI_URLS`。
2. **pi_agent + 4B 后端**：:38082 / :38090，已 healthy 则跳过。
3. **MiniMax 云 TTS 出口**：`ssh -fN -D 127.0.0.1:17890`（经 2222 回连），
   建好后用 `curl -x socks5h://127.0.0.1:17890 https://api.minimaxi.com/v1/t2a_v2` 探链，
   返回 HTTP 40x 即视为链路 OK。
4. **gateway + TTS sidecar + web**：`scripts/deploy/demo.sh up`（gateway :8100 + web :20941，tmux 单会话多窗口）。
5. **入口转发**：`ssh -fN -R 127.0.0.1:20941:127.0.0.1:20941`（经 2222 回连 CPU），
   并从 rtunnel 进程命令行抠出公网 URL 打印。转发失败不影响服务本体，只影响浏览器入口。

### 2.2 一键停止（GPU 节点）

```bash
./stop_demo.sh
```

按 demo.sh down → omni 实例（按 `logs/sglang_omni/omni_*_p*.pid`）→ memory 整套
（pi_agent :38082 + 4B :38090）→ 两条转发（-R 20941 / -D 17890）顺序关停。

### 2.3 状态查看

```bash
bash scripts/deploy/demo.sh status    # tmux 会话 + api/web 健康一览
bash scripts/deploy/demo.sh doctor    # 端口监听 / 野进程 / 隧道 / boot-stamp（排查转发指向陈旧 pod）
bash scripts/deploy/demo.sh logs api  # 跟随日志（api|web|build|backend|workers|tts|sglang）
curl -s http://127.0.0.1:8100/v1/realtime/health   # 透传平面池水位（instances/active_sessions/capacity/replicas）
curl -s http://127.0.0.1:8100/v1/realtime/metrics  # P4 指标 + pool 摘要
curl -s http://127.0.0.1:8100/api/status           # demo 平面状态
```

### 2.4 deploy.conf 布局键

优先级：命令行 env > deploy.conf > 脚本内建默认值。持久布局改 `deploy.conf`，别改脚本。

| 键 | 默认 | 说明 |
|---|---|---|
| `OMNI_GPUS` | `0` | 用哪几张卡铺 omni 实例；`0,2,3` → 3 实例 :18500/:18501/:18502 |
| `OMNI_TP_SIZE` | `1` | TP 模式：`OMNI_GPUS=0,1 OMNI_TP_SIZE=2` → 1 个 TP2 实例 :18500（\|GPUS\| 必须能被整除） |
| `OMNI_PORT_BASE` | `18500` | 实例端口基址，端口 = 基址 + 序号 |
| `MAX_RUNNING_REQUESTS` | `1` | 远端 omni 实例的 `--max-running-requests`（单实例会话数 N） |
| `SGLANG_OMNI_SESSIONS_PER_REPLICA` | `1` | 网关侧每副本 slot 数；**必须与 MAX_RUNNING_REQUESTS 一致**，两值由 deploy.conf 一起 export |
| `DECIDE_LLM_GPU` | `1` | memory 4B decide/compact 放哪张卡 |
| `DECIDE_LLM_MEM_FRAC` | `0.8` | 4B 显存占比（独占卡时给大，KV 池更大并发更稳） |

### 2.5 强制重启开关

| 开关 | 作用 |
|---|---|
| `FORCE_OMNI=1 ./start_demo.sh` | 即使全部 healthy 也强制重启 omni 实例 |
| `FORCE_PI=1 ./start_demo.sh` | 即使 healthy 也强制重启 pi_agent + 4B |
| `START_4B=0` | pi_agent 照起但不拉 4B（decide/compact 会失败，仅调试用途） |
| `CPU_PORT=xxxxx` | 换 CPU 侧入口端口（默认与 GPU 侧同号 20941） |

单进程热重启：`bash scripts/deploy/demo.sh restart api`（gateway 与 demo 后端同进程，restart api
会同时断掉两个平面的所有会话）。

## 3. 容量规划

- **容量 = 实例数 × MAX_RUNNING_REQUESTS**（= `SGLANG_OMNI_SESSIONS_PER_REPLICA` 对应槽位总和，
  见 `server/gateway/pool.py` 的 slot 记账）。网关侧 `GET /v1/realtime/health` 的
  `capacity` / `active_sessions` 就是这套口径。
- **KV 规划**：单卡 KV 池 ÷ N = 每会话安全上下文（GATEWAY_PLAN P2）。omni 启动预检放不下即拒绝启动；
  运行期 KV 紧张按「最重会话中止」降级，透传 `response_failed`。
- **调 N 前必须先压测**：安全 N 由 P5 压测定，未定前保持 1（deploy.conf 注释口径）。
  用 `tools/stress_realtime.py`：

  ```bash
  # 走网关平面（REST 建会话 → WSS 推帧）
  .venv/bin/python tools/stress_realtime.py \
      --mode gateway --url http://127.0.0.1:8100 \
      --ramp 1,2,4,8 --duration 60 --fps 2 --video /path/to/frames_dir \
      --prompt-interval 10 --gpu-sampling --out reports/stress_gateway
  ```

  口径（§7，与 tools/stress_realtime.py docstring 一致）：对外并发/SLA **只认满足时延与稳定性
  目标下的「稳定并发」**，不接瞬时峰值。压测报告需覆盖 §7 八维度（硬件/视频输入/会话负载/
  并发/时延 P50/P95/P99/稳定性/资源/容量策略）。

## 4. 故障预案

### 4.1 omni 实例宕机

- **现象**：实例进程退出 / `/health` 不答；其上会话中断。
- **自动行为**：
  - demo 平面（/api/*）：`server/session/orchestrator.py` 的 `_mark_vlm_dead()` 触发
    `_reseat_vlm(trigger="failure")` —— RolloverManager `build_prefix()`（摘要 + facts + 原文尾巴）
    → 健康实例新建 omni 会话 → memory 前缀 prefill 接续，**语义上下文保留、原始帧历史丢失**；
    prefix 未就绪回退 `response_failed`。
  - 透传平面（/v1/realtime/*）：该平面无重连语义（omni 无法恢复会话），会话被
    **1011** 关闭终结（`WS_CLOSE_OMNI_DEAD`），对账落 `end_reason=omni_dead`；
    副本被 `release(transport_dead=True)` 隔离为 DOWN，prober 按
    `SGLANG_OMNI_HEALTH_INTERVAL_S`（默认 10s）周期重探，`/health` 恢复即自动解除隔离。
- **人工处置**：看 `logs/sglang_omni/omni_gpu{gpu}_p{port}.log` 定位死因；重启实例
  （重跑 `./start_demo.sh`，健康门会只拉起死掉的实例）。网关客户端需重新建会话；仅 Demo 的会话接力按 Demo 验收口径排查。
- **客户端影响**：demo 平面基本无感（接力期间有停顿）；透传平面收到 1011，需走
  REST 重建会话（新建 → 拿新 ws_token → 重新 attach）。

### 4.2 gateway 重启

- **现象**：8100 进程重启（发版、`demo.sh restart api`、OOM 等）。
- **自动行为**：无 —— 一期单副本语义（GATEWAY_PLAN §0）：**所有会话全断**，
  对账批量落 `end_reason=shutdown`。
- **人工处置**：`bash scripts/deploy/demo.sh up`（或 `restart api`）；起后确认
  `/v1/realtime/health` 池水位正常。
- **客户端影响**：所有会话需重连新建（透传平面 REST 重建；demo 平面重开页面会话）。
  注意 restart api 会同时断 demo 平面。

### 4.3 4B 后端宕机（:38090）

本节仅影响 Demo memory 链路；薄网关 `/v1/realtime` 不依赖 4B 摘要/记忆编排。

- **现象**：`curl http://127.0.0.1:38090/health` 不通；memory decide/compact 全部失败。
- **自动行为**：memory 子系统降级 —— **会话继续，但无记忆能力**（decide/compact 调用失败被吞，
  不阻塞推理链路）；4B 挂也意味着**故障接力拿不到 memory 前缀**，omni 宕机场景退化为
  `response_failed`。
- **人工处置**：看 `logs/pi_agent/decide_llm.log`；恢复执行
  `DECIDE_LLM_GPU=<卡> bash scripts/gpu/start_pi_agent.sh`（或直接 `./start_demo.sh`，
  健康门自动只补死掉的组件）。pi_agent 自身日志在 `logs/pi_agent/pi_agent_38082.log`。
- **客户端影响**：推理/语音正常，长期记忆与故障接力能力缺失。

### 4.4 MiniMax 出口断（SOCKS5 :17890）

- **现象**：选 minimax TTS 的会话无声；`curl -x socks5h://127.0.0.1:17890
  https://api.minimaxi.com/v1/t2a_v2` 不返回 40x。
- **自动行为**：gateway 启动时用 `MINIMAX_PROXY`（.env.deploy：`socks5://127.0.0.1:17890`）
  探测 minimax lane，不通则 lane 卡在 not-ready；会话创建时对 not-ready 的 lane 有一次惰性重试。
  降级口径两处不一致：start_demo.sh 注释称「选/默认 minimax 会回落本地 nano」，而
  `server/routers/sessions.py` 的运行时代码是「lane 不就绪 → 该会话只发字幕、无音频」——
  **具体运行时行为待确认**（见 §9）。本地 nano TTS（:18100+）不受此故障影响。
- **人工处置**：查 2222 回连是否存活（rtunnel 进程）、重建转发
  `ssh -fN -D 127.0.0.1:17890 root@127.0.0.1 -p 2222`，或重跑 `./start_demo.sh` 第 3 步。
- **客户端影响**：选 minimax 的会话无云端音质 TTS（无声或回落 nano，待确认）。

### 4.5 副本假满（capacity 满标记）

- **现象**：某副本被标 `capacity_limited`（omni 握手回 `session_capacity_exceeded`，
  即服务端被网关不追踪的会话占满），acquire 跳过它，可用容量看似收缩，但实例其实健康。
- **自动行为**：prober 周期重探被标记的副本，`/health` 应答正常即清除满标记
  （`server/gateway/pool.py` `mark_full` → prober 清标；误清会在下一次容量拒绝时自愈）。
- **人工处置**：一般无需干预；若长期不清，查该实例上是否有非网关来源的残留会话。
- **客户端影响**：容量水位暂时下降，新建会话可能 503 `session_capacity_exceeded`。

## 5. 扩容流程

1. **加卡**：确认新卡空闲（`nvidia-smi`）。
2. **改布局**：`deploy.conf` 的 `OMNI_GPUS` 加卡（如 `0,1` → `0,1,2`）；
   如需调 N，同步改 `MAX_RUNNING_REQUESTS` 与 `SGLANG_OMNI_SESSIONS_PER_REPLICA`（两值必须一致，
   且 N 必须在 P5 压测安全值内）。
3. **拉起**：`./start_demo.sh` —— 健康门自动跳过已健康实例只铺新实例，并自动把新的
   `SGLANG_OMNI_URLS` 写进 `.env.deploy`。**注意**：第 4 步 `demo.sh up` 会重启 gateway
   进程，存量会话全断（一期单副本语义），请选择低峰窗口执行。
4. **灰量验证**：
   - `GET /v1/realtime/health` 确认 `instances` 与 `capacity` 涨到新值、新副本 READY；
   - 建几条会话验证路由命中新实例（`GET /v1/realtime/sessions/{id}` 看 `replica` 字段）；
   - 观察 `docs/gateway_alerting.md` 规则 1/2 指标无异常后放量。

## 6. 回滚

网关层（`server/gateway/`）与 omni 推理实例解耦，**回滚只动网关，不动 omni**：

1. `git checkout <上一稳定 commit> -- server/gateway/ server/config.py`
   （或整仓回退到上一稳定 tag/commit，并核对配置字段兼容性）。
2. `bash scripts/deploy/demo.sh restart api`（重启 8100 进程加载回滚后代码）。
3. 验证：`GET /v1/realtime/health` 正常 + 走通「建会话 → 推帧 → delta → 销毁」闭环。

omni 实例、pi_agent、4B、转发均不受网关回滚影响，无需操作。回滚会断存量会话（同 §4.2）。

## 7. 监控对接

- **指标抓取**：`GET http://<gpu-node>:8100/v1/realtime/metrics`，暴露 gauges
  （`gateway_active_sessions` / `gateway_replica_slots_total` / `gateway_replica_slots_used` /
  `gateway_replicas_down`）+ counters（`gateway_sessions_created_total` /
  `gateway_frames_accepted_total` / `gateway_text_chars_total` / `gateway_errors_total{code=...}` /
  `gateway_abnormal_disconnects_total` / `gateway_attach_timeouts_total`）+ `pool` 摘要。
- **对账**：每次会话终结/reset 追加一条 JSONL 到 `{data_dir}/gateway_usage.jsonl`
  （默认 `data/gateway_usage.jsonl`，可用 `GATEWAY_USAGE_LOG` 覆盖），字段含
  `trace_id / session_id / request_id / model / replica_url / duration_s / frames_accepted /
  prompts / text_deltas / text_chars / end_reason`；`trace_id` 为平台侧对账主键。
  计量写失败只记日志不影响会话。
- **告警规则**：见 `docs/gateway_alerting.md`（副本不可用 / 容量水位 >0.8 / 异常断连率突增 /
  attach 超时 / 错误码突增，含阈值、级别、处置与触发测试记录）。

## 8. 演练记录

> 三项演练均**待 P5 压测窗口执行后回填**。执行窗口需选低峰（演练 2/3 会断存量会话）。

| 日期 | 演练项 | 操作 | 预期 | 实际结果 | 结论 |
|---|---|---|---|---|---|
| 待回填 | kill 实例接力 | `kill <omni pid>`（pid 见 `logs/sglang_omni/*.pid`） | demo 平面 memory 接力恢复（语义上下文保留、原始帧历史丢失）；透传平面 1011 + 副本隔离 + prober 自愈 | | |
| 待回填 | gateway 重启 | `demo.sh restart api` | 所有会话断开，对账落 `end_reason=shutdown`，客户端重连新建恢复 | | |
| 待回填 | 扩容 | `deploy.conf` 加 `OMNI_GPUS` → `start_demo.sh` | 健康门跳过健康实例、只铺新实例；`capacity` 上涨；灰量验证通过 | | |

## 9. 待确认项

1. **MiniMax 出口断时的运行时降级行为**：`start_demo.sh` 注释口径是「回落本地 nano」，
   `server/routers/sessions.py` 代码口径是「lane 不就绪 → 该会话仅字幕无音频」。需实测确认。
2. **平台网关对接路径**：平台 → 本网关是直连 GPU 节点 :8100，还是经 CPU 转发入口 / 平台自有
   隧道，待与平台侧确认（影响 WSS `/v1/realtime` 的对外地址与 `ws_url` 下发形态）。
3. **gateway 独立回滚粒度**：gateway 与 demo 后端同进程（8100），`demo.sh restart api`
   会同时断两个平面；若线上只跑透传平面，是否拆进程待确认。
