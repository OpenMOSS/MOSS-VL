# Gateway 平面告警规则（P4）

> 依据：GATEWAY_PLAN.md §2-P4。指标由 `GET /v1/realtime/metrics` 暴露
> （`server/gateway/metrics.py`，gauges + counters + pool 摘要）。
> 本表是评审文档上线准入项「监控和告警已生效」的对应交付：每条规则给出
> 指标、阈值、级别、含义与处置，触发条件→指标变化的自动化测试记录见
> `server/tests/test_gateway_metrics.py`。

## 指标清单

Gauges（瞬时值）：

| 指标 | 含义 |
|---|---|
| `gateway_active_sessions` | 网关活跃会话数（创建 +1 / 销毁 -1） |
| `gateway_replica_slots_total` | 副本总槽位（实例数 × max_running_requests） |
| `gateway_replica_slots_used` | 已占用槽位 |
| `gateway_replicas_down` | DOWN（探活失败/传输异常被隔离）的副本数 |

Counters（单调递增）：

| 指标 | 含义 |
|---|---|
| `gateway_sessions_created_total` | 累计创建会话数 |
| `gateway_frames_accepted_total` | 累计 accepted 输入帧数 |
| `gateway_text_chars_total` | 累计输出字符数（网关无 tokenizer，字符数近似输出 token） |
| `gateway_errors_total{code=...}` | 按错误码分：omni error 事件透传 + 网关自身错误码（ws_token_invalid/expired、session_not_found、session_already_attached、session_capacity_exceeded 等） |
| `gateway_abnormal_disconnects_total` | 异常断连（omni 传输死亡导致的会话终结） |
| `gateway_attach_timeouts_total` | 建会话后未在窗口内 attach 被 janitor GC 的次数 |

## 告警规则

| # | 规则 | 阈值 | 级别 | 含义 | 处置 |
|---|---|---|---|---|---|
| 1 | 副本不可用 | `gateway_replicas_down > 0` 持续 1min | **critical** | 实例未通过启动/周期探活或被传输错误隔离，可用容量收缩 | 查实例进程/GPU与探活；恢复服务后由客户端新建会话，网关不做 memory 接力 |
| 2 | 容量水位高 | `gateway_replica_slots_used / gateway_replica_slots_total > 0.8` 持续 5min | **warning** | 会话容量接近打满，新建会话将 503（session_capacity_exceeded） | 评估扩容（加实例/调 max_running_requests，需 P5 安全 N 内）；核对是否有泄漏会话（对照 `gateway_active_sessions` 与平台侧） |
| 3 | 异常断连率突增 | `rate(gateway_abnormal_disconnects_total[5m]) > 0.1/s`（或 5min 增量 > 5） | **critical** | omni 传输成片死亡，会话被 1011 终结 | 与规则 1 联动查日志；客户端重新 REST 建会话，不等待网关接力 |
| 4 | attach 超时 | `increase(gateway_attach_timeouts_total[5m]) > 0` | **warning** | 平台建了会话但客户端没来连（token 过期/客户端失败/链路问题） | 查平台侧建连链路；偶发可忽略，持续出现查 `gateway_ws_token_ttl_s` 与 `gateway_attach_timeout_s` 配置 |
| 5 | 错误码突增 | `rate(gateway_errors_total[5m])` 按 code 环比突增（如 >3x 基线） | **warning** | 某类错误集中爆发：透传码（response_failed 等）指向推理侧；网关码（ws_token_invalid 等）指向接入侧 | 按 code 分流：透传码查 omni，网关码查平台接入与 token 发放 |

## 触发测试记录

「触发条件 → 指标变化」的自动化验证在 `server/tests/test_gateway_metrics.py`：

| 规则 | 测试 | 触发方式 → 断言 |
|---|---|---|
| 1 副本不可用 | `test_metrics_omni_death_alerts` | kill fake omni 的 WS → 副本被隔离 → `gateway_replicas_down == 1` |
| 3 异常断连 | `test_metrics_omni_death_alerts` | 同上 → `gateway_abnormal_disconnects_total +1`，对账记录 `end_reason=omni_dead` |
| 5 错误码 | `test_metrics_error_event_passthrough_counted` | fake omni 发 `error{code:response_failed}` → 透传且 `gateway_errors_total{response_failed} +1` |
| 4 attach 超时 | `test_metrics_janitor_attach_timeout` | 建会话不 attach → janitor GC → `gateway_attach_timeouts_total +1`，对账记录 `end_reason=attach_timeout` |
| 2 容量水位 | `test_metrics_endpoint_counters_and_gauges` | 占满槽位 → `slots_used/slots_total == 1.0`（水位比例的分子分母均由端点暴露，比例计算在告警侧） |

运行：`.venv/bin/python -m pytest server/tests/test_gateway_metrics.py -q`
