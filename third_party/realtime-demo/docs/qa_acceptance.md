# 薄网关验收清单

> 本文将接口验收要求对应到自动化测试和目标设备上的验证项目。测试位于
> `server/tests/test_gateway_qa.py`（复用 test_gateway_rest.py 的
> in-process uvicorn + FakeSglangOmniServer harness），每个用例 docstring
> 标注对应 §8.x 条目。
>
> 运行：`.venv/bin/python -m pytest server/tests/test_gateway_qa.py -q`

## §8.1 功能验收

| 验收条目 | 自动化测试 | 状态 |
|---|---|---|
| 建连（REST 建 Session → WS attach → session.created 回放） | `server/tests/test_gateway_rest.py::test_rest_lifecycle`；`server/tests/test_gateway_ws.py::test_ws_attach_passthrough` | 已自动化 |
| 推帧 + 时间戳（两段式：input.frame → ready → 二进制 → accepted/processed，timestamp 透传） | `server/tests/test_gateway_ws.py::test_ws_attach_passthrough` | 已自动化 |
| 随时提问（推帧途中 input.prompt，accepted/processed + delta 透传） | `server/tests/test_gateway_qa.py::test_qa_8_1_prompt_anytime` | 已自动化 |
| 增量输出 + 正确结束（delta 逐条透传，response.done/session.done 原样到达，状态落 done） | `server/tests/test_gateway_qa.py::test_qa_8_1_done_events` | 已自动化 |
| 静默不展示为普通回答（response.turn.silence 为事件，无 response.text.delta，状态落 parked） | `server/tests/test_gateway_qa.py::test_qa_8_1_silence_not_an_answer` | 已自动化 |
| 鉴权（无效/过期 token → error + 1008） | `server/tests/test_gateway_ws.py::test_ws_token_gate` | 已自动化 |
| 并发控制（容量满 REST 503 / omni 1013 拒绝 → 503 + 副本标记 full） | `server/tests/test_gateway_rest.py::test_rest_capacity`、`test_rest_capacity_rejected_by_omni`、`test_rest_multislot_same_replica`、`test_rest_capacity_reject_skips_to_next_replica` | 已自动化 |

## §8.2 边界异常

| 验收条目 | 自动化测试 | 状态 |
|---|---|---|
| 非法格式（坏 JSON → omni error{invalid_request} 透传，会话保活） | `server/tests/test_gateway_qa.py::test_qa_8_2_malformed_json` | 已自动化 |
| 超大帧（> gateway_max_frame_bytes → error{invalid_request} + 1009） | `server/tests/test_gateway_ws.py::test_ws_oversize_frame` | 已自动化 |
| seq 乱序（omni invalid_request 透传，会话保活、后续帧正常） | `server/tests/test_gateway_qa.py::test_qa_8_2_seq_out_of_order` | 已自动化 |
| timestamp 回退（omni invalid_request 透传，会话保活、后续帧正常） | `server/tests/test_gateway_qa.py::test_qa_8_2_timestamp_regression` | 已自动化 |
| 帧队列满背压（input.frame.ready 延迟到达期间网关不丢帧、不改序） | `server/tests/test_gateway_qa.py::test_qa_8_2_backpressure_delayed_ready` | 已自动化 |
| 客户端异常断开（gateway 平面：断连即销毁、槽位释放；demo 平面 grace+replay 恢复） | gateway 侧：`server/tests/test_gateway_ws.py::test_ws_attach_passthrough`（断连销毁段）；demo 侧：`server/tests/test_session_ws.py`（B7 reconnect + replay 段，脚本式） | 已自动化（demo 侧为脚本式用例） |
| park 超时（omni error{response_failed} 透传 → 服务端关闭 → 客户端 1011、会话销毁） | 协议语义：`server/tests/test_gateway_qa.py::test_qa_8_2_park_timeout_response_failed`；真实 300s 计时：真机清单 | 已自动化（协议语义）+ 需真机（计时行为） |
| 容量超限 | 同 §8.1 并发控制各行 | 已自动化 |
| 长时间无输入 | 与 park 超时同路径，合并于 `test_qa_8_2_park_timeout_response_failed` | 已自动化（协议语义）+ 需真机（计时行为） |
| 实例宕机隔离与新建会话 | 网关：`server/tests/test_gateway_ws.py::test_ws_omni_death`；memory 接力仅属 Demo：`server/tests/test_failover.py::test_failure_relay` | 分平面验收，不互相替代 |

## §8.3 性能稳定

使用 `tools/stress_realtime.py` 测量性能：`--mode direct` 直连推理后端，
`--mode gateway` 通过 REST 建立会话并使用 WS 转发。
报告应记录硬件、视频输入、会话负载、并发数、延迟 P50/P95/P99、
稳定性、资源占用和容量策略，并列出各项测试条件、目标值与实际值。

## 目标设备验收

以下条目依赖真实模型实例 / 真实计时 / 真实部署拓扑，in-process fake
无法覆盖，应在目标部署环境执行并记录结果：

1. **真实 park 超时行为**：omni 默认 parked 超时（约 300s）期间不发任何
   输入，验证 error{response_failed} + 服务端关闭的实际时序与字段，
   及客户端收到的事件序列与 `test_qa_8_2_park_timeout_response_failed`
   的协议语义一致。
2. **真实分辨率/FPS 组合表现**：deploy.conf 支持的分辨率 × FPS 矩阵下的
   首帧时延、单轮时延和丢帧率。
3. **长会话稳定性**：单会话连续运行（数小时级）成功率、断开率、内存/
   显存曲线和队列积压。
4. **kill 实例演练**：薄网关验证 1011 终止、故障副本隔离和健康实例新建。
   Demo 的 memory 接力另测，不作为薄网关能力承诺。
5. **gateway 重启会话断开语义**：重启 gateway 进程，验证全部在途会话
   断开、客户端重连需重新走 REST 建 Session（gateway 平面无
   grace/reconnect）。

## 会话结束语义

后端在 `session.done` 后正常关闭时，客户端收到 close 1000，连接池释放
该会话的 slot，不隔离副本，计量记录 `end_reason=session_done`。
会话进行中的异常断连使用 close 1011，并隔离对应副本。
相关实现为 `server/gateway/session.py:_on_omni_close`，自动化验证见
`test_qa_8_1_done_events`。
