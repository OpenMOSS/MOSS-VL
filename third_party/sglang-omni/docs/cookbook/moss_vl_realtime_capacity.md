# MOSS-VL Realtime 容量规划

## 推荐起点

使用 `deployment/moss_vl_realtime/start.sh` 启动单卡实例时，默认 4 个会话、
128K context、`mem_fraction_static=0.5`，开启 60 秒原始视觉 KV 滑窗，关闭 pooling。
参数由 [config.json](../../deployment/moss_vl_realtime/config.json) 管理。

| 后端参数 | Demo 对应配置 |
| --- | --- |
| `max_sessions=4` | `SGLANG_OMNI_SESSIONS_PER_REPLICA=4` |
| `context_length=131072` | `SGLANG_OMNI_CONTEXT_LENGTH=131072` |
| 服务基础 URL | `SGLANG_OMNI_URLS` |

一键入口要求至少 65 GiB 空闲显存，且已有显存占用不超过 2048 MiB。这是启动筛选条件，
不是任何输入长度下的并发保证。会话满额返回 `session_capacity_exceeded`。

## 显存组成

模型权重由实例共享；每个会话有自己的文本/视觉 KV 映射和状态。
显存还包含 CUDA Graph、临时计算缓冲和分配器保留空间。

- 更高并发、更长文本、更高视频分辨率或帧率都会增加负载。
- `mem_fraction_static` 用于规划 SGLang 静态内存，不是整卡显存的硬上限。
- 调整 context 和并发时，应一起检查启动日志中的 KV pool 容量。
- 启动时若一个完整 context 都无法容纳，服务会报错；运行中 KV 不足可能终止会话。

显存峰值及 1/2/4/8/16 路性能参考见[测试结果](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/deployment/moss_vl_realtime/README_zh.md#参考结果)。
80 GB 级别是容量参考目标，测试结果以明确的硬件、输入和参数为条件。

## 长会话

视觉滑窗只回收较早帧的物理视觉 KV，不会清除历史 token 位置；文本 KV 也会持续增长。
接近 context 上限时，客户端需要创建新会话。跨 context 保留交互记忆由 Demo 的 memory
rollover 负责，不是本后端自动执行的功能。

每次输入恢复的生成额度仍受剩余 context 限制，用尽时按长度结束会话。
追加和 decode 分配前均检查历史 token 与物理 KV 边界，并计入 pending token
及异步 lookahead；单个会话耗尽容量不会越界写入其他会话的页表。

接入 Demo 时，还需为其编码、检索、摘要以及可选 ASR/TTS 单独规划资源。
纯 VLM 的吞吐结果不能直接替代整条应用链路的容量评估。

## 扩容

先在目标输入规格下运行[实时多路测试](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/deployment/moss_vl_realtime/README.md)，
观察每路 TPS、首段可见文本 TTFT、待处理事件和显存峰值，再决定扩容方式：
增加单实例会话数（`--max-running-requests`）、使用 TP、用原生 `--dp-size` 起多副本，
或部署多个独立实例（配合 sgl-omni-router / 多 URL）。

- `--dp-size N` 是单命令、单实例内的副本扩容：入口 stage 复制 N 份，每份独立 TP 组、
  独占 `tp_size` 张卡（需要 `tp_size * dp_size` 张卡），会话按副本粒度均衡分配并在生命周期内
  绑定单一副本，KV 与 `shm://` 帧不跨副本。会话上限 = `max_running_requests * dp_size`，
  只有所有副本都满时才返回 `session_capacity_exceeded`；启动会等每个副本完成 warmup 后才监听。
  机内容纳多副本/多实例时注意用 `SGLANG_OMNI_NCCL_PORT_BASE` 错开 NCCL 端口。
- 多实例应使用不同 GPU 和端口，Demo 中为每个实例填写正确的 URL 与 slot 数。

协议和高级启动参数见 [Realtime Cookbook](./moss_vl_realtime.md)。
