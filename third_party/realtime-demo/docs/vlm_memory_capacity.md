# VLM 显存与并发配置

## 四会话部署

后端与 Demo 的会话上限和 context 长度应保持一致：

| 参数 | 配置 |
| --- | --- |
| 后端 `--max-running-requests` | `4` |
| Demo `SGLANG_OMNI_SESSIONS_PER_REPLICA` | `4` |
| 后端 `--context-length` / Demo `SGLANG_OMNI_CONTEXT_LENGTH` | `131072` |
| 后端 `--mem-fraction-static` | `0.60`，根据设备可用显存核对 KV 池大小 |
| 视频输入 | 每路 1 FPS，最长边 512，JPEG quality 60 |
| 视觉窗口 | raw window 60 秒，pooling 关闭 |
| 生成速率目标 | 每路 4 tokens/s |
| Memory | 文本与图像 memory 开启，编码在 CPU，rollover 阈值 idle 8K / hard 12K |

在后端启动前设置视觉窗口：

```bash
export REALTIME_FRAME_WINDOW_ENABLED=1
export REALTIME_FRAME_WINDOW_RAW_S=60
export REALTIME_FRAME_POOLING_ENABLED=0
```

Memory 服务配置见 [运行手册](./ops_runbook.md)。最小 quickstart 使用
单会话且关闭 memory，四会话长时视频服务需要同时配置上述项目。

## 测量条件

以下只统计 VLM 后端，不包含 memory 摘要模型、ASR 或 TTS。
测试模型为 [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG)，
BF16、单卡、CUDA Graph 开启、async decode 关闭，后端代码 `22b671a`。
每路 1 FPS、最长边 512、JPEG quality 60、生成目标 4 tokens/s；
视觉 raw window 60 秒，pooling 关闭，context 131,072。
Demo 开启文本/图像 memory，CPU 编码，idle 8K / hard 12K rollover。

测试使用 H200。缩减预算实验把 PyTorch 分配上限设为 78 GiB，
并按 `78 GiB * 0.60` 计算静态预算，得到约 25.4 GiB 的 KV 池；
同时记录整个 VLM 进程的 NVML 显存，以覆盖 PyTorch 以外的占用。
该实验用于 80GB 级别的显存容量规划；实际设备吞吐应在目标硬件验证。
本文统一用 GiB（2^30 字节）统计，避免与十进制 GB 混用。

## 显存构成

一个后端实例内，模型权重共享，KV 池预先分配，会话从池内领取 slots。
因此关闭 session 后，KV slots 会回到空闲池，但 `nvidia-smi` 占用通常不会下降。

| 项目 | 实测含义 |
| --- | --- |
| 共享权重及加载时常驻张量 | 约 21.3 GiB，每个后端实例一份，不是每路一份 |
| H200、static fraction 0.60 的 KV 池 | 61.7 GiB，336,874 slots |
| 78 GiB 预算、static fraction 0.60 的 KV 池 | 25.4 GiB，138,662 slots |
| H200 配置 4 路、20 分钟的实际 KV 峰值 | 合计 13.1 GiB |
| H200 配置长测单路 KV 分布 | 中位约 2.53 GiB，P95 约 3.36 GiB，最大约 3.69 GiB |
| 缩减预算单路、180 秒短测 | KV 峰值约 2.07 GiB，完成一次 rollover，关闭后池完全回收 |
| 缩减预算 4 路、20 分钟长测 | KV 峰值约 12.97 GiB，VLM 进程 NVML 峰值约 50.33 GiB |

单路分布取输入开始 60 秒后的观测，包含 rollover 后重新填充窗口的阶段；
不同视频形状、文本长度及所处阶段会改变这个数，不能把均值当作硬上限。

当前池的实际 K/V buffers 是 48 层、8 个 KV heads、head dimension 128、BF16：

```text
每 slot = 48 * 8 * 128 * 2 (K+V) * 2 bytes = 192 KiB
每路实际 KV = (仍保留的视觉 slots + 文本 decoder slots) * 192 KiB
```

CUDA Graph、图像编码临时张量、分配器缓存和 CUDA runtime 是另外的开销。
进程占用与实际 session KV 不是同一个指标。

模型包含 12 层 cross-attention 和 36 层 self-attention，视觉与文本
slots 使用统一池。容量计算以实际分配的 48 层 buffers 为准。

## 预算规划

H200 的 0.60 配置预分配 61.7 GiB KV，四会话参考负载的长测峰值使用
约 21%。静态比例可按部署负载调整；较大的池用于容纳更多会话或更长上下文，
不直接增加固定负载的推理吞吐。

启动检查要求 KV 池至少容纳一个完整 128K context，对应 24 GiB KV。
25.4 GiB 池接近该启动下限，应保留设备初始化余量。
H200 上约 0.34-0.35 的静态比例与 78 GiB 预算实验接近；
实际设置应以启动日志中的 pool slots、权重占用和可用显存为准。

视觉滑窗回收物理 KV，却不清除历史 context 位置；文本 KV 也会增长，
因此这个容量结论依赖 memory rollover 正常运行。4 路都完整保留 128K
物理 KV 则仅 KV 就需要 96 GiB，不能用本测试证明这种负载也能放进 80GB。
分辨率、FPS、视觉窗口长度改变时应重新测量。

## 容量参考

缩减到 78 GiB 预算、静态比例 0.60 后，4 路连续运行 20 分钟：

- 每路完成两次自然 rollover，4 路均存活并持续处理视频。
- 4,800 个摄像头输入帧中转发 4,790 个（99.79%）；切换期间不是零丢帧。
- 模型收到的 4,962 帧全部处理完成，包含问题附带帧和 rollover 恢复帧。
- memory 写入丢弃为 0，结束时队列为 0；关闭后 138,662 个 KV slots 全部归还。
- 单路 KV 中位约 2.55 GiB、P95 约 3.36 GiB、最大约 3.69 GiB。
- VLM 进程 NVML 峰值 50.33 GiB；PyTorch reserved 峰值 49.53 GiB、
  allocated 峰值 48.84 GiB。这些口径不能相加，实际 session KV 已包含在池内。

因此，只计算 VLM、保持本页输入和 memory 配置时，80GB 级别的显存可以
支持当前 4 路需求。同样的 `--mem-fraction-static 0.60` 会在较小的卡上
重算池大小，不会沿用 H200 的 61.7 GiB 池。部署时显式设置 context 131,072
及后端 `--max-running-requests 4`、Demo `SGLANG_OMNI_SESSIONS_PER_REPLICA=4`，
并确认启动日志中的 pool slots 不少于 131,072。具体 80GB 卡的吞吐和
软件兼容性应在目标硬件验证。

模型质量与容量分别评估：4 路 memory 均保留自己的标记，未检测到跨会话标记。
最后一次标记问答中 3 路准确回显，另一路在 10 秒观察窗口内未回显，
未回显的原因尚未确定。应用验收应单独定义回答质量和响应时限指标。

## 连接池观测

连接池分别记录已建立会话、握手预留和远端满额冷却。状态接口中，
`tracked_sessions` 为已建立会话数，`pending` 为握手预留数，`used` 为
准入控制使用的占用计数，包含满额冷却期间的保守占用。
握手成功后预留转为已建立会话，失败或会话结束后释放对应资源。
