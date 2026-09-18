# VL API v2 验证结果

[English](./VALIDATION.md) | **简体中文**

[使用说明](./README_zh.md) | [API 参考](./API_zh.md)

本页为 2026-09-11 的协议验证记录。Ascend/main 合并候选结果见[候选验证](../deployment/npu/VALIDATION_zh.md)；下述历史 GPU 结果不是候选版本的新一轮硬件验收。

## 环境

验证日期：2026-09-11。对照版本：原 main `68a0eef`、当前分支原生 v1、当前分支 v2。

| 项目 | 配置 |
| --- | --- |
| 硬件 / 精度 | H200，每个后端单卡 TP=1；BF16 |
| 软件 | SGLang 0.5.16、Transformers 5.12.1 |
| 模型 | 本地 MOSS-VL-Realtime-SGLANG |
| 显存 / 容量 | `mem_fraction_static=0.5`；context 131072；4 会话 |
| 原版精度测试 | HF eager、SGLang FlashInfer；关闭滑窗、pooling、async decode |
| 协议测试 | 生产启动配置，60 秒视觉 KV 滑窗；不含 memory/ASR/TTS |

## 精度与协议对齐

**结果：PASS。** 原版测试保留原始 prompt、完整 PNG、时间戳、事件序列和任务规则，启用 `--strict-tokens`。每个事件生成到 silence 后提交下一事件；每个后端的全部并发会话共享单卡。

| 会话数 | main 精度 | 当前分支精度 | 协议文本 | 单/多路一致 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4/4 | 4/4 | 4/4 | 4/4 |
| 2 | 4/4 | 4/4 | 4/4 | 4/4 |
| 4 | 4/4 | 4/4 | 4/4 | 4/4 |

- **精度**：HF/SGLang 任务、原始 token 和逐事件文本检查均通过；两端各自与单路一致。
- **协议文本**：同一 manifest 在 main/v1/v2 的 36 个 WebSocket 会话中，完整文本和逐事件文本一致，并与原版引擎输出一致。WebSocket 不暴露原始 token ID。
- **跨分支对照**：HF 12/12、SGLang 12/12 的原始 token 与逐事件文本一致。

## 生命周期与计量

| 检查 | 结果 |
| --- | --- |
| CPU 回归 | 499 通过，2 跳过 |
| 单路、续传、打断、视觉滑窗 | v1/v2 文本与最终用量一致 |
| 同轮多段回答 | 回答 ID 独立；重复静默不重复结算 |
| 被打断段 | 不发送正常 done；用量进入后续结算 |
| 输入校验 | 非法字段、跳号、损坏图片被拒；同序号修正重试成功 |
| 中止与超时 | 空闲、输出中、输入在途 abort 及缺失二进制超时通过 |
| 上下文上限 | 初始/追加超长输入正确终止；未提交内容不计量 |
| 容量与恢复 | v1/v2 共享容量；超额拒绝、断连后重建会话通过 |
| 结算与清理 | 算术、单调性、唯一终态、≤1 Hz 观测检查通过；无残留请求或子进程 |

## 运行命令

在仓库根目录执行：

```bash
# 原版 HF/SGLang 精度测试
bash deployment/moss_vl_realtime/test_accuracy.sh /path/to/model --strict-tokens

# CPU 回归
python -m pytest tests/unit_test/moss_vl_realtime \
  vl_api_adapter/tests \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/serve/test_video_realtime_lifecycle.py \
  vl_legacy_adapter/tests -q
```

上述命令不包含 WebSocket manifest 对照。两项 CPU 跳过项分别需要 CUDA、显式模型路径。未验证 TP 多卡/NPU、完整 Demo 链路、长时间压测或真实 worker 崩溃；最终快照不可用等故障分支由 CPU 测试覆盖。
