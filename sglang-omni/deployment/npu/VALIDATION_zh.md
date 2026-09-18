# Ascend 集成验证

[English](./VALIDATION.md) | **简体中文**

## 范围

日期：2026-09-13。main 合并候选由 main `194bcc8` 与 Ascend 分支 `790b96e` 整合而成；此前 Ascend 整合基线为 `c0bacdd` 与 PR #1 `fd921c2`。以下区分两个验证阶段，均不代表完整硬件验收。

吸收 PR 的显式 Torch cross-attention 路径及部署配置集中管理。保留分平台启动锁、配置异常处理、实例内版本兼容、补丁失败拒绝安装及受控进程清理。两套版本补丁使用一致的 mask 命名、形状校验、空 encoder 处理和全遮挡行归零。CUDA 生产默认值不变。

## Ascend 分支结果

| 检查 | 结果 |
| --- | --- |
| ServerArgs、版本兼容、NPU 锁、部署配置、补丁安装器、平台守卫 | 48 通过 |
| Stage 进程环境、GPU 显存辅助逻辑、执行桥、旧协议性能探针 | 56 通过 |
| 0.5.16 私有源码首次安装及重复安装 | 通过；重复安装不改变文件 |
| 0.5.14 官方源码私有副本首次安装及重复安装 | 通过；重复安装不改变文件 |
| 0.5.16 补丁后 attention CPU 测试 | 9 通过 |
| 0.5.14 补丁后 attention CPU 测试 | 9 通过 |

Attention 检查覆盖不同 Q/KV 长度、GQA、FP32/BF16、softcapping、全遮挡行、空 encoder、非法 mask、随机输入参考对照及 self-attention 路径保持。Cross-attention 若调用 fused SDPA，测试会失败。安装器检查覆盖显式目录、API 识别、重复安装和不兼容时原文件不变。

源码来源为既有后端环境中的 SGLang 0.5.16，以及官方 SGLang `v0.5.14` 标签。测试仅修改私有副本，没有修改已安装的模型环境；验证的是源码集成与 CPU 算术，不是厂商 NPU 二进制。

## main 候选结果

| 检查 | 结果 |
| --- | --- |
| 与 main 合并 | 无文本冲突 |
| 相关 CPU 回归矩阵 | 616 通过，11 跳过 |
| 新增历史预填充/v2 计量契约 | 通过，计入上述 CPU 矩阵 |
| 0.5.16 / 0.5.14 补丁后 attention 单独启用 CPU 测试 | 各 9 项通过 |
| GPU 生产配置、依赖锁、原版精度入口 | 与 main 保持一致 |
| 历史预填充、v1/v2 实现及计量文件 | 保留 main 版本，无需额外修改生产逻辑 |

矩阵覆盖实时模型与运行时、v1/v2/legacy 会话、版本兼容、设备映射、启动锁、GPU 显存辅助逻辑及执行桥。11 项默认跳过包括 9 项需显式启用的补丁 attention 测试、1 项 CUDA 模型步测试、1 项需模型路径的 processor 测试；9 项 attention 已针对两套补丁源码分别补跑。

此前超时的交叉回归在使用可写缓存、允许本地 socket 的环境中完成，未确定此前超时的精确原因。在仓库根目录运行 CPU 矩阵：

```bash
python -m pytest tests/unit_test/moss_vl_realtime vl_api_adapter/tests \
  vl_legacy_adapter/tests tests/unit_test/vendor \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/serve/test_video_realtime_lifecycle.py \
  tests/unit_test/pipeline/test_stage_process_env.py \
  tests/unit_test/pipeline/test_npu_startup_lock.py \
  tests/unit_test/pipeline/test_gpu_memory.py \
  tests/unit_test/model_runner/test_sglang_execution.py -q
```

## 待验收

仍需完成 CUDA/NPU 实机推理、CUDA TP 启动、NPU 与 HF 原始样例对齐、不同长度的 1/2/4 会话、视觉滑窗及设备显存回收，并分别验证两套 NPU 运行时。本次 GPU SSH 通道不可用，未使用 NPU 设备。

命令和环境要求见[部署说明](./README_zh.md)。候选分支独立发布；完成硬件验收、再次核查 main 是否有新增提交后，再更新 main。
