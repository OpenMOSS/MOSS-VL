# MOSS-VL Realtime Demo

[English](./README.md) | 简体中文

MOSS-VL 实时视频与语音交互应用，提供浏览器界面，以及可选语音识别、语音合成和长期记忆。

Memory 是会话内长期记忆：宽限期内重连保留，最终结束会话后清理检索记录、向量和临时帧。
已保存的聊天归档采用独立生命周期。

“最早／最后”等事件查询会补充按时间顺序检索的候选；有限候选未命中不再阻止模型使用已有上下文。
记忆数据、向量缓存、待处理载荷和临时帧分别设置预算，超限会记录降级状态，不会自动删除已有记忆。
归档队列也有容量限制，过载会标记归档不完整；历史归档自动删除默认关闭。
清理先用 `.venv/bin/python scripts/history_retention.py --days 90` 预览，只有确认保留策略并停止服务后，
才能添加 `--apply --offline` 删除已结束且符合条件的会话。详见[资源预算与归档维护](./server/README.md#resource-budgets)。

## 配套项目

| 项目 | 职责 |
| --- | --- |
| [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) | 模型权重、配置、tokenizer 和 processor |
| [sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime) | 持续会话、流式推理及 GPU 调度 |
| 本仓库 | 浏览器界面、ASR/TTS、memory 和 REST/WebSocket 网关 |

体验完整应用，从本页开始；只部署模型服务，使用[后端 README](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/README_zh.md)。

## 功能

- 摄像头、屏幕、图片和视频文件输入。
- 流式字幕、文字及语音提问、回答打断。
- 可选 SenseVoice ASR、MOSS-TTS-Nano 本地语音及文本/图像 memory。
- 多会话后端接入、上下文观测与 memory rollover。
- 面向外部客户端的 REST 会话管理和 WebSocket 网关。

## 快速开始

需要 Linux x86_64、Python 3.12、兼容 CUDA 13 的 NVIDIA 驱动、Git、C/C++ 编译器、CMake 和 FFmpeg。安装器负责应用依赖，不安装系统软件包或驱动。

在仓库根目录执行：

```bash
bash bootstrap.sh --doctor-only
bash bootstrap.sh --with-memory --with-asr --with-tts
.venv/bin/python scripts/repro/run.py up --main-gpu 0 --memory-gpu 1
```

安装器自动准备 Python/Node 工具、独立的 Demo 与后端环境、npm 依赖、后端源码和模型。完整配置使用两张独立 GPU；不带 `--with-*` 只安装视频/文本配置，仅使用一张主 GPU。已有的非托管 `.venv` 不会被覆盖。

使用同时包含 `demo/` 和 `backend/` 的源码包时，安装命令改为：

```bash
bash bootstrap.sh --backend-source ../backend --with-memory --with-asr --with-tts
```

浏览器打开 **http://localhost:18502**，切换到视频通话，选择媒体源并连接。语音默认按住说话，自动收音可选 VAD。远程摄像头与麦克风需要 HTTPS。当前配置不启用离线聊天，也不需要云端 TTS 凭据。

```bash
.venv/bin/python scripts/repro/run.py status
.venv/bin/python scripts/repro/smoke.py
.venv/bin/python scripts/repro/run.py down
```

首次启动需要加载模型和编译内核，请等待就绪。端口冲突时使用 `--base-port 19500` 切换整组端口，网页相应为 `19502`。

## 上下文指令

实时会话中，将 `/compact` 或 `/clear` 作为完整文本消息发送。
两者都会中断当前生成和语音；执行期间收到的输入不提交给模型，请等待完成后继续。

- `/compact`：生成摘要并重建上下文，保留会话 Memory 和近期对话。需要启用
  Memory 并配置摘要服务（通常为 pi_agent）；摘要失败会明确报错，不替换上下文。
- `/clear`：清除模型上下文、摘要、会话检索 Memory 和待处理任务。保留系统提示、
  连接和语音配置以及聊天归档，从后续新收到的画面继续；不需要摘要服务。

指令依赖后端会话重建能力。压缩或清空上下文不保证消除视觉幻觉。

## 配置

| 组件 | 默认值 |
| --- | --- |
| 主 VLM | GPU 0；4 会话；131072 context；显存比例 0.5 |
| Memory 4B | GPU 1；4 并发；16384 context；65536 KV token；显存比例 0.2 |
| ASR / 本地 TTS / 检索模型 | CPU |
| 后端 / API / 网页 | 18500 / 18501 / 18502 |
| pi-agent / 4B / TTS | 18503 / 18504 / 18505 |

请按硬件调整显存和并发参数，详见[安装指南](./deployment/repro/README.md)。

## 兼容性与更新

安装清单只标明仓库，不固定后端或模型提交。新安装获取后端默认分支及当前模型仓库；
重跑安装沿用已安装版本，添加 `--update` 才显式更新后端和已启用模型。
运行依赖仍单独锁定：Demo 使用 CPU 环境，后端使用自身仓库的依赖锁。

| 环境 | 运行依赖 |
| --- | --- |
| Demo | Python 3.12；Torch 2.8.0 CPU；Transformers 4.57.1；Node 22.12.0 |
| CUDA 后端 | Python 3.12；Torch 2.11.0；Transformers 5.12.1；SGLang 0.5.16；FlashInfer 0.6.14 |
| CUDA 工具链 | 编译器/CRT/NVVM 13.0.88，配合 CUDA 13.0 运行库 |

不要将原版 Transformers 4.57 自定义模型文件混入 SGLANG 权重目录。
模型上下文 256K 不等于服务配置的 131072；ASR/TTS 和跨 context 记忆恢复由 Demo 提供。
Ascend 使用后端独立的 NPU 安装说明，不安装本 CUDA 依赖锁。

更新前停止托管服务，并先提交或保留本地修改；在仓库根目录执行：

```bash
.venv/bin/python scripts/repro/run.py down
git pull --ff-only
bash bootstrap.sh --update
.venv/bin/python scripts/repro/run.py up --main-gpu 0 --memory-gpu 1
.venv/bin/python scripts/repro/smoke.py
```

更新保留已启用组件，不覆盖后端本地修改。实际安装版本记录在
`.repro/managed-install.json` 和 `.repro/models-verified.json`，用于排错，不约束下次更新。
源码包使用随包后端，不自动从 Git 更新。旧模型快照保留，更新前应预留磁盘空间。

## 接口与部署

```text
浏览器 -> /api/session/{id}/ws -> 会话编排 -> 后端 /v1/video/realtime
外部客户端 -> 网关 /v1/realtime -> 后端 /v1/video/realtime
```

浏览器协议与底层模型协议不同。网页仅代理 `/api`；外部网关客户端访问 API 端口或显式配置 `/v1` 反向代理。浏览器与网关的实例池不共享全局准入计数。

推荐安装的默认端口示例：

```bash
curl --fail http://127.0.0.1:18501/api/status
curl --fail http://127.0.0.1:18501/v1/realtime/health
curl --fail -X POST http://127.0.0.1:18501/v1/realtime/sessions
```

最后一条创建薄网关会话，不包含 Demo memory；使用后通过
`DELETE /v1/realtime/sessions/{session_id}` 关闭。实时帧和问题使用 WebSocket，
不是普通 chat-completions curl 请求。后端可选的 VL API v2 使用独立监听端口（默认 18610），
不替换原有 Demo/native 接口，且回答结束事件语义不同；不要将 Demo 的后端 URL 切到 v2。

启动时会打印实际浏览器/API 地址及健康检查命令。手动部署只需设置 API 的 `PORT`
和 pi-agent 的 `PI_PORT`；未指定 URL 时，前端代理与 `MEMORY_PI_URL` 自动跟随。
已有的 `VITE_BACKEND_ORIGIN` 或 `MEMORY_PI_URL` 显式配置仍优先，删除该覆盖项才恢复自动联动。
独立 pi-agent 与 Demo 统一默认使用 38082；推荐的 `run.py` 继续通过 `--base-port`
统一生成所有组件端口和 URL。

服务默认绑定 loopback。公开部署必须补充鉴权、会话归属校验、TLS 和限流；一次性 WebSocket token 不等同于完整鉴权。

## 更多文档

- [网关协议](./docs/gateway_contract.md)与[运维及历史环境说明](./docs/ops_runbook.md)。
- [Memory 服务](./services/pi_agent/README.md)与[容量规划](./docs/vlm_memory_capacity.md)。
- [手工与历史部署](./docs/manual_deployment.md)、[部署运维](./docs/deployment_operations.md)。旧版 HF/NPU 环境与推荐配置分开使用。

## 开发

在独立开发环境安装 `requirements-dev.txt` 后执行：

```bash
python scripts/run_tests.py
python scripts/dev/check_env.py --check
npm run build
```

测试入口同时支持脚本式套件和 pytest，不应直接用一次 pytest 调用替代全部测试入口。

构建前端并安装 Playwright Chromium 后，可用 `node scripts/tests/realtime_recovery.cjs` 检查浏览器恢复行为。该测试使用模拟媒体和假后端。

## 许可证

模型、后端和 vendored 组件分别遵循各自许可证。模型项目见 [OpenMOSS/MOSS-VL](https://github.com/OpenMOSS/MOSS-VL)，推理框架上游见 [SGLang-Omni](https://github.com/sgl-project/sglang-omni)。
