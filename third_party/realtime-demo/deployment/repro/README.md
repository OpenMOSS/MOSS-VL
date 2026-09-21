# 安装与启动

本入口安装完整的 SGLang-Omni 实时应用，可选 memory、SenseVoice ASR 和 MOSS-TTS-Nano 本地语音。组件关系见[中文首页](../../README_zh.md)或[English README](../../README.md)。

## 前置条件

- Linux x86_64、Python 3.12，且可创建虚拟环境。
- NVIDIA GPU 与兼容 CUDA 13 的驱动；常规安装要求 R580 或更新版本。
- Git、C/C++ 编译器、CMake、FFmpeg 及相关系统动态库。
- 可访问 PyPI、PyTorch wheel 源、npm、GitHub、nodejs.org 和 Hugging Face。
- 基础配置预留 60 GiB 磁盘；启用全部组件时预留 120 GiB。完整配置使用两张独立 GPU，基础视频/文本配置使用一张。

Ubuntu 系统依赖可由管理员安装：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake ffmpeg libsndfile1 libnuma1 libibverbs1 python3-venv
```

确认 `python3 --version` 为 3.12，否则设置 `PYTHON_BOOTSTRAP=/path/to/python3.12`。安装器不修改系统 Python、CUDA 或驱动。网络受限时按所在环境设置 `https_proxy` / `http_proxy`。

## 安装

在新源码目录执行：

```bash
bash bootstrap.sh --doctor-only
bash bootstrap.sh --with-memory --with-asr --with-tts
```

不带 `--with-*` 只安装视频/文本配置。源码包同时包含 `demo/`、`backend/` 时，添加 `--backend-source ../backend` 使用包内后端。

安装器创建 Demo `.venv`、后端 `.repro/.venv-main`，并在 `.repro` 下准备模型、Node 和 CUDA 工具链。组件与依赖说明见[README](../../README_zh.md#兼容性与更新)。

新安装从命名仓库获取当前源码和模型，不固定提交；运行依赖继续使用独立依赖锁。
重跑安装沿用已下载版本。停止托管服务后执行 `git pull --ff-only` 和
`bash bootstrap.sh --update`，显式更新后端、已启用模型和依赖，再启动并执行 smoke 检查。
后端本地修改会阻止更新，不会被覆盖。源码包更新需重新提供 `--backend-source`，
不会把随包源码自动替换为远端源码。

下载先解析仓库当前版本，再下载完整快照，避免更新过程中混入两个版本的文件。
模型目录指向已下载的完整快照，旧快照不自动删除。旧安装的实体模型目录会保留为
同级 `.previous-*` 目录；确认新版本可用后再自行清理不再需要的副本。

已有非托管 `.venv` 时会拒绝覆盖，请使用新目录。安装中断后可重跑相同命令；源码下载失败会继续获取目标提交。更新依赖前须先用 `run.py down` 停止托管部署，安装与启动不能并行执行。`--skip-model-download` 仅准备依赖，不能直接视为完整安装。

## 启停与检查

```bash
.venv/bin/python scripts/repro/run.py up --main-gpu 0 --memory-gpu 1
.venv/bin/python scripts/repro/run.py status
.venv/bin/python scripts/repro/smoke.py
.venv/bin/python scripts/repro/run.py down
```

打开 **http://localhost:18502**，切换到实时通话后连接。语音默认为按住说话，可选择 VAD 自动收音；本地 TTS 不需要云端凭据。本配置不启用离线聊天。

默认后端/API/网页端口为 18500/18501/18502，pi-agent/4B/TTS 为 18503/18504/18505。`--base-port 19500` 将整组端口切到 19500 起。

本入口不读取 `.env.deploy`，也不使用 `start_demo.sh` 的旧端口和内部转发配置。
健康检查使用 `curl --fail http://127.0.0.1:18501/api/status`。

远程摄像头和麦克风需要 HTTPS。内部服务默认只监听 loopback，公开部署需自行配置鉴权、TLS 和访问控制。

## 资源与故障排查

- 主 VLM 默认 4 会话、131072 context、显存比例 0.5。
- Memory 默认 4 并发、16384 context、65536 KV token、显存比例 0.2。
- 用 `--main-memory-fraction` / `--memory-fraction` 调整显存比例，`--cpu-threads` 调整 CPU 线程数。
- 首次启动需要编译 GPU 内核，耗时高于后续启动；日志位于 `.repro/logs`，数据位于 `.repro/data`。
- GPU 或端口占用时请选择空闲资源；启动器不清理其他部署。旧版本状态若无法确认孤儿进程归属，会保留状态并要求人工确认，不会静默宣告清理成功。安装器配置不会读取旧 `.env.deploy`。
- 模型缺失时重新运行安装命令；运行时不自动下载权重。

本入口不涵盖 HF worker、NPU 和其他 TTS provider。容器部署需要单独准备 Docker 与 NVIDIA Container Toolkit，容器路径尚未验证。
