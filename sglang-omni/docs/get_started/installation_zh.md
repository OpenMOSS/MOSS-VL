# 安装指南

[English](./installation.md) | **简体中文**

本页用于独立部署 MOSS-VL Realtime 推理后端。需要完整浏览器、语音和 memory 时，使用 [Demo 安装器](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/blob/main/deployment/repro/README.md)，无需再执行本页安装步骤。

## 系统准备

| 项目 | 要求 |
| --- | --- |
| 系统 / Python | Linux x86_64 / Python 3.12 |
| GPU / 驱动 | NVIDIA GPU、CUDA 13 兼容驱动；常规安装要求 R580 或更新版本 |
| 编译及媒体工具 | Git、C/C++ 编译器、CMake、FFmpeg |
| 网络 | GitHub、PyPI、Hugging Face；首次 JIT 可能需要下载内核资源 |

驱动支持能力与 Toolkit 版本不同，见 [NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。系统依赖可由管理员安装：

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential cmake pkg-config python3-venv \
  ffmpeg libsndfile1 libsox-dev libnuma-dev libibverbs1 librdmacm1 libucx-dev
```

确认 Python 3.12 可用，并安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。驱动由管理员准备，以下命令不会修改系统 CUDA。

## 安装后端

尚无源码时先克隆仓库：

```bash
git clone https://github.com/fnlp-vision/sglang-omni-realtime.git
cd sglang-omni-realtime
```

从仓库根目录创建独立环境，不要复用 Demo 或其他推理引擎的虚拟环境：

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip sync --require-hashes \
  --build-constraint deployment/repro/build-constraints.txt \
  --index-url https://pypi.org/simple deployment/repro/requirements.lock
uv pip install --no-deps --no-build-isolation -e .
uv pip check
```

[requirements.lock](../../deployment/repro/requirements.lock) 固定全部依赖版本与发行包哈希，与 Demo 推荐路径使用的后端依赖锁一致。`pyproject.toml` 定义开发依赖范围；`constraints.txt` 仅保留版本约束参考，不是另一套推荐安装方法。

| 依赖 | 固定版本 |
| --- | --- |
| SGLang / Transformers | 0.5.16 / 5.12.1 |
| Torch / FlashInfer | 2.11.0 / 0.6.14 |
| CUDA 编译器、CRT、NVVM | 13.0.88 |
| CUDA 运行库 | 13.0.96 |

## 下载模型

```bash
export MODEL_PATH="$HOME/models/MOSS-VL-Realtime-SGLANG"
hf download OpenMOSS-Team/MOSS-VL-Realtime-SGLANG \
  --local-dir "$MODEL_PATH"
```

该模型仓库公开可访问。出现 401/403 时检查访问策略，并在确有需要时使用自己的 `hf auth login`；不要把 token 放进源码。模型目录应保留配置、自定义代码、tokenizer、processor 和全部权重分片。

`MODEL_PATH` 也可以指向完整的 Hugging Face 缓存快照。权重软链接应指向存在且非空的文件；权重索引中的分片名须为相对路径，不能包含 `..`。

## 配置工具链并启动

在启动终端中执行以下命令。重新打开终端后也需激活环境并设置工具链：

```bash
source .venv/bin/activate
export CUDA_HOME="$(python deployment/repro/cuda_toolkit.py)"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
python deployment/moss_vl_realtime/check_env.py "$MODEL_PATH"
bash deployment/moss_vl_realtime/start.sh "$MODEL_PATH"
```

工具链来自已安装的 NVIDIA wheel，目录适配在仓库 `.repro/cuda-toolkit/` 下按环境隔离创建链接，补齐 JIT 所需的 `lib64` 和 `libcudart.so`；不依赖系统 `/usr/local/cuda` 的版本。

自检不加载模型权重；无 GPU 时可加 `--no-gpu`，但不能据此认定 GPU 部署通过。首次启动需要编译与视觉预热，等待 `/health` 就绪后按[中文 README](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/README_zh.md#调用示例)运行实际请求。

## 版本与更新

配套仓库与运行依赖见 [Demo README](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/blob/main/README_zh.md#兼容性与更新)。不固定源码或模型提交。更新前停止服务，使用 `git pull --ff-only` 更新代码后重新安装依赖锁；完整模型下载到新目录，验证后再切换 `MODEL_PATH`，不要把不同版本文件混在一起。

升级代码、模型或依赖后，需要重新验证；不要只替换模型目录里的个别自定义文件。旧版 Transformers 4.57 参考实现不用于此后端环境。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| CUDA 不可用 | 检查驱动、GPU 可见性和 CUDA 版 Torch |
| 编译器与头文件不匹配 | 使用完整依赖锁，重新设置上述 CUDA_HOME，不混用系统 Toolkit |
| `cannot find -lcudart` | 检查工具链目录视图是否生成，以及启动终端的 CUDA_HOME |
| 动态库或 FFmpeg 缺失 | 补齐系统依赖；不复制其他环境的库路径 |
| KV 容量不足 | 降低 context 或并发，按硬件调整显存比例 |
| GPU / 端口已占用 | 用 `--gpus` / `--port` 选择空闲资源，不停止其他部署 |
| 下载超时 | 先检查直连；必要时配置可信软件源，不把凭据写入仓库 |

Docker 入口见 [容器说明](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/deployment/repro/README_zh.md)。容器构建和 GPU 运行尚未验收，原生路径通过不代表容器通过。
