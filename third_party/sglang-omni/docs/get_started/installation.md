# Installation

**English** | [简体中文](./installation_zh.md)

This guide installs the standalone MOSS-VL Realtime backend. For browser interaction, voice, and memory, use the [Demo installer](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo/blob/main/deployment/repro/README.md) instead of installing the backend twice.

## Prerequisites

| Item | Requirement |
| --- | --- |
| OS / Python | Linux x86_64 / Python 3.12 |
| GPU / driver | NVIDIA GPU and a CUDA 13-compatible driver; normally R580 or newer |
| Build and media tools | Git, C/C++ compiler, CMake, FFmpeg |
| Network | GitHub, PyPI, Hugging Face; first-time JIT may download kernel assets |

Driver compatibility is different from the installed Toolkit version; see [NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). An administrator can install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential cmake pkg-config python3-venv \
  ffmpeg libsndfile1 libsox-dev libnuma-dev libibverbs1 librdmacm1 libucx-dev
```

Prepare Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). The following commands do not modify the system CUDA installation or driver.

## Install the Backend

Clone the repository if you do not already have a checkout:

```bash
git clone https://github.com/fnlp-vision/sglang-omni-realtime.git
cd sglang-omni-realtime
```

From the repository root, create a dedicated environment:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip sync --require-hashes \
  --build-constraint deployment/repro/build-constraints.txt \
  --index-url https://pypi.org/simple deployment/repro/requirements.lock
uv pip install --no-deps --no-build-isolation -e .
uv pip check
```

[requirements.lock](../../deployment/repro/requirements.lock) pins package versions and hashes and is shared with the Demo's backend installation path. `pyproject.toml` defines development dependency ranges; `constraints.txt` is a version reference, not a second recommended installation method.

| Dependency | Pinned version |
| --- | --- |
| SGLang / Transformers | 0.5.16 / 5.12.1 |
| Torch / FlashInfer | 2.11.0 / 0.6.14 |
| CUDA compiler, CRT, NVVM | 13.0.88 |
| CUDA runtime | 13.0.96 |

## Download the Model

```bash
export MODEL_PATH="$HOME/models/MOSS-VL-Realtime-SGLANG"
hf download OpenMOSS-Team/MOSS-VL-Realtime-SGLANG \
  --local-dir "$MODEL_PATH"
```

The model is public. For 401/403 errors, check access policy and use your own `hf auth login` when necessary. Never put tokens in source files. Retain configuration, custom Python code, tokenizer, processors, and all weight shards.

`MODEL_PATH` may also point to a complete Hugging Face cache snapshot. Weight symlinks must resolve to nonempty files; shard names in the weight index must be relative and must not contain `..`.

## Configure CUDA and Start

Activate the environment and configure the toolkit in each new terminal:

```bash
source .venv/bin/activate
export CUDA_HOME="$(python deployment/repro/cuda_toolkit.py)"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
python deployment/moss_vl_realtime/check_env.py "$MODEL_PATH"
bash deployment/moss_vl_realtime/start.sh "$MODEL_PATH"
```

The toolkit comes from the installed NVIDIA wheels. The helper creates an environment-specific link view under `.repro/cuda-toolkit/`, including `lib64` and `libcudart.so` for JIT. It does not depend on the system `/usr/local/cuda` version.

The check does not load weights. `--no-gpu` checks packages and files only. Startup compiles kernels and warms up vision; wait for `/health`, then run the [client example](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/README.md#client-example).

## Versions and Updates

Companion repositories and runtime requirements are listed in the [Demo README](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo#compatibility-and-updates). No fixed source/model commit is required. Stop the service before updating code with `git pull --ff-only`, reinstall its dependency lock, and download a complete model snapshot into a new directory before switching `MODEL_PATH`. Keep the previous model until validation succeeds; do not mix files from different versions. The Transformers 4.57 reference implementation is not the backend environment.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| CUDA unavailable | Check the driver, visible GPUs, and CUDA-enabled Torch |
| Compiler/header mismatch | Install the complete lock and configure CUDA_HOME; do not mix Toolkits |
| `cannot find -lcudart` | Check the generated toolkit view and CUDA_HOME |
| Missing libraries or FFmpeg | Install system dependencies; do not copy another environment's library paths |
| Insufficient KV capacity | Lower context/concurrency and adjust the memory fraction |
| GPU or port occupied | Select idle resources with `--gpus` / `--port`; do not stop other deployments |
| Download timeout | Check direct connectivity first; configure a trusted package mirror if needed |

Container instructions are separate in the [dependency guide](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/deployment/repro/README.md). Container build and GPU execution are not covered by native-environment validation.
