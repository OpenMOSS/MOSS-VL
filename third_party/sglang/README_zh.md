<div align="center">
  <img src="../../assets/sglang_moss_logo.png" alt="Moss-VL on SGLang" width="360" />
</div>

<div align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">中文</a>
</div>

# Moss-VL on SGLang

`mossvl sglang项目` 提供基于 SGLang 的 Moss-VL 推理与服务化部署说明。

[Quick Start](#quick-start) | [上游 README](./README.upstream.md) | [SGLang 文档](https://docs.sglang.io/) | [模型地址](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Instruct-0408)

## 🚀 Overview

SGLang 官方代码已经支持 Moss-VL。新的部署场景可以直接拉取 SGLang 官方仓库的最新代码使用。

当前目录保留 Moss-VL 在 SGLang 上的使用说明，以及此前同步到本仓库中的代码快照。

当前对应的 SGLang 基线信息如下：

- SGLang 版本：`0.5.10.post2.dev438+gcf9845f8e`
- 基线提交：`cf9845f8e3797f59137ef272705b768c6e1dd3c8`
- 当前分支：`moss-vl`
- 上游项目：`https://github.com/sgl-project/sglang`

该目录中的代码最初同步自 `moss-vl` 分支。后续建议直接使用 SGLang 官方最新代码。

如需查看当前目录中保留的原始 SGLang 说明文档，可参考 [README.upstream.md](./README.upstream.md)。

## 📚 Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Acknowledgement](#acknowledgement)

## ⚡ Quick Start

### 🧩 Create Conda Environment

```bash
conda create -n sglang-moss-vl python=3.12 -y
conda activate sglang-moss-vl
```

### 📦 Install Dependencies

进入 `sglang` 目录后，执行以下安装步骤：

```bash
cd sglang

python -m pip install --upgrade pip setuptools wheel
pip install -e "python"
pip install nvidia-cudnn-cu12==9.16.0.29
pip install joblib
```

### 🤗 Download Model

可以使用 Hugging Face CLI 拉取 `MOSS-VL-Instruct-0408` 模型文件。

```bash
huggingface-cli download OpenMOSS-Team/MOSS-VL-Instruct-0408 \
    --local-dir /path/to/MOSS-VL-Instruct-0408
```

### 🖥️ Launch Service

> 请将 `/path/to/MOSS-VL-Instruct-0408` 替换为本地模型目录，并根据实际硬件情况调整 `--tp`、`--dp`、`--port` 和显存相关参数。

下面是一个服务启动示例：

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/MOSS-VL-Instruct-0408 \
    --host 0.0.0.0 \
    --port 30000 \
    --dp 1 \
    --tp 1 \
    --mem-fraction-static 0.7 \
    --trust-remote-code \
    --attention-backend flashinfer
```

参数说明：

- `--model-path`：模型目录
- `--host` / `--port`：服务监听地址与端口
- `--dp`：数据并行大小
- `--tp`：张量并行大小
- `--mem-fraction-static`：静态显存占用比例
- `--trust-remote-code`：允许加载模型仓库中的自定义代码
- `--attention-backend`：attention 后端。由于 Moss-VL 使用了 custom cross attention mask，prefill 阶段必须使用 `flashinfer` 后端。可以直接使用 `--attention-backend flashinfer`，也可以使用 `--prefill-attention-backend flashinfer --decode-attention-backend fa3`，仅在 prefill 阶段指定 `flashinfer`。

## 🙏 Acknowledgement

`mossvl sglang项目` 构建在 SGLang 官方项目的工程能力之上。SGLang 为高性能大模型与多模态模型推理提供了稳定且可扩展的基础设施。

感谢 SGLang 官方团队及开源社区的持续贡献：

- 官方仓库：`https://github.com/sgl-project/sglang`
- 官方文档：`https://docs.sglang.io/`
