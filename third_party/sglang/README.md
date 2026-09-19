<div align="center">
  <img src="../../assets/sglang_moss_logo.png" alt="Moss-VL on SGLang" width="360" />
</div>

<div align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">中文</a>
</div>

# Moss-VL on SGLang

`mossvl sglang project` provides deployment instructions for serving Moss-VL with SGLang.

[Quick Start](#quick-start) | [Upstream README](./README.upstream.md) | [SGLang Docs](https://docs.sglang.io/) | [Model](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Instruct-0408)

## 🚀 Overview

Official SGLang has already supported Moss-VL. For new deployments, you can pull the latest code from the official SGLang repository directly.

This directory keeps the Moss-VL SGLang usage notes and the code snapshot previously synchronized into this repository.

The current SGLang baseline is:

- SGLang version: `0.5.10.post2.dev438+gcf9845f8e`
- Baseline commit: `cf9845f8e3797f59137ef272705b768c6e1dd3c8`
- Branch: `moss-vl`
- Upstream project: `https://github.com/sgl-project/sglang`

The code in this directory was originally synchronized from the `moss-vl` branch. Going forward, the recommended path is to use the latest official SGLang code.

For the original upstream introduction and general SGLang documentation in this directory, see [README.upstream.md](./README.upstream.md).

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

Run the following commands in the `sglang` directory:

```bash
cd sglang

python -m pip install --upgrade pip setuptools wheel
pip install -e "python"
pip install nvidia-cudnn-cu12==9.16.0.29
pip install joblib
```

### 🤗 Download Model

Use Hugging Face CLI to download `MOSS-VL-Instruct-0408`:

```bash
huggingface-cli download OpenMOSS-Team/MOSS-VL-Instruct-0408 \
    --local-dir /path/to/MOSS-VL-Instruct-0408
```

### 🖥️ Launch Service

> Replace `/path/to/MOSS-VL-Instruct-0408` with the local model directory on your machine. Adjust `--tp`, `--dp`, `--port`, and memory settings according to your hardware.

Example command:

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

Arguments:

- `--model-path`: model directory
- `--host` / `--port`: service bind address and port
- `--dp`: data parallelism size
- `--tp`: tensor parallelism size
- `--mem-fraction-static`: static GPU memory fraction
- `--trust-remote-code`: allow custom code from the model repository
- `--attention-backend`: attention backend. Since Moss-VL uses a custom cross-attention mask, `flashinfer` must be used during the prefill stage. You can either use `--attention-backend flashinfer`, or specify `--prefill-attention-backend flashinfer --decode-attention-backend fa3` to use `flashinfer` only for prefill.

## 🙏 Acknowledgement

`mossvl sglang project` builds on top of the engineering foundation provided by SGLang. SGLang offers a stable and extensible infrastructure for high-performance LLM and multimodal inference.

Thanks to the SGLang team and open-source community for their continued contributions:

- Repository: `https://github.com/sgl-project/sglang`
- Documentation: `https://docs.sglang.io/`
