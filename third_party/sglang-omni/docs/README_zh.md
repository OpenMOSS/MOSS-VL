# 文档

[English](./README.md) | **简体中文**

## MOSS-VL Realtime

| 指南 | 内容 |
| --- | --- |
| [安装](./get_started/installation_zh.md) | 独立环境、固定依赖、模型下载与启动自检 |
| [启动与测试](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/deployment/moss_vl_realtime/README_zh.md) | 三项公开测试、命令、参考结果与折线图 |
| [实时协议](./cookbook/moss_vl_realtime.md) | WebSocket 事件、背压、TP 与原生 DP 副本，以及高级参数 |
| [容量规划](./cookbook/moss_vl_realtime_capacity.md) | 会话容量、显存、视觉窗口与长会话 |
| [架构](./developer_reference/main.md) | Pipeline、调度与通信 |
| [示例](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/examples/README_zh.md) | 模型启动器与客户端 |

浏览器应用、ASR/TTS 和 memory 见 [Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo)。

## 本地构建

使用独立文档环境，在仓库根目录执行：

```bash
uv venv .venv-docs --python 3.12
uv pip install --python .venv-docs/bin/python -r docs/requirements.txt
.venv-docs/bin/sphinx-build -b html docs docs/_build/html
```

实时预览：

```bash
PATH="$PWD/.venv-docs/bin:$PATH" make -C docs serve PORT=8080
```

普通文档构建不执行 Notebook；仅在构建需要转换的 Notebook 内容时安装系统 Pandoc。

## 维护

优先使用 Markdown 和相对链接；新增站点页面写入 [index.rst](./index.rst)。README 使用 `README.md`（英文）与 `README_zh.md`（中文），互相提供入口。两种语言中的命令、默认参数与结果数字应一致。提交前运行相关回归测试与文档检查。
