# Documentation

**English** | [简体中文](./README_zh.md)

## MOSS-VL Realtime

| Guide | Contents |
| --- | --- |
| [Installation](./get_started/installation.md) | Independent environment, pinned dependencies, model download, and startup checks |
| [Launch and tests](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/deployment/moss_vl_realtime/README.md) | Three public tests, commands, reference tables, and charts |
| [Realtime protocol](./cookbook/moss_vl_realtime.md) | WebSocket events, backpressure, TP and native DP replicas, and advanced settings |
| [Capacity planning](./cookbook/moss_vl_realtime_capacity.md) | Session capacity, memory, visual windows, and long conversations |
| [Architecture](./developer_reference/main.md) | Pipeline, scheduling, and communication |
| [Examples](https://github.com/fnlp-vision/sglang-omni-realtime/blob/main/examples/README.md) | Model launchers and clients |

For the browser application, ASR/TTS, and memory, see the [Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo).

## Build Locally

Use a separate documentation environment. From the repository root:

```bash
uv venv .venv-docs --python 3.12
uv pip install --python .venv-docs/bin/python -r docs/requirements.txt
.venv-docs/bin/sphinx-build -b html docs docs/_build/html
```

Live preview:

```bash
PATH="$PWD/.venv-docs/bin:$PATH" make -C docs serve PORT=8080
```

Notebook execution is disabled in the normal documentation build. Install system Pandoc only when building notebook content that requires it.

## Contributing

Prefer Markdown and relative links. Register new site pages in [index.rst](./index.rst). README pairs use `README.md` (English) and `README_zh.md` (Chinese), with reciprocal links. Keep commands, defaults, and numerical results identical between languages. Run the relevant regression tests and documentation checks before submitting changes.
