# SGLang-Omni Realtime for MOSS-VL

**English** | [简体中文](./README_zh.md)

A realtime video-understanding backend for MOSS-VL, built on [SGLang-Omni](https://github.com/sgl-project/sglang-omni) and powered by [SGLang](https://github.com/sgl-project/sglang). Clients stream frames and questions over WebSocket and receive text, silence, and input-processing events.

[Installation](./docs/get_started/installation.md) | [Launch and tests](./deployment/moss_vl_realtime/README.md) | [WebSocket protocol](./docs/cookbook/moss_vl_realtime.md) | [Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo)

- Incremental visual features and KV, with JPEG, PNG, and WebP frames.
- Persistent conversations, prompt interruption, and wake-up after silence.
- Dynamic multi-session scheduling; single-GPU, tensor-parallel, and data-parallel (`--dp-size`) replica inference.
- Decode CUDA Graphs, a sliding visual KV window, and bounded input queues.

> **Required checkpoint**: this backend only loads the SGLang-format realtime checkpoint [`OpenMOSS-Team/MOSS-VL-Realtime-SGLANG`](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG). Download it and point `MODEL_PATH` at the local copy — it does **not** work with the Transformers-format `MOSS-VL-Realtime` checkpoint.

The separate [Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo) provides browser interaction, ASR/TTS, text memory, and the REST gateway.

## Installation

For Ascend integration, use the separate [NPU setup and validation guide](./deployment/npu/README.md). The instructions below target CUDA.

Follow the [installation guide](./docs/get_started/installation.md) to create a Python 3.12 environment and install the hashed dependency lock. For the complete application, use the [Demo installer](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo#readme) without installing the backend twice.

Companion repositories and runtime requirements are in the [Demo README](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo#compatibility-and-updates). Installation follows current repositories, not a fixed source commit; record actual installed revisions when reporting issues. Validate memory settings and JIT on the target hardware.

## Start the Server

An opt-in [VL API v2 listener](./vl_api_adapter/README.md) adds per-response settlement on a separate port. Existing clients retain the native endpoint and semantics.

After installation, run from the repository root:

```bash
source .venv/bin/activate
export MODEL_PATH="$HOME/models/MOSS-VL-Realtime-SGLANG"
export CUDA_HOME="$(python deployment/repro/cuda_toolkit.py)"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
python deployment/moss_vl_realtime/check_env.py "$MODEL_PATH"
bash deployment/moss_vl_realtime/start.sh "$MODEL_PATH"
```

The launcher selects an idle GPU, loads the model, and warms up vision processing. Use `--gpus 0` to select a device, `--port 18510` for another port, or `--dry-run` to inspect the configuration.

| Default | Value |
| --- | --- |
| Service address | `http://127.0.0.1:18500` |
| Session capacity / context | 4 / 131072 |
| Static memory fraction | 0.5 |
| Visual KV window | Enabled, 60 seconds |
| Pooling / async decode | Disabled |

These defaults come from [config.json](./deployment/moss_vl_realtime/config.json) and apply to `start.sh`. The lower-level Python launcher and Demo-managed deployment have separate defaults.

```bash
curl --fail http://127.0.0.1:18500/health
curl --fail http://127.0.0.1:18500/v1/models
```

The served model ID is `moss-vl-realtime`; the model path selects the checkpoint. Configure authentication, TLS, and access control before exposing the service. See the [cookbook](./docs/cookbook/moss_vl_realtime.md#start-the-server) for TP and advanced options.

## Client Example

This example uses two included video frames:

```bash
python examples/moss_vl_realtime_client.py \
  --url ws://127.0.0.1:18500/v1/video/realtime \
  --prompt "Describe the visible scene." \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0000.png --timestamp 0.0 \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0001.png --timestamp 1.0
```

The client replays timestamps and marks the last frame as `final`. Use the Demo or a custom client for continuous camera input:

```text
session.created -> session.configure -> session.configured / session.ready
input.frame -> input.frame.ready -> binary frame -> input.frame.accepted
input.frame.processed -> response.text.delta / response.turn.silence
```

Frames and `input.prompt` share an increasing `seq_no`. Wait for `input.frame.ready` before sending an image. `session.abort` ends the session even while input processing waits for capacity. The configuration deadline defaults to 180 seconds, excluding model prefill.

`input.prompt` also works before any frame is sent. The server consumes the
trained assistant opener before generation, so it is not mistaken for an idle
decision. Generated silence remains valid, including an explicit request to stay quiet.

`max_tokens_per_turn` is a soft per-session tokens/s target, not an answer-length limit. `max_new_tokens` is a generation allowance re-anchored after each input. See the [protocol](./docs/cookbook/moss_vl_realtime.md#websocket-protocol) for fields, errors, and usage events.

## Demo Integration

For manual/legacy Demo deployment only, set these values in `.env.deploy`:

```dotenv
VLM_DEPLOY=sglang_omni
SGLANG_OMNI_URLS=http://127.0.0.1:18500
SGLANG_OMNI_SESSIONS_PER_REPLICA=4
SGLANG_OMNI_CONTEXT_LENGTH=131072
MODEL_PATH=/absolute/path/to/MOSS-VL-Realtime-SGLANG
```

Match the backend URL, session capacity, and context limit. The model directory must be readable on the Demo host. The Demo uses its own environment and ports; see its [README](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo#readme).

The recommended Demo `bootstrap.sh` / `scripts/repro/run.py` setup does not read
`.env.deploy`: it manages the backend and API itself (default ports 18500/18501,
browser 18502). Use its CLI configuration instead; do not start a duplicate
backend using this section. The optional VL API v2 listener is separate and does
not replace the native endpoint used by Demo.

The visual window releases old frame KV, but historical positions and text context keep growing. Demo memory rollover manages conversations across context limits; see [capacity planning](./docs/cookbook/moss_vl_realtime_capacity.md).

## Tests and Development

The three public tests cover accuracy alignment, single-session performance, and realtime concurrency. Commands, tables, and charts are in the [test guide](./deployment/moss_vl_realtime/README.md).

```bash
bash deployment/moss_vl_realtime/test_accuracy.sh "$MODEL_PATH"
bash deployment/moss_vl_realtime/test_latency.sh "$MODEL_PATH"
bash deployment/moss_vl_realtime/test_concurrency.sh "$MODEL_PATH"
```

Run regression tests without loading weights:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_video_realtime*.py
```

| Path | Contents |
| --- | --- |
| [deployment/moss_vl_realtime](./deployment/moss_vl_realtime/) | Environment checks, launchers, tests, and fixtures |
| [models/moss_vl_realtime](./sglang_omni/models/moss_vl_realtime/) | Model integration, incremental input, scheduling, and KV management |
| [serve/video_realtime.py](./sglang_omni/serve/video_realtime.py) | WebSocket, backpressure, and session lifecycle |
| [Realtime cookbook](./docs/cookbook/moss_vl_realtime.md) | Advanced deployment and protocol reference |

### Restoring Text History

`GET /v1/video/realtime/capabilities` advertises `prefill_messages: true`.
Clients may pass `prefill_messages` in `session.configure` to restore text history
as `{role, content}` messages (`system`, `user`, or `assistant`). The limit is
64 messages and 131072 total text characters; the model context limit still applies.
Assistant openers are restored automatically. This reconstructs text context,
not cached KV or past image/video tensors. Clients must probe support before
sending this field to older backends.

## Upstream and License

Based on [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni), retaining its framework, Git history, and [Apache License 2.0](./LICENSE). Earlier development used [CloudRipple/sglang-omni](https://github.com/CloudRipple/sglang-omni); this fork is maintained at [fnlp-vision/sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime).

Report issues in [this repository](https://github.com/fnlp-vision/sglang-omni-realtime/issues). Thanks to the SGLang-Omni, SGLang, and MOSS-VL teams.
