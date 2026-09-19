# VL API v2

**English** | [简体中文](./README_zh.md)

A WebSocket adapter for MOSS-VL Realtime with per-response completion, usage accounting and session-terminal events. It shares the native v1 inference service through a separate listener.

[API Reference](./API.md) | [Validation](./VALIDATION.md) | [Installation](../docs/get_started/installation.md)

## Start

Prepare the backend environment and local model using the installation guide. Run from the repository root:

```bash
source .venv/bin/activate
export MODEL_PATH=/path/to/MOSS-VL-Realtime-SGLANG
export CUDA_HOME="$(python deployment/repro/cuda_toolkit.py)"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

bash vl_api_adapter/start.sh "$MODEL_PATH"
```

The launcher selects one idle GPU by default. Add `--gpus 0` to select a device or `--dry-run` to inspect configuration.

| Setting | Default |
| --- | --- |
| v2 WebSocket | `ws://127.0.0.1:18610/v1/video/realtime` |
| Native v1 port | 18500 |
| Session capacity | 4, shared by v1/v2 |

| Option | Purpose |
| --- | --- |
| `--host`, `--port` | Listener address and native v1 port |
| `VL_API_V2_PORT` | v2 port; default 18610 |
| `VL_API_V2_API_KEY` | v2 authentication key; clients send `Authorization: Bearer <key>` |
| `VL_API_V2_MODEL_VERSION` | Optional deployment version label |

Without a key, use a trusted network. Public access requires TLS and access control. For TP or custom deployment, see the [cookbook](../docs/cookbook/moss_vl_realtime.md#start-the-server) and add `--vl-api-v2-port` to the Python launcher.

## Client

```bash
python vl_api_adapter/client.py \
  --url ws://127.0.0.1:18610/v1/video/realtime \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0000.png \
  --prompt "Describe the scene."
```

`--frame` and `--prompt` can repeat. For authentication, the client reads the same `VL_API_V2_API_KEY` environment variable.

- `response.done` completes one response and keeps the connection open; `session.done` terminates the session.
- Associate responses by `response_id`. One `turn_id` may contain multiple responses.
- `include_usage` controls telemetry only, not settlement usage.

See the [API reference](./API.md) for fields, accounting and abnormal disconnect handling. Existing v1/Demo/legacy clients retain their native endpoint; adapt response-completion handling before switching to v2.

## Tests

```bash
python -m pytest vl_api_adapter/tests -q
```

The original accuracy suite and 1/2/4-session protocol comparisons are documented in [validation results](./VALIDATION.md).

## Layout

| Path | Contents |
| --- | --- |
| `adapter/` | Protocol implementation, installed with the backend package |
| `tests/` | CPU protocol regressions |
| `start.sh`, `client.py` | Launcher and example client |
| `API*.md`, `VALIDATION*.md` | API reference and validation records |
