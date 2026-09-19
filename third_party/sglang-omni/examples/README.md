# Examples

**English** | [简体中文](./README_zh.md)

Run commands from the repository root in the installed backend environment.

## MOSS-VL Realtime

Use the [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) checkpoint and complete the [installation guide](../docs/get_started/installation.md).

```bash
bash deployment/moss_vl_realtime/start.sh "$MODEL_PATH"
python examples/moss_vl_realtime_client.py \
  --url ws://127.0.0.1:18500/v1/video/realtime \
  --prompt "Describe the visible scene." \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0000.png --timestamp 0.0 \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0001.png --timestamp 1.0
```

Run the server and client in separate terminals. The default server supports four sessions on one GPU. The example replays frames at their timestamps and closes input after the final frame. Reference semantic tests use 1 FPS.

For explicit single-GPU/TP settings, use `examples/run_moss_vl_realtime_server.py --help`. See the [cookbook](../docs/cookbook/moss_vl_realtime.md) for the protocol and the [test guide](../deployment/moss_vl_realtime/README.md) for performance.

## Other Model Launchers

`run_omni.py` groups Qwen3-Omni and Ming-Omni launch options into presets:

| Preset | Workload |
| --- | --- |
| `qwen3-text-server` | Qwen3-Omni server, text output |
| `qwen3-speech-server` | Qwen3-Omni server, text and audio |
| `qwen3-speech` | Offline Qwen3-Omni speech |
| `ming-text-server` | Ming-Omni server, text output |
| `ming-speech-server` | Ming-Omni server, text and audio |
| `ming-speech` / `ming-text` | Offline Ming-Omni speech/text |

```bash
python examples/run_omni.py --help
python examples/run_omni.py qwen3-text-server --help
python examples/run_omni.py qwen3-text-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000 --model-name qwen3-omni
python examples/run_omni.py ming-text-server \
  --model-path inclusionAI/Ming-flash-omni-2.0 --port 8001 --model-name ming-omni
```

Launch only the service you need. Model-specific GPU placement and configs are in the [cookbooks](../docs/README.md). Older `run_qwen3_omni_*.py` and `run_ming_omni_*.py` scripts remain compatibility wrappers.

## Adding a Launcher

Add a model-local preset map under [launchers](./launchers/) and register it in [_omni_launcher.py](./_omni_launcher.py). Keep model defaults, stage mutations, and request schemas in that module; the shared launcher owns only registry and CLI dispatch.
