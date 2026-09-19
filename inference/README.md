# Inference Examples

Ready-to-run inference examples for MOSS-VL, organized by workload. Both subdirectories are reference implementations based on Hugging Face Transformers — for production serving, use the specialized backends under [`third_party/`](../third_party/).

## Directory Structure

```
inference/
├── offline/     # Offline image/video inference: run_inference.py + JSON query examples
└── realtime/    # Realtime streaming inference: run_online_inference.py + WebSocket service
```

## Offline Inference

[`offline/`](./offline/) runs full-modality queries on finished inputs — pure text, single or multiple images, single or multiple videos, and interleaved image-video inputs — through `model.offline_generate(...)`. See [`offline/README.md`](./offline/README.md) for usage.

## Realtime Inference

[`realtime/`](./realtime/) runs MOSS-VL-Realtime on a continuous stream: timestamped frames arrive from a video file, camera, screen capture, or remote producers over WebSocket, and answers are generated incrementally through `model.create_realtime_session(...)` or `model.online_generate(...)`. One model process serves one session at a time. See [`realtime/README.md`](./realtime/README.md) for the CLI, input format, and protocol.

## Production Serving

For higher throughput, use the backends under [`third_party/`](../third_party/): [SGLang](../third_party/sglang/) for offline deployment, and [SGLang-Omni](../third_party/sglang-omni/) for realtime serving — its dynamic multi-session scheduling and data-parallel replicas serve multiple concurrent streams with substantially higher throughput.
