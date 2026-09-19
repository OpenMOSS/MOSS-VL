# Inference

Ready-to-run inference examples for MOSS-VL, organized by workload.

## Contents

- [`offline/`](./offline/) — offline image and video inference examples (`run_inference.py` with JSON query files), built directly on the Hugging Face runtime. Covers pure text, single or multiple images, single or multiple videos, and interleaved image-video inputs through `model.offline_generate(...)`.
- [`realtime/`](./realtime/) — realtime streaming inference for MOSS-VL-Realtime (`run_online_inference.py`). Consumes timestamped frames incrementally from video files, cameras, screen capture, synthetic sources, or external frames over a FastAPI WebSocket service, through `model.create_realtime_session(...)` or `model.online_generate(...)`.

Both are reference implementations based on Hugging Face Transformers. For production-scale serving, use the vendored specialized backends under [`third_party/`](../third_party/): [SGLang](../third_party/sglang/) for offline deployment and [SGLang-Omni](../third_party/sglang-omni/) for concurrent multi-stream realtime serving.
