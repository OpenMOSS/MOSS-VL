# Realtime Inference

This directory contains ready-to-run realtime inference and deployment examples for MOSS-VL.
The entry point supports timestamped frame-by-frame inference through the model APIs:

- `model.create_realtime_session(...)` for direct session control
- `model.online_generate(...)` for queue-based inference workers
- FastAPI WebSocket service for remote frame producers

Supported input sources include:

- offline video replayed against its media clock
- streaming training or validation samples in JSONL format
- physical or virtual cameras
- screen capture
- synthetic frames for headless debugging
- external JPEG or PNG frames over WebSocket

## High-Performance Serving with SGLang-Omni

The scripts in this directory are a reference implementation built directly on the Hugging Face runtime: one model process serves one active realtime session. For production deployments that must serve many concurrent realtime streams, use the vendored specialized backend in [`../../third_party/sglang-omni/`](../../third_party/sglang-omni/), synchronized from [fnlp-vision/sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime):

- **Multi-stream concurrency**: dynamic multi-session scheduling serves several realtime video streams per replica (4 sessions by default), and data-parallel replicas (`--dp-size`) scale throughput further; tensor parallelism is also supported.
- **Substantially faster inference**: incremental visual features and KV caching, decode CUDA Graphs, a 60-second sliding visual KV window, and bounded input queues.
- **Persistent sessions**: prompt interruption, wake-up after silence, and text-history restoration across context limits.

The SGLang-Omni backend loads the SGLang-format checkpoint `OpenMOSS-Team/MOSS-VL-Realtime-SGLANG`, runs in its own Python environment, and exposes a different WebSocket protocol (`/v1/video/realtime`). See [`../../third_party/sglang-omni/README.md`](../../third_party/sglang-omni/README.md) and its [realtime cookbook](../../third_party/sglang-omni/docs/cookbook/moss_vl_realtime.md) for installation, launch, tests, and protocol details.

## Supported Checkpoint

By default, `run_online_inference.py` loads the public Hugging Face checkpoint:

```text
OpenMOSS-Team/MOSS-VL-Realtime
```

A local checkpoint or a different Hugging Face model ID can be selected with `--checkpoint`. The checkpoint must expose `create_realtime_session(...)` and `online_generate(...)` in its remote modeling code.

## Run

Run commands from the root of this repository.

Offline video simulated as a realtime stream:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/realtime/run_online_inference.py \
  --source video \
  --video /path/to/video.mp4 \
  --sample-fps 1 \
  --playback-speed 1 \
  --max-frames 256
```

`--playback-speed 1` follows the video timeline and is the default. Keep this value for model inference: disabling pacing or accelerating the stream can change realtime generation behavior. Use `--playback-speed 0` only for source-only diagnostics such as `--dry-run`.

`--max-frames 256` is an upper bound: shorter videos finish naturally and longer videos are truncated. Use `--max-frames 0` to remove the frame cap.

The repository includes a streaming JSONL schema example with a dummy video path. Replace `/path/to/example.mp4` in the JSONL file with a local video before running:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/realtime/run_online_inference.py \
  --dataset inference/realtime/data/aerial_xinjiang_tour_guide_30s.jsonl \
  --dataset-index 0 \
  --playback-speed 1 \
  --max-frames 256
```

Synthetic headless smoke test without loading the model:

```bash
python inference/realtime/run_online_inference.py \
  --dry-run \
  --source synthetic \
  --sample-fps 4 \
  --synthetic-duration 2 \
  --playback-speed 1
```

Camera input:

```bash
pip install opencv-python-headless
CUDA_VISIBLE_DEVICES=0 python inference/realtime/run_online_inference.py \
  --source camera \
  --camera-index 0 \
  --sample-fps 1
```

Screen input:

```bash
pip install mss
CUDA_VISIBLE_DEVICES=0 python inference/realtime/run_online_inference.py \
  --source screen \
  --monitor-index 1 \
  --sample-fps 1
```

## Service Deployment

Install the service dependencies in the same environment as the model:

```bash
pip install fastapi uvicorn websockets
```

`wsproto` can be used instead of `websockets`. Installing `uvicorn[standard]` also provides a supported WebSocket transport.

Start one model process and keep it resident on the selected GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/realtime/run_online_inference.py \
  --serve \
  --host 0.0.0.0 \
  --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

The WebSocket endpoint is:

```text
ws://127.0.0.1:8000/v1/realtime
```

The first client message must create the session:

```json
{
  "type": "start",
  "system_prompt": "You are a helpful realtime video assistant.",
  "prompt": "Describe important changes in the video.",
  "frame_queue_size": 256,
  "max_tokens_per_second": 12,
  "max_new_tokens": 4096,
  "do_sample": false,
  "repetition_penalty": 1.0
}
```

When `do_sample` is true, `temperature`, `top_k`, and `top_p` can also be supplied. The server replies with:

```json
{"type":"ready"}
```

### Send External Frames

For every frame, first send a JSON metadata message:

```json
{"type":"frame","timestamp":12.5}
```

Immediately send one binary WebSocket message containing the corresponding JPEG or PNG bytes. The server replies with a frame acknowledgement:

```json
{"type":"frame_ack","dropped_oldest":false,"pending_frames":0}
```

To align a new user prompt with the current frame, include it in the metadata and then send the image bytes:

```json
{"type":"frame","timestamp":12.5,"prompt":"Describe this frame now."}
```

Frame timestamps are seconds relative to the client stream and must be non-decreasing within a session. The default maximum encoded frame size is 20 MiB and can be changed with `--max-frame-bytes`.

### Send Prompts

A text-only follow-up can be sent while frames continue arriving:

```json
{"type":"prompt","text":"Focus on the people and their actions."}
```

The server replies with:

```json
{"type":"prompt_ack"}
```

### Replay a Server-local Video

A video file visible to the service host can be decoded and replayed by the server:

```json
{
  "type": "video",
  "path": "/path/to/video.mp4",
  "sample_fps": 1,
  "playback_speed": 1,
  "max_frames": 256
}
```

The service reports `video_started` and `video_end`. Send `{"type":"stop_video"}` to stop only the active video source.

### Receive Output and Stop

Generated chunks are returned incrementally:

```json
{"type":"output","text":"The camera moves toward the river."}
```

Output is raw model text and can contain control tokens such as `<|response|>`, `<|silence|>`, and `<|round_start|>`. A client may hide these tokens when rendering output without changing model decoding.

Close the current model session while keeping the model loaded:

```json
{"type":"stop"}
```

The service replies with `{"type":"stopping"}`. Disconnecting the WebSocket also closes its session.

## WebSocket Protocol

One WebSocket connection owns one realtime session. The supported messages are:

| Direction | Message type | Purpose |
| --- | --- | --- |
| Client to server | `start` | Create the model session; must be first. |
| Client to server | `frame` + binary bytes | Push one JPEG or PNG frame. |
| Client to server | `prompt` | Append a text-only user turn. |
| Client to server | `video` | Replay a server-local video file. |
| Client to server | `stop_video` | Stop the active server-local replay. |
| Client to server | `ping` | Check the WebSocket connection. |
| Client to server | `stop` | Close the session but keep the model loaded. |
| Server to client | `ready` | Session creation completed. |
| Server to client | `output` | Incremental raw model output. |
| Server to client | `frame_ack` / `prompt_ack` | Input acknowledgement. |
| Server to client | `video_started` / `video_end` | Server-local replay state. |
| Server to client | `pong` / `stopping` / `session_end` | Connection and lifecycle state. |
| Server to client | `error` | Request or inference error. |

The example service accepts one active session per model instance; for concurrent multi-session serving use the [SGLang-Omni backend](#high-performance-serving-with-sglang-omni). It has no authentication and binds to `127.0.0.1` unless `--host` is changed. For remote deployment, place it behind an authenticated reverse proxy with TLS and WebSocket upgrade support.

Browser `MediaStream` objects are not sent directly. A browser should sample the camera or screen stream, encode each selected frame as JPEG or PNG, and send the metadata and binary image over the WebSocket connection. A high-frame-rate product can replace this transport with WebRTC while continuing to call `session.push_frame(...)` after decoding.

## JSONL Input Format

The input file is a JSONL file where each line is one streaming sample. The loader supports records containing:

- `messages`: system, user, and assistant turns
- `videos`: a video path or video object
- `segments`: frame timestamps associated with the video

The first system and user turns initialize the realtime session before frames arrive. Assistant `<|video|>` placeholders determine how many frame events are replayed, and later user turns are inserted while the stream is active. Assistant target text is not supplied to the model during inference.

Use `--no-realtime-template` to ignore the assistant placeholder script and replay the available timestamps directly.

Relative video paths are resolved relative to the JSONL file location when supported by the record.

## Generation Options

Common options include:

- `--max-frames`: maximum streamed frames; `0` means no cap
- `--sample-fps`: sampling rate for ordinary video, camera, screen, and synthetic sources
- `--playback-speed`: wall-clock replay speed; the default `1` preserves realtime model behavior
- `--frame-queue-size`: maximum pending frame events
- `--max-tokens-per-second`: model token pacing
- `--max-new-tokens`: generation limit
- `--do-sample`, `--temperature`, `--top-k`, and `--top-p`: sampling controls
- `--attention-backend eager`: fallback when FlashAttention is unavailable
- `--raw-output`: display model control tokens in CLI mode
- `--show-silence`: display `<|silence|>` in CLI mode

## Files

- `run_online_inference.py`: CLI inference and FastAPI WebSocket service
- `video_sources.py`: video, camera, screen, synthetic, and wall-clock pacing helpers
- `data/aerial_xinjiang_tour_guide_30s.jsonl`: streaming JSONL schema example with a dummy video path
- `README.md`: deployment and protocol documentation
