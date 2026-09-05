from __future__ import annotations

import argparse
import importlib.util
import asyncio
import json
import queue
import re
import sys
import time
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, Iterable

from PIL import Image

try:
    from fastapi import WebSocket
except ImportError:  # FastAPI remains optional outside --serve mode.
    WebSocket = Any

from video_sources import (
    VideoFrame,
    iter_camera_frames,
    iter_screen_frames,
    iter_synthetic_frames,
    iter_video_file,
    iter_video_timestamps,
    pace_frames,
)


DEFAULT_CHECKPOINT = "OpenMOSS-Team/MOSS-VL-Realtime"
DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in real-time video analysis. "
    "The video streams to you frame by frame. At every frame, you decide independently "
    "whether to respond or stay silent — output `<|silence|>` when nothing relevant has happened, "
    "and respond when the visual content warrants it."
)
DEFAULT_PROMPT = (
    "As the video streams frame by frame, respond to this question whenever the visual evidence "
    "lets you answer or updates a previous answer; otherwise stay silent.\n\n"
    "Question: What happens in the video?"
)
HIDDEN_OUTPUT_TOKENS = ("<|response|>", "<|assistant|>", "<go>")
_VISION_TOKEN_PATTERN = re.compile(
    r"<\|vision_start\|>|<\|vision_end\|>|<\|image_pad\|>|<\|video_pad\|>"
    r"|<\|image\|>|<\|video\|>|<image>|<video>"
)


def extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content") or item.get("value")
                if value is not None:
                    parts.append(str(value))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def strip_vision_tokens(text: str) -> str:
    text = _VISION_TOKEN_PATTERN.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def build_qz_realtime_script(record: Dict[str, Any], max_frames: int | None = None) -> tuple[list[Dict[str, Any]], int]:
    """Build the same training-shaped realtime event script as the board qz_data path."""
    messages = [message for message in record.get("messages", []) or [] if isinstance(message, dict)]
    script: list[Dict[str, Any]] = []
    frame_cursor = 0
    seen_first_user = False

    for message in messages:
        role = message.get("role")
        content = extract_text_from_content(message.get("content", ""))

        if role == "system":
            continue
        if role == "user":
            prompt = strip_vision_tokens(content)
            if not seen_first_user:
                seen_first_user = True
                continue
            if prompt:
                script.append({"type": "prompt", "prompt": prompt, "after_frames": frame_cursor})
            continue
        if role != "assistant":
            continue

        frame_count = content.count("<|video|>")
        if max_frames is not None:
            frame_count = max(0, min(frame_count, max_frames - frame_cursor))
        if frame_count > 0:
            script.append({"type": "frames", "start": frame_cursor, "count": frame_count})
            frame_cursor += frame_count
        if max_frames is not None and frame_cursor >= max_frames:
            break

    return script, frame_cursor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run incremental MOSS-VL online inference from a timestamped frame source."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Local checkpoint path or Hugging Face model ID. Defaults to "
            "OpenMOSS-Team/MOSS-VL-Realtime. Not used with --dry-run."
        ),
    )
    parser.add_argument("--serve", action="store_true", help="Load the model once and run the WebSocket service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-frame-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument(
        "--source",
        choices=("video", "synthetic", "camera", "screen"),
        default="synthetic",
        help="Frame source. Synthetic and video work on headless Linux.",
    )
    parser.add_argument("--video", default=None, help="Video path used when --source video.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Training-compatible JSONL record. Overrides source, video, prompt, and system prompt.",
    )
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument(
        "--no-realtime-template",
        action="store_false",
        dest="realtime_template",
        help=(
            "For --dataset, ignore the assistant <|video|> evolution script and stream "
            "the selected timestamps directly. The default matches the board qz_data path."
        ),
    )
    parser.set_defaults(realtime_template=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--monitor-index", type=int, default=1)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Video/synthetic replay speed. 1.0 simulates realtime; values >1 are accelerated tests; 0 disables pacing.",
    )
    parser.add_argument("--max-frames", type=int, default=256, help="Maximum frames to stream; zero means all available frames.")
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--synthetic-duration", type=float, default=10.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompt-frame-first",
        action="store_true",
        dest="prompt_frame_first",
        help=(
            "Send --prompt together with the first frame as a live-interaction batch. "
            "By default, --prompt is prefilled before any frames to match qz_data SFT."
        ),
    )
    parser.add_argument(
        "--no-prompt-frame-first",
        action="store_false",
        dest="prompt_frame_first",
        help="Prefill --prompt before frames, matching qz_data SFT timing.",
    )
    parser.set_defaults(prompt_frame_first=False)
    parser.add_argument("--system-prompt", default=DEFAULT_BOARD_REALTIME_SYSTEM_PROMPT)
    parser.add_argument("--frame-queue-size", type=int, default=256)
    parser.add_argument("--drain-seconds", type=float, default=15.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens-per-second", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attention-backend", default="flash_attention_2")
    parser.add_argument("--show-silence", action="store_true")
    parser.add_argument("--raw-output", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the frame source without loading a model.",
    )
    args = parser.parse_args()

    if not args.serve and not args.dataset and args.source == "video" and not args.video:
        parser.error("--video is required when --source video")
    if args.serve and args.dry_run:
        parser.error("--serve and --dry-run cannot be used together")
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.max_frame_bytes < 1:
        parser.error("--max-frame-bytes must be positive")
    if args.dataset_index < 0:
        parser.error("--dataset-index must be non-negative")
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive")
    if args.playback_speed < 0:
        parser.error("--playback-speed must be non-negative")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if args.frame_queue_size < 1:
        parser.error("--frame-queue-size must be at least 1")
    if args.drain_seconds <= 0:
        parser.error("--drain-seconds must be positive")
    return args


def resolve_checkpoint(value: str) -> str:
    local_path = Path(value).expanduser()
    return str(local_path.resolve()) if local_path.exists() else value


def load_dataset_record(args: argparse.Namespace) -> None:
    args.dataset_timestamps = None
    args.dataset_event_script = None
    args.dataset_record = None
    args.dataset_golden_answer = ""
    if not args.dataset:
        return

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    record = None
    with dataset_path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index == args.dataset_index:
                record = json.loads(line)
                break
    if record is None:
        raise IndexError(f"Dataset index {args.dataset_index} is out of range: {dataset_path}")

    messages = [message for message in record.get("messages", []) or [] if isinstance(message, dict)]
    system_message = next(
        (
            extract_text_from_content(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        ),
        None,
    )
    user_message = next(
        (
            extract_text_from_content(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
        ),
        "",
    )
    golden_answer = next(
        (
            extract_text_from_content(message.get("content", ""))
            for message in messages
            if message.get("role") == "assistant"
        ),
        "",
    )

    videos = record.get("videos") or []
    if not videos or not isinstance(videos[0], dict):
        raise ValueError("Dataset record must contain videos[0] with video_path and segments")
    video = videos[0]
    video_path = Path(str(video.get("video_path") or video.get("video") or video.get("path") or "")).expanduser()
    if not video_path.is_absolute():
        video_path = (dataset_path.parent / video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Dataset video file not found: {video_path}")

    all_timestamps = []
    for segment in video.get("segments") or video.get("segment") or []:
        if isinstance(segment, (list, tuple)) and segment:
            all_timestamps.append(float(segment[0]))
        elif isinstance(segment, (int, float)):
            all_timestamps.append(float(segment))
        else:
            raise ValueError(f"Unsupported video segment: {segment!r}")
    if not all_timestamps:
        raise ValueError("Dataset record has no usable video segment timestamps")

    max_frames = args.max_frames or None
    event_script = None
    script_frames = None
    if args.realtime_template:
        event_script, script_frames = build_qz_realtime_script(record, max_frames)
        if script_frames <= 0:
            raise ValueError("Dataset realtime template has no <|video|> frame events")
        timestamps = all_timestamps[:script_frames]
        args.prompt_frame_first = False
    else:
        timestamps = all_timestamps[:max_frames] if max_frames is not None else all_timestamps

    if not timestamps:
        raise ValueError("Dataset record has no timestamps after max-frame filtering")

    args.source = "video"
    args.video = str(video_path)
    args.prompt = user_message
    args.system_prompt = system_message
    args.dataset_timestamps = timestamps
    args.dataset_event_script = event_script
    args.dataset_record = record
    args.dataset_golden_answer = golden_answer
    print(
        f"dataset={dataset_path} index={args.dataset_index} segments={len(timestamps)} "
        f"realtime_template={bool(event_script)} script_events={len(event_script or [])} "
        f"script_frames={script_frames if script_frames is not None else 'n/a'}",
        file=sys.stderr,
    )

def build_source(args: argparse.Namespace, stop_event: Event) -> Iterable[VideoFrame]:
    if getattr(args, "dataset_timestamps", None) is not None:
        frames = iter_video_timestamps(
            args.video,
            args.dataset_timestamps,
            decode_batch_size=args.decode_batch_size,
        )
        return pace_frames(frames, args.playback_speed, stop_event)
    if args.source == "video":
        frames = iter_video_file(
            args.video,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            decode_batch_size=args.decode_batch_size,
        )
        return pace_frames(frames, args.playback_speed, stop_event)
    if args.source == "synthetic":
        frames = iter_synthetic_frames(
            sample_fps=args.sample_fps,
            duration_seconds=args.synthetic_duration,
            max_frames=args.max_frames,
        )
        return pace_frames(frames, args.playback_speed, stop_event)
    if args.source == "camera":
        return iter_camera_frames(args.camera_index, args.sample_fps, stop_event)
    return iter_screen_frames(args.monitor_index, args.sample_fps, stop_event)


def load_model(checkpoint: str, args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    checkpoint = resolve_checkpoint(checkpoint)
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        frame_extract_num_threads=1,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        device_map=args.device_map,
        dtype=dtype,
        attn_implementation=args.attention_backend,
    )
    model.eval()
    missing_interfaces = [
        name
        for name in ("create_realtime_session", "online_generate")
        if not callable(getattr(model, name, None))
    ]
    if missing_interfaces:
        missing = ", ".join(missing_interfaces)
        raise RuntimeError(
            f"The checkpoint does not expose {missing}. "
            "Use the MOSS-VL streaming final release model code."
        )
    return model, processor


def render_output(chunk: str, raw_output: bool, show_silence: bool) -> bool:
    contains_silence = "<|silence|>" in chunk
    if raw_output:
        print(chunk, end="", flush=True)
        return contains_silence

    if "<|round_start|>" in chunk:
        print("\nAssistant: ", end="", flush=True)
    cleaned = chunk.replace("<|round_start|>", "")
    cleaned = cleaned.replace("<|round_end|>", "")
    cleaned = cleaned.replace("<|silence|>", "")
    for token in HIDDEN_OUTPUT_TOKENS:
        cleaned = cleaned.replace(token, "")
    if cleaned:
        print(cleaned, end="", flush=True)
    if contains_silence and show_silence:
        print("[silence]", end="", flush=True)
    return contains_silence


def run_source_dry_run(args: argparse.Namespace) -> None:
    stop_event = Event()
    started_at = time.monotonic()
    count = 0
    try:
        for frame in build_source(args, stop_event):
            count += 1
            print(
                f"frame={frame.index} timestamp={frame.timestamp:.3f}s "
                f"size={frame.image.width}x{frame.image.height}"
            )
    except KeyboardInterrupt:
        stop_event.set()
    print(f"dry-run complete: frames={count} wall_time={time.monotonic() - started_at:.3f}s")


def run_online_inference(args: argparse.Namespace) -> None:
    model, processor = load_model(args.checkpoint, args)
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generate_kwargs.update(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    input_queue: "queue.Queue[dict]" = queue.Queue()
    output_queue: "queue.Queue[str]" = queue.Queue()
    stop_event = Event()
    source_done = Event()
    producer_errors = []
    online_errors = []
    stats = {"sent": 0}

    def run_model_online() -> None:
        try:
            model.online_generate(
                processor,
                input_queue,
                output_queue,
                frame_queue_size=args.frame_queue_size,
                max_tokens_per_turn=args.max_tokens_per_second,
                **generate_kwargs,
            )
        except BaseException as exc:
            online_errors.append(exc)

    def feed_frames() -> None:
        sent_prompt_frame = False

        def submit_frame(frame: VideoFrame, prompt: str | None = None, drop_oldest: bool = True) -> None:
            payload: Dict[str, Any] = {"frame": frame.image, "timestamp": frame.timestamp}
            if prompt:
                payload["prompt"] = prompt
                payload["drop_oldest"] = drop_oldest
            input_queue.put(payload)
            stats["sent"] += 1

        def feed_dataset_script(event_script: list[Dict[str, Any]]) -> None:
            frame_iter = iter(build_source(args, stop_event))
            frame_cache: list[VideoFrame] = []

            def get_frame(frame_index: int) -> VideoFrame:
                while len(frame_cache) <= frame_index:
                    frame_cache.append(next(frame_iter))
                return frame_cache[frame_index]

            frames_sent = 0
            prompt_events = 0
            total_frames = len(getattr(args, "dataset_timestamps", []) or [])
            for event in event_script:
                if stop_event.is_set():
                    break
                event_type = event.get("type")
                if event_type == "frames":
                    start_index = int(event.get("start", frames_sent))
                    count = int(event.get("count", 0))
                    end_index = min(start_index + count, total_frames)
                    for frame_index in range(start_index, end_index):
                        if stop_event.is_set():
                            break
                        submit_frame(get_frame(frame_index))
                        frames_sent += 1
                    continue
                if event_type == "prompt":
                    prompt = str(event.get("prompt") or "")
                    if prompt:
                        input_queue.put({"prompt": prompt, "drop_oldest": False})
                        prompt_events += 1
            stats["script_prompts"] = prompt_events

        try:
            event_script = getattr(args, "dataset_event_script", None)
            if event_script:
                feed_dataset_script(event_script)
                return

            for frame in build_source(args, stop_event):
                if args.prompt_frame_first and args.prompt and not sent_prompt_frame:
                    submit_frame(frame, args.prompt, drop_oldest=False)
                    sent_prompt_frame = True
                else:
                    submit_frame(frame)
        except BaseException as exc:
            producer_errors.append(exc)
        finally:
            source_done.set()

    online_worker = Thread(target=run_model_online, name="mossvl-online-generate", daemon=True)
    online_worker.start()
    input_queue.put(
        {
            "initial_prompt": "" if args.prompt_frame_first else args.prompt,
            "system_prompt": args.system_prompt,
            "frame_queue_size": args.frame_queue_size,
            "max_tokens_per_turn": args.max_tokens_per_second,
        }
    )

    producer = Thread(target=feed_frames, name="mossvl-frame-source", daemon=True)
    producer.start()
    source_done_at = None
    last_output_at = time.monotonic()
    interrupted = False
    active_error = None

    print(
        f"source={args.source} sample_fps={args.sample_fps} "
        f"playback_speed={args.playback_speed}",
        file=sys.stderr,
    )
    try:
        while True:
            if producer_errors:
                raise RuntimeError("Frame source failed") from producer_errors[0]
            if online_errors:
                raise RuntimeError("Model online_generate failed") from online_errors[0]

            try:
                chunk = output_queue.get(timeout=0.1)
            except queue.Empty:
                chunk = None

            now = time.monotonic()
            saw_silence = False
            if chunk is not None:
                last_output_at = now
                saw_silence = render_output(chunk, args.raw_output, args.show_silence)

            if source_done.is_set():
                if source_done_at is None:
                    source_done_at = now
                # Match the board backend: <|silence|> completes a frame, but it
                # does not close the realtime session. Keep polling for the full
                # drain window so late non-silence output is not cut off.
                if now - max(source_done_at, last_output_at) >= args.drain_seconds:
                    break
            else:
                source_done_at = None

            if not online_worker.is_alive() and source_done.is_set():
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted.", file=sys.stderr)
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        stop_event.set()
        producer.join(timeout=2.0)
        input_queue.put({"stop_online_generate": True})
        online_worker.join(timeout=30.0)
        while True:
            try:
                chunk = output_queue.get_nowait()
            except queue.Empty:
                break
            render_output(chunk, args.raw_output, args.show_silence)
        if online_worker.is_alive() and active_error is None and not interrupted:
            raise TimeoutError("model.online_generate did not stop before the timeout")

    print(f"\nframes_sent={stats['sent']}", file=sys.stderr)


def decode_frame_bytes(raw: bytes) -> Image.Image:
    image = Image.open(BytesIO(raw)).convert("RGB")
    image.load()
    return image


def feed_video_to_session(
    session,
    payload: Dict[str, Any],
    stop_event: Event,
    default_args: argparse.Namespace,
) -> Dict[str, int]:
    video_path = str(payload.get("path") or "").strip()
    if not video_path:
        raise ValueError("video.path must not be empty")

    decode_batch_size = int(payload.get("decode_batch_size", default_args.decode_batch_size))
    max_frames = int(payload.get("max_frames", 0))
    playback_speed = float(payload.get("playback_speed", 1.0))
    timestamps = payload.get("timestamps")
    if timestamps is not None:
        timestamp_list = [float(timestamp) for timestamp in timestamps]
        if max_frames:
            timestamp_list = timestamp_list[:max_frames]
        frames = iter_video_timestamps(
            video_path,
            timestamp_list,
            decode_batch_size=decode_batch_size,
        )
    else:
        frames = iter_video_file(
            video_path,
            sample_fps=float(payload.get("sample_fps", default_args.sample_fps)),
            max_frames=max_frames,
            decode_batch_size=decode_batch_size,
        )

    sent = 0
    dropped = 0
    for frame in pace_frames(frames, playback_speed, stop_event):
        dropped += int(session.push_frame(frame.image, timestamp=frame.timestamp))
        sent += 1
    return {"frames_sent": sent, "frames_dropped": dropped}


def create_service_app(model, processor, default_args: argparse.Namespace):
    try:
        from fastapi import FastAPI, WebSocketDisconnect
    except ImportError as exc:
        raise RuntimeError("Service mode requires fastapi and uvicorn") from exc

    app = FastAPI(title="MOSS-VL Realtime Inference", version="1")
    connection_lock = asyncio.Lock()
    service_state = {"active": False}

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": True,
            "session_active": service_state["active"],
        }

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        if connection_lock.locked():
            await websocket.send_json({"type": "error", "message": "A realtime session is already active"})
            await websocket.close(code=1013)
            return

        await connection_lock.acquire()
        service_state["active"] = True
        session = None
        send_lock = asyncio.Lock()
        stop_signal = asyncio.Event()
        video_control: Dict[str, Any] = {"task": None, "stop": None}

        async def send_json(payload: Dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_text(json.dumps(payload, ensure_ascii=False))

        try:
            raw_start = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            start_payload = json.loads(raw_start)
            if start_payload.get("type") != "start":
                raise ValueError("The first WebSocket message must have type=start")

            do_sample = bool(start_payload.get("do_sample", default_args.do_sample))
            generate_kwargs = {
                "max_new_tokens": int(start_payload.get("max_new_tokens", default_args.max_new_tokens)),
                "repetition_penalty": float(
                    start_payload.get("repetition_penalty", default_args.repetition_penalty)
                ),
                "do_sample": do_sample,
            }
            if do_sample:
                generate_kwargs.update(
                    temperature=float(start_payload.get("temperature", default_args.temperature)),
                    top_k=int(start_payload.get("top_k", default_args.top_k)),
                    top_p=float(start_payload.get("top_p", default_args.top_p)),
                )

            session = model.create_realtime_session(
                processor,
                initial_prompt=str(start_payload.get("prompt", default_args.prompt) or ""),
                system_prompt=start_payload.get("system_prompt", default_args.system_prompt),
                frame_queue_size=int(
                    start_payload.get("frame_queue_size", default_args.frame_queue_size)
                ),
                max_tokens_per_turn=int(
                    start_payload.get(
                        "max_tokens_per_second",
                        default_args.max_tokens_per_second,
                    )
                ),
                **generate_kwargs,
            )
            await asyncio.to_thread(session.start)
            await send_json({"type": "ready"})

            async def output_loop() -> None:
                try:
                    while not stop_signal.is_set():
                        chunk = await asyncio.to_thread(session.poll_output, 0.1)
                        if chunk is not None:
                            await send_json({"type": "output", "text": chunk})
                            continue
                        if not session.active:
                            await send_json({"type": "session_end"})
                            stop_signal.set()
                            return
                except Exception as exc:
                    with suppress(Exception):
                        await send_json({"type": "error", "message": str(exc)})
                    stop_signal.set()

            async def start_video(payload: Dict[str, Any]) -> None:
                if video_control["task"] is not None and not video_control["task"].done():
                    await send_json({"type": "error", "message": "A video source is already running"})
                    return

                source_stop = Event()
                video_control["stop"] = source_stop

                async def run_video() -> None:
                    await send_json({"type": "video_started", "path": payload.get("path")})
                    try:
                        result = await asyncio.to_thread(
                            feed_video_to_session,
                            session,
                            payload,
                            source_stop,
                            default_args,
                        )
                        await send_json({"type": "video_end", **result})
                    except Exception as exc:
                        await send_json({"type": "error", "message": f"Video source failed: {exc}"})

                video_control["task"] = asyncio.create_task(run_video())

            async def input_loop() -> None:
                pending_frame: Dict[str, Any] | None = None
                while not stop_signal.is_set():
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        return

                    text = message.get("text")
                    frame_bytes = message.get("bytes")
                    if text is not None:
                        payload = json.loads(text)
                        message_type = payload.get("type")
                        if message_type == "ping":
                            await send_json({"type": "pong"})
                        elif message_type == "prompt":
                            prompt = str(payload.get("text") or "").strip()
                            if not prompt:
                                await send_json({"type": "error", "message": "prompt.text must not be empty"})
                                continue
                            await asyncio.to_thread(session.push_prompt, prompt)
                            await send_json({"type": "prompt_ack"})
                        elif message_type == "frame":
                            pending_frame = payload
                        elif message_type == "video":
                            await start_video(payload)
                        elif message_type == "stop_video":
                            source_stop = video_control.get("stop")
                            if source_stop is not None:
                                source_stop.set()
                            await send_json({"type": "stop_video_ack"})
                        elif message_type == "stop":
                            await send_json({"type": "stopping"})
                            return
                        else:
                            await send_json(
                                {
                                    "type": "error",
                                    "message": f"Unsupported message type: {message_type}",
                                }
                            )
                        continue

                    if frame_bytes is None:
                        continue
                    if pending_frame is None:
                        await send_json(
                            {
                                "type": "error",
                                "message": "Binary frame must follow a type=frame metadata message",
                            }
                        )
                        continue
                    metadata = pending_frame
                    pending_frame = None
                    if len(frame_bytes) > default_args.max_frame_bytes:
                        await send_json({"type": "error", "message": "Frame exceeds max-frame-bytes"})
                        continue

                    image = await asyncio.to_thread(decode_frame_bytes, frame_bytes)
                    timestamp = metadata.get("timestamp")
                    frame_prompt = str(metadata.get("prompt") or metadata.get("text") or "").strip()
                    if frame_prompt:
                        dropped = await asyncio.to_thread(
                            session.push_prompt_frame,
                            frame_prompt,
                            image,
                            None if timestamp is None else float(timestamp),
                        )
                    else:
                        dropped = await asyncio.to_thread(
                            session.push_frame,
                            image,
                            None if timestamp is None else float(timestamp),
                        )
                    await send_json(
                        {
                            "type": "frame_ack",
                            "dropped_oldest": bool(dropped),
                            "pending_frames": session.pending_frames,
                        }
                    )

            output_task = asyncio.create_task(output_loop())
            input_task = asyncio.create_task(input_loop())
            done, pending = await asyncio.wait(
                {output_task, input_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                with suppress(WebSocketDisconnect, asyncio.CancelledError):
                    task.result()
            stop_signal.set()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            with suppress(Exception):
                await send_json({"type": "error", "message": str(exc)})
        finally:
            stop_signal.set()
            source_stop = video_control.get("stop")
            if source_stop is not None:
                source_stop.set()
            video_task = video_control.get("task")
            if video_task is not None:
                with suppress(Exception):
                    await asyncio.wait_for(video_task, timeout=2.0)
            if session is not None:
                with suppress(Exception):
                    await asyncio.to_thread(session.close, 30.0)
            service_state["active"] = False
            connection_lock.release()
            with suppress(Exception):
                await websocket.close()

    return app


def run_service(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Service mode requires fastapi and uvicorn") from exc

    if not any(importlib.util.find_spec(name) for name in ("websockets", "wsproto")):
        raise RuntimeError(
            "Service mode requires a WebSocket transport. "
            "Install it with: pip install 'uvicorn[standard]'"
        )

    model, processor = load_model(args.checkpoint, args)
    app = create_service_app(model, processor, args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main() -> None:
    args = parse_args()
    if args.serve:
        run_service(args)
        return
    load_dataset_record(args)
    if args.dry_run:
        run_source_dry_run(args)
        return
    run_online_inference(args)


if __name__ == "__main__":
    main()
