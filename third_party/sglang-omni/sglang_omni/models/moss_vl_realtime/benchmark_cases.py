"""Compile realtime training records into backend-neutral benchmark cases."""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path
from typing import Any

VIDEO_TOKEN = "<|video|>"


def _require_text(message: dict[str, Any], *, index: int) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        raise TypeError(f"message {index} content must be a string")
    return content


def _singleton_timestamps(record: dict[str, Any]) -> tuple[str, list[list[float]]]:
    videos = record.get("videos")
    if not isinstance(videos, list) or len(videos) != 1:
        raise ValueError("benchmark records must contain exactly one video")
    video = videos[0]
    if not isinstance(video, dict):
        raise TypeError("video entry must be a dictionary")
    video_path = video.get("video_path")
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("video_path must be a non-empty string")
    raw_segments = video.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("video segments must be a non-empty list")

    segments: list[list[float]] = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, list) or len(segment) != 1:
            raise ValueError(f"segment {index} must contain one frame timestamp")
        timestamp = segment[0]
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise TypeError(f"segment {index} timestamp must be numeric")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(
                f"segment {index} timestamp must be finite and non-negative"
            )
        segments.append([timestamp])
    return video_path, segments


def compile_training_record(
    record: dict[str, Any],
    *,
    case_id: str,
    source_tag: str,
    task_shape: str,
    source_path: str,
    source_line: int,
) -> dict[str, Any]:
    """Preserve one training record as an ordered prompt/frame event timeline."""
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    video_path, segments = _singleton_timestamps(record)

    system_prompt: str | None = None
    initial_prompt: str | None = None
    pending_prompt: str | None = None
    initial_expected = ""
    events: list[dict[str, Any]] = []
    frame_cursor = 0
    saw_assistant = False

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"message {message_index} must be a dictionary")
        role = message.get("role")
        content = _require_text(message, index=message_index)
        if role == "system":
            if system_prompt is not None or initial_prompt is not None:
                raise ValueError("system prompt must appear once before user messages")
            system_prompt = content
            continue
        if role == "user":
            if initial_prompt is None:
                initial_prompt = content
            else:
                if pending_prompt is not None:
                    raise ValueError(
                        "consecutive user messages cannot map to one frame event"
                    )
                pending_prompt = content
            continue
        if role != "assistant":
            raise ValueError(f"unsupported message role: {role!r}")

        parts = content.split(VIDEO_TOKEN)
        frame_count = len(parts) - 1
        if frame_count == 0:
            if pending_prompt is None:
                raise ValueError("text-only assistant turn has no pending user prompt")
            events.append(
                {
                    "type": "prompt",
                    "prompt": pending_prompt,
                    "expected_after": content,
                }
            )
            pending_prompt = None
            continue
        if not saw_assistant:
            initial_expected = parts[0]
            saw_assistant = True

        for local_index in range(frame_count):
            if frame_cursor >= len(segments):
                raise ValueError("messages contain more video tokens than segments")
            timestamp = segments[frame_cursor][0]
            event: dict[str, Any] = {
                "type": "frame",
                "frame_index": frame_cursor,
                "timestamp": timestamp,
                "segment": [timestamp],
                "expected_before": parts[local_index],
                "expected_after": parts[local_index + 1],
            }
            if local_index == 0 and pending_prompt is not None:
                event["prompt"] = pending_prompt
                pending_prompt = None
            events.append(event)
            frame_cursor += 1

    if initial_prompt is None:
        raise ValueError("record must contain an initial user prompt")
    if pending_prompt is not None:
        raise ValueError("final user prompt has no following frame")
    if frame_cursor != len(segments):
        raise ValueError("segments contain more frames than messages reference")

    for seq_no, event in enumerate(events):
        event["seq_no"] = seq_no
        event["final"] = seq_no == len(events) - 1

    timestamps = [event["timestamp"] for event in events if event["type"] == "frame"]
    intervals = [right - left for left, right in pairwise(timestamps)]
    one_fps = all(math.isclose(value, 1.0, abs_tol=1e-6) for value in intervals)
    return {
        "schema_version": 1,
        "case_id": case_id,
        "source_tag": source_tag,
        "task_shape": task_shape,
        "source": {"path": source_path, "line": source_line},
        "video_path": video_path,
        "frame_count": len(segments),
        "frame_interval_seconds": 1.0 if one_fps else None,
        "system_prompt": system_prompt,
        "initial_prompt": initial_prompt,
        "initial_expected": initial_expected,
        "events": events,
        "training_messages": messages,
        "training_metadata": record.get("metadata", {}),
    }


def materialize_case_frames(case: dict[str, Any], output_dir: Path) -> None:
    """Decode each singleton segment once and save lossless frames for both backends."""
    from PIL import Image
    from torchcodec.decoders import VideoDecoder

    case_dir = output_dir / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    events = [event for event in case["events"] if event["type"] == "frame"]
    timestamps = [float(event["timestamp"]) for event in events]
    decoder = VideoDecoder(str(case["video_path"]), num_ffmpeg_threads=0)
    try:
        frame_batch = decoder.get_frames_played_at(timestamps)
        frame_data = frame_batch.data
    finally:
        del decoder
    if len(frame_data) != len(events):
        raise RuntimeError("decoded frame count does not match event count")

    for event, tensor in zip(events, frame_data, strict=True):
        frame_index = int(event["frame_index"])
        frame_path = case_dir / f"frame_{frame_index:04d}.png"
        array = tensor.permute(1, 2, 0).contiguous().cpu().numpy()
        Image.fromarray(array).save(frame_path, format="PNG")
        event["frame_path"] = str(frame_path)
