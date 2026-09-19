from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Iterable, Iterator, Optional

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class VideoFrame:
    image: Image.Image
    timestamp: float
    index: int


def _tensor_to_pil(frame) -> Image.Image:
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu()
    if hasattr(frame, "ndim") and frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = frame.permute(1, 2, 0)

    array = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
    if array.dtype != np.uint8:
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] >= 4:
        array = array[..., :3]
    return Image.fromarray(array).convert("RGB")


def iter_video_file(
    video_path: str,
    sample_fps: float,
    max_frames: int = 0,
    decode_batch_size: int = 8,
) -> Iterator[VideoFrame]:
    """Decode a file incrementally in small timestamp batches."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    if decode_batch_size < 1:
        raise ValueError("decode_batch_size must be at least 1")

    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as exc:
        raise RuntimeError("Video-file input requires torchcodec") from exc

    decoder = VideoDecoder(str(path), num_ffmpeg_threads=0)
    try:
        metadata = decoder.metadata
        begin_value = getattr(metadata, "begin_stream_seconds_from_content", None)
        begin = max(0.0, float(begin_value)) if begin_value is not None else 0.0
        end_value = getattr(metadata, "end_stream_seconds_from_content", None)
        duration_value = getattr(metadata, "duration_seconds", None)

        if end_value is not None:
            safe_end = max(begin, float(end_value) - 1e-6)
            duration = max(0.0, float(end_value) - begin)
        elif duration_value is not None:
            duration = max(0.0, float(duration_value))
            safe_end = max(begin, begin + duration - 1e-6)
        else:
            duration = None
            safe_end = None

        if duration is not None and duration > 0:
            frame_count = max(1, int(math.ceil(duration * sample_fps - 1e-6)))
            if max_frames:
                frame_count = min(frame_count, max_frames)
        elif max_frames:
            frame_count = max_frames
        else:
            raise RuntimeError("Video duration is unavailable; pass --max-frames")

        for batch_start in range(0, frame_count, decode_batch_size):
            batch_end = min(frame_count, batch_start + decode_batch_size)
            requested_timestamps = [begin + index / sample_fps for index in range(batch_start, batch_end)]
            if safe_end is not None:
                requested_timestamps = [min(timestamp, safe_end) for timestamp in requested_timestamps]

            batch = decoder.get_frames_played_at(requested_timestamps)
            pts_value = getattr(batch, "pts_seconds", None)
            if hasattr(pts_value, "detach"):
                pts_value = pts_value.detach().cpu()
            decoded_timestamps = pts_value.tolist() if hasattr(pts_value, "tolist") else []

            for offset, frame in enumerate(batch.data):
                source_index = batch_start + offset
                timestamp = requested_timestamps[offset]
                if offset < len(decoded_timestamps):
                    timestamp = float(decoded_timestamps[offset])
                yield VideoFrame(
                    image=_tensor_to_pil(frame),
                    timestamp=timestamp,
                    index=source_index,
                )
    finally:
        close = getattr(decoder, "close", None)
        if callable(close):
            close()


def iter_video_timestamps(
    video_path: str,
    timestamps: Iterable[float],
    decode_batch_size: int = 8,
) -> Iterator[VideoFrame]:
    """Decode explicit training/evaluation segment timestamps incrementally."""
    if decode_batch_size < 1:
        raise ValueError("decode_batch_size must be at least 1")

    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    timestamp_list = [float(timestamp) for timestamp in timestamps]
    if not timestamp_list:
        raise ValueError("timestamps must not be empty")
    if any(timestamp < 0 for timestamp in timestamp_list):
        raise ValueError("timestamps must be non-negative")
    if any(current < previous for previous, current in zip(timestamp_list, timestamp_list[1:])):
        raise ValueError("timestamps must be non-decreasing")

    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as exc:
        raise RuntimeError("Video-file input requires torchcodec") from exc

    decoder = VideoDecoder(str(path), num_ffmpeg_threads=0)
    try:
        metadata = decoder.metadata
        begin_value = getattr(metadata, "begin_stream_seconds_from_content", None)
        safe_begin = max(0.0, float(begin_value)) if begin_value is not None else 0.0
        end_value = getattr(metadata, "end_stream_seconds_from_content", None)
        if end_value is not None:
            safe_end = max(safe_begin, float(end_value) - 1e-6)
            timestamp_list = [max(safe_begin, min(timestamp, safe_end)) for timestamp in timestamp_list]
        else:
            timestamp_list = [max(safe_begin, timestamp) for timestamp in timestamp_list]

        for batch_start in range(0, len(timestamp_list), decode_batch_size):
            batch_timestamps = timestamp_list[batch_start : batch_start + decode_batch_size]
            batch = decoder.get_frames_played_at(batch_timestamps)
            pts_value = getattr(batch, "pts_seconds", None)
            if hasattr(pts_value, "detach"):
                pts_value = pts_value.detach().cpu()
            decoded_timestamps = pts_value.tolist() if hasattr(pts_value, "tolist") else []

            for offset, frame in enumerate(batch.data):
                source_index = batch_start + offset
                timestamp = batch_timestamps[offset]
                if offset < len(decoded_timestamps):
                    timestamp = float(decoded_timestamps[offset])
                yield VideoFrame(
                    image=_tensor_to_pil(frame),
                    timestamp=timestamp,
                    index=source_index,
                )
    finally:
        close = getattr(decoder, "close", None)
        if callable(close):
            close()


def iter_synthetic_frames(
    sample_fps: float,
    duration_seconds: float,
    max_frames: int = 0,
    width: int = 640,
    height: int = 360,
) -> Iterator[VideoFrame]:
    """Generate a deterministic moving scene for headless smoke tests."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    frame_count = max(1, int(math.ceil(duration_seconds * sample_fps)))
    if max_frames:
        frame_count = min(frame_count, max_frames)

    box_width = max(48, width // 8)
    box_height = max(48, height // 5)
    travel = max(1, width - box_width - 40)
    for index in range(frame_count):
        timestamp = index / sample_fps
        phase = index / max(1, frame_count - 1)
        left = 20 + int(travel * phase)
        top = height // 2 - box_height // 2

        image = Image.new("RGB", (width, height), color=(24, 28, 36))
        draw = ImageDraw.Draw(image)
        draw.rectangle((left, top, left + box_width, top + box_height), fill=(230, 74, 70))
        draw.text((20, 20), "MOSS-VL realtime synthetic source", fill=(245, 245, 245))
        draw.text((20, 46), f"frame={index}  time={timestamp:.1f}s", fill=(165, 205, 255))
        yield VideoFrame(image=image, timestamp=timestamp, index=index)


def iter_camera_frames(
    camera_index: int,
    sample_fps: float,
    stop_event: Optional[Event] = None,
) -> Iterator[VideoFrame]:
    """Capture a physical or virtual V4L2 camera through OpenCV."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Camera input requires opencv-python-headless") from exc

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera index {camera_index}")

    interval = 1.0 / sample_fps
    started_at = time.monotonic()
    next_capture_at = started_at
    index = 0
    try:
        while stop_event is None or not stop_event.is_set():
            delay = max(0.0, next_capture_at - time.monotonic())
            if stop_event is not None and stop_event.wait(delay):
                break
            if stop_event is None and delay:
                time.sleep(delay)

            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Camera {camera_index} stopped returning frames")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield VideoFrame(
                image=Image.fromarray(rgb),
                timestamp=time.monotonic() - started_at,
                index=index,
            )
            index += 1
            next_capture_at = max(next_capture_at + interval, time.monotonic())
    finally:
        capture.release()


def iter_screen_frames(
    monitor_index: int,
    sample_fps: float,
    stop_event: Optional[Event] = None,
) -> Iterator[VideoFrame]:
    """Capture an X11/Wayland desktop through the optional mss package."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    try:
        import mss
    except ImportError as exc:
        raise RuntimeError("Screen input requires mss") from exc

    interval = 1.0 / sample_fps
    started_at = time.monotonic()
    next_capture_at = started_at
    index = 0
    with mss.mss() as capture:
        if monitor_index < 0 or monitor_index >= len(capture.monitors):
            raise ValueError(
                f"monitor_index must be between 0 and {len(capture.monitors) - 1}"
            )
        monitor = capture.monitors[monitor_index]
        while stop_event is None or not stop_event.is_set():
            delay = max(0.0, next_capture_at - time.monotonic())
            if stop_event is not None and stop_event.wait(delay):
                break
            if stop_event is None and delay:
                time.sleep(delay)

            shot = capture.grab(monitor)
            yield VideoFrame(
                image=Image.frombytes("RGB", shot.size, shot.rgb),
                timestamp=time.monotonic() - started_at,
                index=index,
            )
            index += 1
            next_capture_at = max(next_capture_at + interval, time.monotonic())


def pace_frames(
    frames: Iterable[VideoFrame],
    playback_speed: float,
    stop_event: Optional[Event] = None,
) -> Iterator[VideoFrame]:
    """Replay timestamped frames against a monotonic wall clock.

    A playback speed of 1.0 simulates realtime. Zero disables pacing and is
    useful for source-only smoke tests.
    """
    if playback_speed < 0:
        raise ValueError("playback_speed must be non-negative")

    first_timestamp = None
    wall_started_at = None
    for frame in frames:
        if stop_event is not None and stop_event.is_set():
            break
        if playback_speed > 0:
            if first_timestamp is None:
                first_timestamp = frame.timestamp
                wall_started_at = time.monotonic()
            target = wall_started_at + (frame.timestamp - first_timestamp) / playback_speed
            delay = max(0.0, target - time.monotonic())
            if stop_event is not None and stop_event.wait(delay):
                break
            if stop_event is None and delay:
                time.sleep(delay)
        yield frame
