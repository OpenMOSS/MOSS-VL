"""Shared-memory frame transport for the video realtime API."""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from multiprocessing import resource_tracker, shared_memory
from urllib.parse import urlparse

from PIL import Image

MAX_FRAME_BYTES = 32 * 1024 * 1024


class SharedMemoryFrameStore:
    """Create bounded one-consumer frame objects and clean unconsumed refs."""

    def __init__(self, *, max_frame_bytes: int = MAX_FRAME_BYTES) -> None:
        self.max_frame_bytes = int(max_frame_bytes)
        if self.max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")
        self._names_by_request: dict[str, set[str]] = defaultdict(set)

    def put(self, request_id: str, data: bytes) -> str:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if not data:
            raise ValueError("frame payload must be non-empty")
        if len(data) > self.max_frame_bytes:
            raise ValueError(f"frame payload exceeds {self.max_frame_bytes} byte limit")
        shm = shared_memory.SharedMemory(create=True, size=len(data))
        try:
            shm.buf[: len(data)] = data
            name = shm.name
            # The stage process becomes unlink owner after it opens the frame.
            resource_tracker.unregister(shm._name, "shared_memory")
        finally:
            shm.close()
        self._names_by_request[request_id].add(name)
        return f"shm://{name}"

    def discard(self, request_id: str, frame_ref: str) -> None:
        name = _shared_memory_name(frame_ref)
        names = self._names_by_request.get(request_id)
        if names is not None:
            names.discard(name)
            if not names:
                self._names_by_request.pop(request_id, None)
        _unlink_if_present(name)

    def forget(self, request_id: str, frame_ref: str) -> None:
        """Forget a ref already unlinked by its one consumer."""
        name = _shared_memory_name(frame_ref)
        names = self._names_by_request.get(request_id)
        if names is None:
            return
        names.discard(name)
        if not names:
            self._names_by_request.pop(request_id, None)

    def cleanup(self, request_id: str) -> None:
        for name in self._names_by_request.pop(request_id, ()):
            _unlink_if_present(name)

    def close(self) -> None:
        for request_id in tuple(self._names_by_request):
            self.cleanup(request_id)


def resolve_shared_memory_frame(
    frame_ref: str, *, max_frame_bytes: int = MAX_FRAME_BYTES
) -> Image.Image:
    """Read, decode, and unlink one shared-memory image reference."""
    name = _shared_memory_name(frame_ref)
    shm = shared_memory.SharedMemory(name=name, create=False)
    try:
        if shm.size <= 0 or shm.size > max_frame_bytes:
            raise ValueError("shared-memory frame has invalid size")
        payload = bytes(shm.buf[: shm.size])
        with Image.open(BytesIO(payload)) as image:
            image.load()
            return image.convert("RGB")
    finally:
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def _shared_memory_name(frame_ref: str) -> str:
    parsed = urlparse(frame_ref)
    if parsed.scheme != "shm":
        raise ValueError("frame_ref must use shm://")
    name = parsed.netloc or parsed.path.lstrip("/")
    if not name or "/" in name or "\\" in name:
        raise ValueError("invalid shared-memory frame name")
    return name


def _unlink_if_present(name: str) -> None:
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    try:
        shm.unlink()
    finally:
        shm.close()
