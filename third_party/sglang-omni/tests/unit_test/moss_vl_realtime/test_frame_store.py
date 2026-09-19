from __future__ import annotations

from io import BytesIO
from multiprocessing import shared_memory

import pytest
from PIL import Image

from sglang_omni.models.moss_vl_realtime.frame_store import (
    SharedMemoryFrameStore,
    resolve_shared_memory_frame,
)


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def test_shared_memory_frame_is_consumed_and_unlinked() -> None:
    store = SharedMemoryFrameStore(max_frame_bytes=1024)
    frame_ref = store.put("req-1", _png_bytes())
    image = resolve_shared_memory_frame(frame_ref)
    assert image.size == (4, 3)
    assert image.getpixel((0, 0)) == (10, 20, 30)
    name = frame_ref.removeprefix("shm://")
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name, create=False)
    store.cleanup("req-1")


def test_shared_memory_frame_store_rejects_oversized_payload() -> None:
    store = SharedMemoryFrameStore(max_frame_bytes=4)
    with pytest.raises(ValueError, match="exceeds"):
        store.put("req-1", b"12345")
