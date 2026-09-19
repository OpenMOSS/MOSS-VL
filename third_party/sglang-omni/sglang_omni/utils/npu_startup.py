"""Ascend startup locks, independent of CUDA's device mapping and lock names."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path


def get_npu_startup_lock_path(
    logical_device_id: int,
    *,
    env: Mapping[str, str] | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve a process-local index through ASCEND_RT_VISIBLE_DEVICES."""
    if isinstance(logical_device_id, bool) or not isinstance(logical_device_id, int):
        raise ValueError("NPU device index must be an integer")
    if logical_device_id < 0:
        raise ValueError("NPU device index must be non-negative")
    source = os.environ if env is None else env
    visible = source.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    device_id = logical_device_id
    if visible:
        entries = [entry.strip() for entry in visible.split(",")]
        if any(not entry.isascii() or not entry.isdecimal() for entry in entries):
            raise ValueError(
                "ASCEND_RT_VISIBLE_DEVICES must contain non-negative device IDs"
            )
        if logical_device_id >= len(entries):
            raise ValueError(
                f"NPU device index {logical_device_id} exceeds "
                f"ASCEND_RT_VISIBLE_DEVICES={visible!r}"
            )
        device_id = int(entries[logical_device_id])
    directory = Path(tempfile.gettempdir()) if base_dir is None else Path(base_dir)
    return directory / f"sglang_omni_npu_{device_id}_startup.lock"


@contextmanager
def npu_startup_lock(logical_device_id: int):
    """Serialize startup on the same NPU, but not across different TP cards."""
    import fcntl

    path = get_npu_startup_lock_path(logical_device_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
