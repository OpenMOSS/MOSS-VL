"""Device abstraction layer for CUDA / Ascend NPU backends.

Auto-detects the accelerator at import time and exposes a unified API so the
rest of the server code never branches on ``torch.cuda`` vs ``torch.npu``.

Usage::

    from ..device_compat import (is_npu, is_available, device_str,
                              mem_get_info, memory_allocated,
                              empty_cache, get_device_properties,
                              select_attn_impl, set_visible_device)

All functions are no-ops / zeros when no accelerator is present (CPU box).
"""
from __future__ import annotations

from typing import Any, Tuple

# ---------------------------------------------------------------------------
# Backend detection (run once at import)
# ---------------------------------------------------------------------------

_HAS_NPU = False
_HAS_CUDA = False
_TORCH_NPU = None

try:
    import torch  # noqa: F401
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

if not _HAS_CUDA:
    try:
        import torch_npu  # noqa: F401
        import torch
        _TORCH_NPU = torch_npu
        _HAS_NPU = torch.npu.is_available()
    except ImportError:
        pass


def is_npu() -> bool:
    return _HAS_NPU


def is_available() -> bool:
    return _HAS_NPU or _HAS_CUDA


def device_str(index: int = 0) -> str:
    if _HAS_NPU:
        return f"npu:{index}"
    if _HAS_CUDA:
        return f"cuda:{index}"
    return "cpu"


def get_device_properties(index: int = 0) -> Any:
    """Return an object with .name, .major, .minor attributes."""
    if _HAS_NPU:
        name = torch.npu.get_device_name(index)
        return type("Props", (), {"name": name, "major": 0, "minor": 0})()
    if _HAS_CUDA:
        return torch.cuda.get_device_properties(index)
    return type("Props", (), {"name": "cpu", "major": 0, "minor": 0})()


def mem_get_info(index: int = 0) -> Tuple[int, int]:
    """Return (free_bytes, total_bytes)."""
    if _HAS_NPU:
        return torch.npu.mem_get_info(index)
    if _HAS_CUDA:
        return torch.cuda.mem_get_info(index)
    return (0, 0)


def memory_allocated(index: int = 0) -> int:
    if _HAS_NPU:
        return torch.npu.memory_allocated(index)
    if _HAS_CUDA:
        return torch.cuda.memory_allocated(index)
    return 0


def memory_reserved(index: int = 0) -> int:
    if _HAS_NPU:
        return torch.npu.memory_reserved(index)
    if _HAS_CUDA:
        return torch.cuda.memory_reserved(index)
    return 0


def empty_cache() -> None:
    if _HAS_NPU:
        torch.npu.empty_cache()
    elif _HAS_CUDA:
        torch.cuda.empty_cache()


def current_device() -> int:
    if _HAS_NPU:
        return torch.npu.current_device()
    if _HAS_CUDA:
        return torch.cuda.current_device()
    return 0


# ---------------------------------------------------------------------------
# Attention implementation selection
# ---------------------------------------------------------------------------

def default_attn_impl() -> str:
    if _HAS_NPU:
        return "eager"
    if _HAS_CUDA:
        return "flash_attention_2"
    return "eager"


def select_attn_impl(override: str = "auto") -> str:
    override = (override or "auto").strip().lower()
    if override and override != "auto":
        if override == "flash_attention_2" and not _HAS_CUDA:
            return "eager"
        return override
    return default_attn_impl()


# ---------------------------------------------------------------------------
# Worker process device visibility
# ---------------------------------------------------------------------------

def visible_devices_env() -> str:
    """Return the env-var name used to pin a child process to one device."""
    if _HAS_NPU:
        return "ASCEND_RT_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES"


def _npu_physical_ids() -> list:
    """Physical NPU ids in npu-smi order == the mapping base of logical ids.

    ASCEND_RT_VISIBLE_DEVICES takes LOGICAL ids (0..N-1 in npu-smi listing
    order), but the placement plan works with PHYSICAL ids (npu-smi's NPU
    column — sparse on shared pods, e.g. 0,1,8,9,10,12,14,15). Map between
    them by position: physical[i] → logical i.
    """
    import subprocess
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True,
                             text=True, timeout=8, check=False)
    except Exception:  # noqa: BLE001 — probe failure falls back to identity
        return []
    ids = []
    for line in (out.stdout or "").splitlines():
        # NOTE: health-agnostic on purpose. ASCEND_RT_VISIBLE_DEVICES logical
        # ids are positions in the FULL npu-smi listing — every card on the
        # bus counts, whatever its health state. Filtering by health ("OK")
        # would shift every logical id after an unhealthy card: a pod with one
        # Warning card would pin workers onto the WRONG card. Row identity is
        # structural instead — a CARD row's field 1 is "<id> <model-name>"
        # (e.g. "0 910B2C": 2+ tokens, leading digit, model NOT pure digits),
        # which excludes: chip rows ("| 0 | 0000:5A:00.0 |" — single token),
        # PROCESS-table rows ("| 4 0 | <pid> | python | ..." — second token
        # IS a digit, the chip id; appears whenever processes run on cards),
        # header/banner rows (leading token not a digit), and the
        # "No running processes found in NPU N" filler lines.
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4 and parts[1]:
            head = parts[1].split()
            if (len(head) >= 2 and head[0].isdigit()
                    and not head[1].isdigit()):
                ids.append(int(head[0]))
    return ids


def set_visible_device(env: dict, gpu_index: int) -> None:
    """Pin ``env`` to the given device index (physical → logical on NPU).

    On CUDA the index is passed through unchanged (CUDA ids are already the
    same space both layers use). On NPU the placement plan hands in the
    PHYSICAL id from npu-smi; ASCEND_RT_VISIBLE_DEVICES needs the LOGICAL id
    (listing position). When the mapping cannot be probed, fall back to the
    raw value (identity — correct on dense numbering like 0..7).
    """
    if _HAS_NPU:
        physical = _npu_physical_ids()
        if gpu_index in physical:
            logical = physical.index(gpu_index)
            env[visible_devices_env()] = str(logical)
            return
    env[visible_devices_env()] = str(gpu_index)
