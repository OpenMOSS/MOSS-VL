"""Device abstraction layer for MOSS-VL.

Automatically detects and supports both NVIDIA CUDA and Huawei Ascend NPU
backends.  Import this module before any heavy model operations to ensure
the correct device backend is initialised.

Usage::

    from device_utils import get_device_type, synchronize, get_default_attn_impl

    device = get_device_type()                  # "npu" / "cuda" / "cpu"
    attn_impl = get_default_attn_impl()         # "flash_attention_2" / "eager"
    synchronize()                               # torch.cuda.synchronize() or torch.npu.synchronize()
"""

from __future__ import annotations

import torch

_HAS_NPU = False
_HAS_CUDA = False

try:
    import torch_npu  # noqa: F401
    _HAS_NPU = torch.npu.is_available()
except ImportError:
    pass

_HAS_CUDA = torch.cuda.is_available()


def get_device_type() -> str:
    if _HAS_NPU:
        return "npu"
    if _HAS_CUDA:
        return "cuda"
    return "cpu"


def get_device_count() -> int:
    if _HAS_NPU:
        return torch.npu.device_count()
    if _HAS_CUDA:
        return torch.cuda.device_count()
    return 0


def get_device_name(index: int = 0) -> str:
    if _HAS_NPU:
        return torch.npu.get_device_name(index)
    if _HAS_CUDA:
        return torch.cuda.get_device_name(index)
    return "cpu"


def synchronize() -> None:
    if _HAS_NPU:
        torch.npu.synchronize()
    elif _HAS_CUDA:
        torch.cuda.synchronize()


def get_default_attn_impl() -> str:
    if _HAS_NPU:
        return "eager"
    if _HAS_CUDA:
        return "flash_attention_2"
    return "eager"


def resolve_attn_impl(requested: str | None = None) -> str:
    if requested is None or requested == "auto":
        return get_default_attn_impl()
    if requested == "flash_attention_2" and not _HAS_CUDA:
        print(
            f"[device_utils] flash_attention_2 is not available on "
            f"{get_device_type().upper()}; falling back to 'eager'."
        )
        return "eager"
    return requested


def get_device_str(index: int = 0) -> str:
    if _HAS_NPU:
        return f"npu:{index}"
    if _HAS_CUDA:
        return f"cuda:{index}"
    return "cpu"


def patch_hf_device_map(device_map: str | dict | None = None):
    if device_map is not None:
        return device_map
    if _HAS_NPU:
        return "npu:0"
    if _HAS_CUDA:
        return "auto"
    return "cpu"


def print_device_info() -> None:
    dt = get_device_type()
    count = get_device_count()
    print(f"[device_utils] device_type={dt}  device_count={count}")
    for i in range(min(count, 4)):
        print(f"[device_utils]   device {i}: {get_device_name(i)}")
    print(f"[device_utils] default attn_impl={get_default_attn_impl()}")
