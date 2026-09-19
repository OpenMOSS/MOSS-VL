# SPDX-License-Identifier: Apache-2.0
"""Platform-specific failure policies shared by scheduler and stage runtime."""


def is_fatal_npu_oom(exc: BaseException) -> bool:
    """Apply the Ascend fail-fast policy without changing CUDA error recovery."""
    from sglang_omni.platforms import current_platform

    if current_platform.device_type != "npu":
        return False

    import torch

    return isinstance(exc, torch.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )
