"""Platform-aware defaults for the MOSS-VL realtime pipeline.

The pipeline was built for CUDA (FlashInfer attention, decode CUDA graphs).
On Ascend NPU the equivalents resolve through ``sglang_omni.platforms``:
the ascend attention backend replaces FlashInfer, decode graphs stay off
by default, and the upstream Moss-VL FlashInfer guard is relaxed.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CUDA_ATTENTION_BACKEND = "flashinfer"
NPU_ATTENTION_BACKEND = "ascend"

_FLASHINFER_GUARD_MESSAGE = (
    "MossVLForConditionalGeneration requires flashinfer prefill "
    "attention backend for cross-attention custom mask support."
)


def is_npu_platform() -> bool:
    from sglang_omni.platforms import current_platform

    return current_platform.device_type == "npu"


def platform_device_type() -> str:
    from sglang_omni.platforms import current_platform

    return current_platform.device_type


def device_spec(index: int) -> str:
    """Return the concrete single-card device spec for a placement index."""
    return f"{platform_device_type()}:{int(index)}"


def preferred_attention_backend() -> str:
    """Attention backend serving cross-attention on the resolved platform.

    FlashInfer is the CUDA decode backend (it also re-plans the packed
    cross-attention custom mask on every replay). The ascend backend is the
    NPU equivalent registered in sglang's attention registry.
    """
    return NPU_ATTENTION_BACKEND if is_npu_platform() else CUDA_ATTENTION_BACKEND


def relax_mossvl_flashinfer_guard() -> None:
    """Allow non-FlashInfer attention backends for Moss-VL on NPU.

    Ascend must consume the same per-query frame visibility as FlashInfer.
    Only relax the CUDA-specific backend assertion after verifying the
    installed Ascend patches. CUDA validation is left untouched.
    """
    if not is_npu_platform():
        return
    for name in ("ASCEND_USE_FA", "ASCEND_USE_FIA"):
        if os.environ.get(name, "false").lower() not in ("", "0", "false", "no", "off"):
            raise RuntimeError(f"MOSS-VL frame visibility requires {name}=false")
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )
    from sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend import (
        AscendTorchNativeAttnBackend,
    )
    from sglang.srt.server_args import ServerArgs

    parameters = inspect.signature(
        AscendTorchNativeAttnBackend.run_sdpa_forward_extend
    ).parameters
    if (
        not getattr(AscendAttnBackend, "_moss_vl_visibility_mask_supported", False)
        or not getattr(
            AscendTorchNativeAttnBackend,
            "_moss_vl_torch_cross_attention_supported",
            False,
        )
        or "cross_attention_custom_mask" not in parameters
    ):
        raise RuntimeError(
            "MOSS-VL on Ascend requires frame-visibility attention support. "
            "Apply patches/npu/apply_npu_patches.sh in the serving environment."
        )

    for method_name in (
        "_handle_model_specific_adjustments",
        "_handle_attention_backend_compatibility",
    ):
        original = getattr(ServerArgs, method_name, None)
        if original is None or getattr(original, "_moss_npu_relaxed", False):
            continue

        @functools.wraps(original)
        def wrapper(self: Any, _original=original) -> None:
            try:
                _original(self)
            except AssertionError as exc:
                if _FLASHINFER_GUARD_MESSAGE not in str(exc):
                    raise
                logger.info(
                    "NPU: using Ascend cross-attention with frame visibility support"
                )

        wrapper._moss_npu_relaxed = True
        setattr(ServerArgs, method_name, wrapper)


__all__ = [
    "CUDA_ATTENTION_BACKEND",
    "NPU_ATTENTION_BACKEND",
    "device_spec",
    "is_npu_platform",
    "platform_device_type",
    "preferred_attention_backend",
    "relax_mossvl_flashinfer_guard",
]
