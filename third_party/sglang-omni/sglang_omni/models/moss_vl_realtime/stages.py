"""Pipeline stage factory for MOSS-VL realtime."""

from __future__ import annotations

from typing import Any


def create_sglang_moss_vl_realtime_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    dp_rank: int = 0,
    dp_size: int = 1,
    nccl_port: int | None = None,
    dtype: str = "bfloat16",
    max_running_requests: int = 1,
    max_new_tokens: int = 4096,
    context_length: int = 262144,
    mem_fraction_static: float | None = 0.40,
    server_args_overrides: dict[str, Any] | None = None,
    parked_request_timeout_s: float = 300.0,
    disable_cuda_graph: bool = False,
    page_size: int = 1,
    enable_async_decode: bool = False,
    frame_resolver: Any = None,
    realtime_frame_window_enabled: bool | None = None,
    realtime_frame_window_raw_s: float | None = None,
    realtime_frame_pool_window_s: float | None = None,
    realtime_frame_pool_ratio: int | None = None,
    realtime_frame_pooling_enabled: bool | None = None,
):
    from sglang_omni.models.moss_vl_realtime.engine_builder import (
        MossVLRealtimeEngineBuilder,
    )
    from sglang_omni.models.moss_vl_realtime.frame_window import (
        RealtimeFrameWindowConfig,
    )

    frame_window_config = RealtimeFrameWindowConfig.resolve(
        enabled=realtime_frame_window_enabled,
        raw_window_s=realtime_frame_window_raw_s,
        pool_window_s=realtime_frame_pool_window_s,
        pool_ratio=realtime_frame_pool_ratio,
        pooling_enabled=realtime_frame_pooling_enabled,
    )
    return MossVLRealtimeEngineBuilder(
        max_running_requests=max_running_requests,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        mem_fraction_static=mem_fraction_static,
        parked_request_timeout_s=parked_request_timeout_s,
        disable_cuda_graph=disable_cuda_graph,
        page_size=page_size,
        enable_async_decode=enable_async_decode,
        frame_resolver=frame_resolver,
        frame_window_config=frame_window_config,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=tp_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
        nccl_port=nccl_port,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


__all__ = ["create_sglang_moss_vl_realtime_executor"]
