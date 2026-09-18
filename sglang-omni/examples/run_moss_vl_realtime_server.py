#!/usr/bin/env python3
"""Launch the MOSS-VL realtime pipeline and binary-frame WebSocket API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_python_bin_on_path() -> None:
    """Expose venv console scripts to FlashInfer JIT subprocesses."""
    python_bin = str(Path(sys.executable).parent)
    path_entries = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and entry != python_bin
    ]
    os.environ["PATH"] = os.pathsep.join([python_bin, *path_entries])


def parse_args() -> argparse.Namespace:
    from sglang_omni.models.moss_vl_realtime.platform_compat import (
        is_npu_platform,
        preferred_attention_backend,
    )

    is_npu = is_npu_platform()
    backend = preferred_attention_backend()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--vl-api-v2-port",
        type=int,
        default=None,
        help="Opt-in separate VL API v2 listener sharing the same model",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids for TP deployment, one GPU per rank. "
        "With --dp-size, one GPU per (replica, rank): exactly "
        "tp_size * dp_size ids.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data-parallel replicas of the entry stage; each replica is an "
        "independent engine (its own TP group when --tp-size > 1) with its own "
        "session slots. Sessions are pinned to one replica for their lifetime.",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.40)
    parser.add_argument("--context-length", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--parked-request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=1,
        help="Maximum concurrent realtime sessions (one live request each). "
        "Default 1; values above 1 enable multi-session serving.",
    )
    parser.add_argument(
        "--enable-decode-cuda-graph",
        dest="decode_cuda_graph",
        action="store_true",
        default=not is_npu,
        help="Capture CUDA graphs for stable-shape decode steps "
        "(frame extend stays eager). Default on for CUDA, off for NPU; "
        "validated in P11.",
    )
    parser.add_argument(
        "--disable-decode-cuda-graph",
        dest="decode_cuda_graph",
        action="store_false",
        help="Fall back to eager decode.",
    )
    parser.add_argument(
        "--mm-attention-backend",
        default=None,
        help="Optional server_args override for the multimodal (vision) "
        "attention backend, e.g. ascend_attn on NPU for memory-efficient "
        "fused ViT attention.",
    )
    parser.add_argument(
        "--decode-attention-backend",
        default=None,
        help=f"Optional server_args override for the decode attention backend "
        f"(default {backend}). Must be {backend} when decode CUDA graph is on.",
    )
    parser.add_argument(
        "--enable-async-decode",
        dest="enable_async_decode",
        action="store_true",
        default=False,
        help="Launch decode step N+1 before resolving step N (lookahead). "
        "Default off; validated in P10.5.",
    )
    parser.add_argument(
        "--enable-benchmark-mode",
        action="store_true",
        default=False,
        help="Allow benchmark-only WebSocket options such as "
        "benchmark_ignore_eos. Never enable for production traffic.",
    )
    parser.add_argument(
        "--disable-startup-warmup",
        action="store_true",
        help="Skip the default internal frame warmup for diagnostics.",
    )
    args = parser.parse_args()
    if args.tp_size < 1:
        parser.error("--tp-size must be at least 1")
    if args.dp_size < 1:
        parser.error("--dp-size must be at least 1")
    if args.max_running_requests < 1:
        parser.error("--max-running-requests must be at least 1")
    parallelism = args.tp_size * args.dp_size
    if parallelism > 1:
        if args.gpus is None:
            parser.error("--tp-size/--dp-size above 1 requires --gpus")
        try:
            args.gpus = [int(value.strip()) for value in args.gpus.split(",")]
        except ValueError:
            parser.error("--gpus must be a comma-separated list of integers")
        if len(args.gpus) != parallelism:
            parser.error(
                f"--gpus must contain exactly tp_size * dp_size "
                f"({args.tp_size} * {args.dp_size} = {parallelism}) GPU ids"
            )
        if len(set(args.gpus)) != len(args.gpus):
            parser.error("--gpus must not contain duplicate GPU ids")
    elif args.gpus is not None:
        parser.error(
            "--gpus only applies when --tp-size/--dp-size > 1; "
            "use --gpu for a single replica"
        )
    if args.decode_cuda_graph and args.decode_attention_backend not in (
        None,
        backend,
    ):
        # fa3 decode-graph replay overflows req_to_token rows for the
        # encoder-prefix KV layout (see perf_p10_3/server_graph_blocking.log).
        parser.error(
            f"--decode-attention-backend must be {backend} when decode CUDA "
            "graph is enabled"
        )
    return args


def main() -> None:
    args = parse_args()
    _ensure_python_bin_on_path()
    from sglang_omni.models.moss_vl_realtime.config import MossVLRealtimePipelineConfig
    from sglang_omni.models.moss_vl_realtime.platform_compat import (
        device_spec,
        preferred_attention_backend,
    )
    from sglang_omni.serve import launch_server

    backend = preferred_attention_backend()
    config = MossVLRealtimePipelineConfig(model_path=args.model_path)
    stage = config.stages[0]
    stage.gpu = args.gpus if args.tp_size * args.dp_size > 1 else args.gpu
    stage.tp_size = args.tp_size
    stage.parallelism.tp = args.tp_size
    stage.dp_size = args.dp_size
    stage.parallelism.dp = args.dp_size
    factory_args = dict(stage.factory_args)
    factory_args.update(
        {
            "device": (
                device_spec(0) if args.tp_size * args.dp_size > 1
                else device_spec(args.gpu)
            ),
            "mem_fraction_static": args.mem_fraction_static,
            "context_length": args.context_length,
            "max_new_tokens": args.max_new_tokens,
            "parked_request_timeout_s": args.parked_request_timeout,
            "max_running_requests": args.max_running_requests,
            "disable_cuda_graph": not args.decode_cuda_graph,
            "page_size": 1,
            "enable_async_decode": args.enable_async_decode,
        }
    )
    server_args_overrides = dict(factory_args.get("server_args_overrides") or {})
    if args.mm_attention_backend is not None:
        server_args_overrides.update(
            {"mm_attention_backend": args.mm_attention_backend}
        )
    if args.decode_attention_backend is not None:
        server_args_overrides.update(
            {
                # Setting any single backend dimension stops the upstream MossVL
                # override from injecting its prefill default; pin both.
                "prefill_attention_backend": backend,
                "decode_attention_backend": args.decode_attention_backend,
            }
        )
    if server_args_overrides:
        factory_args["server_args_overrides"] = server_args_overrides
    stage.factory_args = factory_args
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name="moss-vl-realtime",
        video_realtime_warmup=not args.disable_startup_warmup,
        video_realtime_benchmark_mode=args.enable_benchmark_mode,
        vl_api_v2_port=args.vl_api_v2_port,
    )


if __name__ == "__main__":
    main()
