#!/usr/bin/env python3
"""Build the MOSS-VL realtime engine and report its selected runtime types."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--mem-fraction-static", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("gpu build: import", flush=True)
    import torch

    from sglang_omni.models.moss_vl_realtime.stages import (
        create_sglang_moss_vl_realtime_executor,
    )

    print("gpu build: loading", flush=True)
    try:
        scheduler = create_sglang_moss_vl_realtime_executor(
            args.model_path,
            device=args.device,
            max_running_requests=1,
            max_new_tokens=args.max_new_tokens,
            context_length=args.context_length,
            mem_fraction_static=args.mem_fraction_static,
        )
        print(
            "gpu build: ready",
            f"scheduler={type(scheduler).__module__}.{type(scheduler).__name__}",
            f"runner={type(scheduler._model_runner).__name__}",
            f"model={type(scheduler.tp_worker.model_runner.model).__name__}",
            flush=True,
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
