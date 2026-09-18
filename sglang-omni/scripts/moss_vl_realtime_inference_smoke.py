#!/usr/bin/env python3
"""Run one timestamped frame through the MOSS-VL realtime scheduler."""

from __future__ import annotations

import argparse
import queue
import threading
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--second-frame")
    parser.add_argument("--prompt", default="Describe what is visible.")
    parser.add_argument("--timestamp", type=float, default=0.0)
    parser.add_argument("--second-timestamp", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--mem-fraction-static", type=float, default=0.40)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from sglang_omni.models.moss_vl_realtime.payload_types import FramePromptEvent
    from sglang_omni.models.moss_vl_realtime.stages import (
        create_sglang_moss_vl_realtime_executor,
    )
    from sglang_omni.proto import OmniRequest, StagePayload
    from sglang_omni.scheduling.messages import IncomingMessage

    request_id = "moss-vl-realtime-smoke"
    session_id = "moss-vl-realtime-smoke-session"
    scheduler = None
    scheduler_thread = None
    print("gpu inference: loading", flush=True)
    try:
        scheduler = create_sglang_moss_vl_realtime_executor(
            args.model_path,
            device=args.device,
            max_running_requests=1,
            max_new_tokens=args.max_new_tokens,
            context_length=args.context_length,
            mem_fraction_static=args.mem_fraction_static,
        )
        payload = StagePayload(
            request_id=request_id,
            request=OmniRequest(
                inputs={
                    "initial_prompt": args.prompt,
                    "session_id": session_id,
                },
                params={
                    "stream": True,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": 0.0,
                },
            ),
            data={},
        )
        event = FramePromptEvent(
            request_id=request_id,
            session_id=session_id,
            seq_no=0,
            timestamp=args.timestamp,
            frame_ref=args.frame,
            final=args.second_frame is None,
        )
        scheduler.inbox.put(IncomingMessage(request_id, "new_request", payload))
        scheduler.inbox.put(
            IncomingMessage(request_id, "request_update", event.to_dict())
        )
        if args.second_frame is not None:
            second_event = FramePromptEvent(
                request_id=request_id,
                session_id=session_id,
                seq_no=1,
                timestamp=args.second_timestamp,
                frame_ref=args.second_frame,
                final=True,
            )
            scheduler.inbox.put(
                IncomingMessage(
                    request_id,
                    "request_update",
                    second_event.to_dict(),
                )
            )
        scheduler_thread = threading.Thread(
            target=scheduler.start,
            name="moss-vl-realtime-smoke-scheduler",
            daemon=True,
        )
        scheduler_thread.start()
        print("gpu inference: request submitted", flush=True)

        deadline = time.monotonic() + args.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("realtime inference did not finish before timeout")
            try:
                message = scheduler.outbox.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if not scheduler_thread.is_alive():
                    raise RuntimeError("scheduler exited without a terminal message")
                continue
            if message.type == "stream":
                print(f"gpu inference: stream {message.data!r}", flush=True)
                continue
            if message.type == "error":
                raise RuntimeError(f"scheduler error: {message.data}")
            if message.type == "result":
                result = message.data
                print(f"gpu inference: result {result.data!r}", flush=True)
                break
    finally:
        if scheduler is not None:
            scheduler.stop()
        if scheduler_thread is not None:
            scheduler_thread.join(timeout=30)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
