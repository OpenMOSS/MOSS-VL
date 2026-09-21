"""Entry point: python -m server.vlm_worker --port 9000 --worker-id 0

The supervisor pins the GPU via CUDA_VISIBLE_DEVICES before spawn, so the
worker always addresses cuda:0. Standalone/debug runs work the same way:

    CUDA_VISIBLE_DEVICES=3 MODEL_PATH=... AUTOLOAD_VLM=1 \
        python -m server.vlm_worker --port 9003 --worker-id 3
"""
from __future__ import annotations

import argparse

import uvicorn

from .app import create_worker_app


def main() -> int:
    parser = argparse.ArgumentParser(description="MOSS-VL realtime worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    args = parser.parse_args()

    # log_config=None: uvicorn installs NO handlers of its own — its records
    # (banner, access lines) propagate to the root logger, i.e. into this
    # worker's rotating MOSS_LOG_FILE. Raw stdout is discarded by the spawner.
    uvicorn.run(create_worker_app(worker_id=args.worker_id),
                host=args.host, port=args.port, log_level="info",
                log_config=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
