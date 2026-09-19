"""Single-GPU MOSS-VL Realtime server launcher."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import sys
import traceback
from pathlib import Path

from common import (
    CONFIG,
    environment,
    free_port,
    gpu_inventory,
    select_gpus,
    server_command,
    validate_model,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("model_path", type=Path)
    serve.add_argument("--gpus", help="Physical GPU index/UUID; default: one idle GPU")
    serve.add_argument("--dry-run", action="store_true")
    serve.add_argument("--host", default=CONFIG["host"])
    serve.add_argument("--port", type=int, default=CONFIG["port"])
    serve.add_argument("--vl-api-v2-port", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.model_path = validate_model(args.model_path)
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.vl_api_v2_port is not None and (
        not 1 <= args.vl_api_v2_port <= 65535 or args.vl_api_v2_port == args.port
    ):
        raise ValueError("--vl-api-v2-port must be valid and distinct from --port")
    if args.dry_run:
        details = dict(
            model_path=str(args.model_path),
            config=CONFIG,
            host=args.host,
            port=args.port,
            gpus=args.gpus or "one idle GPU",
        )
        if args.vl_api_v2_port is not None:
            details["vl_api_v2_port"] = args.vl_api_v2_port
        print(
            json.dumps(
                details,
                indent=2,
            )
        )
        return 0
    if importlib.metadata.version("transformers") != "5.12.1":
        raise ValueError(
            "Activate the repository backend environment (Transformers 5.12.1)"
        )
    gpus = select_gpus(
        args.gpus, gpu_inventory(), os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    dp_size = int(CONFIG.get("dp_size", 1))
    if len(gpus) != dp_size:
        if dp_size == 1:
            raise ValueError(
                "start.sh launches a single-GPU instance; "
                "provide at most one --gpus value"
            )
        raise ValueError(
            f"start.sh needs {dp_size} GPUs for dp_size={dp_size}; "
            f"got {len(gpus)} from --gpus"
        )
    free_port(args.host, args.port)
    if args.vl_api_v2_port is not None:
        free_port(args.host, args.vl_api_v2_port)
    command = server_command(args.model_path, args.port, args.host)
    if args.vl_api_v2_port is not None:
        command.extend(["--vl-api-v2-port", str(args.vl_api_v2_port)])
    print(
        f'GPU(s) {[gpu["index"] for gpu in gpus]}; '
        f'{CONFIG["max_sessions"]} sessions x {dp_size} replica(s); '
        f"http://{args.host}:{args.port}/health",
        flush=True,
    )
    # environment() narrows CUDA_VISIBLE_DEVICES; replica placement then
    # addresses device-local ids 0..dp_size-1 (see server_command).
    os.execve(
        sys.executable,
        command,
        environment(",".join(gpu["uuid"] for gpu in gpus)),
    )


if __name__ == "__main__":

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
