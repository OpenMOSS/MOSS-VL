"""Supervise native Ascend backends and the shared legacy VL adapters."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment/moss_vl_realtime"))
from common import Child, free_port, validate_model

from deployment.npu.config import recommended_deploy_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("start", "start4", "start8"), default="start"
    )
    parser.add_argument("--model-path", type=Path, default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument(
        "--adapter-port", type=int, default=int(os.environ.get("ADAPTER_PORT", "18600"))
    )
    parser.add_argument("--startup-timeout", type=float, default=600)
    args = parser.parse_args()
    if args.model_path is None:
        parser.error("provide --model-path or MODEL_PATH")
    model = validate_model(args.model_path)
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("startup-timeout must be finite and positive")

    defaults = {
        "start": os.environ.get("GPUS") or os.environ.get("GPU", "0"),
        "start4": "0,1;2,3",
        "start8": "0,1,2,3;4,5,6,7",
    }
    try:
        groups = [
            [int(value) for value in group.split(",")]
            for group in os.environ.get("NPU_GROUPS", defaults[args.mode]).split(";")
        ]
    except ValueError:
        parser.error(
            "NPU_GROUPS must contain semicolon-separated groups of integer device IDs"
        )
    devices = [device for group in groups for device in group]
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        parser.error("each logical device must appear in exactly one group")
    try:
        profiles = [
            recommended_deploy_params(len(group), os.environ) for group in groups
        ]
    except ValueError as exc:
        parser.error(str(exc))
    backend_ports = [args.port + index for index in range(len(groups))]
    adapter_ports = [args.adapter_port + index for index in range(len(groups))]
    ports = backend_ports + adapter_ports
    if any(not 1 <= port <= 65535 for port in ports) or len(set(ports)) != len(ports):
        parser.error("backend and adapter ports must be distinct valid ports")
    for port in ports:
        free_port(port=port)

    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in ("http_proxy", "https_proxy", "all_proxy")
    }
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    log_dir = Path(tempfile.mkdtemp(prefix="moss-npu-"))
    children = []
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    probe_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    if probe_host == "::":
        probe_host = "::1"
    authority = f"[{probe_host}]" if ":" in probe_host else probe_host
    print(f"Logs: {log_dir}", flush=True)
    try:
        for index, group in enumerate(groups):
            if stop.is_set():
                break
            port, adapter_port = backend_ports[index], adapter_ports[index]
            backend_env = dict(env)
            # Give independent TP groups separate rendezvous/socket ranges.
            backend_env["HCCL_IF_BASE_PORT"] = str(
                int(env.get("HCCL_IF_BASE_PORT", "61000")) + index * 100
            )
            base = 62000 + index * 200
            backend_env["HCCL_NPU_SOCKET_PORT_RANGE"] = f"{base}-{base + 199}"
            profile = profiles[index]
            command = [
                sys.executable,
                str(ROOT / "examples/run_moss_vl_realtime_server.py"),
                "--model-path",
                str(model),
                "--host",
                args.host,
                "--port",
                str(port),
                "--context-length",
                str(profile["context_length"]),
                "--mem-fraction-static",
                str(profile["mem_fraction_static"]),
                "--max-running-requests",
                str(profile["max_running_requests"]),
            ]
            command += shlex.split(os.environ.get("EXTRA_SERVER_ARGS", ""))
            if len(group) == 1:
                command += ["--gpu", str(group[0])]
            else:
                command += [
                    "--tp-size",
                    str(len(group)),
                    "--gpus",
                    ",".join(map(str, group)),
                ]
            child = Child(command, backend_env, log_dir / f"backend-{port}.log")
            children.append(child)
            deadline = time.monotonic() + args.startup_timeout
            healthy = False
            while not stop.is_set() and time.monotonic() < deadline:
                if any(process.process.poll() is not None for process in children):
                    raise RuntimeError(f"a child exited during startup; see {log_dir}")
                try:
                    with opener.open(
                        f"http://{authority}:{port}/health", timeout=2
                    ) as response:
                        health = json.load(response)
                    healthy = (
                        health.get("status") == "healthy"
                        and health.get("running") is True
                    )
                except (OSError, ValueError):
                    pass
                if healthy:
                    break
                stop.wait(0.5)
            if stop.is_set():
                break
            if not healthy:
                raise TimeoutError(
                    f"backend {port} did not become healthy; see {log_dir}"
                )
            adapter_env = dict(
                env,
                LISTEN_HOST=args.host,
                LISTEN_PORT=str(adapter_port),
                OMNI_WS_URL=f"ws://{authority}:{port}/v1/video/realtime",
                MAX_INFLIGHT=env.get("MAX_INFLIGHT", "1"),
            )
            adapter_env["PYTHONPATH"] = (
                str(ROOT / "vl_legacy_adapter") + os.pathsep + env["PYTHONPATH"]
            )
            children.append(
                Child(
                    [sys.executable, "-m", "adapter"],
                    adapter_env,
                    log_dir / f"adapter-{adapter_port}.log",
                )
            )
            print(
                f"Native :{port}/v1/video/realtime; legacy :{adapter_port}/v1/realtime; "
                f"logical devices {group}",
                flush=True,
            )
        while not stop.wait(0.5):
            if any(child.process.poll() is not None for child in children):
                raise RuntimeError(f"a serving child exited; see {log_dir}")
    finally:
        # Reuse the repository's parent-first, owned-process cleanup.
        errors = []
        for child in reversed(children):
            try:
                child.stop()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("serving cleanup failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
