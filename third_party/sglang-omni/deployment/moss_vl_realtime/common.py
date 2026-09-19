"""Shared model validation, GPU selection and process management."""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE / "config.json").read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def validate_model(path):
    path = path.expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(
            "Provide a downloaded local model directory containing config.json"
        )
    config = json.loads((path / "config.json").read_text())
    if not any("MossVL" in name for name in config.get("architectures", [])):
        raise ValueError("Expected a MOSS-VL checkpoint")
    return path


def environment(gpu):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.lower().endswith("_proxy")
        and not k.startswith(("REALTIME_FRAME_", "MOSS_DELIVERY_"))
    }
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        NO_PROXY="*",
        no_proxy="*",
        PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=os.pathsep.join([str(ROOT), str(HERE)]),
        REALTIME_FRAME_WINDOW_ENABLED="1",
        REALTIME_FRAME_WINDOW_RAW_S=str(CONFIG["raw_window_seconds"]),
        REALTIME_FRAME_POOLING_ENABLED="0",
        SGLANG_OMNI_STRICT_PORT="1",
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def gpu_inventory():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    return [
        dict(
            index=r[0].strip(),
            uuid=r[1].strip(),
            name=r[2].strip(),
            total_mib=int(r[3]),
            used_mib=int(r[4]),
        )
        for r in csv.reader(output.splitlines())
    ]


def select_gpus(requested, inventory, visible=None):
    ids = (
        None
        if visible is None
        else [v.strip() for v in visible.split(",") if v.strip()]
    )
    available = [
        r for r in inventory if ids is None or r["index"] in ids or r["uuid"] in ids
    ]
    if requested:
        wanted = [v.strip() for v in requested.split(",")]
        if len(wanted) != len(set(wanted)) or not all(wanted):
            raise ValueError("--gpus requires distinct GPU indices or UUIDs")
        picked = []
        for gpu in wanted:
            matches = [r for r in available if gpu in (r["index"], r["uuid"])]
            if len(matches) != 1:
                raise ValueError(f"GPU {gpu} is not visible")
            picked.append(matches[0])
        if len({r["uuid"] for r in picked}) != len(picked):
            raise ValueError("--gpus resolves to duplicate physical GPUs")
    else:
        picked = [
            r
            for r in available
            if r["used_mib"] <= CONFIG["maximum_existing_memory_mib"]
            and (r["total_mib"] - r["used_mib"]) / 1024 >= CONFIG["minimum_free_gib"]
        ][:1]
    if not picked:
        raise ValueError(
            "No idle GPU has sufficient free VRAM; release a GPU or specify --gpus"
        )
    for r in picked:
        if r["used_mib"] > CONFIG["maximum_existing_memory_mib"]:
            raise ValueError(
                f"GPU {r['index']} is occupied; existing processes are not stopped"
            )
        if (r["total_mib"] - r["used_mib"]) / 1024 < CONFIG["minimum_free_gib"]:
            raise ValueError(
                f"GPU {r['index']} has insufficient free VRAM for the 128K profile"
            )
    return picked


def free_port(host="127.0.0.1", port=0):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return sock.getsockname()[1]


def server_command(model, port, host="127.0.0.1"):
    entry = ROOT / "examples/run_moss_vl_realtime_server.py"
    command = [
        sys.executable,
        "-u",
        str(entry),
        "--model-path",
        str(model),
        "--gpu",
        "0",
        "--host",
        host,
        "--port",
        str(port),
        "--context-length",
        str(CONFIG["context_length"]),
        "--mem-fraction-static",
        str(CONFIG["mem_fraction_static"]),
        "--max-running-requests",
        str(CONFIG["max_sessions"]),
        "--parked-request-timeout",
        str(CONFIG["parked_timeout_seconds"]),
    ]
    # Replica GPUs are device-local once CUDA_VISIBLE_DEVICES is narrowed to
    # the selected GPU set, so the launcher pwids run 0..dp_size-1.
    dp_size = int(CONFIG.get("dp_size", 1))
    if dp_size > 1:
        command.extend(
            [
                "--dp-size",
                str(dp_size),
                "--gpus",
                ",".join(str(index) for index in range(dp_size)),
            ]
        )
    return command


class Child:
    def __init__(self, command, env, log, *, new_group=True):
        self.log = Path(log).open("w")
        self.new_group = new_group
        self._stopped = False
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=new_group,
            )
            self._owner = psutil.Process(self.process.pid)
            self._owner_started = self._owner.create_time()
        except BaseException:
            try:
                process = getattr(self, "process", None)
                if process is not None:
                    try:
                        if self.new_group:
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
            finally:
                self.log.close()
            raise

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Child {self.process.pid} exceeded {timeout}s")
            time.sleep(0.2)
        return self.process.returncode

    def _live_owned_processes(self):
        self.process.poll()
        if not self.new_group:
            candidates = [self._owner]
        else:
            # A recycled group-leader PID belongs to a different run. Never
            # signal it, even if stop() is called long after our leader exited.
            try:
                if (
                    psutil.Process(self.process.pid).create_time()
                    != self._owner_started
                ):
                    return []
            except psutil.NoSuchProcess:
                pass
            candidates = []
            for process in psutil.process_iter():
                try:
                    if os.getpgid(process.pid) == self.process.pid:
                        candidates.append(process)
                except (ProcessLookupError, PermissionError):
                    continue
        alive = []
        for process in candidates:
            try:
                if (
                    process.is_running()
                    and process.create_time() >= self._owner_started
                    and process.status() != psutil.STATUS_ZOMBIE
                ):
                    alive.append(process)
            except psutil.NoSuchProcess:
                pass
        return alive

    def _wait_owned_processes(self, timeout, *, kill=False):
        deadline = time.monotonic() + timeout
        signalled_orphans = set()
        while True:
            alive = self._live_owned_processes()
            if kill and alive:
                # Catch a descendant forked between the initial snapshot and
                # the first KILL, rather than leaving it behind on timeout.
                self._signal_processes(alive, signal.SIGKILL)
            elif alive and self.process.poll() is not None:
                for process in alive:
                    try:
                        identity = (process.pid, process.create_time())
                    except psutil.NoSuchProcess:
                        continue
                    if identity not in signalled_orphans:
                        self._signal_processes([process], signal.SIGTERM)
                        signalled_orphans.add(identity)
            remaining = deadline - time.monotonic()
            if not alive or remaining <= 0:
                return alive
            time.sleep(min(0.05, remaining))

    @staticmethod
    def _signal_processes(processes, sig):
        for process in processes:
            try:
                # psutil checks PID identity before delivering the signal.
                process.send_signal(sig)
            except psutil.NoSuchProcess:
                pass

    def stop(self, timeout=30.0, kill_timeout=10.0):
        if any(
            not math.isfinite(value) or value < 0 for value in (timeout, kill_timeout)
        ):
            raise ValueError("shutdown timeouts must be finite and non-negative")
        if self._stopped:
            return
        try:
            alive = self._live_owned_processes()
            leaders = [p for p in alive if p.pid == self.process.pid]
            # Let the parent coordinate worker/IPC cleanup before escalation.
            self._signal_processes(leaders or alive, signal.SIGTERM)
            alive = self._wait_owned_processes(timeout)
            if alive:
                self._signal_processes(alive, signal.SIGKILL)
                alive = self._wait_owned_processes(kill_timeout, kill=True)
            if alive:
                raise TimeoutError(
                    f"Child processes did not exit: {[p.pid for p in alive]}"
                )
            self.process.wait(timeout=kill_timeout)
            self._stopped = True
        finally:
            self.log.close()
