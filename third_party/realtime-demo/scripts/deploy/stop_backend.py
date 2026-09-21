"""Stop this checkout's Demo API with a five-second grace period (Linux)."""

import argparse
import math
from pathlib import Path
import signal
import socket
import sys
import time

import psutil


def is_owned(proc, repo, roles, port=None):
    """Only explicitly tagged children belong to a managed component.

    psutil.Process retains creation time and checks PID reuse before signaling.
    Unmarked legacy sidecars are deliberately not adopted by shutdown.
    """
    try:
        env = proc.environ()
        return (env.get("MOSS_DEPLOY_REPO") == str(repo)
                and env.get("MOSS_DEPLOY_ROLE") in roles
                and (port is None or env.get("MOSS_DEPLOY_PORT") == str(port))
                and proc.pid != psutil.Process().pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def require_free_port(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(f"Port {port} is occupied; refusing to kill an unowned listener: {exc}")


def is_backend(proc, repo, port=None):
    try:
        cwd = Path(proc.cwd()).resolve(strict=True)
        args = proc.cmdline()
    except (FileNotFoundError, psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    if cwd != repo and repo not in cwd.parents:
        return False
    # Match an actual uvicorn invocation, not shell command text mentioning it.
    module = any(args[i:i + 2] == ["-m", "uvicorn"] for i in range(len(args) - 1))
    executable = any(arg.rsplit("/", 1)[-1] == "uvicorn" for arg in args[:2])
    if not (module or executable) or "server.app:app" not in args:
        return False
    if port is not None:
        configured = "8000"
        for i, arg in enumerate(args):
            if arg == "--port" and i + 1 < len(args):
                configured = args[i + 1]
            elif arg.startswith("--port="):
                configured = arg.split("=", 1)[1]
        if configured != str(port):
            return False
    return True


def still_running(proc):
    try:
        # is_running checks process identity, including PID reuse. A zombie
        # has exited and owns no threads, even if its parent has not reaped it.
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def wait_exited(processes, timeout):
    pending = list(processes)
    deadline = time.monotonic() + timeout
    while pending:
        pending = [proc for proc in pending if still_running(proc)]
        remaining = deadline - time.monotonic()
        if not pending or remaining <= 0:
            break
        time.sleep(min(0.05, remaining))
    return pending


def stop_processes(processes, timeout=5.0):
    """TERM, then KILL on expiry; report survivors without blocking restart."""
    for proc in processes:
        try:
            proc.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess:
            pass
    pending = wait_exited(processes, timeout)
    for proc in pending:
        print(f"Demo API pid {proc.pid} exceeded {timeout:g}s; sending SIGKILL", flush=True)
        try:
            proc.send_signal(signal.SIGKILL)
        except psutil.NoSuchProcess:
            pass
    # KILL is asynchronous. Bound confirmation too: a task stuck in kernel
    # I/O must not make the deployment script wait indefinitely.
    pending = wait_exited(pending, 2.0)
    if pending:
        pids = [proc.pid for proc in pending]
        print(f"WARNING: Demo API processes have not exited after SIGKILL: {pids}; "
              "continuing restart attempt; ports/resources may still be occupied", file=sys.stderr, flush=True)
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--require-free-port", type=int)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("--timeout must be finite and non-negative")
    repo = args.repo.resolve(strict=True)
    def matches(proc):
        return (is_owned(proc, repo, args.role, args.port) if args.role else is_backend(proc, repo, args.port))
    processes = [proc for proc in psutil.process_iter() if matches(proc)]
    if processes:
        print(f"Stopping Demo API {[proc.pid for proc in processes]}; grace period {args.timeout:g}s", flush=True)
    survivors = stop_processes(processes, args.timeout)
    known = {proc.pid for proc in survivors}
    remaining = [proc.pid for proc in psutil.process_iter()
                 if proc.pid not in known and matches(proc)]
    if remaining:
        print(f"WARNING: Demo API processes appeared during shutdown: {remaining}; "
              "continuing restart attempt; ports/resources may still be occupied", file=sys.stderr, flush=True)
    if args.require_free_port:
        require_free_port(args.require_free_port)


if __name__ == "__main__":
    main()
