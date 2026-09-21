#!/usr/bin/env python3
"""tee with real size rotation — stdin → rotated file (+ stdout echo).

The box ships no logrotate/rotatelogs/svlogd, and plain `tee -a` grows
unbounded, so every raw stdout capture (tmux windows, the pytorch TTS
sidecar pipeline) goes through this instead:

    producer 2>&1 | rotating_tee.py [--quiet] [--max-bytes N] [--backups K] PATH

Contract (the pipe is load-bearing — a dead tee SIGPIPEs the producer):
- lifecycle is stdin EOF ONLY: SIGTERM/SIGINT/SIGHUP are ignored so a tmux
  respawn-window -k or a killpg teardown can't sever the pipe before the
  producer is gone; SIGKILL escalation still reaps us.
- file errors (rotation race, full disk) never abort the pump — drop the
  chunk and keep reading.
- a closed downstream (pane gone, `| head`) permanently disables the echo
  but the file keeps rotating.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="log file (parent dirs are created)")
    parser.add_argument("--max-bytes", type=int, default=32 * 2**20,
                        help="rotate when the file would exceed this (default 32 MiB)")
    parser.add_argument("--backups", type=int, default=5,
                        help="rotated generations to keep (.1 newest; 0 = truncate in place)")
    parser.add_argument("--quiet", action="store_true", help="no stdout echo")
    args = parser.parse_args()

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_IGN)

    path = os.path.abspath(args.path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = open(path, "ab")  # noqa: SIM115 — long-lived, reopened on rotation
    echo = not args.quiet

    def rotate():
        nonlocal out
        try:
            out.close()
        except OSError:
            pass
        try:
            if args.backups > 0:
                for i in range(args.backups - 1, 0, -1):
                    if os.path.exists(f"{path}.{i}"):
                        os.replace(f"{path}.{i}", f"{path}.{i + 1}")
                os.replace(path, f"{path}.1")
            else:
                os.replace(path, path + ".tmp")  # truncate via rename+drop
                os.unlink(path + ".tmp")
        except OSError:
            pass
        try:
            out = open(path, "ab")  # noqa: SIM115
        except OSError:
            out = None

    for chunk in iter(lambda: sys.stdin.buffer.readline(1 << 20), b""):
        try:
            if out is None:
                out = open(path, "ab")  # noqa: SIM115 — retry after a failed rotate
            if out.tell() > 0 and out.tell() + len(chunk) > args.max_bytes:
                rotate()
            if out is not None:
                out.write(chunk)
                out.flush()
        except (OSError, ValueError):
            out = None  # full disk / racing writer — keep pumping regardless
        if echo:
            try:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            except (BrokenPipeError, OSError):
                echo = False
                # silence the interpreter-exit flush of the broken pipe too
                try:
                    os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
                except OSError:
                    pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
