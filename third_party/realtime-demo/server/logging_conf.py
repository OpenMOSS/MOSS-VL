"""Minimal structured logging, shared by every module.

Always logs to stdout. When MOSS_LOG_FILE is set, ALSO writes a size-rotated
file (32 MiB × 5) — the logs/handler/ tree. One file per PROCESS — the gateway
and each vlm_worker set their own path (run_backend.sh / gpu/supervisor.py);
rotation is not safe on a file shared across processes. Unset (tests, ad-hoc
scripts) = stdout only, exactly the pre-file behavior.

uvicorn's loggers (error AND access) are taken over: their own handlers are
removed and records propagate to the root handlers instead, so access lines
land in the rotating file rather than on block-buffered raw stdout. The
takeover re-runs on every configure_logging() call because an in-process
`uvicorn.Config` (tests, embedded servers) re-installs uvicorn's dictConfig
after import time — the lifespan's configure_logging() undoes that.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_CONFIGURED = False

LOG_FILE_MAX_BYTES = 32 * 2**20
LOG_FILE_BACKUPS = 5
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def _takeover_uvicorn() -> None:
    """Route uvicorn/uvicorn.error/uvicorn.access through the root handlers."""
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(name)
        uv.handlers.clear()
        uv.propagate = True
        uv.setLevel(logging.NOTSET)  # root's LOG_LEVEL governs


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        _takeover_uvicorn()  # re-assert after any later uvicorn dictConfig
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handlers: list[logging.Handler] = [handler]
    log_file = os.getenv("MOSS_LOG_FILE", "")
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUPS, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers[:] = handlers
    # third-party noise
    for noisy in ("httpx", "urllib3", "websockets", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _takeover_uvicorn()
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
