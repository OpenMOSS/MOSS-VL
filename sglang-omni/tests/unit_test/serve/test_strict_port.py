# SPDX-License-Identifier: Apache-2.0
"""SGLANG_OMNI_STRICT_PORT turns the port fallback into a hard error."""

from __future__ import annotations

import socket

import pytest

from sglang_omni.serve.launcher import _find_available_port


def test_free_port_is_returned_unchanged(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_STRICT_PORT", raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    assert _find_available_port("127.0.0.1", free) == free


def test_busy_port_falls_back_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_STRICT_PORT", raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        busy = holder.getsockname()[1]
        assert _find_available_port("127.0.0.1", busy) != busy


def test_busy_port_hard_errors_under_strict(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_STRICT_PORT", "1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        busy = holder.getsockname()[1]
        with pytest.raises(RuntimeError, match="STRICT_PORT"):
            _find_available_port("127.0.0.1", busy)


def test_time_wait_port_does_not_look_busy(monkeypatch):
    """A port with only TIME_WAIT leftovers must still be considered free.

    Regression: restarting the server right after a stop hard-failed under
    STRICT_PORT because the probe socket did not set SO_REUSEADDR while the
    real server does.
    """
    monkeypatch.setenv("SGLANG_OMNI_STRICT_PORT", "1")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    client = socket.create_connection(("127.0.0.1", port))
    conn, _ = listener.accept()
    # The accepted connection owns the port; closing the server side first
    # leaves it in TIME_WAIT.
    conn.close()
    client.close()
    listener.close()
    assert _find_available_port("127.0.0.1", port) == port
