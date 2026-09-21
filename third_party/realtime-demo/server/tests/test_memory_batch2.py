"""Resource config, readiness and end-to-end deadline regressions."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from server.memory.pi_client import _post_json

DEPLOY = Path(__file__).resolve().parents[2] / "scripts/deploy"
sys.path.insert(0, str(DEPLOY))
spec = importlib.util.spec_from_file_location("memory_backend", DEPLOY / "memory_backend.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
sys.path.pop(0)


@pytest.fixture
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            srv = self.server
            srv.calls.append(float(self.headers["X-Memory-Timeout-Ms"]))
            self.rfile.read(int(self.headers["Content-Length"]))
            try:
                if srv.delay:
                    time.sleep(srv.delay)
                self.send_response(srv.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if srv.drip:
                    for byte in b'{"ok": true}':
                        self.wfile.write(bytes([byte])); self.wfile.flush()
                        time.sleep(0.08)
                else:
                    self.wfile.write(srv.body)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.status, server.delay, server.drip, server.body, server.calls = 200, 0, False, b'{"ok":true}', []
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server, f"http://127.0.0.1:{server.server_port}/decide"
    server.shutdown(); server.server_close(); worker.join(2)


@pytest.mark.parametrize("mode", ["headers", "body", "retry"])
def test_total_deadline_includes_slow_headers_body_and_backoff(endpoint, mode):
    server, url = endpoint
    if mode == "headers": server.delay = 0.8
    elif mode == "body": server.drip = True
    else: server.status = 503
    start = time.monotonic()
    assert _post_json(url, {}, 0.2) is None
    assert time.monotonic() - start < 0.55
    assert len(server.calls) == 1


def test_retry_uses_remaining_budget(endpoint):
    server, url = endpoint
    server.status = 503
    assert _post_json(url, {}, 2) is None
    assert len(server.calls) == 3
    assert server.calls[0] > server.calls[1] > server.calls[2]


@pytest.mark.parametrize("status", [400, 401, 413, 429])
def test_client_errors_are_not_retried(endpoint, status):
    server, url = endpoint
    server.status = status
    assert _post_json(url, {}, 2) is None
    assert len(server.calls) == 1


def test_malformed_json_is_not_retried(endpoint):
    server, url = endpoint
    server.body = b"not JSON"
    assert _post_json(url, {}, 2) is None
    assert len(server.calls) == 1


def test_defaults_and_overrides():
    _, port, desired, gpu, _ = backend.config({"OMNI_ROOT": "/tmp"})
    assert (port, gpu) == (38090, "1")
    assert [desired[k] for k in ("mem_fraction_static", "context_length", "max_running_requests", "max_total_tokens")] == [.2, 16384, 4, 65536]
    assert backend.config({"OMNI_ROOT": "/tmp", "DECIDE_LLM_CONTEXT_LENGTH": "8192"})[2]["context_length"] == 8192


def test_mismatch_and_gpu_identity(monkeypatch):
    desired = backend.config({"OMNI_ROOT": "/tmp"})[2]
    monkeypatch.setattr(backend, "get", lambda base, path: dict(desired))
    monkeypatch.setattr(backend.psutil, "net_connections", lambda **kw:
        [SimpleNamespace(status=backend.psutil.CONN_LISTEN, laddr=SimpleNamespace(port=38090), pid=123)])
    monkeypatch.setattr(backend.psutil, "Process", lambda pid: SimpleNamespace(environ=lambda: {"CUDA_VISIBLE_DEVICES": "1"}))
    assert backend.backend_differences("http://127.0.0.1:38090", desired, "1") == {}
    assert "CUDA_VISIBLE_DEVICES" in backend.backend_differences("http://127.0.0.1:38090", desired, "2")
    changed = dict(desired, context_length=8192)
    assert "context_length" in backend.backend_differences("http://127.0.0.1:38090", changed, "1")


def test_readiness_does_not_treat_failure_as_success():
    with pytest.raises(TimeoutError):
        backend.wait_ready(lambda: False, None, .01)
    with pytest.raises(RuntimeError, match="child exited"):
        backend.wait_ready(lambda: True, SimpleNamespace(poll=lambda: 1, returncode=1), 1)


def test_ssh_forwards_resource_overrides_without_secrets_in_argv(tmp_path):
    ssh = tmp_path / "ssh-stub"
    ssh.write_text('#!/bin/bash\nprintf "ARGS: %s\\n" "$*"\ncat\n')
    ssh.chmod(0o755)
    repo = DEPLOY.parents[1]
    env = dict(os.environ, GPU_SSH=str(ssh), _PI_ON_GPU="0", MOSS_DEPLOY_REPO=str(repo),
               DECIDE_LLM_MEM_FRAC="0.31", DECIDE_LLM_CONTEXT_LENGTH="8192",
               DECIDE_LLM_MAX_RUNNING_REQUESTS="2", DECIDE_LLM_MAX_TOTAL_TOKENS="32768",
               AIGW_API_KEY="test-only-secret")
    output = subprocess.check_output(["bash", str(repo / "scripts/gpu/start_pi_agent.sh")], env=env, text=True)
    assert "test-only-secret" not in output.splitlines()[0]
    for key in ("DECIDE_LLM_MEM_FRAC", "DECIDE_LLM_CONTEXT_LENGTH", "DECIDE_LLM_MAX_RUNNING_REQUESTS", "DECIDE_LLM_MAX_TOTAL_TOKENS"):
        assert f"export {key}={env[key]}" in output
