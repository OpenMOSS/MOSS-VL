"""Exercise process termination without touching any deployed services."""

import importlib.util
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]
STOP = REPO / "scripts/deploy/stop_backend.py"
spec = importlib.util.spec_from_file_location("stop_backend", STOP)
shutdown = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shutdown)
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux deployment tests")


@pytest.fixture
def launch(tmp_path):
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "uvicorn.py").write_text(
        "import os, signal, time\n"
        "def stop(*_):\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN if os.environ['STUB_STUCK'] == '1' else stop)\n"
        "print('ready', flush=True)\n"
        "while True: time.sleep(1)\n"
    )
    children = []

    def start(repo, *, stuck=False, port=8000, owner=None, role="demo"):
        repo.mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(modules), STUB_STUCK=str(int(stuck)),
                   PYTHONDONTWRITEBYTECODE="1")
        if owner is not None:
            env.update(MOSS_DEPLOY_REPO=str(owner), MOSS_DEPLOY_ROLE=role,
                       MOSS_DEPLOY_PORT=str(port))
        child = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(port)],
            cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        children.append(child)
        assert select.select([child.stdout], [], [], 5)[0], "stub did not become ready"
        assert child.stdout.readline().strip() == "ready"
        return child

    yield start
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdout.close()
        child.stderr.close()


def run_stop(repo, *args):
    return subprocess.run([sys.executable, str(STOP), "--repo", str(repo), *args],
                          capture_output=True, text=True, timeout=12)


def test_graceful_exit_does_not_wait_out_the_budget(tmp_path, launch):
    repo = tmp_path / "repo"
    child = launch(repo)
    begin = time.monotonic()
    result = run_stop(repo)
    assert result.returncode == 0, result.stderr
    assert child.wait(timeout=1) == 0
    assert "SIGKILL" not in result.stdout
    assert time.monotonic() - begin < 3


def test_default_five_second_deadline_kills_stuck_process(tmp_path, launch):
    repo = tmp_path / "repo"
    child = launch(repo, stuck=True)
    begin = time.monotonic()
    result = run_stop(repo)
    elapsed = time.monotonic() - begin
    assert result.returncode == 0, result.stderr
    assert child.wait(timeout=1) == -signal.SIGKILL
    assert "exceeded 5s" in result.stdout
    assert 5 <= elapsed < 9


def test_scope_keeps_other_checkout_and_other_port_running(tmp_path, launch):
    repo = tmp_path / "repo"
    target = launch(repo, port=8100)
    other_port = launch(repo, port=8101)
    other_repo = launch(tmp_path / "other", port=8100)
    result = run_stop(repo, "--port", "8100", "--timeout", "0.2")
    assert result.returncode == 0, result.stderr
    assert target.wait(timeout=1) == 0
    assert other_port.poll() is None and other_repo.poll() is None


def test_repeated_stop_after_exit_succeeds(tmp_path, launch):
    repo = tmp_path / "repo"
    child = launch(repo)
    assert run_stop(repo).returncode == 0
    assert child.wait(timeout=1) == 0
    assert run_stop(repo).returncode == 0


def test_unconfirmed_death_warns_and_allows_restart(monkeypatch, capsys):
    signals = []
    proc = SimpleNamespace(pid=456, send_signal=signals.append)
    monkeypatch.setattr(shutdown, "wait_exited", lambda processes, timeout: processes)
    assert shutdown.stop_processes([proc], timeout=0) == [proc]
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert "continuing restart attempt" in capsys.readouterr().err


def test_shell_command_mention_is_not_a_backend(tmp_path):
    proc = SimpleNamespace(cwd=lambda: str(tmp_path),
                           cmdline=lambda: ["bash", "-c", "python -m uvicorn server.app:app"])
    assert not shutdown.is_backend(proc, tmp_path)


def test_controller_returns_normally_with_a_surviving_process(tmp_path, monkeypatch, capsys):
    proc = SimpleNamespace(pid=456, send_signal=lambda sig: None,
                           cwd=lambda: str(tmp_path),
                           cmdline=lambda: ["python", "-m", "uvicorn", "server.app:app"])
    monkeypatch.setattr(shutdown.psutil, "process_iter", lambda: iter([proc]))
    monkeypatch.setattr(shutdown, "wait_exited", lambda processes, timeout: processes)
    monkeypatch.setattr(sys, "argv", [str(STOP), "--repo", str(tmp_path)])
    assert shutdown.main() is None
    assert "continuing restart attempt" in capsys.readouterr().err


@pytest.mark.parametrize("command", [["down"], ["up"], ["restart", "api"]])
def test_deployment_stops_on_shutdown_failure(tmp_path, command):
    result, calls = run_deployment_stub(tmp_path, command, stop_code=1)
    assert result.returncode != 0
    assert "stop\n" in calls
    assert "kill-session" not in calls and "respawn-window" not in calls
    assert "new-session" not in calls and "new-window" not in calls


def test_restart_waits_for_stop_before_respawn(tmp_path):
    result, calls = run_deployment_stub(tmp_path, ["restart", "api"], stop_code=0)
    assert result.returncode == 0, result.stderr
    assert calls.index("stop\n") < calls.index("respawn-window")


def run_deployment_stub(tmp_path, command, stop_code):
    deploy = tmp_path / "scripts/deploy"
    deploy.mkdir(parents=True)
    script = deploy / "demo.sh"
    script.write_text((REPO / "scripts/deploy/demo.sh").read_text())
    (deploy / "env_lib.sh").write_text("load_env_deploy() { ENV_DEPLOY_KEYS=(); }\n")
    (deploy / "env_manifest.sh").write_text("MOSS_ENV_VARS=()\n")
    calls = tmp_path / "calls"
    pybin = tmp_path / "stub_python"
    pybin.write_text(f'#!/bin/bash\necho stop >> "$CALLS"\nexit {stop_code}\n')
    pybin.chmod(0o755)
    tmux = tmp_path / "tmux"
    tmux.write_text('#!/bin/bash\necho "$*" >> "$CALLS"\n'
                    'if [ "$1" = show-option ]; then dirname "$CALLS"; fi\n'
                    'if [ "$1" = list-windows ]; then echo api; fi\nexit 0\n')
    tmux.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", PYBIN=str(pybin),
               CALLS=str(calls), DEMO_SKIP_GPU="1")
    result = subprocess.run(["bash", str(script), *command], env=env, capture_output=True,
                            text=True, timeout=5)
    return result, calls.read_text()


def test_owner_tags_keep_foreign_and_unmarked_services(tmp_path, launch):
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    shared = tmp_path / "shared-backend"
    repo.mkdir()
    other.mkdir()
    own = launch(shared, owner=repo, role="omni", port=18500)
    foreign = launch(shared, owner=other, role="omni", port=18500)
    unmarked = launch(shared, port=18500)
    another_port = launch(shared, owner=repo, role="omni", port=18501)
    result = run_stop(repo, "--role", "omni", "--port", "18500", "--timeout", "0.2")
    assert result.returncode == 0, result.stderr
    assert own.wait(timeout=1) == 0
    assert all(p.poll() is None for p in [foreign, unmarked, another_port])


def test_occupied_port_is_not_killed(tmp_path):
    import socket
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        result = run_stop(tmp_path, "--role", "omni", "--require-free-port", str(port))
        assert result.returncode != 0
        assert "refusing to kill" in result.stderr
        assert listener.getsockname()[1] == port
