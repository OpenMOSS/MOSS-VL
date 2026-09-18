"""Only test-owned process groups are created and signalled by these tests."""

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

HERE = Path(__file__).resolve().parents[3] / "deployment/moss_vl_realtime"
spec = importlib.util.spec_from_file_location(
    "delivery_common_cleanup", HERE / "common.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process groups")


def alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.fixture
def spawn(tmp_path):
    children = []

    def start(code, *, new_group=True):
        log = tmp_path / f"child-{len(children)}.log"
        child = common.Child(
            [sys.executable, "-u", "-c", code],
            dict(os.environ),
            log,
            new_group=new_group,
        )
        children.append(child)
        return child, log

    yield start
    for child in children:
        child.stop(timeout=0.05, kill_timeout=2)


@pytest.mark.parametrize("parent_exits_first", [False, True])
def test_stop_reaps_descendants_even_after_leader_exits(spawn, parent_exits_first):
    grandchild_code = (
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(60)"
    )
    code = (
        "import subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-u','-c',{grandchild_code!r}],stdout=subprocess.PIPE,text=True); "
        "p.stdout.readline(); print(p.pid,flush=True); "
        + ("sys.exit(0)" if parent_exits_first else "time.sleep(60)")
    )
    child, log = spawn(code)
    deadline = time.monotonic() + 5
    while not log.read_text().strip():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    descendant = psutil.Process(int(log.read_text().strip()))
    assert alive(descendant)
    if parent_exits_first:
        assert child.wait(2) == 0
        assert alive(descendant)
    child.stop(timeout=0.05, kill_timeout=2)
    assert not alive(descendant)
    assert child.process.poll() is not None
    assert child.log.closed
    child.stop(timeout=0, kill_timeout=0)


@pytest.mark.parametrize("new_group", [False, True])
def test_stop_does_not_signal_unrelated_process(spawn, new_group):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        child, _ = spawn("import time; time.sleep(60)", new_group=new_group)
        child.stop(timeout=0.1, kill_timeout=2)
        assert child.process.poll() is not None
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=2)


def test_stop_allows_parent_to_coordinate_descendant_exit(spawn):
    worker = (
        "import signal,sys; "
        "signal.signal(signal.SIGTERM, lambda *args: sys.exit(42)); "
        "print('ready',flush=True); sys.stdin.readline()"
    )
    code = f"""
import signal,subprocess,sys,time
p = subprocess.Popen([sys.executable, '-u', '-c', {worker!r}],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
p.stdout.readline()
def shutdown(*args):
    time.sleep(0.1)
    p.stdin.write('stop\\n')
    p.stdin.flush()
    code = p.wait(timeout=2)
    print('worker-exit:' + str(code), flush=True)
    sys.exit(code)
signal.signal(signal.SIGTERM, shutdown)
print('parent-ready', flush=True)
time.sleep(60)
"""
    child, log = spawn(code)
    deadline = time.monotonic() + 5
    while "parent-ready" not in log.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    child.stop(timeout=3, kill_timeout=2)
    assert child.process.returncode == 0, log.read_text()
    assert "worker-exit:0" in log.read_text()
    assert not child._live_owned_processes()


def test_stuck_parent_and_descendants_are_force_killed(spawn):
    worker = (
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready',flush=True); time.sleep(60)"
    )
    code = (
        "import signal,subprocess,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"p=subprocess.Popen([sys.executable,'-u','-c',{worker!r}],stdout=subprocess.PIPE,text=True); "
        "p.stdout.readline(); print(p.pid,flush=True); time.sleep(60)"
    )
    child, log = spawn(code)
    deadline = time.monotonic() + 5
    while not log.read_text().strip():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    descendant = psutil.Process(int(log.read_text().strip()))
    started = time.monotonic()
    child.stop(timeout=0.05, kill_timeout=2)
    assert time.monotonic() - started < 3
    assert child.process.returncode == -signal.SIGKILL
    assert not alive(descendant)


def test_reused_group_leader_pid_is_not_signalled(monkeypatch):
    child = common.Child.__new__(common.Child)
    child.process = SimpleNamespace(pid=123, poll=lambda: 0)
    child.new_group = True
    child._owner_started = 100.0
    monkeypatch.setattr(
        common.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 101.0)
    )

    def forbidden():
        raise AssertionError("must not scan a replacement process group")

    monkeypatch.setattr(common.psutil, "process_iter", forbidden)
    assert child._live_owned_processes() == []


def test_failed_shutdown_keeps_retry_possible(monkeypatch):
    child = common.Child.__new__(common.Child)
    child._stopped = False
    child.process = SimpleNamespace(pid=123, wait=lambda timeout: None)
    closed, signals = [], []
    child.log = SimpleNamespace(close=lambda: closed.append(True))
    process = SimpleNamespace(pid=123)
    monkeypatch.setattr(child, "_live_owned_processes", lambda: [process])
    monkeypatch.setattr(
        child, "_wait_owned_processes", lambda *args, **kwargs: [process]
    )
    monkeypatch.setattr(
        child, "_signal_processes", lambda procs, sig: signals.append(sig)
    )
    with pytest.raises(TimeoutError, match="did not exit"):
        child.stop(timeout=0, kill_timeout=0)
    assert not child._stopped and closed
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    monkeypatch.setattr(child, "_wait_owned_processes", lambda *args, **kwargs: [])
    child.stop(timeout=0, kill_timeout=0)
    assert child._stopped


def test_identity_initialization_failure_cleans_up_spawn(monkeypatch, tmp_path):
    calls = []
    process = SimpleNamespace(pid=123, wait=lambda timeout: calls.append("wait"))
    monkeypatch.setattr(common.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(common.os, "killpg", lambda pid, sig: calls.append((pid, sig)))

    def fail(pid):
        raise RuntimeError("identity unavailable")

    monkeypatch.setattr(common.psutil, "Process", fail)
    with pytest.raises(RuntimeError, match="identity unavailable"):
        common.Child(["unused"], {}, tmp_path / "child.log")
    assert calls == [(123, signal.SIGKILL), "wait"]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_shutdown_deadlines_are_rejected(value):
    child = common.Child.__new__(common.Child)
    with pytest.raises(ValueError):
        child.stop(timeout=value)
