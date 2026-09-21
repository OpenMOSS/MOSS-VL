"""Shutdown completion is observed, never inferred from a fixed sleep."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location('shutdown_run', ROOT/'scripts/repro/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zombie_does_not_require_an_environment(monkeypatch, capsys):
    module = load()
    def forbidden():
        raise AssertionError('A zombie environment must not be inspected')
    process = SimpleNamespace(pid=123, status=lambda: psutil.STATUS_ZOMBIE, environ=forbidden)
    monkeypatch.setattr(module.psutil, 'Process', lambda pid: process)
    module.stop([{'pid':123, 'created':0}])
    assert 'already stopped (zombie)' in capsys.readouterr().out


def test_live_foreign_pid_remains_protected(monkeypatch):
    module = load()
    process = SimpleNamespace(pid=123, status=lambda:psutil.STATUS_RUNNING,
                              create_time=lambda:2, environ=lambda:{})
    monkeypatch.setattr(module.psutil, 'Process', lambda pid:process)
    with pytest.raises(RuntimeError, match='ownership mismatch'):
        module.stop([{'pid':123,'created':1}])


def test_wait_observes_port_release():
    module = load()
    listener = socket.socket()
    listener.bind(('127.0.0.1',0))
    listener.listen()
    port = listener.getsockname()[1]
    timer = threading.Timer(.15, listener.close)
    timer.start()
    started = time.monotonic()
    try:
        module.wait_ports_released([port], timeout=2)
        assert time.monotonic()-started >= .1
    finally:
        listener.close()
        timer.join()


def test_busy_port_times_out_without_stopping_its_owner():
    module = load()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match='Ports still occupied'):
            module.wait_ports_released([port], timeout=.02)
        assert listener.getsockname()[1] == port


def test_next_operation_waits_for_lock(tmp_path):
    module = load()
    with (tmp_path/'run.lock').open('a') as owner, (tmp_path/'run.lock').open('a') as next_op:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        timer = threading.Timer(.15, lambda:fcntl.flock(owner, fcntl.LOCK_UN))
        timer.start()
        try:
            started = time.monotonic()
            module.acquire_run_lock(next_op, timeout=2)
            assert time.monotonic()-started >= .1
        finally:
            timer.join()


def test_busy_lock_has_bounded_wait(tmp_path):
    module = load()
    with (tmp_path/'run.lock').open('a') as owner, (tmp_path/'run.lock').open('a') as next_op:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='no new operation started'):
            module.acquire_run_lock(next_op, timeout=.02)


@pytest.mark.parametrize('fail', [False, True])
def test_state_cleared_only_after_stop_succeeds(tmp_path, monkeypatch, fail):
    module = load()
    state = tmp_path/'run-state.json'
    state.write_text(json.dumps({'processes':[], 'ports':{'api':18501}}))
    monkeypatch.setattr(module, 'HOME', tmp_path)
    monkeypatch.setattr(module, 'STATE', state)
    monkeypatch.setattr(sys, 'argv', ['run.py','down'])
    def stop(records, ports):
        assert state.exists() and list(ports)==[18501]
        if fail:
            raise RuntimeError('still occupied')
    monkeypatch.setattr(module, 'stop', stop)
    if fail:
        with pytest.raises(RuntimeError, match='still occupied'):
            module.main()
        assert state.exists()
    else:
        module.main()
        assert not state.exists()


def test_real_owned_process_exits_before_stop_returns(tmp_path, monkeypatch):
    module = load()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    process = subprocess.Popen([sys.executable, '-c',
        'import time; print("ready",flush=True); time.sleep(60)'],
        stdout=subprocess.PIPE, text=True,
        env={**os.environ, 'MOSS_REPRO_ROOT':str(tmp_path)})
    try:
        assert process.stdout.readline().strip()=='ready'
        module.stop([{'pid':process.pid,'created':psutil.Process(process.pid).create_time()}], timeout=2)
        assert process.wait(timeout=1) is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_process_becoming_zombie_does_not_wait_for_reaping(monkeypatch):
    module = load()
    process = SimpleNamespace(is_running=lambda:True, status=lambda:psutil.STATUS_ZOMBIE)
    def forbidden(*args, **kwargs):
        raise AssertionError('No need to wait for zombie PID disappearance')
    monkeypatch.setattr(module.psutil, 'wait_procs', forbidden)
    assert module.wait_stopped([process], 30)==[]
