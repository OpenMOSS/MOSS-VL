"""Regression coverage for reconnect and portable deployment recovery."""
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location('recovery_' + name, ROOT / f'scripts/repro/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disconnect_cancels_queue_getter():
    from server.routers import session_ws as module
    async def check():
        queue, closed, welcome = asyncio.Queue(), asyncio.Event(), asyncio.Event()
        state = SimpleNamespace(out_queue=queue, seq=SimpleNamespace(next=lambda: 1),
            created_event_fields=lambda: {'session_id': 'test-only'}, orchestrator=None, ws_token='test')
        class Manager:
            async def attach_ws(self, *args, **kwargs): return state, 'test', closed, []
            async def detach_ws(self, *args): closed.set()
        class Socket:
            query_params = {}
            async def accept(self): pass
            async def send_text(self, text): welcome.set()
            async def receive(self):
                await welcome.wait()
                await asyncio.sleep(.02)
                return {'type': 'websocket.disconnect'}
        runtime = SimpleNamespace(session_manager=Manager(), tts=None,
            settings=SimpleNamespace(asr_sample_rate=16000, tts_sample_rate=48000, tts_channels=2))
        before = asyncio.all_tasks()
        try:
            with patch.object(module, 'get_runtime', return_value=runtime):
                await module.session_ws(Socket(), 'test-only')
            await asyncio.sleep(0)
            assert not queue._getters
            queue.put_nowait('next-connection-event')
            await asyncio.sleep(0)
            assert queue.get_nowait() == 'next-connection-event'
        finally:
            tasks = asyncio.all_tasks() - before
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.run(check())


def test_install_refuses_run_state(tmp_path, monkeypatch):
    module = load('bootstrap')
    home = tmp_path / '.repro'
    home.mkdir()
    (home / 'run-state.json').write_text('{}')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'doctor', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['bootstrap'])
    with pytest.raises(RuntimeError, match='Stop the managed deployment'):
        module.main()
    assert not (home / 'managed-install.json').exists()


def test_install_shares_launch_lock(tmp_path, monkeypatch):
    module = load('bootstrap')
    home = tmp_path / '.repro'
    home.mkdir()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'doctor', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['bootstrap'])
    with (home / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): module.main()
    assert not (home / 'managed-install.json').exists()


def test_retry_interrupted_git_fetch(tmp_path, monkeypatch):
    module = load('bootstrap')
    source, home = tmp_path / 'origin', tmp_path / 'home'
    home.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '--allow-empty', '-qm', 'fixture'], check=True)
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    spec = {'backend': {'url': str(source), 'revision': revision}}
    fetches = []
    def command(args, **kwargs):
        args = list(map(str, args))
        if 'fetch' in args:
            fetches.append(args)
            if len(fetches) == 1: raise subprocess.CalledProcessError(128, args)
        subprocess.run(args, cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setattr(module, 'command', command)
    with pytest.raises(subprocess.CalledProcessError): module.backend_source(home, spec, None, {})
    target = module.backend_source(home, spec, None, {})
    assert subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip() == revision
    assert module.backend_source(home, spec, None, {}) == target
    assert len(fetches) == 2


def test_packaging_rejects_invalid_backend_before_writing(tmp_path, monkeypatch):
    module = load('make_release')
    directory = tmp_path / 'deployment/repro'
    directory.mkdir(parents=True)
    (directory / 'manifest.json').write_text(json.dumps({'backend': {'revision': 'a' * 40}}))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *args, **kwargs: 'b' * 40)
    monkeypatch.setattr(sys, 'argv', ['release', '--backend', str(tmp_path), '--output', str(tmp_path / 'out/package.tgz')])
    with pytest.raises(RuntimeError, match='pyproject.toml'): module.main()
    assert not (tmp_path / 'out').exists()


def test_interrupted_bundle_copy_can_retry(tmp_path, monkeypatch):
    module = load('bootstrap')
    source, home = tmp_path / 'source', tmp_path / 'home'
    source.mkdir()
    home.mkdir()
    content = b'[project]\nname="fixture"\n'
    (source / 'pyproject.toml').write_bytes(content)
    revision = 'a' * 40
    (source / 'source-manifest.json').write_text(json.dumps({'revision': revision,
        'files': {'pyproject.toml': hashlib.sha256(content).hexdigest()}}))
    spec = {'backend': {'revision': revision}}
    copy = module.shutil.copy2
    def interrupted(src, dst):
        if Path(src).name == 'source-manifest.json': raise OSError('simulated copy interruption')
        return copy(src, dst)
    monkeypatch.setattr(module.shutil, 'copy2', interrupted)
    with pytest.raises(OSError, match='interruption'): module.backend_source(home, spec, source, {})
    assert not (home / 'sglang-omni-main').exists()
    monkeypatch.setattr(module.shutil, 'copy2', copy)
    target = module.backend_source(home, spec, source, {})
    assert (target / 'pyproject.toml').read_bytes() == content


def test_stop_finds_owned_orphan_without_harming_foreign_process(tmp_path, monkeypatch):
    module = load('run')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    token = uuid.uuid4().hex
    foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
        env={**os.environ, 'MOSS_REPRO_ROOT': str(tmp_path), 'MOSS_REPRO_PROCESS_ID': 'different-token'})
    parent = subprocess.Popen([sys.executable, '-c',
        'import subprocess,sys,time; child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],'
        'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(child.pid,flush=True); time.sleep(.3)'],
        stdout=subprocess.PIPE, text=True, start_new_session=True,
        env={**os.environ, 'MOSS_REPRO_ROOT': str(tmp_path), 'MOSS_REPRO_PROCESS_ID': token})
    child = None
    try:
        created = psutil.Process(parent.pid).create_time()
        child = psutil.Process(int(parent.stdout.readline()))
        parent.wait(timeout=5)
        records = [{'pid': parent.pid, 'created': created, 'owner_token': token}]
        module.stop(records)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert foreign.poll() is None
        module.stop(records)
    finally:
        if child and child.is_running():
            try: child.kill()
            except psutil.NoSuchProcess: pass
        for process in (parent, foreign):
            if process.poll() is None: process.kill()
            process.wait(timeout=5)


def test_missing_legacy_parent_does_not_silently_succeed(tmp_path, monkeypatch):
    module = load('run')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='manual cleanup'):
        module.stop([{'pid': 2**30, 'created': 0}])
