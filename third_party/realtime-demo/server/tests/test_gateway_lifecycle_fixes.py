"""Gateway lifecycle, admission, limits, health and version regressions."""
import asyncio
import json
import os
import subprocess
import threading
from types import SimpleNamespace

import pytest
import requests
from fastapi import HTTPException

from server.config import Settings
from server.gateway import rest
from server.gateway.pool import GatewayPool, GatewayCapacityError, GatewayUnavailableError, READY, BUSY, DOWN
from server.gateway.session import GatewayRegistry, GatewaySession
from server.gateway.tokens import TokenIssuer


def registry(**kwargs):
    settings = Settings(sglang_omni_urls='http://one,http://two', **kwargs)
    pool = GatewayPool(settings)
    pool._probe_health = lambda url: {'ok': True}
    pool.probe_all()
    ledger = []
    reg = GatewayRegistry(settings, pool, TokenIssuer(), usage=SimpleNamespace(record=ledger.append))
    return reg, pool, ledger


def fake_client(version=None):
    class Client:
        closed = False
        def __init__(self, *args):
            pass
        def open(self):
            return {'type': 'session.created', 'session_id': 'upstream', 'request_id': 'r',
                    'model': 'moss', **({'model_version': version} if version else {})}
        def start_receiver(self, *args, **kwargs):
            pass
        def close(self):
            self.closed = True
        def abort_transport(self):
            self.close()
    return Client


def test_cancelled_create_returns_unregistered_reservation(monkeypatch):
    async def run():
        reg, pool, _ = registry()
        entered = asyncio.Event()
        async def opening(self, index):
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(GatewaySession, 'open_on_replica', opening)
        task = asyncio.create_task(reg.create())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.busy == 0 and len(reg) == 0
        await reg.aclose()
    asyncio.run(run())


def test_late_handshake_is_closed_even_after_repeated_cancellation(monkeypatch):
    async def run():
        reg, pool, _ = registry()
        entered, proceed = threading.Event(), threading.Event()
        closed = []
        class Client(fake_client()):
            def open(self):
                entered.set()
                assert proceed.wait(2)
                return super().open()
            def close(self):
                closed.append(True)
        monkeypatch.setattr('server.gateway.session.SglangOmniClient', Client)
        task = asyncio.create_task(reg.create())
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        proceed.set()
        await asyncio.gather(task, return_exceptions=True)
        await reg.aclose()
        assert closed and pool.busy == 0 and len(reg) == 0
        assert not reg._cleanup_tasks
    asyncio.run(run())


def test_handshake_retry_visits_each_replica_only_once(monkeypatch):
    async def run():
        reg, pool, _ = registry()
        attempts = []
        async def opening(self, index):
            attempts.append(index)
            for replica in pool.replicas:
                if replica.state == DOWN:
                    replica.state = READY
            raise ConnectionError('WS fails while health recovers')
        monkeypatch.setattr(GatewaySession, 'open_on_replica', opening)
        with pytest.raises(GatewayUnavailableError):
            await reg.create()
        assert attempts == [0, 1] and pool.busy == 0
        await reg.aclose()
    asyncio.run(run())


def test_create_has_an_overall_deadline(monkeypatch):
    async def run():
        reg, pool, _ = registry(gateway_create_timeout_s=0.05)
        async def opening(self, index):
            await asyncio.Event().wait()
        monkeypatch.setattr(GatewaySession, 'open_on_replica', opening)
        start = asyncio.get_running_loop().time()
        with pytest.raises(GatewayUnavailableError):
            await reg.create()
        assert asyncio.get_running_loop().time() - start < 0.5
        assert pool.busy == 0
        await reg.aclose()
    asyncio.run(run())


def test_short_creation_budget_still_allows_a_fast_handshake(monkeypatch):
    async def run():
        monkeypatch.setattr('server.gateway.session.SglangOmniClient', fake_client())
        reg, pool, _ = registry(gateway_create_timeout_s=0.5)
        session = await reg.create()
        assert 0 < session._client.connect_timeout_s < 0.5
        await reg.aclose()
        assert pool.busy == 0
    asyncio.run(run())


def test_shutdown_cancels_creates_before_draining_sessions(monkeypatch):
    async def run():
        reg, pool, _ = registry()
        entered = asyncio.Event()
        async def opening(self, index):
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(GatewaySession, 'open_on_replica', opening)
        task = asyncio.create_task(reg.create())
        await entered.wait()
        await reg.aclose()
        assert task.cancelled() and pool.busy == 0
        with pytest.raises(GatewayUnavailableError):
            await reg.create()
    asyncio.run(run())


def test_tokens_are_bound_to_epoch_and_revoked_on_reset_and_delete(monkeypatch):
    async def run():
        monkeypatch.setattr('server.gateway.session.SglangOmniClient', fake_client())
        reg, pool, _ = registry()
        session = await reg.create()
        old = session.mint_token()
        # Even a binding consumed concurrently before reset cannot attach later.
        consumed = reg.tokens.consume_binding(session.mint_token())[0]
        reset = await session.reset()
        assert reg.tokens.consume_binding(old) == (None, 'invalid')
        assert not session.token_epoch_is_current(consumed[1])
        assert not session.try_attach(consumed[1])
        binding, reason = reg.tokens.consume_binding(reset['ws_token'])
        assert reason is None and binding == (session.session_id, 1)
        token = session.mint_token()
        await session.adestroy('delete')
        assert reg.tokens.consume_binding(token) == (None, 'invalid')
        assert pool.busy == 0
        await reg.aclose()
    asyncio.run(run())


def test_reset_cancel_releases_slot_and_revokes_tokens(monkeypatch):
    async def run():
        monkeypatch.setattr('server.gateway.session.SglangOmniClient', fake_client())
        reg, pool, _ = registry()
        session = await reg.create()
        token = session.mint_token()
        entered = asyncio.Event()
        async def opening(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        session._open_client = opening
        task = asyncio.create_task(session.reset())
        await entered.wait()
        assert not session.try_attach()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.busy == 0 and reg.get(session.session_id) is None
        assert reg.tokens.consume(token) is None
        await reg.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('upstream,local,expected', [(32, 8, 8), (4, 32, 4), (32, 32, 32)])
def test_configured_advertises_effective_frame_limit(upstream, local, expected):
    async def run():
        reg, pool, _ = registry(gateway_max_frame_bytes=local * 1024 * 1024)
        session = GatewaySession(reg, 's')
        session._queue = asyncio.Queue()
        message = {'type': 'session.configured', 'max_frame_bytes': upstream * 1024 * 1024,
                   'future_field': 'preserved'}
        raw = json.dumps(message, separators=(',', ':'))
        session._deliver_omni_event(0, raw, message)
        forwarded = session._queue.get_nowait()
        assert json.loads(forwarded)['max_frame_bytes'] == expected * 1024 * 1024
        assert json.loads(forwarded)['future_field'] == 'preserved'
        if upstream == expected:
            assert forwarded == raw
        assert session._effective_max_frame_bytes == expected * 1024 * 1024
        await reg.aclose()
    asyncio.run(run())


def test_health_tracks_idle_and_busy_failures_without_losing_slots():
    pool = GatewayPool(Settings(sglang_omni_urls='http://one', sglang_omni_sessions_per_replica=2))
    assert pool.replicas[0].state == DOWN
    with pytest.raises(GatewayCapacityError):
        pool.acquire()
    pool._probe_health = lambda url: {'ok': True}
    pool.probe_all()
    index = pool.acquire()
    pool._probe_health = lambda url: None
    pool.probe_all()
    assert pool.replicas[0].state == DOWN and pool.busy == 1
    pool.release(index)
    assert pool.replicas[0].state == DOWN and pool.busy == 0
    pool._probe_health = lambda url: {'ok': True}
    pool.probe_all()
    assert pool.replicas[0].state == READY


def test_stale_probe_does_not_overwrite_a_new_transport_failure():
    pool = GatewayPool(Settings(sglang_omni_urls='http://one'))
    def probe(url):
        pool.mark_unhealthy(0)
        return {'ok': True}
    pool._probe_health = probe
    pool.probe_all()
    assert pool.replicas[0].state == DOWN


@pytest.mark.parametrize('backend,expected,source', [('backend-v2', 'backend-v2', 'backend'),
                                                   (None, 'deployed-v1', 'deployment')])
def test_model_version_is_preserved_in_wire_rest_snapshot_and_ledger(monkeypatch, backend, expected, source):
    async def run():
        monkeypatch.setattr('server.gateway.session.SglangOmniClient', fake_client(backend))
        reg, pool, ledger = registry(gateway_model_version='deployed-v1')
        session = await reg.create()
        assert json.loads(session._queue.get_nowait())['model_version'] == expected
        assert session.create_payload(session.mint_token())['model_version'] == expected
        assert session.snapshot()['model_version_source'] == source
        await session.adestroy('done')
        assert ledger[-1]['model_version'] == expected
        await reg.aclose()
    asyncio.run(run())


def test_model_version_mapping_and_unknown_are_explicit():
    pool = GatewayPool(Settings(sglang_omni_urls='http://one,http://two',
        gateway_model_versions='{"http://one":"revision-a"}'))
    assert pool.replica_version(0) == 'revision-a' and pool.replica_version(1) is None
    with pytest.raises(ValueError):
        GatewayPool(Settings(sglang_omni_urls='http://one', gateway_model_versions='{"http://other":"v"}'))


def test_reset_deleted_reference_maps_to_404(monkeypatch):
    async def run():
        monkeypatch.setattr('server.gateway.session.SglangOmniClient', fake_client())
        reg, pool, _ = registry()
        session = await reg.create()
        original = session.reset
        async def race():
            await session.adestroy('delete won')
            return await original()
        session.reset = race
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(gateway_registry=reg)))
        with pytest.raises(HTTPException) as caught:
            await rest.reset_session(request, session.session_id)
        assert caught.value.status_code == 404
        assert caught.value.detail['code'] == 'session_not_found'
        assert pool.busy == 0
        await reg.aclose()
    asyncio.run(run())


def test_models_retries_without_releasing_a_live_session_slot(monkeypatch):
    async def run():
        reg, pool, _ = registry()
        pool.acquire()
        attempts = []
        class Response:
            status_code = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def json(self): return {'data': [{'id': 'moss'}]}
        def get(url, **kwargs):
            attempts.append(url)
            if url.startswith('http://two'):
                raise requests.ConnectionError('metadata connection failed')
            return Response()
        monkeypatch.setattr(rest.requests, 'get', get)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(gateway_pool=pool)))
        response = await rest.list_models(request)
        assert response.status_code == 200
        assert attempts == ['http://two/v1/models', 'http://one/v1/models']
        assert pool.busy == 1
        pool.release(0)
        await reg.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('frame,transport,expected', [(None, None, '67108864'),
                                                    ('8388608', None, '16777216'),
                                                    ('33554432', '16777216', None)])
def test_real_launch_script_sets_transport_limit(tmp_path, frame, transport, expected):
    from pathlib import Path
    source = Path(__file__).resolve().parents[2] / 'scripts/deploy/run_backend.sh'
    script = tmp_path / 'scripts/deploy/run_backend.sh'
    script.parent.mkdir(parents=True)
    script.write_text(source.read_text())
    (script.parent / 'env_lib.sh').write_text('load_env_deploy() { :; }\n')
    python = tmp_path / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$OUT_ARGS"\n')
    python.chmod(0o755)
    (tmp_path / '.venv/lib/ffmpeg').mkdir(parents=True)
    output = tmp_path / 'args'
    env = {'PATH': os.environ['PATH'], 'OUT_ARGS': str(output)}
    if frame is not None: env['GATEWAY_MAX_FRAME_BYTES'] = frame
    if transport is not None: env['WS_MAX_SIZE'] = transport
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True)
    if expected is None:
        assert result.returncode != 0 and not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        args = output.read_text().splitlines()
        assert args[args.index('--ws-max-size') + 1] == expected
