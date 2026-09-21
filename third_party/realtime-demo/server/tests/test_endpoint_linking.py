"""Companion addresses derive from ports without rewriting explicit URLs."""
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def resolve(tmp_path, content='', **overrides):
    env_file = tmp_path / 'deploy.env'
    env_file.write_text(content)
    command = ['bash', '-c',
               'set -euo pipefail; source "$1"; load_env_deploy "$2"; resolve_service_endpoints; '
               'printf "%s\\n" "$VITE_BACKEND_ORIGIN" "$PI_PORT" "$MEMORY_PI_URL"',
               'test', str(ROOT / 'scripts/deploy/env_lib.sh'), str(ROOT)]
    result = subprocess.run(command, text=True, capture_output=True, check=True,
                            env={'PATH': os.environ['PATH'], 'ENV_DEPLOY_FILE': str(env_file), **overrides})
    return result.stdout.splitlines(), result.stderr


def test_default_addresses(tmp_path):
    values, _ = resolve(tmp_path)
    assert values == ['http://127.0.0.1:8000', '38082', 'http://127.0.0.1:38082']


def test_file_ports_link_both_companions(tmp_path):
    values, _ = resolve(tmp_path, 'PORT=8100\nPI_PORT=39082\n')
    assert values == ['http://127.0.0.1:8100', '39082', 'http://127.0.0.1:39082']


def test_caller_ports_win_over_file(tmp_path):
    values, _ = resolve(tmp_path, 'PORT=8000\nPI_PORT=38082\n', PORT='8010', PI_PORT='39083')
    assert values == ['http://127.0.0.1:8010', '39083', 'http://127.0.0.1:39083']


def test_explicit_urls_are_preserved_and_local_mismatch_is_visible(tmp_path):
    values, warnings = resolve(tmp_path, 'PORT=8100\nVITE_BACKEND_ORIGIN=http://127.0.0.1:8000\n',
                               MEMORY_PI_URL='http://pi.example:1234')
    assert values[0] == 'http://127.0.0.1:8000'
    assert values[2] == 'http://pi.example:1234'
    assert 'differs from local API port 8100' in warnings


@pytest.mark.parametrize('port,url,expected', [
    (None, None, 'http://127.0.0.1:38082'),
    ('39082', None, 'http://127.0.0.1:39082'),
    ('39082', 'http://remote:9999', 'http://remote:9999'),
    ('39082', '', ''),
])
def test_direct_python_config_matches_shell_defaults(monkeypatch, port, url, expected):
    for key, value in [('PI_PORT', port), ('MEMORY_PI_URL', url)]:
        monkeypatch.delenv(key, raising=False)
        if value is not None:
            monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location('endpoint_config', ROOT / 'server/config.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.Settings().memory_pi_url == expected


def test_all_manual_entrypoints_resolve_after_loading_env():
    for name in ['scripts/deploy/demo.sh', 'scripts/deploy/run_backend.sh',
                 'scripts/deploy/run_web.sh', 'scripts/gpu/start_pi_agent.sh', 'start_demo.sh']:
        text = (ROOT / name).read_text()
        assert text.index('load_env_deploy "$REPO"') < text.index('\nresolve_service_endpoints\n')


def test_vite_and_node_default_contract():
    vite = (ROOT / 'vite.config.ts').read_text()
    assert "process.env.VITE_BACKEND_ORIGIN || `http://127.0.0.1:${process.env.PORT || '8000'}`" in vite
    node = (ROOT / 'services/pi_agent/service.mjs').read_text()
    assert re.search(r'process.env.PI_PORT \|\| 38082', node)
