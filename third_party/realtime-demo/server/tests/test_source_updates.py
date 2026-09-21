import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/repro' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def commit(repo, text):
    (repo / 'source.txt').write_text(text)
    git(repo, 'add', 'source.txt')
    git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', text)
    return git(repo, 'rev-parse', 'HEAD')


def test_backend_install_resume_update_and_local_edits(tmp_path):
    bootstrap = load('bootstrap')
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    git(upstream, 'init', '--initial-branch=main')
    first = commit(upstream, 'first')
    home = tmp_path / 'home'
    home.mkdir()
    spec = {'backend': {'url': str(upstream)}}
    target = bootstrap.backend_source(home, spec, None, None)
    assert bootstrap.backend_revision(target) == first
    second = commit(upstream, 'second')
    bootstrap.backend_source(home, spec, None, None)
    assert bootstrap.backend_revision(target) == first
    bootstrap.backend_source(home, spec, None, None, update=True)
    assert bootstrap.backend_revision(target) == second
    (target / 'source.txt').write_text('local work')
    with pytest.raises(RuntimeError, match='local changes'):
        bootstrap.backend_source(home, spec, None, None, update=True)
    assert (target / 'source.txt').read_text() == 'local work'


def test_bundle_revision_is_provenance_not_a_required_pin(tmp_path):
    bootstrap = load('bootstrap')
    source = tmp_path / 'bundle'
    source.mkdir()
    file = source / 'pyproject.toml'
    file.write_text('[project]\nname="test"\n')
    (source / 'source-manifest.json').write_text(json.dumps({
        'revision': 'bundle-version', 'files': {'pyproject.toml': bootstrap.sha256(file)}}))
    home = tmp_path / 'home'
    home.mkdir()
    target = bootstrap.backend_source(home, {'backend': {'url': 'unused'}}, source, None)
    assert bootstrap.backend_revision(target) == 'bundle-version'


def test_backend_uses_its_own_dependency_lock(tmp_path, monkeypatch):
    bootstrap = load('bootstrap')
    calls = []
    monkeypatch.setattr(bootstrap, 'command', lambda args, **kw: calls.append(list(map(str, args))))
    bootstrap.install_python('uv', tmp_path / 'home', tmp_path / 'backend', {})
    assert any(str(tmp_path / 'backend/deployment/repro/requirements.lock') in c for c in calls)
    assert not any(str(ROOT / 'deployment/repro/locks/backend.lock') in c for c in calls)


def model_module(monkeypatch, tmp_path):
    current = {'revision': 'a'}
    for rev, names in [('a', ['old.py', 'model.bin']), ('b', ['new.py', 'model.bin'])]:
        snapshot = tmp_path / rev
        snapshot.mkdir()
        for name in names:
            (snapshot / name).write_text('test')
    def info(*args, **kwargs):
        rev = current['revision']
        return SimpleNamespace(sha=rev, siblings=[SimpleNamespace(rfilename=f.name, size=4)
                                                 for f in (tmp_path / rev).iterdir()])
    def download(**kwargs):
        assert kwargs['revision'] == current['revision']
        return str(tmp_path / kwargs['revision'])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda **kwargs: SimpleNamespace(model_info=info), snapshot_download=download))
    return load('models'), current


def test_models_resolve_current_snapshot_without_stale_files(tmp_path, monkeypatch):
    models, current = model_module(monkeypatch, tmp_path)
    model = {'repo_id': 'Example/Model'}
    destination = tmp_path / 'installed'
    record = models.download_model(model, destination, tmp_path / 'cache')
    assert record['revision'] == 'a' and (destination / 'old.py').is_file()
    current['revision'] = 'b'
    assert models.download_model(model, destination, tmp_path / 'cache', record)['revision'] == 'a'
    record = models.download_model(model, destination, tmp_path / 'cache', record, update=True)
    assert record['revision'] == 'b' and (destination / 'new.py').is_file()
    assert not (destination / 'old.py').exists()


def test_failed_model_download_keeps_previous_snapshot(tmp_path, monkeypatch):
    models, current = model_module(monkeypatch, tmp_path)
    destination = tmp_path / 'installed'
    model = {'repo_id': 'Example/Model'}
    record = models.download_model(model, destination, tmp_path / 'cache')
    current['revision'] = 'b'
    (tmp_path / 'b/model.bin').write_text('incomplete')
    with pytest.raises(RuntimeError, match='incomplete'):
        models.download_model(model, destination, tmp_path / 'cache', record, update=True)
    assert destination.resolve() == tmp_path / 'a'


def test_legacy_model_directory_is_preserved(tmp_path, monkeypatch):
    models, _ = model_module(monkeypatch, tmp_path)
    destination = tmp_path / 'installed'
    destination.mkdir()
    (destination / 'old-local-file').write_text('keep')
    models.download_model({'repo_id': 'Example/Model'}, destination, tmp_path / 'cache', update=True)
    previous = list(tmp_path.glob('installed.previous-*'))
    assert len(previous) == 1 and (previous[0] / 'old-local-file').read_text() == 'keep'
