"""Portable release inputs, isolation and failure behavior."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "scripts/repro/bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_manifest_names_sources_without_pinning_revisions():
    manifest = json.loads((ROOT / "deployment/repro/manifest.json").read_text())
    assert 'revision' not in manifest['backend']
    assert manifest["backend"]["url"] == "https://github.com/fnlp-vision/sglang-omni-realtime.git"
    assert len(manifest["node"]["sha256"]) == 64
    for model in manifest["models"].values():
        assert model['repo_id'] and 'revision' not in model and 'files' not in model
        assert not Path(model["directory"]).is_absolute()
        assert '..' not in Path(model['directory']).parts


def test_installation_environment_does_not_inherit_internal_indexes(tmp_path, monkeypatch):
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://private.invalid/simple")
    monkeypatch.setenv("UV_INDEX", "https://private.invalid/simple")
    monkeypatch.setenv("PYTHONPATH", "/developer/environment")
    env = bootstrap.environment(tmp_path)
    assert "PIP_EXTRA_INDEX_URL" not in env and "UV_INDEX" not in env and "PYTHONPATH" not in env
    assert env["UV_PYTHON_BIN_DIR"].startswith(str(tmp_path))
    assert env["NPM_CONFIG_CACHE"].startswith(str(tmp_path))
    assert env["NPM_CONFIG_GLOBALCONFIG"] != env["NPM_CONFIG_USERCONFIG"]


def test_refuses_existing_unmanaged_environment(tmp_path, monkeypatch):
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "doctor", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["bootstrap"])
    with pytest.raises(RuntimeError, match="unmanaged"):
        bootstrap.main()


def test_all_dependency_entries_are_pinned_and_hashed():
    for lock in (ROOT / "deployment/repro/locks").glob("*.lock"):
        text = lock.read_text()
        assert "/inspire/" not in text and "nexus" not in text.lower()
        logical = text.replace("\\\n", " ")
        for line in logical.splitlines():
            if not line.strip() or line.startswith(("#", "--")):
                continue
            assert "==" in line and "--hash=sha256:" in line, (lock.name, line)


def test_pi_agent_is_part_of_source_delivery():
    pi = ROOT / "services/pi_agent"
    assert (pi / "service.mjs").is_file() and (pi / "package-lock.json").is_file()
    assert "/inspire/" not in (pi / "aigw.mjs").read_text()


def test_cuda_compiler_is_locked_to_runtime_minor():
    lock = (ROOT / "deployment/repro/locks/backend.lock").read_text()
    for name in ("nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm"):
        assert f"{name}==13.0.88" in lock
    assert "nvidia-cuda-runtime==13.0." in lock


def test_cuda_toolkit_layout_is_owned_and_repeatable(tmp_path):
    import runpy
    prepare = runpy.run_path(str(ROOT / "scripts/repro/cuda_toolkit.py"))["prepare"]
    sdk = tmp_path / "sdk"
    for name in ("bin/nvcc", "include/cuda_runtime.h", "lib/libcudart.so.13", "nvvm/libdevice"):
        path = sdk / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    target = prepare(tmp_path / "managed", sdk)
    assert (target / "lib64/libcudart.so").resolve() == sdk / "lib/libcudart.so.13"
    assert (target / "lib/libcudart.so").is_file()
    assert prepare(tmp_path / "managed", sdk) == target
    assert not (sdk / "lib64").exists()
    (target / "lib64/libcudart.so").unlink()
    (target / "lib64/libcudart.so").touch()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        prepare(tmp_path / "managed", sdk)
