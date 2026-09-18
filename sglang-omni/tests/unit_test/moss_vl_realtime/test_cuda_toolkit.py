"""The local CUDA view must be isolated by environment and safe to reuse."""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "cuda_toolkit", ROOT / "deployment/repro/cuda_toolkit.py"
)
toolkit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(toolkit)


def make_sdk(root):
    for name in ("bin/nvcc", "include/cuda_runtime.h", "lib/libcudart.so.13"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (root / "nvvm").mkdir()
    return root


def test_environment_views_do_not_conflict(tmp_path):
    first = make_sdk(tmp_path / "env1")
    second = make_sdk(tmp_path / "env2")
    one = toolkit.prepare(tmp_path / "views", first)
    two = toolkit.prepare(tmp_path / "views", second)
    assert one != two
    assert (one / "bin/nvcc").resolve() == first / "bin/nvcc"
    assert (two / "lib64/libcudart.so").resolve() == second / "lib/libcudart.so.13"
    assert toolkit.prepare(tmp_path / "views", first) == one


def test_concurrent_prepare_is_idempotent(tmp_path):
    sdk = make_sdk(tmp_path / "env")
    with ThreadPoolExecutor(max_workers=8) as pool:
        views = list(
            pool.map(lambda _: toolkit.prepare(tmp_path / "views", sdk), range(32))
        )
    assert len(set(views)) == 1
    assert (views[0] / "lib/libcudart.so").is_file()


def test_relative_home_produces_valid_links(tmp_path, monkeypatch):
    sdk = make_sdk(tmp_path / "env")
    monkeypatch.chdir(tmp_path)
    view = toolkit.prepare(Path("views"), sdk)
    assert view.is_absolute()
    assert (view / "lib/libcudart.so").resolve() == sdk / "lib/libcudart.so.13"


def test_conflicting_file_is_not_overwritten(tmp_path):
    sdk = make_sdk(tmp_path / "env")
    view = toolkit.prepare(tmp_path / "views", sdk)
    (view / "bin").unlink()
    (view / "bin").write_text("owned by someone else")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        toolkit.prepare(tmp_path / "views", sdk)
    assert (view / "bin").read_text() == "owned by someone else"


def test_incomplete_sdk_does_not_create_view(tmp_path):
    with pytest.raises(RuntimeError, match="Incomplete"):
        toolkit.prepare(tmp_path / "views", tmp_path / "missing")
    assert not (tmp_path / "views").exists()
