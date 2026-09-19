"""The documented preflight must identify incomplete or mismatched installs."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

tomllib = pytest.importorskip("tomllib")


ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "deployment/moss_vl_realtime/check_env.py"
spec = importlib.util.spec_from_file_location("delivery_check_env", PATH)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def model(tmp_path):
    config = dict(
        architectures=["MossVLForConditionalGeneration"],
        auto_map={"AutoConfig": "configuration_moss_vl.MossVLConfig"},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.json",
    ):
        (tmp_path / name).write_text("{}")
    (tmp_path / "configuration_moss_vl.py").write_text("# fixture\n")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001.safetensors"}})
    )
    (tmp_path / "model-00001.safetensors").write_bytes(
        b"not loaded by the file checker"
    )
    return tmp_path


def test_complete_model_does_not_need_processor_config(tmp_path):
    assert "1 weight file(s)" in check.check_model(model(tmp_path))


def test_model_accepts_cached_weight_symlink(tmp_path):
    snapshot = tmp_path / "models--test--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    model(snapshot)
    blob = snapshot.parent.parent / "blobs" / "weight-hash"
    blob.parent.mkdir()
    shard = snapshot / "model-00001.safetensors"
    shard.rename(blob)
    shard.symlink_to("../../blobs/weight-hash")
    assert "1 weight file(s)" in check.check_model(snapshot)


@pytest.mark.parametrize("name", ["/tmp/outside.safetensors", "../outside.safetensors"])
def test_weight_index_rejects_nonlocal_names(tmp_path, name):
    directory = model(tmp_path)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": name}})
    )
    with pytest.raises(ValueError, match="invalid weight shard"):
        check.check_model(directory)


def test_broken_weight_symlink_is_rejected(tmp_path):
    directory = model(tmp_path)
    shard = directory / "model-00001.safetensors"
    shard.unlink()
    shard.symlink_to("../missing-weight")
    with pytest.raises(ValueError, match="invalid weight shard"):
        check.check_model(directory)


@pytest.mark.parametrize(
    "name", ["tokenizer.json", "configuration_moss_vl.py", "model-00001.safetensors"]
)
def test_incomplete_model_is_rejected(tmp_path, name):
    directory = model(tmp_path)
    (directory / name).unlink()
    with pytest.raises(ValueError, match="Missing"):
        check.check_model(directory)


def test_weight_index_cannot_escape_model_directory(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    model(directory)
    (tmp_path / "outside.safetensors").write_bytes(b"external")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="invalid weight shard"):
        check.check_model(directory)


def test_core_versions_and_constraints_match_project():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = {
        canonicalize_name(r.name): r
        for r in map(Requirement, project["project"]["dependencies"])
    }
    constraints = [
        Requirement(line)
        for line in (ROOT / "deployment/moss_vl_realtime/constraints.txt")
        .read_text()
        .splitlines()
        if line and not line.startswith("#")
    ]
    pinned = {
        canonicalize_name(r.name): next(iter(r.specifier)).version for r in constraints
    }
    assert len(pinned) == len(constraints)
    assert all(r.url is None for r in constraints)
    for name, expected in check.CORE_VERSIONS.items():
        assert expected in requirements[canonicalize_name(name)].specifier
        assert pinned[canonicalize_name(name)].partition("+")[0] == expected
    for name, requirement in requirements.items():
        assert name in pinned and pinned[name] in requirement.specifier


@pytest.mark.parametrize("missing", [False, True])
def test_no_gpu_preflight_reports_package_errors(monkeypatch, capsys, missing):
    def version(name):
        if missing and name == "torch":
            raise check.importlib.metadata.PackageNotFoundError(name)
        return check.CORE_VERSIONS.get(name, "13.610.43")

    monkeypatch.setattr(check.importlib.metadata, "version", version)
    monkeypatch.setattr(check.platform, "system", lambda: "Linux")
    monkeypatch.setattr(check.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))

    def load(name):
        assert name == "sglang_omni", "GPU runtime imported during --no-gpu"
        return SimpleNamespace(__file__=str(ROOT / "sglang_omni/__init__.py"))

    monkeypatch.setattr(check.importlib, "import_module", load)
    assert check.main(["--no-gpu"]) == int(missing)
    output = capsys.readouterr().out
    assert "SKIP  GPU runtime" in output
    assert ("ERROR torch" in output) is missing
