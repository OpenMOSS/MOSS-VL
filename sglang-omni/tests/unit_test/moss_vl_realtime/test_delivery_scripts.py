"""Server launcher and shared GPU safety checks without loading a model."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[3] / "deployment/moss_vl_realtime"


@pytest.fixture
def modules(monkeypatch):
    modules = []
    for name, filename in (
        ("common", "common.py"),
        ("delivery_test_entry", "entry.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, HERE / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


@pytest.fixture
def model(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["MossVLForConditionalGeneration"]})
    )
    return tmp_path


def inventory():
    return [
        dict(
            index=str(i),
            uuid=f"GPU-{i}",
            name="fixture",
            total_mib=81920,
            used_mib=40000 if i == 0 else 735,
        )
        for i in range(4)
    ]


def test_single_gpu_default_and_explicit_selection(modules):
    common, _ = modules
    assert [g["index"] for g in common.select_gpus(None, inventory())] == ["1"]
    assert [g["index"] for g in common.select_gpus("2,3", inventory())] == ["2", "3"]


@pytest.mark.parametrize(
    "requested,visible",
    [("0", None), ("2,2", None), ("2,GPU-2", None), ("2", "1"), (None, "")],
)
def test_reject_busy_duplicate_or_invisible_gpu(modules, requested, visible):
    with pytest.raises(ValueError):
        modules[0].select_gpus(requested, inventory(), visible)


def test_visibility_and_low_memory(modules):
    common, _ = modules
    assert common.select_gpus(None, inventory(), "GPU-3")[0]["index"] == "3"
    devices = inventory()[1:2]
    devices[0]["total_mib"] = 24000
    with pytest.raises(ValueError):
        common.select_gpus("1", devices)


def test_server_is_single_gpu_four_sessions_and_environment_is_direct(
    modules, monkeypatch
):
    common, _ = modules
    command = common.server_command("/model with spaces", 18500)
    assert Path(command[2]).is_file()
    expected = {
        "--gpu": "0",
        "--max-running-requests": "4",
        "--context-length": "131072",
        "--mem-fraction-static": "0.5",
        "--parked-request-timeout": "3600",
    }
    for option, value in expected.items():
        assert command[command.index(option) + 1] == value
    assert "--tp-size" not in command and "/model with spaces" in command
    monkeypatch.setenv("HTTPS_PROXY", "http://unused")
    monkeypatch.setenv("REALTIME_FRAME_POOLING_ENABLED", "1")
    env = common.environment("GPU-2")
    assert "HTTPS_PROXY" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-2"
    assert env["REALTIME_FRAME_POOLING_ENABLED"] == "0"
    assert env["REALTIME_FRAME_WINDOW_RAW_S"] == "60"


def test_model_validation_and_dry_run(modules, model, tmp_path, capsys):
    common, entry = modules
    with pytest.raises(ValueError):
        common.validate_model(tmp_path / "missing")
    assert entry.main(["serve", str(model), "--port", "18510", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["port"] == 18510


@pytest.mark.parametrize("port", ["0", "65536"])
def test_invalid_port_is_rejected(modules, model, port):
    with pytest.raises(ValueError, match="port"):
        modules[1].main(["serve", str(model), "--port", port, "--dry-run"])


@pytest.mark.parametrize("mode", ["test", "worker"])
def test_server_entry_only_accepts_serving(modules, model, mode):
    with pytest.raises(SystemExit):
        modules[1].parse_args([mode, str(model)])


@pytest.mark.parametrize("gpus", ["2", "2,3"])
def test_launch_dispatch_and_multi_gpu_rejection(modules, model, monkeypatch, gpus):
    common, entry = modules
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(entry, "gpu_inventory", inventory)
    monkeypatch.setattr(entry.importlib.metadata, "version", lambda name: "5.12.1")
    ports, launched = [], []
    monkeypatch.setattr(
        entry, "free_port", lambda host, port: ports.append((host, port))
    )
    monkeypatch.setattr(entry.os, "execve", lambda *args: launched.append(args))
    if "," in gpus:
        with pytest.raises(ValueError, match="single-GPU"):
            entry.main(["serve", str(model), "--gpus", gpus])
        assert not ports and not launched
    else:
        entry.main(["serve", str(model), "--gpus", gpus, "--port", "18510"])
        assert ports == [("127.0.0.1", 18510)]
        executable, command, env = launched[0]
        assert executable == sys.executable
        assert command == common.server_command(model.resolve(), 18510)
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-2"


def test_occupied_port_is_not_reused(modules):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        with pytest.raises(OSError):
            modules[0].free_port(port=sock.getsockname()[1])


@pytest.mark.parametrize(
    "script", ["start.sh", "test_accuracy.sh", "test_latency.sh", "test_concurrency.sh"]
)
def test_shell_entry_dry_run_with_explicit_python(model, script):
    env = {**os.environ, "PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        ["bash", str(HERE / script), str(model), "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model_path"] == str(model.resolve())
    if script == "test_concurrency.sh":
        settings = json.loads(result.stdout)
        assert settings["sessions"] == [1, 2, 4, 8]
        assert settings["token_rate"] == 10
        assert settings["fps"] == 1


@pytest.mark.parametrize("suite", ["accuracy", "latency", "concurrency"])
def test_help_only_shows_relevant_parameters(suite):
    env = {**os.environ, "PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        ["bash", str(HERE / f"test_{suite}.sh"), "--help"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert f"usage: test_{suite}.sh" in result.stdout
    assert ("--token-rate" in result.stdout) == (suite == "concurrency")
    assert ("--strict-tokens" in result.stdout) == (suite == "accuracy")
    assert ("--hf-attention" in result.stdout) == (suite != "concurrency")
