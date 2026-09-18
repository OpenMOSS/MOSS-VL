"""NVML accounting uses bytes and never hides a missing process measurement."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[3] / "deployment/moss_vl_realtime"


@pytest.fixture
def monitor(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    for name in ("common", "memory_monitor"):
        spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    state = dict(process=0, device=700_000_000, closed=False, uuid=None)

    def by_uuid(uuid):
        state["uuid"] = uuid
        return "device"

    fake = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: state.update(closed=True),
        nvmlDeviceGetHandleByUUID=by_uuid,
        nvmlDeviceGetHandleByIndex=lambda i: "device",
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(
            total=150_000_000_000, used=state["device"]
        ),
        nvmlDeviceGetComputeRunningProcesses=lambda h: [
            SimpleNamespace(pid=os.getpid(), usedGpuMemory=state["process"]),
            SimpleNamespace(pid=os.getpid() + 1, usedGpuMemory=700_000_000),
        ],
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    return module, state


def test_process_peak_separate_from_device_peak(monitor, tmp_path):
    module, state = monitor
    path = tmp_path / "memory.json"
    with module.MemoryMonitor(path, interval=100) as sampler:
        state.update(process=70_000_000_000, device=71_000_000_000)
        sampler.sample()
        state.update(process=0, device=700_000_000)
    result = json.loads(path.read_text())
    assert state["closed"] and state["uuid"] == "GPU-test"
    assert result["process_peak_bytes"] == 70_000_000_000
    assert result["device_peak_bytes"] == 71_000_000_000
    assert result["within_budget"]


@pytest.mark.parametrize("used", [0, 2**64 - 1, 81_000_000_000])
def test_measurement_validity_is_separate_from_reference_capacity(
    monitor, tmp_path, used
):
    module, state = monitor
    path = tmp_path / "memory.json"
    with module.MemoryMonitor(path, interval=100):
        state.update(
            process=used, device=82_000_000_000 if used < 2**64 - 1 else 700_000_000
        )
    result = json.loads(path.read_text())
    assert not result["within_budget"]
    assert result["measurement_valid"] is (used == 81_000_000_000)
    assert result["budget_bytes"] == 80_000_000_000
