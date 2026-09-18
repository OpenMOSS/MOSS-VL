from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from sglang_omni.models.moss_vl_realtime.platform_compat import is_npu_platform

_REPO_ROOT = Path(__file__).parents[3]
_LAUNCHER_PATH = _REPO_ROOT / "examples" / "run_moss_vl_realtime_server.py"
_SPEC = importlib.util.spec_from_file_location(
    "moss_vl_realtime_launcher", _LAUNCHER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_LAUNCHER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LAUNCHER)


def test_launcher_exposes_venv_console_scripts_without_activation(monkeypatch) -> None:
    python = "/project/.venv-main/bin/python"
    monkeypatch.setattr(_LAUNCHER.sys, "executable", python)
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")

    _LAUNCHER._ensure_python_bin_on_path()
    _LAUNCHER._ensure_python_bin_on_path()

    assert os.environ["PATH"].split(os.pathsep) == [
        "/project/.venv-main/bin",
        "/usr/local/bin",
        "/usr/bin",
    ]


def _parse(monkeypatch, *argv: str):
    monkeypatch.setattr(
        _LAUNCHER.sys, "argv", ["run_moss_vl_realtime_server.py", *argv]
    )
    return _LAUNCHER.parse_args()


def test_launcher_async_decode_flag_defaults_off(monkeypatch) -> None:
    args = _parse(monkeypatch, "--model-path", "/models/x")

    assert args.enable_async_decode is False
    assert args.decode_cuda_graph is (not is_npu_platform())
    assert args.context_length == 262144
    assert args.mem_fraction_static == 0.40
    assert args.enable_benchmark_mode is False
    assert args.disable_startup_warmup is False
    assert args.tp_size == 1
    assert args.gpus is None


def test_launcher_accepts_distinct_gpu_per_tp_rank(monkeypatch) -> None:
    args = _parse(
        monkeypatch,
        "--model-path",
        "/models/x",
        "--tp-size",
        "2",
        "--gpus",
        "3,5",
    )

    assert args.tp_size == 2
    assert args.gpus == [3, 5]


@pytest.mark.parametrize(
    "argv",
    [
        ("--tp-size", "2"),
        ("--tp-size", "2", "--gpus", "0"),
        ("--tp-size", "2", "--gpus", "0,0"),
        ("--gpus", "0"),
    ],
)
def test_launcher_rejects_invalid_tp_placement(monkeypatch, argv) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--model-path", "/models/x", *argv)


def test_launcher_async_decode_flag_opt_in(monkeypatch) -> None:
    args = _parse(monkeypatch, "--model-path", "/models/x", "--enable-async-decode")

    assert args.enable_async_decode is True


def test_launcher_benchmark_mode_is_explicit_opt_in(monkeypatch) -> None:
    args = _parse(monkeypatch, "--model-path", "/models/x", "--enable-benchmark-mode")

    assert args.enable_benchmark_mode is True


def test_launcher_does_not_expose_paged_kv(monkeypatch) -> None:
    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            "--model-path",
            "/models/x",
            "--kv-page-size",
            "16",
        )


def test_launcher_accepts_dp_replicas_with_distinct_gpus(monkeypatch) -> None:
    args = _parse(
        monkeypatch,
        "--model-path",
        "/models/x",
        "--dp-size",
        "2",
        "--gpus",
        "0,2",
    )

    assert args.dp_size == 2
    assert args.gpus == [0, 2]


def test_launcher_accepts_tp_times_dp_placement(monkeypatch) -> None:
    args = _parse(
        monkeypatch,
        "--model-path",
        "/models/x",
        "--tp-size",
        "2",
        "--dp-size",
        "2",
        "--gpus",
        "0,1,4,5",
    )

    assert args.gpus == [0, 1, 4, 5]


@pytest.mark.parametrize(
    "argv",
    [
        ("--dp-size", "0", "--gpus", "0,1"),
        ("--dp-size", "2"),
        ("--dp-size", "2", "--gpu", "1"),
        ("--dp-size", "2", "--gpus", "0"),
        ("--dp-size", "2", "--gpus", "0,0"),
    ],
)
def test_launcher_rejects_invalid_dp_placement(monkeypatch, argv) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--model-path", "/models/x", *argv)


def test_dp2_boot_resolves_per_replica_factory_args(monkeypatch, tmp_path) -> None:
    """The ws_benchmark/start.sh boot path: --dp-size 2 --gpus 0,1 must reach
    each replica as per-replica factory args (dp_rank/dp_size/nccl_port)."""
    import tempfile as _tempfile

    from sglang_omni.pipeline.mp_runner import _build_stage_groups
    from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
    from sglang_omni.serve import launcher as serve_launcher
    from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext

    captured = {}

    def fake_launch_server(config, **kwargs):
        captured["config"] = config

    monkeypatch.setattr(serve_launcher, "launch_server", fake_launch_server)
    monkeypatch.setattr(
        _LAUNCHER.sys,
        "argv",
        [
            "run_moss_vl_realtime_server.py",
            "--model-path",
            "/models/x",
            "--dp-size",
            "2",
            "--gpus",
            "0,1",
        ],
    )

    _LAUNCHER.main()
    config = captured["config"]

    with _tempfile.TemporaryDirectory() as run_dir:
        config.endpoints.base_path = run_dir
        prep = prepare_pipeline_runtime(config)
        try:
            groups = _build_stage_groups(
                config,
                ctx=FakeMpContext(),
                stages_cfg=prep.stages_cfg,
                name_map=prep.name_map,
                endpoints=prep.endpoints,
                placement_plan=prep.placement_plan,
                process_plan=prep.process_plan,
            )
        finally:
            prep.runtime_dir.close()

    assert len(groups) == 2
    for dp_rank, group in enumerate(groups):
        spec = group.specs[0]
        assert spec.factory_args["dp_rank"] == dp_rank
        assert spec.factory_args["dp_size"] == 2
        assert isinstance(spec.factory_args["nccl_port"], int)
        other = groups[1 - dp_rank].specs[0].factory_args["nccl_port"]
        assert spec.factory_args["nccl_port"] != other
