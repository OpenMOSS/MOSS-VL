# SPDX-License-Identifier: Apache-2.0
"""Native data parallelism: endpoint fan-out, per-replica process specs, and
Coordinator replica routing."""
from __future__ import annotations

import asyncio
import socket
from types import SimpleNamespace

import pytest
import typer

from sglang_omni.cli.serve import apply_parallelism_cli_overrides
from sglang_omni.config import ParallelismConfig, PipelineConfig
from sglang_omni.config.schema import EndpointsConfig, StageConfig
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.mp_runner import (
    MultiProcessPipelineRunner,
    _build_stage_groups,
)
from sglang_omni.pipeline.runtime_config import (
    allocate_endpoints,
    prepare_pipeline_runtime,
)
from sglang_omni.proto import AdminResult, CompleteMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeMpContext,
    RecordingCoordinatorControlPlane,
    fake_factory_path,
)

_FACTORY = fake_factory_path("runtime_factory")


def _dp_stage(
    name: str, *, tp: int, dp: int, gpu, entry: bool = True, factory=_FACTORY
) -> StageConfig:
    return StageConfig(
        name=name,
        factory=factory,
        gpu=gpu,
        tp_size=tp,
        parallelism=ParallelismConfig(tp=tp, dp=dp),
        # Unreplicated single-rank stages still declare a process group.
        process="pipeline" if tp == 1 and dp == 1 else None,
        terminal=True,
    )


def test_allocate_endpoints_dp1_keys_are_unchanged(tmp_path) -> None:
    stages = [_dp_stage("ar", tp=2, dp=1, gpu=[0, 1])]

    endpoints = allocate_endpoints(stages=stages, ipc_base_dir=tmp_path)

    assert endpoints == {
        "completion": f"ipc://{tmp_path}/completion.sock",
        "abort": f"ipc://{tmp_path}/abort.sock",
        "stage_ar": f"ipc://{tmp_path}/stage_ar.sock",
        "comm_ar_rank0": f"ipc://{tmp_path}/comm_ar_rank0.sock",
        "comm_ar_rank1": f"ipc://{tmp_path}/comm_ar_rank1.sock",
    }


def test_allocate_endpoints_splits_per_replica_socket_keys(tmp_path) -> None:
    stages = [_dp_stage("ar", tp=2, dp=2, gpu=[0, 1, 2, 3])]

    endpoints = allocate_endpoints(stages=stages, ipc_base_dir=tmp_path)

    assert endpoints == {
        "completion": f"ipc://{tmp_path}/completion.sock",
        "abort": f"ipc://{tmp_path}/abort.sock",
        "stage_ar_dp0": f"ipc://{tmp_path}/stage_ar_dp0.sock",
        "stage_ar_dp1": f"ipc://{tmp_path}/stage_ar_dp1.sock",
        "comm_ar_dp0_rank0": f"ipc://{tmp_path}/comm_ar_dp0_rank0.sock",
        "comm_ar_dp0_rank1": f"ipc://{tmp_path}/comm_ar_dp0_rank1.sock",
        "comm_ar_dp1_rank0": f"ipc://{tmp_path}/comm_ar_dp1_rank0.sock",
        "comm_ar_dp1_rank1": f"ipc://{tmp_path}/comm_ar_dp1_rank1.sock",
    }


def test_dp_stage_process_specs_are_per_replica(tmp_path) -> None:
    config = PipelineConfig(
        model_path="global-model",
        name="dpcontract",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[_dp_stage("ar", tp=2, dp=2, gpu=[0, 1, 2, 3])],
    )
    # Same route the serve CLI takes: --dp-size 2 on the entry stage.
    config = apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        dp_size=2,
        talker_gpu=None,
        code2wav_gpu=None,
    )
    assert config.stages[0].parallelism.dp == 2

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

    assert [group.group_name for group in groups] == ["ar_dp0", "ar_dp1"]
    assert sum(group.process_count for group in groups) == 4
    for dp_rank, group in enumerate(groups):
        specs = sorted(group.specs, key=lambda spec: spec.tp_rank)
        leader, follower = specs
        assert leader.owns_external_io and not follower.owns_external_io
        assert leader.recv_endpoint == prep.endpoints[f"stage_ar_dp{dp_rank}"]
        assert follower.recv_endpoint == ""
        assert (leader.gpu_id, follower.gpu_id) == (dp_rank * 2, dp_rank * 2 + 1)
        assert (leader.dp_rank, leader.dp_size) == (dp_rank, 2)
        assert leader.factory_args["dp_rank"] == dp_rank
        assert follower.factory_args["dp_size"] == 2
        assert follower.factory_args["tp_rank"] == 1
        assert "ar" not in leader.stage_endpoints
        assert leader.rank_endpoints["ar"] == (
            prep.endpoints[f"comm_ar_dp{dp_rank}_rank0"],
            prep.endpoints[f"comm_ar_dp{dp_rank}_rank1"],
        )
        assert group.stage_replica_control_endpoints == [
            ("ar", dp_rank, prep.endpoints[f"stage_ar_dp{dp_rank}"])
        ]
    assert groups[0].specs[0].nccl_port != groups[1].specs[0].nccl_port
    # Follower queues are per replica, not shared across the dp axis.
    assert (
        groups[0].process_specs[0].stage_specs[0].follower_work_queues
        is not groups[1].process_specs[0].stage_specs[0].follower_work_queues
    )


def test_dp_replicas_get_distinct_nccl_ports_at_tp1(tmp_path) -> None:
    """tp=1 dp=2 replicas each need a unique process-group rendezvous port."""
    config = PipelineConfig(
        model_path="global-model",
        name="dpcontract",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[_dp_stage("ar", tp=1, dp=2, gpu=[0, 1])],
    )
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
    specs = [group.specs[0] for group in groups]
    assert specs[0].nccl_port is not None
    assert specs[0].nccl_port != specs[1].nccl_port
    assert [spec.factory_args["nccl_port"] for spec in specs] == [
        specs[0].nccl_port,
        specs[1].nccl_port,
    ]


def test_dp1_tp1_stage_keeps_legacy_nccl_port_resolution(tmp_path) -> None:
    """dp=1/tp=1 must not pin a port in advance (legacy MASTER_PORT flow)."""
    config = PipelineConfig(
        model_path="global-model",
        name="dpcontract",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[
            _dp_stage("ar", tp=1, dp=1, gpu=0),
        ],
    )
    # dp=1/tp=1 single stages run in a shared process group.
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

    spec = groups[0].specs[0]
    assert spec.nccl_port is None
    assert "nccl_port" not in spec.factory_args


def test_dp1_process_specs_are_unchanged(tmp_path) -> None:
    config = PipelineConfig(
        model_path="global-model",
        name="tpcontract",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[_dp_stage("ar", tp=2, dp=1, gpu=[0, 1])],
    )
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

    assert [group.group_name for group in groups] == ["ar"]
    specs = sorted(groups[0].specs, key=lambda spec: spec.tp_rank)
    assert specs[0].recv_endpoint == prep.endpoints["stage_ar"]
    assert specs[0].stage_endpoints["ar"] == prep.endpoints["stage_ar"]
    assert specs[0].rank_endpoints["ar"] == (
        prep.endpoints["comm_ar_rank0"],
        prep.endpoints["comm_ar_rank1"],
    )
    assert "dp_rank" not in specs[0].factory_args
    assert groups[0].stage_replica_control_endpoints == [
        ("ar", 0, prep.endpoints["stage_ar"])
    ]


def test_dp_size_over_one_is_rejected_off_entry_stage(tmp_path) -> None:
    with pytest.raises(ValueError, match="only supported on the entry stage"):
        PipelineConfig(
            model_path="dummy",
            endpoints=EndpointsConfig(base_path=str(tmp_path)),
            stages=[
                StageConfig(
                    name="pre",
                    process="pipeline",
                    factory=_FACTORY,
                    terminal=True,
                ),
                _dp_stage("ar", tp=1, dp=2, gpu=[0, 1], entry=False),
            ],
            entry_stage="pre",
        )
    # Invalid CLI wiring still fails before any process plan is built.
    with pytest.raises(typer.BadParameter, match="must provide 2 GPU ids"):
        apply_parallelism_cli_overrides(
            PipelineConfig(
                model_path="dummy",
                endpoints=EndpointsConfig(base_path=str(tmp_path)),
                stages=[
                    StageConfig(
                        name="ar",
                        process="pipeline",
                        factory=_FACTORY,
                        gpu=0,
                        terminal=True,
                    )
                ],
            ),
            thinker_tp_size=None,
            thinker_gpus=None,
            dp_size=2,
            talker_gpu=None,
            code2wav_gpu=None,
        )


def _make_coordinator_with_replicas(*, capacities=(None, None)):
    coordinator = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="ar",
        terminal_stages=["ar"],
    )
    control_plane = RecordingCoordinatorControlPlane()
    coordinator.control_plane = control_plane
    for dp_rank, capacity in enumerate(capacities):
        coordinator.register_stage(
            "ar", f"inproc://ar_dp{dp_rank}", dp_rank=dp_rank, capacity=capacity
        )
    return coordinator, control_plane


def test_coordinator_round_robins_and_stickily_updates_replicas() -> None:
    async def _run() -> None:
        coordinator, control_plane = _make_coordinator_with_replicas()

        for index in range(3):
            await coordinator._submit_request(f"req-{index}", {"x": index})
        assert [s[1] for s in control_plane.submitted] == [
            "inproc://ar_dp0",
            "inproc://ar_dp1",
            "inproc://ar_dp0",
        ]
        info = coordinator.get_request_info("req-1")
        assert info is not None
        assert (info.owner_endpoint, info.owner_dp_rank) == ("inproc://ar_dp1", 1)
        assert [r.inflight for r in coordinator.replicas("ar")] == [2, 1]

        # Updates follow the owning replica, not the round-robin cursor.
        await coordinator.update_request("req-1", {"frame": 7})
        assert control_plane.submitted[-1][1] == "inproc://ar_dp1"

        # Completion releases the replica's inflight counter.
        await coordinator._handle_completion(
            CompleteMessage("req-1", "ar", True, result={"ok": True})
        )
        assert coordinator._completion_futures["req-1"].result() == {"ok": True}
        assert [r.inflight for r in coordinator.replicas("ar")] == [2, 0]

        assert await coordinator.abort("req-2") is True
        assert [a.request_id for a in control_plane.aborts] == ["req-2"]
        assert [r.inflight for r in coordinator.replicas("ar")] == [1, 0]

    asyncio.run(_run())


def test_coordinator_selection_skips_full_replicas() -> None:
    async def _run() -> None:
        coordinator, control_plane = _make_coordinator_with_replicas(
            capacities=(1, None)
        )

        for index in range(4):
            await coordinator._submit_request(f"req-{index}", {"x": index})
        assert [s[1] for s in control_plane.submitted] == [
            "inproc://ar_dp0",
            "inproc://ar_dp1",
            "inproc://ar_dp1",
            "inproc://ar_dp1",
        ]

    asyncio.run(_run())


def test_coordinator_submit_to_replica_targets_an_explicit_rank() -> None:
    async def _run() -> None:
        coordinator, control_plane = _make_coordinator_with_replicas()

        with pytest.raises(ValueError, match="no replica"):
            await coordinator.submit_to_replica("req-x", {"x": 1}, dp_rank=7)

        task = asyncio.create_task(
            coordinator.submit_to_replica("req-warm", {"warm": True}, dp_rank=1)
        )
        await asyncio.sleep(0)
        assert control_plane.submitted[-1][1] == "inproc://ar_dp1"
        await coordinator._handle_completion(
            CompleteMessage("req-warm", "ar", True, result="warm-up")
        )
        assert await task == "warm-up"

    asyncio.run(_run())


def test_coordinator_admin_fans_out_to_all_replicas() -> None:
    async def _run() -> None:
        coordinator, control_plane = _make_coordinator_with_replicas()
        await coordinator.start()

        task = asyncio.create_task(coordinator.admin("model_info"))
        for _ in range(100):
            if len(control_plane.submitted) == 2:
                break
            await asyncio.sleep(0)
        assert [s[1] for s in control_plane.submitted] == [
            "inproc://ar_dp0",
            "inproc://ar_dp1",
        ]
        op_id = control_plane.submitted[0][2].operation.op_id
        assert not task.done()
        coordinator._handle_admin_result(
            AdminResult(
                op_id=op_id, stage="ar", action="model_info", success=True, dp_rank=0
            )
        )
        assert not task.done()
        coordinator._handle_admin_result(
            AdminResult(
                op_id=op_id,
                stage="ar",
                action="model_info",
                success=True,
                data={"replica": 1},
                dp_rank=1,
            )
        )
        result = await task
        assert result["success"]
        assert len(result["results"]) == 1
        merged = result["results"][0]
        assert merged["stage"] == "ar"
        assert [item["dp_rank"] for item in merged["data"]["replica_results"]] == [0, 1]

    asyncio.run(_run())


def test_coordinator_admin_fails_when_any_replica_fails() -> None:
    async def _run() -> None:
        coordinator, control_plane = _make_coordinator_with_replicas()
        await coordinator.start()

        task = asyncio.create_task(coordinator.admin("pause_generation"))
        for _ in range(100):
            if len(control_plane.submitted) == 2:
                break
            await asyncio.sleep(0)
        op_id = control_plane.submitted[0][2].operation.op_id
        coordinator._handle_admin_result(
            AdminResult(
                op_id=op_id, stage="ar", action="pause_generation", success=True,
                dp_rank=1,
            )
        )
        coordinator._handle_admin_result(
            AdminResult(
                op_id=op_id,
                stage="ar",
                action="pause_generation",
                success=False,
                error="replica-0 busy",
                dp_rank=0,
            )
        )

        result = await task
        assert result["success"] is False
        assert "replica-0 busy" in result["message"]
        merged = result["results"][0]
        assert merged["success"] is False
        assert [item["dp_rank"] for item in merged["data"]["replica_results"]] == [0, 1]

    asyncio.run(_run())


def test_admin_result_dicts_are_dp1_identical() -> None:
    single = AdminResult(op_id="op", stage="ar", action="model_info", success=True)
    as_dict = single.to_dict()
    assert "dp_rank" not in as_dict

    pinned = AdminResult(
        op_id="op", stage="ar", action="model_info", success=True, dp_rank=1
    )
    assert pinned.to_dict()["dp_rank"] == 1
    assert AdminResult.from_dict(as_dict).dp_rank is None


def test_runner_stage_control_endpoints_key_replicas(tmp_path) -> None:
    config = PipelineConfig(
        model_path="global-model",
        name="dpcontract",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[_dp_stage("ar", tp=1, dp=2, gpu=[0, 1])],
    )
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

    runner = MultiProcessPipelineRunner.__new__(MultiProcessPipelineRunner)
    runner._groups = list(groups)
    runner._started = True

    assert runner.stage_control_endpoints == {
        "ar_dp0": prep.endpoints["stage_ar_dp0"],
        "ar_dp1": prep.endpoints["stage_ar_dp1"],
    }


def test_nccl_port_allocator_honors_env_base(monkeypatch) -> None:
    from sglang_omni.pipeline.mp_runner import _NcclPortAllocator

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
        reserved.bind(("127.0.0.1", 0))
        base = reserved.getsockname()[1]

    monkeypatch.setenv("SGLANG_OMNI_NCCL_PORT_BASE", str(base))
    port = _NcclPortAllocator().allocate()
    assert port >= base

    monkeypatch.setenv("SGLANG_OMNI_NCCL_PORT_BASE", str(port + 10))
    assert _NcclPortAllocator().allocate() == port + 10


def test_engine_factory_forwards_nccl_port_at_tp1_for_replicas(monkeypatch) -> None:
    """build() must forward a replica-picked nccl_port even without TP ranks."""
    import pytest

    from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
    from sglang_omni.scheduling import engine_factory, sglang_backend

    captured: dict = {}

    class _Stop(Exception):
        pass

    def fake_server_args(_checkpoint, **kwargs):
        return SimpleNamespace(
            tp_size=1,
            disable_cuda_graph=True,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled")
            ),
        )

    monkeypatch.setattr(sglang_backend, "build_sglang_server_args", fake_server_args)

    def fake_infra(server_args, gpu_id, **kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(
        scheduling_bootstrap, "create_sglang_infrastructure_defer_cuda_graph",
        fake_infra,
    )

    class _Builder(engine_factory.SGLangGenerationEngineBuilder):
        model_name = "probe"
        context_length = 16

        def generation_defaults(self, *, dtype):
            del dtype
            return {"max_running_requests": 4}

        def make_adapters(self, *args, **kwargs):
            raise NotImplementedError

        def make_model_runner(self, *args, **kwargs):
            raise NotImplementedError

    with pytest.raises(_Stop):
        _Builder().build(
            "unused", gpu_id=0, dp_rank=1, dp_size=2, nccl_port=42571
        )

    assert captured["nccl_port"] == 42571
    assert captured["dp_rank"] == 1


def test_construct_scheduler_threads_replica_args_at_tp1(tmp_path) -> None:
    """The real worker seam: replica specs feed dp_rank/dp_size/nccl_port
    through _construct_scheduler into the stage factory."""
    import logging

    from sglang_omni.pipeline import stage_workers
    from tests.unit_test.fixtures.pipeline_fakes import fake_factory_path

    config = PipelineConfig(
        model_path="global-model",
        name="dpcontract",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[
            _dp_stage(
                "ar",
                tp=1,
                dp=2,
                gpu=[0, 1],
                factory=fake_factory_path("dummy_factory"),
            ),
        ],
    )
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

    for dp_rank, group in enumerate(groups):
        spec = group.specs[0]
        received = stage_workers._construct_scheduler(
            spec, None, logging.getLogger("test-dp-replication")
        )
        assert received["dp_rank"] == dp_rank == spec.dp_rank
        assert received["dp_size"] == 2 == spec.dp_size
        assert received["nccl_port"] == spec.nccl_port


def test_moss_vl_builder_scheduler_receives_dp_rank(monkeypatch) -> None:
    """Regression: the model-specific builder override forwards dp_rank."""
    import sglang_omni.models.moss_vl_realtime.engine_builder as engine_builder_mod

    captured: dict = {}

    class _FakeScheduler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        engine_builder_mod, "MossVLRealtimeScheduler", _FakeScheduler
    )

    builder = engine_builder_mod.MossVLRealtimeEngineBuilder(
        max_running_requests=1,
        max_new_tokens=8,
        context_length=128,
        mem_fraction_static=0.4,
    )
    builder.dp_rank = 1
    builder.dp_size = 2
    builder.silence_token_ids = (1,)
    builder.segment_builder = None

    builder._make_scheduler(
        model_worker=None,
        tree_cache=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        server_args=None,
        model_config=None,
        prefill_manager=None,
        decode_manager=None,
        model_runner=None,
        request_builder=None,
        result_adapter=None,
        extra_scheduler_kwargs={},
    )

    assert captured["dp_rank"] == 1


def test_tts_builder_make_scheduler_receives_dp_rank(monkeypatch) -> None:
    """TTS builders construct schedulers through the compatibility path."""
    from sglang_omni.scheduling import engine_factory, omni_scheduler

    captured: dict = {}

    class _FakeScheduler:
        def __init__(self, *a, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(omni_scheduler, "OmniScheduler", _FakeScheduler)

    class _Builder(engine_factory.TtsEngineBuilder):
        model_name = "probe"
        context_length = 16

        def generation_defaults(self, *, dtype):
            del dtype
            return {"max_running_requests": 4}

        def setup_model(self, **kwargs):
            del kwargs

        def make_model_runner(self, *a, **k):
            raise NotImplementedError

        def make_adapters(self, *a, **k):
            raise NotImplementedError

    builder = _Builder()
    builder.dp_rank = 1

    builder.make_scheduler(
        model_worker=None,
        tree_cache=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        server_args=None,
        model_config=None,
        prefill_manager=None,
        decode_manager=None,
        model_runner=None,
        request_builder=None,
        result_adapter=None,
    )

    assert captured["dp_rank"] == 1
