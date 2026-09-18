# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import (
    ParallelismConfig,
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
    build_stage_placement_plan,
    resolve_stage_gpu_ids,
    resolve_stage_replica_gpu_ids,
)

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def _stage(
    name: str,
    *,
    gpu: int | list[int] | None = None,
    fraction: float | None = None,
    tp_size: int = 1,
    terminal: bool = False,
    next_stage: str | None = None,
) -> StageConfig:
    return StageConfig(
        name=name,
        process="pipeline",
        factory=_FACTORY,
        gpu=gpu,
        tp_size=tp_size,
        runtime=StageRuntimeConfig(
            resources=StageResourceConfig(total_gpu_memory_fraction=fraction)
        ),
        next=next_stage,
        terminal=terminal,
    )


def test_same_gpu_placement_records_missing_memory_fraction_stages() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.10, next_stage="thinker"),
            _stage("thinker", gpu=0, terminal=True),
        ],
    )

    plan = build_stage_placement_plan(config)

    assert plan.gpus[0].missing_fraction_stage_names == ("thinker",)


def test_same_gpu_without_budget_records_placement() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("image_encoder", gpu=0, next_stage="thinker"),
            _stage("thinker", gpu=0, terminal=True),
        ],
    )

    plan = build_stage_placement_plan(config)

    assert plan.gpus[0].stage_names == ("image_encoder", "thinker")
    assert plan.gpus[0].missing_fraction_stage_names == (
        "image_encoder",
        "thinker",
    )


def test_untyped_factory_budget_is_rejected_before_placement() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                process="pipeline",
                factory=_FACTORY,
                factory_args={"total_gpu_memory_fraction": 0.50},
                gpu=0,
                terminal=True,
            )
        ],
    )

    with pytest.raises(ValueError, match="runtime.resources.total_gpu_memory_fraction"):
        build_stage_placement_plan(config)


def test_untyped_runtime_override_budget_is_rejected_before_placement() -> None:
    config = PipelineConfig(
        model_path="dummy",
        runtime_overrides={"thinker": {"total_gpu_memory_fraction": 0.50}},
        stages=[_stage("thinker", gpu=0, terminal=True)],
    )

    with pytest.raises(ValueError, match="runtime.resources.total_gpu_memory_fraction"):
        build_stage_placement_plan(config)


def test_same_gpu_colocation_sums_budget() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.10, next_stage="thinker"),
            _stage("thinker", gpu=0, fraction=0.70, terminal=True),
        ],
    )

    plan = build_stage_placement_plan(config)

    assert plan.gpus[0].stage_names == ("preprocess", "thinker")
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.80)


def test_same_gpu_colocation_rejects_over_budget() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.35, next_stage="thinker"),
            _stage("thinker", gpu=0, fraction=0.75, terminal=True),
        ],
    )

    with pytest.raises(ValueError, match="exceeds placement limit"):
        build_stage_placement_plan(config)


def test_tp_rank_gpu_ids_are_preserved() -> None:
    stage = _stage(
        "thinker",
        gpu=[0, 1],
        fraction=0.45,
        tp_size=2,
        terminal=True,
    )
    config = PipelineConfig(model_path="dummy", stages=[stage])

    plan = build_stage_placement_plan(config)

    assert resolve_stage_gpu_ids(plan, stage) == [0, 1]


def test_tp_memory_fraction_is_per_rank_per_assigned_gpu() -> None:
    stage = _stage(
        "thinker",
        gpu=[0, 1],
        fraction=0.30,
        tp_size=2,
        terminal=True,
    )
    config = PipelineConfig(model_path="dummy", stages=[stage])

    plan = build_stage_placement_plan(config)

    assert plan.gpus[0].stage_names == ("thinker",)
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.30)
    assert plan.gpus[1].stage_names == ("thinker",)
    assert plan.gpus[1].total_gpu_memory_fraction == pytest.approx(0.30)


def test_tp_scalar_gpu_is_rejected() -> None:
    stage = _stage(
        "thinker",
        gpu=0,
        fraction=0.45,
        tp_size=2,
        terminal=True,
    )
    config = PipelineConfig(model_path="dummy", stages=[stage])

    with pytest.raises(ValueError, match="requires a list"):
        build_stage_placement_plan(config)


def test_tp_duplicate_gpu_ids_are_rejected() -> None:
    stage = _stage(
        "thinker",
        gpu=[0, 0],
        fraction=0.45,
        tp_size=2,
        terminal=True,
    )
    config = PipelineConfig(model_path="dummy", stages=[stage])

    with pytest.raises(ValueError, match="unique GPU ids"):
        build_stage_placement_plan(config)


def test_placement_policy_hook_runs_after_generic_plan() -> None:
    config = PipelineConfig(
        model_path="dummy",
        placement_policy=(
            "tests.unit_test.fixtures.pipeline_fakes.RejectThinkerPlacementPolicy"
        ),
        stages=[_stage("thinker", gpu=0, terminal=True)],
    )

    with pytest.raises(ValueError, match="policy rejected thinker"):
        build_stage_placement_plan(config)


def _dp_stage(
    name: str,
    *,
    gpu: int | list[int] | None,
    tp_size: int = 1,
    dp_size: int = 2,
    fraction: float | None = None,
    terminal: bool = True,
) -> StageConfig:
    return StageConfig(
        name=name,
        process="pipeline",
        factory=_FACTORY,
        gpu=gpu,
        tp_size=tp_size,
        parallelism=ParallelismConfig(tp=tp_size, dp=dp_size),
        runtime=StageRuntimeConfig(
            resources=StageResourceConfig(total_gpu_memory_fraction=fraction)
        ),
        terminal=terminal,
    )


def test_dp_replica_gpu_ids_are_contiguous_tp_slices() -> None:
    stage = _dp_stage("thinker", gpu=[0, 1, 2, 3], tp_size=2, dp_size=2)
    config = PipelineConfig(model_path="dummy", stages=[stage])

    plan = build_stage_placement_plan(config)

    assert resolve_stage_gpu_ids(plan, stage) == [0, 1, 2, 3]
    assert resolve_stage_replica_gpu_ids(plan, stage) == [[0, 1], [2, 3]]
    assert plan.stages["thinker"].dp_size == 2


def test_dp1_replica_gpu_ids_match_existing_flat_list() -> None:
    stage = _stage("thinker", gpu=[0, 1], fraction=0.45, tp_size=2, terminal=True)
    config = PipelineConfig(model_path="dummy", stages=[stage])

    plan = build_stage_placement_plan(config)

    assert resolve_stage_replica_gpu_ids(plan, stage) == [[0, 1]]
    assert plan.stages["thinker"].dp_size == 1


def test_dp_replicas_without_gpu_resolve_to_none_per_rank() -> None:
    stage = _dp_stage("preprocess", gpu=None, dp_size=2, terminal=True)
    config = PipelineConfig(model_path="dummy", stages=[stage])

    plan = build_stage_placement_plan(config)

    assert resolve_stage_gpu_ids(plan, stage) == [None, None]
    assert resolve_stage_replica_gpu_ids(plan, stage) == [[None], [None]]


def test_dp_placement_rejects_scalar_gpu_for_multiple_replicas() -> None:
    stage = _dp_stage("thinker", gpu=0, dp_size=2)
    config = PipelineConfig(model_path="dummy", stages=[stage])

    with pytest.raises(ValueError, match="requires a list of 2 unique GPU ids"):
        build_stage_placement_plan(config)


def test_dp_placement_requires_tp_times_dp_gpu_ids() -> None:
    stage = _dp_stage("thinker", gpu=[0, 1, 2], tp_size=2, dp_size=2)

    with pytest.raises(ValueError, match="tp_size \\* dp_size = 4"):
        PipelineConfig(model_path="dummy", stages=[stage])


def test_dp_placement_requires_unique_gpu_ids_across_replicas() -> None:
    stage = _dp_stage("thinker", gpu=[0, 0], dp_size=2)
    config = PipelineConfig(model_path="dummy", stages=[stage])

    with pytest.raises(ValueError, match="unique GPU ids"):
        build_stage_placement_plan(config)
