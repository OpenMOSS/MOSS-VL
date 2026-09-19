# SPDX-License-Identifier: Apache-2.0
"""``--dp-size`` CLI override for entry-stage data-parallel replicas."""
from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import apply_parallelism_cli_overrides
from sglang_omni.config import ParallelismConfig, PipelineConfig, StageConfig

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def _config(gpu: int | list[int] | None = 0, tp_size: int = 1) -> PipelineConfig:
    return PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="ar",
                process="pipeline",
                factory=_FACTORY,
                gpu=gpu,
                tp_size=tp_size,
                parallelism=ParallelismConfig(tp=tp_size),
                terminal=True,
            ),
        ],
    )


def _apply(config: PipelineConfig, *, dp_size: int | None) -> PipelineConfig:
    return apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=None,
        thinker_gpus=None,
        dp_size=dp_size,
        talker_gpu=None,
        code2wav_gpu=None,
    )


def test_dp_size_applies_to_the_entry_stage() -> None:
    config = _config(gpu=0)
    # GPU ids for the replica ranks arrive through the config / --set channel,
    # mirroring how --thinker-gpus pairs with --thinker-tp-size.
    config.stages[0].gpu = [0, 1]

    merged = _apply(config, dp_size=2)

    stage = merged.stages[0]
    assert stage.dp_size == 2
    assert stage.parallelism.dp == 2
    assert stage.tp_size == 1
    assert stage.gpu == [0, 1]


def test_dp_size_with_tp_requires_tp_times_dp_gpus() -> None:
    config = _config(gpu=[0, 1], tp_size=2)
    config.stages[0].gpu = [0, 1, 2, 3]

    merged = _apply(config, dp_size=2)

    stage = merged.stages[0]
    assert (stage.tp_size, stage.dp_size) == (2, 2)
    assert stage.parallelism.dp == 2

    with pytest.raises(typer.BadParameter, match="exactly 4 GPU ids"):
        _apply(_config(gpu=[0, 1], tp_size=2), dp_size=2)


def test_dp_size_rejects_scalar_gpu_for_multiple_replicas() -> None:
    with pytest.raises(typer.BadParameter, match="must provide 2 GPU ids"):
        _apply(_config(gpu=0), dp_size=2)


def test_dp_size_must_be_at_least_one() -> None:
    with pytest.raises(typer.BadParameter, match="--dp-size must be >= 1"):
        _apply(_config(gpu=0), dp_size=0)


def test_dp_size_one_is_a_noop() -> None:
    config = _config(gpu=0)
    before = config.model_dump()

    merged = _apply(config, dp_size=1)

    assert merged.model_dump() == before
