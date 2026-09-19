from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.moss_vl_realtime.engine_builder import (
    DECODE_GRAPH_ENCODER_LEN_FILL_VALUE,
    MossVLRealtimeEngineBuilder,
)
from sglang_omni.models.moss_vl_realtime.platform_compat import (
    preferred_attention_backend,
)


def _make_builder(**overrides) -> MossVLRealtimeEngineBuilder:
    kwargs = {
        "max_running_requests": 1,
        "max_new_tokens": 64,
        "context_length": 32768,
        "mem_fraction_static": None,
    }
    kwargs.update(overrides)
    return MossVLRealtimeEngineBuilder(**kwargs)


def _fake_model_worker() -> SimpleNamespace:
    hf_config = SimpleNamespace()
    return SimpleNamespace(
        model_runner=SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))
    )


def test_generation_defaults_enable_decode_graph_by_default() -> None:
    expected_attention_backend = preferred_attention_backend()
    defaults = _make_builder().generation_defaults(dtype="bfloat16")

    assert defaults["disable_cuda_graph"] is False
    assert defaults["disable_prefill_cuda_graph"] is True
    # FlashInfer is the project's decode backend in all modes; setting any
    # backend dimension requires pinning prefill explicitly as well.
    assert defaults["decode_attention_backend"] == expected_attention_backend
    assert defaults["prefill_attention_backend"] == expected_attention_backend


def test_generation_defaults_enable_decode_graph_only() -> None:
    expected_attention_backend = preferred_attention_backend()
    defaults = _make_builder(disable_cuda_graph=False).generation_defaults(
        dtype="bfloat16"
    )

    assert defaults["disable_cuda_graph"] is False
    # Frame extend keeps dynamic shapes and must stay eager.
    assert defaults["disable_prefill_cuda_graph"] is True
    assert defaults["page_size"] == 1
    assert defaults["decode_attention_backend"] == expected_attention_backend


def test_generation_defaults_fix_page_size_to_one() -> None:
    assert _make_builder().generation_defaults(dtype="bfloat16")["page_size"] == 1


def test_tp_uses_nccl_in_one_visible_device_per_rank_topology() -> None:
    overrides = {"tp_size": 2, "disable_custom_all_reduce": False}

    _make_builder().adjust_overrides(overrides)

    assert overrides["disable_custom_all_reduce"] is True


@pytest.mark.parametrize("page_size", [0, 16, 48])
def test_page_size_must_be_one(page_size: int) -> None:
    with pytest.raises(ValueError, match="page_size"):
        _make_builder(page_size=page_size)


def test_async_decode_defaults_off() -> None:
    builder = _make_builder()
    assert builder.enable_async_decode is False


def test_realtime_builder_accepts_multiple_live_requests() -> None:
    builder = _make_builder(max_running_requests=2)
    assert builder.max_running_requests == 2
    assert builder.generation_defaults(dtype="bfloat16")["max_running_requests"] == 2


def test_realtime_builder_rejects_zero_live_requests() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _make_builder(max_running_requests=0)


def test_stage_factory_forwards_tensor_parallel_runtime(monkeypatch) -> None:
    from sglang_omni.models.moss_vl_realtime import stages

    seen = {}

    def fake_build(self, model_path, **kwargs):
        del self
        seen["model_path"] = model_path
        seen.update(kwargs)
        return "scheduler"

    monkeypatch.setattr(MossVLRealtimeEngineBuilder, "build", fake_build)

    result = stages.create_sglang_moss_vl_realtime_executor(
        "/models/moss",
        gpu_id=0,
        tp_rank=1,
        tp_size=2,
        nccl_port=29500,
    )

    assert result == "scheduler"
    assert seen["model_path"] == "/models/moss"
    assert seen["gpu_id"] == 0
    assert seen["tp_rank"] == 1
    assert seen["tp_size"] == 2
    assert seen["nccl_port"] == 29500


def test_extra_scheduler_kwargs_pass_async_decode_through() -> None:
    builder = _make_builder(enable_async_decode=True)
    builder.processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))
    builder.silence_token_ids = (151671,)
    kwargs = builder.extra_scheduler_kwargs()

    assert kwargs["enable_overlap"] is False
    assert kwargs["enable_async_decode"] is True
    assert kwargs["async_decode_min_batch_size"] == 1

    default_builder = _make_builder()
    default_builder.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=0)
    )
    default_builder.silence_token_ids = (151671,)
    default_kwargs = default_builder.extra_scheduler_kwargs()
    assert default_kwargs["enable_async_decode"] is False


def test_setup_model_sets_encoder_len_fill_value_when_graph_enabled() -> None:
    model_worker = _fake_model_worker()

    _make_builder(disable_cuda_graph=False).setup_model(
        model_worker=model_worker,
        checkpoint_dir="",
        device="cuda:0",
        gpu_id=0,
        server_args=None,
    )

    hf_config = model_worker.model_runner.model_config.hf_config
    assert hf_config.max_source_positions == DECODE_GRAPH_ENCODER_LEN_FILL_VALUE


def test_setup_model_leaves_hf_config_untouched_when_graph_disabled() -> None:
    model_worker = _fake_model_worker()

    _make_builder(disable_cuda_graph=True).setup_model(
        model_worker=model_worker,
        checkpoint_dir="",
        device="cuda:0",
        gpu_id=0,
        server_args=None,
    )

    hf_config = model_worker.model_runner.model_config.hf_config
    assert not hasattr(hf_config, "max_source_positions")
