# SPDX-License-Identifier: Apache-2.0
"""Raw ``dp_size`` overrides are rejected in favor of pipeline-level DP.

Native data parallelism sets the process topology, so ``dp_size > 1`` must
come from the pipeline parallelism config; a raw server_args override would
silently compute parallel state for replicas that were never launched.
"""

from __future__ import annotations

from typing import Any

import pytest

from sglang_omni.scheduling.sglang_backend import server_args_builder


class _CapturedServerArgs:
    """Stands in for ServerArgs so no HF checkpoint is needed."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.enable_dp_attention = False
        self.dp_size = kwargs.get("dp_size", 1)


def _build(monkeypatch, **extra: Any) -> dict[str, Any]:
    monkeypatch.setattr(server_args_builder, "ServerArgs", _CapturedServerArgs)
    built = server_args_builder.build_sglang_server_args(
        "model", context_length=128, **extra
    )
    return built.kwargs


def test_default_dp_size_is_accepted(monkeypatch) -> None:
    assert "dp_size" not in _build(monkeypatch)


def test_caller_supplied_dp_size_one_still_passes(monkeypatch) -> None:
    assert _build(monkeypatch, dp_size=1)["dp_size"] == 1


def test_user_supplied_dp_size_above_one_is_rejected(monkeypatch) -> None:
    with pytest.raises(ValueError, match="dp_size"):
        _build(monkeypatch, dp_size=2)


def test_pipeline_injected_dp_size_bypasses_the_guard(monkeypatch) -> None:
    assert _build(monkeypatch, dp_size=2, allow_native_dp=True)["dp_size"] == 2


def test_enable_dp_attention_remains_rejected(monkeypatch) -> None:
    class _DpAttentionServerArgs(_CapturedServerArgs):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.enable_dp_attention = True

    monkeypatch.setattr(server_args_builder, "ServerArgs", _DpAttentionServerArgs)
    with pytest.raises(ValueError, match="enable_dp_attention"):
        server_args_builder.build_sglang_server_args("model", context_length=128)
