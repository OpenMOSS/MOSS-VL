"""Platform guards remain GPU-neutral and require complete NPU patches."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def compat():
    path = (
        Path(__file__).resolve().parents[3]
        / "sglang_omni/models/moss_vl_realtime/platform_compat.py"
    )
    spec = importlib.util.spec_from_file_location("platform_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_does_not_load_or_patch_ascend(compat, monkeypatch):
    monkeypatch.setattr(compat, "is_npu_platform", lambda: False)
    monkeypatch.setenv("ASCEND_USE_FA", "true")
    compat.relax_mossvl_flashinfer_guard()
    assert compat.preferred_attention_backend() == "flashinfer"


@pytest.mark.parametrize("torch_support", [False, True])
def test_npu_requires_torch_and_visibility_support(compat, monkeypatch, torch_support):
    monkeypatch.setattr(compat, "is_npu_platform", lambda: True)
    monkeypatch.delenv("ASCEND_USE_FA", raising=False)
    monkeypatch.delenv("ASCEND_USE_FIA", raising=False)

    class Native:
        _moss_vl_torch_cross_attention_supported = torch_support

        def run_sdpa_forward_extend(self, cross_attention_custom_mask=None):
            pass

    class Args:
        def _handle_model_specific_adjustments(self):
            raise AssertionError(compat._FLASHINFER_GUARD_MESSAGE)

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.hardware_backend.npu.attention.ascend_backend",
        SimpleNamespace(
            AscendAttnBackend=SimpleNamespace(_moss_vl_visibility_mask_supported=True)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend",
        SimpleNamespace(AscendTorchNativeAttnBackend=Native),
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.server_args", SimpleNamespace(ServerArgs=Args)
    )
    if not torch_support:
        with pytest.raises(RuntimeError, match="frame-visibility"):
            compat.relax_mossvl_flashinfer_guard()
    else:
        compat.relax_mossvl_flashinfer_guard()
        Args()._handle_model_specific_adjustments()


@pytest.mark.parametrize("flag", ["ASCEND_USE_FA", "ASCEND_USE_FIA"])
def test_unsupported_npu_dispatch_rejected(compat, monkeypatch, flag):
    monkeypatch.setattr(compat, "is_npu_platform", lambda: True)
    monkeypatch.setenv(flag, "true")
    with pytest.raises(RuntimeError, match=flag):
        compat.relax_mossvl_flashinfer_guard()
