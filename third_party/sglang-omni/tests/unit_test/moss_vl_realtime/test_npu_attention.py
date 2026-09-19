"""Opt-in checks of patched SGLang attention; run on CPU and on an Ascend card."""

import importlib.util
import inspect
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("MOSSVL_TEST_NPU_PATCHES") != "1",
    reason="requires the installed Ascend environment patches",
)


def backend_and_device():
    device = os.environ.get("NPU_TEST_DEVICE", "cpu")
    if device.startswith("npu"):
        import torch_npu  # noqa: F401
    site = os.environ.get("NPU_TEST_SITE_PACKAGES")
    if site:
        root = Path(site) / "sglang"
    else:
        spec = importlib.util.find_spec("sglang")
        root = Path(spec.origin).parent
    path = root / "srt/hardware_backend/npu/attention/ascend_torch_native_backend.py"
    spec = importlib.util.spec_from_file_location("moss_test_ascend_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.AscendTorchNativeAttnBackend()
    assert (
        "cross_attention_custom_mask"
        in inspect.signature(backend.run_sdpa_forward_extend).parameters
    )
    assert backend._moss_vl_torch_cross_attention_supported
    return backend, device


@pytest.mark.parametrize("logit_cap", [0.0, 20.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixed_requests_visibility_and_fully_masked_rows(logit_cap, dtype):
    backend, device = backend_and_device()
    query = torch.zeros((5, 2, 4), device=device, dtype=dtype)
    keys = torch.zeros((5, 1, 4), device=device, dtype=dtype)
    values = (
        torch.tensor([1, 3, 5, 7, 9], device=device, dtype=dtype)[:, None, None]
        .expand(5, 1, 4)
        .contiguous()
    )
    mask = torch.tensor(
        [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0], device=device, dtype=torch.uint8
    )
    actual = backend.run_sdpa_forward_extend(
        query,
        torch.empty_like(query),
        keys,
        values,
        torch.tensor([[0, 1, 0], [2, 3, 4]], device=device),
        torch.tensor([0, 1], device=device),
        torch.tensor([7, 2], device=device),
        torch.tensor([4, 0], device=device),
        torch.tensor([3, 2], device=device),
        torch.tensor([2, 3], device=device),
        is_cross_attention=True,
        enable_gqa=True,
        logit_cap=logit_cap,
        cross_attention_custom_mask=mask,
    )
    expected = torch.tensor([1, 0, 2, 7, 6], dtype=torch.float32)[:, None, None].expand(
        5, 2, 4
    )
    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.05, rtol=0.01)


def test_empty_encoder_produces_zero_output():
    backend, device = backend_and_device()
    query = torch.zeros((2, 1, 4), device=device)
    actual = backend.run_sdpa_forward_extend(
        query,
        torch.empty_like(query),
        torch.empty((0, 1, 4), device=device),
        torch.empty((0, 1, 4), device=device),
        torch.empty((1, 0), dtype=torch.long, device=device),
        torch.tensor([0], device=device),
        torch.tensor([2], device=device),
        torch.tensor([0], device=device),
        torch.tensor([2], device=device),
        torch.tensor([0], device=device),
        is_cross_attention=True,
        cross_attention_custom_mask=torch.empty(0, dtype=torch.uint8, device=device),
    )
    torch.testing.assert_close(actual, torch.zeros_like(query))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_self_attention_extend_reads_complete_text_prefix_after_encoder(dtype):
    backend, device = backend_and_device()
    torch.manual_seed(17)
    query = torch.randn(3, 2, 4, device=device, dtype=dtype)
    keys = torch.randn(10, 1, 4, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    table = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0, 0]], device=device)
    actual = backend.run_sdpa_forward_extend(
        query,
        torch.empty_like(query),
        keys,
        values,
        table,
        torch.tensor([0, 1], device=device),
        torch.tensor([5, 2], device=device),
        torch.tensor([3, 1], device=device),
        torch.tensor([2, 1], device=device),
        torch.tensor([2, 1], device=device),
        is_cross_attention=False,
        causal=True,
        enable_gqa=True,
    )
    expected = []
    for q, indices, prefix in [(query[:2], [2, 3, 4, 5, 6], 3), (query[2:], [8, 9], 1)]:
        mask = torch.arange(len(indices), device=device)[None, :] <= (
            prefix + torch.arange(q.shape[0], device=device)[:, None]
        )
        expected.append(
            torch.nn.functional.scaled_dot_product_attention(
                q.float().transpose(0, 1),
                keys[indices].float().transpose(0, 1),
                values[indices].float().transpose(0, 1),
                attn_mask=mask,
                enable_gqa=True,
            ).transpose(0, 1)
        )
    torch.testing.assert_close(
        actual.float(), torch.cat(expected), atol=0.04, rtol=0.03
    )


def test_mask_length_is_validated():
    backend, device = backend_and_device()
    query = torch.zeros((2, 1, 4), device=device)
    with pytest.raises(ValueError, match="Q/KV lengths"):
        backend.run_sdpa_forward_extend(
            query,
            torch.empty_like(query),
            query,
            query,
            torch.tensor([[0, 1]], device=device),
            torch.tensor([0], device=device),
            torch.tensor([2], device=device),
            torch.tensor([0], device=device),
            torch.tensor([2], device=device),
            torch.tensor([2], device=device),
            is_cross_attention=True,
            cross_attention_custom_mask=torch.ones(1, device=device),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cross_attention_uses_torch_and_preserves_reference(monkeypatch, dtype):
    backend, device = backend_and_device()
    torch.manual_seed(7)
    query = torch.randn(3, 2, 4, device=device, dtype=dtype)
    key = torch.randn(2, 1, 4, device=device, dtype=dtype)
    value = torch.randn(2, 1, 4, device=device, dtype=dtype)
    mask = torch.tensor([[1, 0], [0, 0], [1, 1]], device=device, dtype=torch.bool)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.float().cpu().transpose(0, 1),
        key.float().cpu().transpose(0, 1),
        value.float().cpu().transpose(0, 1),
        attn_mask=mask.cpu(),
        enable_gqa=True,
    ).transpose(0, 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("cross-attention entered fused SDPA")

    monkeypatch.setitem(
        backend.run_sdpa_forward_extend.__globals__,
        "scaled_dot_product_attention",
        forbidden,
    )
    actual = backend.run_sdpa_forward_extend(
        query,
        torch.empty_like(query),
        key,
        value,
        torch.tensor([[0, 1]], device=device),
        torch.tensor([0], device=device),
        torch.tensor([3], device=device),
        torch.tensor([0], device=device),
        torch.tensor([3], device=device),
        torch.tensor([2], device=device),
        is_cross_attention=True,
        enable_gqa=True,
        cross_attention_custom_mask=mask.flatten(),
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.03, rtol=0.03)


def test_self_attention_retains_sdpa_dispatch(monkeypatch):
    backend, device = backend_and_device()
    calls = []
    original = backend.run_sdpa_forward_extend.__globals__[
        "scaled_dot_product_attention"
    ]

    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setitem(
        backend.run_sdpa_forward_extend.__globals__,
        "scaled_dot_product_attention",
        record,
    )
    query = torch.ones(2, 1, 4, device=device)
    actual = backend.run_sdpa_forward_extend(
        query,
        torch.empty_like(query),
        query,
        query,
        torch.tensor([[0, 1]], device=device),
        torch.tensor([0], device=device),
        torch.tensor([2], device=device),
        torch.tensor([0], device=device),
        torch.tensor([2], device=device),
        is_cross_attention=False,
        causal=True,
    )
    assert calls == [True]
    torch.testing.assert_close(actual, query)
