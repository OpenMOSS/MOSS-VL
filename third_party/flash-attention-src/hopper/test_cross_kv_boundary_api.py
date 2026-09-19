"""API and compatibility tests for ``cross_kv_boundary``."""

import inspect

import pytest
import torch

import flash_attn_interface as top_level_interface
from flash_attn_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_cross_kv_boundary_is_appended_to_public_apis():
    for func in (
        flash_attn_qkvpacked_func,
        flash_attn_func,
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
        get_scheduler_metadata,
    ):
        assert list(inspect.signature(func).parameters)[-1] == "cross_kv_boundary"


def test_package_layout_reexports_top_level_interface():
    import flash_attn_3
    from flash_attn_3 import flash_attn_interface as package_interface

    for name in flash_attn_3.__all__:
        assert getattr(flash_attn_3, name) is getattr(top_level_interface, name)
        assert getattr(package_interface, name) is getattr(top_level_interface, name)
    assert (
        package_interface._flash_attn_forward is top_level_interface._flash_attn_forward
    )
    assert (
        package_interface._flash_attn_backward
        is top_level_interface._flash_attn_backward
    )


def test_flash_attn_func_old_positional_arguments_remain_compatible():
    q = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = flash_attn_func(
        q,
        k,
        v,
        None,
        False,
        None,
        None,
        None,
        None,
        (-1, -1),
        0,
        0.0,
        1,
        None,
        False,
        0,
        False,
    )

    assert out.shape == q.shape


@pytest.mark.parametrize("api", ["standard", "qkvpacked", "varlen"])
def test_cross_kv_boundary_saved_tensor_version_check(api):
    torch.manual_seed(20260807)
    batch_size, seqlen, nheads, head_dim = 1, 8, 2, 64
    boundary = torch.full(
        (batch_size, seqlen) if api != "varlen" else (seqlen,),
        seqlen,
        dtype=torch.int32,
        device="cuda",
    )

    if api == "qkvpacked":
        qkv = torch.randn(
            batch_size,
            seqlen,
            3,
            nheads,
            head_dim,
            dtype=torch.float16,
            device="cuda",
            requires_grad=True,
        )
        out = flash_attn_qkvpacked_func(qkv, cross_kv_boundary=boundary)
    else:
        q_shape = (
            (batch_size, seqlen, nheads, head_dim)
            if api == "standard"
            else (seqlen, nheads, head_dim)
        )
        q = torch.randn(q_shape, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        if api == "standard":
            out = flash_attn_func(q, k, v, cross_kv_boundary=boundary)
        else:
            cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device="cuda")
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                seqlen,
                seqlen,
                cross_kv_boundary=boundary,
            )

    boundary.zero_()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


@pytest.mark.parametrize(
    "boundary,error_type,error_match",
    [
        (
            lambda: torch.ones(1, 8, dtype=torch.int64, device="cuda"),
            TypeError,
            "torch.int32",
        ),
        (
            lambda: torch.ones(8, dtype=torch.int32, device="cuda"),
            ValueError,
            "shape",
        ),
        (
            lambda: torch.ones(1, 8, dtype=torch.int32),
            ValueError,
            "device",
        ),
    ],
)
def test_cross_kv_boundary_validation(boundary, error_type, error_match):
    q = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with pytest.raises(error_type, match=error_match):
        flash_attn_func(q, k, v, cross_kv_boundary=boundary())


def test_cross_kv_boundary_out_of_range_values_are_logically_clamped():
    torch.manual_seed(20260807)
    q = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    boundary = torch.tensor(
        [[-4, 0, 1, 3, 8, 9, 64, 2]], dtype=torch.int32, device="cuda"
    )

    out = flash_attn_func(q, k, v, cross_kv_boundary=boundary)
    out_clamped = flash_attn_func(
        q, k, v, cross_kv_boundary=boundary.clamp(0, k.shape[1])
    )

    torch.testing.assert_close(out, out_clamped, atol=0, rtol=0)
