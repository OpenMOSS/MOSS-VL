"""Tests for staircase mask (cross_kv_boundary) in Flash Attention 3."""

import itertools

import pytest
import torch

from cross_kv_boundary_test_utils import (
    make_staircase_boundary,
    staircase_attention_ref,
)

from flash_attn_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
)


def split_qkvpacked(qkv, num_heads_q=None):
    if qkv.dim() == 5:
        return qkv.unbind(dim=2)
    assert num_heads_q is not None
    num_heads_k = (qkv.shape[2] - num_heads_q) // 2
    return qkv.split([num_heads_q, num_heads_k, num_heads_k], dim=2)


def make_raw_nonmonotonic_boundary(batch_size, seqlen_q, seqlen_k, case):
    """Generate raw per-row prefix lengths without applying cummax.

    These patterns exercise raw non-monotonic cross_kv_boundary semantics, including
    right-padding drops. Each row independently attends to [0, cross_kv_boundary[row]).
    """
    row = torch.arange(seqlen_q, device="cuda")

    def prefix_and_tail(preferred_prefix, preferred_tail):
        prefix_len = min(preferred_prefix, max(1, seqlen_q // 8))
        available_after_prefix = max(seqlen_q - prefix_len - 1, 1)
        tail_len = min(preferred_tail, max(8, seqlen_q // 4), available_after_prefix)
        tail_start = seqlen_q - tail_len
        return prefix_len, tail_start

    if case == "right_padding_drop":
        valid_end = max(1, seqlen_q - max(16, seqlen_q // 4))
        ramp = torch.linspace(1, seqlen_k, valid_end, device="cuda").to(torch.int32)
        boundary = torch.zeros((batch_size, seqlen_q), dtype=torch.int32, device="cuda")
        boundary[:, :valid_end] = ramp.clamp(min=1, max=seqlen_k)
        boundary[:, : min(8, valid_end)] = 0

    elif case == "right_padding_block_edges":
        boundary_1d = (((row // 64) + 1) * 64).clamp(max=seqlen_k).to(torch.int32)
        prefix_len, tail_start = prefix_and_tail(64, 128)
        boundary_1d[:prefix_len] = 0
        boundary_1d[tail_start:] = 0
        boundary = boundary_1d.unsqueeze(0).expand(batch_size, -1).contiguous()

    elif case == "right_padding_off_block_edges":
        boundary_1d = (
            (((row // 64) + 1) * 64 - 17).clamp(min=1, max=seqlen_k).to(torch.int32)
        )
        prefix_len, tail_start = prefix_and_tail(17, 119)
        boundary_1d[:prefix_len] = 0
        boundary_1d[tail_start:] = 0
        boundary = boundary_1d.unsqueeze(0).expand(batch_size, -1).contiguous()

    elif case == "per_batch_tail_lengths":
        base = torch.linspace(1, seqlen_k, seqlen_q, device="cuda").to(torch.int32)
        boundary = torch.zeros((batch_size, seqlen_q), dtype=torch.int32, device="cuda")
        for b in range(batch_size):
            tail = max(8, (b + 1) * seqlen_q // (batch_size + 3))
            valid_end = max(1, seqlen_q - tail)
            boundary[b, :valid_end] = base[:valid_end].clamp(min=1, max=seqlen_k)
            boundary[b, : min(8, valid_end)] = 0

    elif case == "interior_zero_island":
        base = torch.linspace(1, seqlen_k, seqlen_q, device="cuda").to(torch.int32)
        boundary = base.unsqueeze(0).expand(batch_size, -1).clone().contiguous()
        island_start = seqlen_q // 3
        island_end = min(seqlen_q, island_start + max(8, seqlen_q // 8))
        boundary[:, island_start:island_end] = 0
        boundary[:, : min(8, seqlen_q)] = 0

    else:
        raise ValueError(f"Unknown raw boundary case: {case}")

    boundary = boundary.clamp(min=0, max=seqlen_k).contiguous()
    assert (boundary[:, 1:] < boundary[:, :-1]).any(), case
    return boundary


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["linear", "step", "constant", "random"])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [(64, 64), (128, 128), (256, 256), (64, 256), (128, 512)],
)
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("nheads_kv", [1, 4])
def test_staircase_mask_fwd(seqlen_q, seqlen_k, d, nheads_kv, pattern, dtype):
    """Forward pass: compare FA3 staircase output against reference dense-mask attention."""
    batch_size = 2
    nheads = 8

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)

    cross_kv_boundary = make_staircase_boundary(batch_size, seqlen_q, seqlen_k, pattern)

    out_ref = staircase_attention_ref(q, k, v, cross_kv_boundary)
    out_fa3 = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )

    atol = 1e-2 if dtype == torch.float16 else 5e-2
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(out_fa3, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["linear", "random"])
@pytest.mark.parametrize("d", [64, 128])
def test_staircase_mask_bwd(d, pattern, dtype):
    """Backward pass: compare FA3 staircase gradients against reference."""
    batch_size = 2
    seqlen_q = 128
    seqlen_k = 128
    nheads = 4
    nheads_kv = 4

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    cross_kv_boundary = make_staircase_boundary(batch_size, seqlen_q, seqlen_k, pattern)

    out_fa3 = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    out_ref = staircase_attention_ref(q_ref, k_ref, v_ref, cross_kv_boundary)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g.float() if out_ref.dtype == torch.float32 else g)

    atol = 1e-2 if dtype == torch.float16 else 5e-2
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(64, 256), (192, 128)])
@pytest.mark.parametrize("pattern", ["linear", "step"])
def test_staircase_mask_bwd_cross_attention_shapes(seqlen_q, seqlen_k, pattern, dtype):
    """Backward pass for non-square cross-attention shapes."""
    batch_size = 2
    nheads = 4
    nheads_kv = 4
    d = 64

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    cross_kv_boundary = make_staircase_boundary(batch_size, seqlen_q, seqlen_k, pattern)

    out_fa3 = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    out_ref = staircase_attention_ref(q_ref, k_ref, v_ref, cross_kv_boundary)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    atol = 1e-2 if dtype == torch.float16 else 5e-2
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_staircase_mask_bwd_gqa(dtype):
    """Backward pass with grouped-query attention."""
    batch_size = 2
    seqlen_q = 128
    seqlen_k = 256
    nheads = 8
    nheads_kv = 2
    d = 64

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    cross_kv_boundary = make_staircase_boundary(
        batch_size, seqlen_q, seqlen_k, "linear"
    )

    out_fa3 = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    out_ref = staircase_attention_ref(q_ref, k_ref, v_ref, cross_kv_boundary)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    atol = 1e-2 if dtype == torch.float16 else 5e-2
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=atol, rtol=rtol)


def test_staircase_mask_bwd_sm90_separate_masking_long():
    """Regression for SM90 hdim=64 backward split masking iterations."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SM90-specific separate masking regression")

    torch.manual_seed(0)
    batch_size = 1
    seqlen_q = 512
    seqlen_k = 512
    nheads = 2
    nheads_kv = 1
    d = 64
    dtype = torch.float16

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    cross_kv_boundary = make_staircase_boundary(
        batch_size, seqlen_q, seqlen_k, "linear"
    )

    out_fa3 = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    out_ref = staircase_attention_ref(q_ref, k_ref, v_ref, cross_kv_boundary)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("case", ["delayed_step", "block_edges", "zero_prefix"])
def test_staircase_mask_bwd_sm90_boundary_cases(case):
    """SM90 hdim=64 backward boundary cases for staircase split masking.

    These patterns are intentionally not diagonal/causal-like. In particular,
    ``delayed_step`` keeps many later Q blocks only partially visible to an
    early K block; a local-window-style no-mask split would incorrectly skip
    staircase masking for those blocks.
    """
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SM90-specific separate masking regression")

    torch.manual_seed(1234)
    batch_size = 1
    seqlen_q = 512
    seqlen_k = 512
    nheads = 2
    nheads_kv = 1
    d = 64
    dtype = torch.float16

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    if case == "delayed_step":
        cross_kv_boundary = torch.full(
            (batch_size, seqlen_q), 96, dtype=torch.int32, device="cuda"
        )
        cross_kv_boundary[:, 384:] = seqlen_k
    elif case == "block_edges":
        row = torch.arange(seqlen_q, device="cuda")
        boundary = ((row // 64) + 1).clamp(max=8) * 64
        cross_kv_boundary = boundary.to(torch.int32).unsqueeze(0).contiguous()
    elif case == "zero_prefix":
        cross_kv_boundary = torch.full(
            (batch_size, seqlen_q), 128, dtype=torch.int32, device="cuda"
        )
        cross_kv_boundary[:, :64] = 0
        cross_kv_boundary[:, 256:] = seqlen_k
    else:
        raise ValueError(f"Unknown case: {case}")

    out_fa3 = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    out_ref = staircase_attention_ref(q_ref, k_ref, v_ref, cross_kv_boundary)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    torch.testing.assert_close(out_fa3, out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case",
    [
        "right_padding_drop",
        "right_padding_block_edges",
        "right_padding_off_block_edges",
        "per_batch_tail_lengths",
        "interior_zero_island",
    ],
)
def test_staircase_mask_raw_nonmonotonic_fwd(case, dtype):
    """Target semantics for raw, non-monotonic cross_kv_boundary in forward.

    This intentionally does not apply cummax. Each row independently attends to
    [0, cross_kv_boundary[row]), including rows that drop back to 0 for right padding.
    """
    batch_size = 2
    seqlen_q = 192
    seqlen_k = 256
    nheads = 4
    nheads_kv = 2
    d = 64

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)

    cross_kv_boundary = make_raw_nonmonotonic_boundary(
        batch_size, seqlen_q, seqlen_k, case
    )

    out_fa3 = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    out_ref = staircase_attention_ref(q, k, v, cross_kv_boundary)

    atol = 1e-2 if dtype == torch.float16 else 5e-2
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(out_fa3, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "case",
    [
        "right_padding_drop",
        "right_padding_block_edges",
        "right_padding_off_block_edges",
        "per_batch_tail_lengths",
        "interior_zero_island",
    ],
)
def test_staircase_mask_raw_nonmonotonic_bwd_sm90(case):
    """Target backward semantics for raw, non-monotonic cross_kv_boundary on SM90."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SM90-specific raw non-monotonic cross_kv_boundary regression")

    torch.manual_seed(5678)
    batch_size = 2
    seqlen_q = 512
    seqlen_k = 512
    nheads = 4
    nheads_kv = 2
    d = 64
    dtype = torch.float16

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    cross_kv_boundary = make_raw_nonmonotonic_boundary(
        batch_size, seqlen_q, seqlen_k, case
    )

    out_fa3 = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    out_ref = staircase_attention_ref(q_ref, k_ref, v_ref, cross_kv_boundary)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    torch.testing.assert_close(out_fa3, out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("case", ["right_padding_drop", "per_batch_tail_lengths"])
def test_staircase_mask_raw_nonmonotonic_varlen_bwd(case):
    """Varlen target semantics for raw non-monotonic cross_kv_boundary."""
    batch_size = 3
    seqlens_q = [96, 128, 80]
    seqlens_k = [160, 256, 192]
    nheads = 4
    nheads_kv = 2
    d = 64
    dtype = torch.float16

    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    cu_seqlens_q = torch.tensor(
        [0] + list(itertools.accumulate(seqlens_q)), dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(itertools.accumulate(seqlens_k)), dtype=torch.int32, device="cuda"
    )

    q = torch.randn(
        sum(seqlens_q), nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        sum(seqlens_k), nheads_kv, d, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        sum(seqlens_k), nheads_kv, d, device="cuda", dtype=dtype, requires_grad=True
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    cross_kv_boundary_list = []
    for i, (sq, sk) in enumerate(zip(seqlens_q, seqlens_k)):
        bnd_case = case if i != 1 else "right_padding_off_block_edges"
        bnd = make_raw_nonmonotonic_boundary(1, sq, sk, bnd_case).squeeze(0)
        cross_kv_boundary_list.append(bnd)
    cross_kv_boundary = torch.cat(cross_kv_boundary_list, dim=0)

    out_fa3 = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
        cross_kv_boundary=cross_kv_boundary,
    )

    out_ref_list = []
    for i in range(batch_size):
        q_i = q_ref[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].unsqueeze(0)
        k_i = k_ref[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0)
        v_i = v_ref[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0)
        bnd_i = cross_kv_boundary_list[i].unsqueeze(0)
        out_ref_list.append(staircase_attention_ref(q_i, k_i, v_i, bnd_i).squeeze(0))
    out_ref = torch.cat(out_ref_list, dim=0)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    torch.testing.assert_close(out_fa3, out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["linear", "step"])
def test_staircase_mask_varlen(pattern, dtype):
    """Varlen path: compare FA3 staircase output against reference."""
    batch_size = 3
    seqlens_q = [32, 64, 48]
    seqlens_k = [64, 128, 96]
    nheads = 4
    nheads_kv = 2
    d = 64

    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    cu_seqlens_q = torch.tensor(
        [0] + list(itertools.accumulate(seqlens_q)), dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(itertools.accumulate(seqlens_k)), dtype=torch.int32, device="cuda"
    )

    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    q = torch.randn(total_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(total_k, nheads_kv, d, device="cuda", dtype=dtype)
    v = torch.randn(total_k, nheads_kv, d, device="cuda", dtype=dtype)

    cross_kv_boundary_list = []
    for i in range(batch_size):
        b = make_staircase_boundary(1, seqlens_q[i], seqlens_k[i], pattern)
        cross_kv_boundary_list.append(b.squeeze(0))
    cross_kv_boundary = torch.cat(cross_kv_boundary_list, dim=0)  # (total_q,)

    out_fa3 = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
        cross_kv_boundary=cross_kv_boundary,
    )

    # Reference: process each batch element separately
    out_ref_list = []
    for i in range(batch_size):
        q_i = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].unsqueeze(0)  # (1, sq, h, d)
        k_i = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0)
        v_i = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0)
        bnd_i = cross_kv_boundary_list[i].unsqueeze(0)  # (1, sq)
        out_i = staircase_attention_ref(q_i, k_i, v_i, bnd_i)
        out_ref_list.append(out_i.squeeze(0))
    out_ref = torch.cat(out_ref_list, dim=0)

    atol = 1e-2 if dtype == torch.float16 else 5e-2
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(out_fa3, out_ref.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["linear", "step"])
def test_staircase_mask_varlen_bwd(pattern, dtype):
    """Varlen backward: compare gradients with per-sequence dense references."""
    batch_size = 3
    seqlens_q = [32, 64, 48]
    seqlens_k = [64, 128, 96]
    nheads = 4
    nheads_kv = 2
    d = 64

    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    cu_seqlens_q = torch.tensor(
        [0] + list(itertools.accumulate(seqlens_q)), dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(itertools.accumulate(seqlens_k)), dtype=torch.int32, device="cuda"
    )

    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    q = torch.randn(total_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(
        total_k, nheads_kv, d, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        total_k, nheads_kv, d, device="cuda", dtype=dtype, requires_grad=True
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    cross_kv_boundary_list = []
    for i in range(batch_size):
        b = make_staircase_boundary(1, seqlens_q[i], seqlens_k[i], pattern)
        cross_kv_boundary_list.append(b.squeeze(0))
    cross_kv_boundary = torch.cat(cross_kv_boundary_list, dim=0)  # (total_q,)

    out_fa3 = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
        cross_kv_boundary=cross_kv_boundary,
    )

    out_ref_list = []
    for i in range(batch_size):
        q_i = q_ref[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].unsqueeze(0)
        k_i = k_ref[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0)
        v_i = v_ref[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0)
        bnd_i = cross_kv_boundary_list[i].unsqueeze(0)
        out_i = staircase_attention_ref(q_i, k_i, v_i, bnd_i)
        out_ref_list.append(out_i.squeeze(0))
    out_ref = torch.cat(out_ref_list, dim=0)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    atol = 1e-2 if dtype == torch.float16 else 5e-2
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_staircase_mask_zero_boundary_rows(dtype):
    """Rows with cross_kv_boundary=0 should produce zero output and matching gradients."""
    batch_size = 2
    seqlen_q = 128
    seqlen_k = 256
    nheads = 4
    d = 64

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen_k, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    cross_kv_boundary = make_staircase_boundary(batch_size, seqlen_q, seqlen_k, "step")
    cross_kv_boundary[:, : seqlen_q // 4] = 0

    out_fa3 = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    out_ref = staircase_attention_ref(q_ref, k_ref, v_ref, cross_kv_boundary)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    torch.testing.assert_close(
        out_fa3[:, : seqlen_q // 4],
        torch.zeros_like(out_fa3[:, : seqlen_q // 4]),
        atol=0,
        rtol=0,
    )
    atol = 1e-2 if dtype == torch.float16 else 5e-2
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(out_fa3, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_staircase_mask_mossvl_multivideo_right_padding_precision(dtype):
    """MossVL-style multi-video rows must not see right-padded KV tokens."""
    torch.manual_seed(20260527)
    batch_size = 1
    seqlen_q = 7
    seqlen_k = 24
    valid_k_end = 18
    nheads = 4
    nheads_kv = 2
    d = 64

    # Two videos:
    #   video 1: 2 frames, cumulative boundaries 3, 6
    #   video 2: 3 frames, cumulative boundaries 10, 14, 18
    # Rows 5 and 6 are right-padded text rows and must produce exact zeros.
    cross_kv_boundary = torch.tensor(
        [[3, 6, 10, 14, 18, 0, 0]], dtype=torch.int32, device="cuda"
    )

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_kv,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_fa3 = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    out_ref = staircase_attention_ref(q_ref, k_ref, v_ref, cross_kv_boundary)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    atol = 1e-2 if dtype == torch.float16 else 5e-2
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(out_fa3, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=atol, rtol=rtol)

    torch.testing.assert_close(
        out_fa3[:, 5:], torch.zeros_like(out_fa3[:, 5:]), atol=0, rtol=0
    )
    torch.testing.assert_close(
        k.grad[:, valid_k_end:],
        torch.zeros_like(k.grad[:, valid_k_end:]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        v.grad[:, valid_k_end:],
        torch.zeros_like(v.grad[:, valid_k_end:]),
        atol=0,
        rtol=0,
    )

    k_poison = k.detach().clone()
    v_poison = v.detach().clone()
    k_poison[:, valid_k_end:] = 1000
    v_poison[:, valid_k_end:] = -1000
    out_poison = flash_attn_func(
        q.detach(),
        k_poison,
        v_poison,
        causal=False,
        cross_kv_boundary=cross_kv_boundary,
    )
    torch.testing.assert_close(out_poison, out_fa3.detach(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_staircase_constant_boundary_matches_no_mask(dtype):
    """When cross_kv_boundary is all seqlen_k, output should match unmasked attention."""
    torch.manual_seed(0)
    batch_size = 2
    seqlen_q = 128
    seqlen_k = 128
    nheads = 4
    d = 64

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device="cuda", dtype=dtype)

    cross_kv_boundary = torch.full(
        (batch_size, seqlen_q), seqlen_k, dtype=torch.int32, device="cuda"
    )

    out_staircase = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    out_no_mask = flash_attn_func(q, k, v, causal=False)

    tol = max(1e-3, torch.finfo(dtype).eps)
    torch.testing.assert_close(out_staircase, out_no_mask, atol=tol, rtol=tol)


def test_staircase_causal_raises():
    """Using cross_kv_boundary with is_causal should raise an error."""
    batch_size = 1
    seqlen = 64
    nheads = 4
    d = 64

    q = torch.randn(batch_size, seqlen, nheads, d, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, seqlen, nheads, d, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, seqlen, nheads, d, device="cuda", dtype=torch.float16)

    cross_kv_boundary = torch.full(
        (batch_size, seqlen), seqlen, dtype=torch.int32, device="cuda"
    )

    with pytest.raises(ValueError, match="cross_kv_boundary.*cannot.*causal"):
        flash_attn_func(q, k, v, causal=True, cross_kv_boundary=cross_kv_boundary)


@pytest.mark.parametrize("packed_layout", ["standard", "gqa"])
@pytest.mark.parametrize("case", ["right_padding_drop", "interior_zero_island"])
def test_staircase_mask_qkvpacked_raw_nonmonotonic_bwd(packed_layout, case):
    """Packed-QKV forward and backward must use cross_kv_boundary for each Q row."""
    torch.manual_seed(20260524 + len(case))
    batch_size = 2
    seqlen = 128
    d = 64
    dtype = torch.float16

    if packed_layout == "standard":
        nheads = 4
        q = torch.randn(batch_size, seqlen, nheads, d, device="cuda", dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, d, device="cuda", dtype=dtype)
        v = torch.randn(batch_size, seqlen, nheads, d, device="cuda", dtype=dtype)
        qkv = torch.stack([q, k, v], dim=2).detach().requires_grad_(True)
        num_heads_q = None
    else:
        nheads = 8
        nheads_kv = 2
        q = torch.randn(batch_size, seqlen, nheads, d, device="cuda", dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads_kv, d, device="cuda", dtype=dtype)
        v = torch.randn(batch_size, seqlen, nheads_kv, d, device="cuda", dtype=dtype)
        qkv = torch.cat([q, k, v], dim=2).detach().requires_grad_(True)
        num_heads_q = nheads

    qkv_ref = qkv.detach().clone().requires_grad_(True)
    cross_kv_boundary = make_raw_nonmonotonic_boundary(batch_size, seqlen, seqlen, case)

    out_fa3 = flash_attn_qkvpacked_func(
        qkv,
        causal=False,
        num_heads_q=num_heads_q,
        cross_kv_boundary=cross_kv_boundary,
    )
    q_ref, k_ref, v_ref = split_qkvpacked(qkv_ref, num_heads_q)
    out_ref = staircase_attention_ref(q_ref, k_ref, v_ref, cross_kv_boundary)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    torch.testing.assert_close(out_fa3, out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(qkv.grad, qkv_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)


def test_staircase_mask_qkvpacked_full_boundary_matches_unmasked():
    """All-full cross_kv_boundary should match the unmasked packed-QKV path."""
    torch.manual_seed(20260525)
    batch_size = 2
    seqlen = 128
    nheads = 4
    d = 64
    dtype = torch.float16

    qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device="cuda", dtype=dtype)
    cross_kv_boundary = torch.full(
        (batch_size, seqlen), seqlen, dtype=torch.int32, device="cuda"
    )

    out_masked = flash_attn_qkvpacked_func(
        qkv, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    out_unmasked = flash_attn_qkvpacked_func(qkv, causal=False)

    torch.testing.assert_close(out_masked, out_unmasked, atol=2e-2, rtol=2e-2)


def test_staircase_mask_qkvpacked_causal_rejects_boundary():
    """Packed-QKV should reject cross_kv_boundary with causal attention."""
    batch_size = 1
    seqlen = 64
    nheads = 4
    d = 64

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device="cuda", dtype=torch.float16
    )
    cross_kv_boundary = torch.full(
        (batch_size, seqlen), seqlen, dtype=torch.int32, device="cuda"
    )

    with pytest.raises(ValueError, match="cross_kv_boundary.*cannot.*causal"):
        flash_attn_qkvpacked_func(qkv, causal=True, cross_kv_boundary=cross_kv_boundary)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_staircase_gqa(dtype):
    """Test staircase mask with grouped-query attention."""
    batch_size = 2
    seqlen_q = 128
    seqlen_k = 256
    nheads = 8
    nheads_kv = 2
    d = 64

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)

    cross_kv_boundary = make_staircase_boundary(
        batch_size, seqlen_q, seqlen_k, "linear"
    )

    out_ref = staircase_attention_ref(q, k, v, cross_kv_boundary)
    out_fa3 = flash_attn_func(
        q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
    )

    atol = 1e-2 if dtype == torch.float16 else 5e-2
    rtol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(out_fa3, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("num_splits", [1, 2])
@pytest.mark.parametrize("case", ["right_padding_drop", "interior_zero_island"])
def test_staircase_mask_pack_gqa_raw_nonmonotonic_fwd(num_splits, case):
    """Packed-GQA forward must index cross_kv_boundary by the real Q row."""
    torch.manual_seed(20260522 + num_splits)
    batch_size = 2
    seqlen_q = 192
    seqlen_k = 320
    nheads_kv = 2
    gqa = 4
    nheads = nheads_kv * gqa
    d = 64
    dtype = torch.float16

    q = torch.randn(batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device="cuda", dtype=dtype)
    cross_kv_boundary = make_raw_nonmonotonic_boundary(
        batch_size, seqlen_q, seqlen_k, case
    )

    out_fa3 = flash_attn_func(
        q,
        k,
        v,
        causal=False,
        num_splits=num_splits,
        pack_gqa=True,
        cross_kv_boundary=cross_kv_boundary,
    )
    out_ref = staircase_attention_ref(q, k, v, cross_kv_boundary)

    torch.testing.assert_close(out_fa3, out_ref, atol=2e-2, rtol=2e-2)


def test_staircase_mask_varlen_pack_gqa_raw_nonmonotonic_fwd():
    """Varlen packed-GQA scheduler and mask must use un-packed Q rows for boundaries."""
    torch.manual_seed(20260523)
    seqlens_q = [65, 96, 129]
    seqlens_k = [96, 160, 192]
    cases = [
        "right_padding_off_block_edges",
        "per_batch_tail_lengths",
        "right_padding_block_edges",
    ]
    nheads_kv = 2
    gqa = 4
    nheads = nheads_kv * gqa
    d = 64
    dtype = torch.float16

    cu_seqlens_q = torch.tensor(
        [0] + list(itertools.accumulate(seqlens_q)), dtype=torch.int32, device="cuda"
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(itertools.accumulate(seqlens_k)), dtype=torch.int32, device="cuda"
    )
    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)

    q = torch.randn(sum(seqlens_q), nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(sum(seqlens_k), nheads_kv, d, device="cuda", dtype=dtype)
    v = torch.randn(sum(seqlens_k), nheads_kv, d, device="cuda", dtype=dtype)

    cross_kv_boundary_list = [
        make_raw_nonmonotonic_boundary(1, sq, sk, case).squeeze(0)
        for sq, sk, case in zip(seqlens_q, seqlens_k, cases)
    ]
    cross_kv_boundary = torch.cat(cross_kv_boundary_list, dim=0).contiguous()

    out_fa3 = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
        num_splits=2,
        pack_gqa=True,
        cross_kv_boundary=cross_kv_boundary,
    )

    out_ref_list = []
    for i, (sq, sk) in enumerate(zip(seqlens_q, seqlens_k)):
        q_i = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].unsqueeze(0)
        k_i = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0)
        v_i = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].unsqueeze(0)
        bnd_i = cross_kv_boundary_list[i].unsqueeze(0)
        out_ref_list.append(staircase_attention_ref(q_i, k_i, v_i, bnd_i).squeeze(0))
    out_ref = torch.cat(out_ref_list, dim=0)

    torch.testing.assert_close(out_fa3, out_ref, atol=2e-2, rtol=2e-2)


def test_staircase_mask_varlen_zero_kv_segment_text_only_doc():
    """Packed varlen cross-attention must support a text-only document with no KV segment."""
    torch.manual_seed(20260526)
    seqlens_q = [3, 4]
    seqlens_k = [0, 20]
    nheads = 4
    nheads_kv = 2
    d = 64
    dtype = torch.float16

    cu_seqlens_q = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor([0, 0, 20], dtype=torch.int32, device="cuda")
    q = torch.randn(sum(seqlens_q), nheads, d, device="cuda", dtype=dtype)
    k = torch.randn(sum(seqlens_k), nheads_kv, d, device="cuda", dtype=dtype)
    v = torch.randn(sum(seqlens_k), nheads_kv, d, device="cuda", dtype=dtype)
    cross_kv_boundary = torch.tensor(
        [0, 0, 0, 0, 5, 20, 0], dtype=torch.int32, device="cuda"
    )

    out_fa3 = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max(seqlens_q),
        max(seqlens_k),
        causal=False,
        cross_kv_boundary=cross_kv_boundary,
    )

    out_ref_text = torch.zeros_like(out_fa3[: seqlens_q[0]])
    out_ref_vision = staircase_attention_ref(
        q[seqlens_q[0] :].unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        cross_kv_boundary[seqlens_q[0] :].unsqueeze(0),
    ).squeeze(0)
    out_ref = torch.cat([out_ref_text, out_ref_vision], dim=0)

    assert torch.isfinite(out_fa3).all()
    torch.testing.assert_close(out_fa3[: seqlens_q[0]], out_ref_text, atol=0, rtol=0)
    torch.testing.assert_close(out_fa3, out_ref, atol=2e-2, rtol=2e-2)


def test_staircase_mask_deterministic_bwd():
    """Deterministic backward should preserve the staircase mask."""
    batch_size = 2
    seqlen_q = 128
    seqlen_k = 256
    nheads = 4
    d = 64
    dtype = torch.float16

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen_k, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    cross_kv_boundary = make_staircase_boundary(
        batch_size, seqlen_q, seqlen_k, "linear"
    )

    out_fa3 = flash_attn_func(
        q, k, v, causal=False, deterministic=True, cross_kv_boundary=cross_kv_boundary
    )
    out_ref = staircase_attention_ref(q_ref, k_ref, v_ref, cross_kv_boundary)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)


def test_staircase_mask_splitkv_bwd():
    """Split-KV forward plus backward should keep using cross_kv_boundary."""
    batch_size = 2
    seqlen_q = 64
    seqlen_k = 512
    nheads = 4
    d = 64
    dtype = torch.float16

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size, seqlen_k, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    cross_kv_boundary = make_staircase_boundary(
        batch_size, seqlen_q, seqlen_k, "random"
    )

    out_fa3 = flash_attn_func(
        q, k, v, causal=False, num_splits=2, cross_kv_boundary=cross_kv_boundary
    )
    out_ref = staircase_attention_ref(q_ref, k_ref, v_ref, cross_kv_boundary)

    g = torch.randn_like(out_fa3)
    out_fa3.backward(g)
    out_ref.backward(g)

    torch.testing.assert_close(out_fa3, out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(q.grad, q_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v.grad, v_ref.grad.to(dtype), atol=2e-2, rtol=2e-2)
