"""Mask-only validation for FA3 ``cross_kv_boundary`` attention."""

import pytest
import torch

from cross_kv_boundary_test_utils import recover_attention_probs, reference_lse
from flash_attn_interface import flash_attn_func


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def make_boundary(case, batch_size, seqlen_q, seqlen_k, device):
    """Build a boundary pattern for mask reconstruction tests."""
    if case == "all_zero":
        return torch.zeros((batch_size, seqlen_q), dtype=torch.int32, device=device)
    if case == "all_visible":
        return torch.full(
            (batch_size, seqlen_q), seqlen_k, dtype=torch.int32, device=device
        )
    if case == "linear":
        boundary = torch.linspace(1, seqlen_k, seqlen_q, device=device).int()
        return boundary.unsqueeze(0).expand(batch_size, -1).contiguous()
    if case == "step_half":
        boundary = torch.zeros(seqlen_q, dtype=torch.int32, device=device)
        boundary[: seqlen_q // 2] = seqlen_k // 2
        boundary[seqlen_q // 2 :] = seqlen_k
        return boundary.unsqueeze(0).expand(batch_size, -1).contiguous()
    if case in {"block_edges", "off_block_edges"}:
        offset = 0 if case == "block_edges" else -17
        row = torch.arange(seqlen_q, device=device)
        boundary = ((row // 64) + 1) * 64 + offset
        boundary = boundary.clamp(min=0, max=seqlen_k).to(torch.int32)
        return boundary.unsqueeze(0).expand(batch_size, -1).contiguous()
    if case == "delayed_step":
        boundary = torch.full(
            (batch_size, seqlen_q), 96, dtype=torch.int32, device=device
        )
        boundary[:, max(seqlen_q - 128, 0) :] = seqlen_k
        return boundary
    if case == "zero_prefix":
        boundary = torch.full(
            (batch_size, seqlen_q),
            seqlen_k // 4,
            dtype=torch.int32,
            device=device,
        )
        boundary[:, : seqlen_q // 8] = 0
        boundary[:, seqlen_q // 2 :] = seqlen_k
        return boundary
    if case == "random_sorted":
        boundary = torch.randint(
            0,
            seqlen_k + 1,
            (batch_size, seqlen_q),
            dtype=torch.int32,
            device=device,
        )
        return boundary.sort(dim=1).values
    raise ValueError(f"Unknown boundary case: {case}")


CASES = [
    (1, 128, 128, 2, 1, 64, torch.float16, "all_visible"),
    (1, 128, 128, 2, 1, 64, torch.float16, "all_zero"),
    (1, 128, 128, 2, 1, 64, torch.float16, "linear"),
    (1, 128, 128, 2, 1, 64, torch.float16, "step_half"),
    (1, 128, 128, 2, 1, 64, torch.float16, "block_edges"),
    (1, 128, 128, 2, 1, 64, torch.float16, "off_block_edges"),
    (1, 256, 256, 2, 1, 64, torch.float16, "delayed_step"),
    (1, 256, 256, 2, 1, 64, torch.float16, "zero_prefix"),
    (2, 256, 256, 4, 2, 64, torch.float16, "random_sorted"),
    (1, 256, 256, 8, 2, 128, torch.float16, "linear"),
    (1, 256, 256, 8, 2, 128, torch.float16, "delayed_step"),
    (1, 256, 256, 8, 2, 128, torch.float16, "off_block_edges"),
    (1, 256, 256, 8, 2, 128, torch.bfloat16, "delayed_step"),
    (2, 384, 512, 8, 2, 128, torch.bfloat16, "random_sorted"),
]


@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_k,nheads,nheads_k,head_dim,dtype,case",
    CASES,
    ids=[case[-1] + f"-sq{case[1]}-sk{case[2]}-{case[6]}" for case in CASES],
)
def test_mask_reconstruction(
    batch_size,
    seqlen_q,
    seqlen_k,
    nheads,
    nheads_k,
    head_dim,
    dtype,
    case,
):
    torch.manual_seed(0)
    boundary = make_boundary(case, batch_size, seqlen_q, seqlen_k, "cuda")
    q = torch.randn(
        batch_size,
        seqlen_q,
        nheads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn_like(k)

    _, lse = flash_attn_func(
        q,
        k,
        v,
        causal=False,
        cross_kv_boundary=boundary,
        return_attn_probs=True,
    )
    lse_ref = reference_lse(q, k, boundary)
    valid_rows = (boundary > 0).unsqueeze(1).expand_as(lse_ref)
    if valid_rows.any():
        torch.testing.assert_close(
            lse.float()[valid_rows], lse_ref[valid_rows], atol=1e-2, rtol=1e-2
        )

    probs = recover_attention_probs(flash_attn_func, q, k, boundary, dtype=dtype)
    cols = torch.arange(seqlen_k, device="cuda")
    masked = cols.view(1, 1, 1, seqlen_k) >= boundary.view(batch_size, 1, seqlen_q, 1)
    if masked.any():
        assert probs.masked_select(masked).abs().max().item() <= 1e-3

    target_sum = (
        (boundary > 0).float().unsqueeze(1).expand(batch_size, nheads, seqlen_q)
    )
    torch.testing.assert_close(probs.sum(dim=-1), target_sum, atol=1e-2, rtol=0)
