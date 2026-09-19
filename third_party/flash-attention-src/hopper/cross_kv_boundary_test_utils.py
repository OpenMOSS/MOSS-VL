"""Shared reference helpers for ``cross_kv_boundary`` tests."""

import math

import torch
from einops import repeat


def staircase_attention_ref(
    q, k, v, cross_kv_boundary, softmax_scale=None, upcast=True
):
    """Compute dense reference attention with per-query KV prefix lengths."""
    dtype = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    batch_size, seqlen_q, nheads, head_dim = q.shape
    seqlen_k = k.shape[1]
    nheads_k = k.shape[2]

    k = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    cols = torch.arange(seqlen_k, device=q.device)
    boundary = cross_kv_boundary.clamp(0, seqlen_k)
    masked = cols.view(1, 1, 1, seqlen_k) >= boundary.view(batch_size, 1, seqlen_q, 1)
    valid_rows = (boundary > 0).view(batch_size, 1, seqlen_q, 1)
    scores.masked_fill_(masked, float("-inf"))
    scores = torch.where(valid_rows, scores, torch.zeros_like(scores))
    attention = torch.softmax(scores, dim=-1)
    attention = torch.where(valid_rows, attention, torch.zeros_like(attention))
    return torch.einsum("bhts,bshd->bthd", attention, v).to(dtype=dtype)


def reference_lse(q, k, cross_kv_boundary, softmax_scale=None):
    """Compute the dense per-row log-sum-exp for a prefix mask."""
    batch_size, seqlen_q, nheads, head_dim = q.shape
    seqlen_k = k.shape[1]
    nheads_k = k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    q = q.float() * softmax_scale
    k = repeat(k.float(), "b s h d -> b s (h g) d", g=nheads // nheads_k)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k)
    cols = torch.arange(seqlen_k, device=q.device)
    boundary = cross_kv_boundary.clamp(0, seqlen_k)
    masked = cols.view(1, 1, 1, seqlen_k) >= boundary.view(batch_size, 1, seqlen_q, 1)
    lse = torch.logsumexp(scores.masked_fill(masked, float("-inf")), dim=-1)
    return torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)


def recover_attention_probs(flash_attn_func, q, k, cross_kv_boundary, *, dtype=None):
    """Recover attention probabilities by evaluating one-hot value chunks."""
    batch_size, seqlen_q, nheads, head_dim = q.shape
    seqlen_k = k.shape[1]
    nheads_k = k.shape[2]
    dtype = q.dtype if dtype is None else dtype
    probs = torch.zeros(
        (batch_size, nheads, seqlen_q, seqlen_k),
        device=q.device,
        dtype=torch.float32,
    )

    for chunk_start in range(0, seqlen_k, head_dim):
        chunk_end = min(chunk_start + head_dim, seqlen_k)
        v = torch.zeros(
            (batch_size, seqlen_k, nheads_k, head_dim),
            device=q.device,
            dtype=dtype,
        )
        for local_idx, col in enumerate(range(chunk_start, chunk_end)):
            v[:, col, :, local_idx] = 1.0
        out = flash_attn_func(
            q, k, v, causal=False, cross_kv_boundary=cross_kv_boundary
        )
        if isinstance(out, tuple):
            out = out[0]
        for local_idx, col in enumerate(range(chunk_start, chunk_end)):
            probs[..., col] = out[..., local_idx].float().permute(0, 2, 1)
    return probs


def make_staircase_boundary(
    batch_size, seqlen_q, seqlen_k, pattern="linear", device="cuda"
):
    """Generate common boundary patterns used by kernel tests."""
    if pattern == "linear":
        boundary = torch.linspace(1, seqlen_k, seqlen_q, device=device).int()
        boundary = boundary.clamp(min=1, max=seqlen_k)
        return boundary.unsqueeze(0).expand(batch_size, -1).contiguous()
    if pattern == "step":
        boundary = torch.zeros(seqlen_q, dtype=torch.int32, device=device)
        mid = seqlen_q // 2
        boundary[:mid] = seqlen_k // 2
        boundary[mid:] = seqlen_k
        return boundary.unsqueeze(0).expand(batch_size, -1).contiguous()
    if pattern == "constant":
        return torch.full(
            (batch_size, seqlen_q), seqlen_k, dtype=torch.int32, device=device
        )
    if pattern == "random":
        boundary = torch.randint(
            1,
            seqlen_k + 1,
            (batch_size, seqlen_q),
            dtype=torch.int32,
            device=device,
        )
        return boundary.sort(dim=1).values
    raise ValueError(f"Unknown pattern: {pattern}")
