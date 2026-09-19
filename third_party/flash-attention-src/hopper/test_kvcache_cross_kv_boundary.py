"""KV-cache coverage for cross_kv_boundary staircase masks."""

import math

import pytest
import torch

import flash_attn_interface


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _repeat_kv_heads(x, nheads_q):
    nheads_k = x.shape[2]
    return x.repeat_interleave(nheads_q // nheads_k, dim=2)


def _boundary_attention_ref(q, k, v, cross_kv_boundary):
    dtype = q.dtype
    q = q.float()
    k = _repeat_kv_heads(k, q.shape[2]).float()
    v = _repeat_kv_heads(v, q.shape[2]).float()
    boundary = cross_kv_boundary.clamp(min=0, max=k.shape[1])
    scores = torch.einsum("bqhd,bkhd->bhqk", q / math.sqrt(q.shape[-1]), k)
    cols = torch.arange(k.shape[1], device=q.device, dtype=torch.int32)
    scores = scores.masked_fill(cols.view(1, 1, 1, -1) >= boundary.view(boundary.shape[0], 1, boundary.shape[1], 1), float("-inf"))
    valid_rows = boundary > 0
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(~valid_rows.view(boundary.shape[0], 1, boundary.shape[1], 1), 0.0)
    return torch.einsum("bhqk,bkhd->bqhd", attn, v).to(dtype)


def _logical_cache(k_cache, v_cache, cache_seqlens, cache_leftpad=None, page_table=None):
    batch = cache_seqlens.shape[0]
    leftpad = torch.zeros_like(cache_seqlens) if cache_leftpad is None else cache_leftpad
    logical_lens = (cache_seqlens - leftpad).tolist()
    max_len = max(logical_lens)
    k_out = torch.zeros((batch, max_len, k_cache.shape[-2], k_cache.shape[-1]), device=k_cache.device, dtype=k_cache.dtype)
    v_out = torch.zeros((batch, max_len, v_cache.shape[-2], v_cache.shape[-1]), device=v_cache.device, dtype=v_cache.dtype)
    if page_table is None:
        for b in range(batch):
            start = int(leftpad[b])
            end = int(cache_seqlens[b])
            k_out[b, : end - start] = k_cache[b, start:end]
            v_out[b, : end - start] = v_cache[b, start:end]
    else:
        page_size = k_cache.shape[1]
        for b in range(batch):
            for pos in range(int(cache_seqlens[b])):
                page = int(page_table[b, pos // page_size])
                offset = pos % page_size
                k_out[b, pos] = k_cache[page, offset]
                v_out[b, pos] = v_cache[page, offset]
    return k_out, v_out


def _append_logical_cache(k_cache, v_cache, k_new, v_new, cache_seqlens, cache_leftpad):
    k_old, v_old = _logical_cache(k_cache, v_cache, cache_seqlens, cache_leftpad)
    return torch.cat([k_old, k_new], dim=1), torch.cat([v_old, v_new], dim=1)


def _metadata_for(q, k_cache, cache_seqlens, cross_kv_boundary, *, cu_seqlens_q=None, cache_leftpad=None,
                  page_table=None, max_seqlen_q=None, max_seqlen_k_new=0, num_splits=0, pack_gqa=None):
    batch = cache_seqlens.shape[0]
    max_q = q.shape[1] if cu_seqlens_q is None else max_seqlen_q
    max_k = k_cache.shape[1] if page_table is None else page_table.shape[1] * k_cache.shape[1]
    return flash_attn_interface.get_scheduler_metadata(
        batch,
        max_q,
        max_k,
        q.shape[-2],
        k_cache.shape[-2],
        q.shape[-1],
        cache_seqlens,
        q.dtype,
        cu_seqlens_q=cu_seqlens_q,
        cache_leftpad=cache_leftpad,
        page_size=None if page_table is None else k_cache.shape[1],
        max_seqlen_k_new=max_seqlen_k_new,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        cross_kv_boundary=cross_kv_boundary,
    )


def _assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


def _kvcache_out(*args, **kwargs):
    result = flash_attn_interface.flash_attn_with_kvcache(*args, **kwargs)
    return result[0] if isinstance(result, tuple) else result


@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize(
    "dtype, headdim, num_splits",
    [
        pytest.param(torch.bfloat16, 64, 2, id="bf16-hdim64-split2"),
        pytest.param(torch.bfloat16, 128, 0, id="bf16-hdim128-auto"),
        pytest.param(torch.float16, 128, 1, id="fp16-hdim128-single"),
    ],
)
def test_kvcache_cross_kv_boundary_no_append_metadata_packgqa(
    dtype, headdim, num_splits, pack_gqa
):
    torch.manual_seed(0)
    batch, seqlen_q, seqlen_k, nheads_k, gqa = 2, 5, 16, 2, 2
    nheads_q = nheads_k * gqa
    q = torch.randn(batch, seqlen_q, nheads_q, headdim, device="cuda", dtype=dtype)
    k_cache = torch.randn(batch, seqlen_k, nheads_k, headdim, device="cuda", dtype=dtype)
    v_cache = torch.randn(batch, seqlen_k, nheads_k, headdim, device="cuda", dtype=dtype)
    cache_seqlens = torch.tensor([11, 8], device="cuda", dtype=torch.int32)
    cross_kv_boundary = torch.tensor([[0, 3, 7, 11, 5], [2, 6, 0, 8, 4]], device="cuda", dtype=torch.int32)

    k_ref, v_ref = _logical_cache(k_cache, v_cache, cache_seqlens)
    out_ref = _boundary_attention_ref(q, k_ref, v_ref, cross_kv_boundary)
    metadata = _metadata_for(q, k_cache, cache_seqlens, cross_kv_boundary, num_splits=num_splits, pack_gqa=pack_gqa)

    for scheduler_metadata in [None, metadata]:
        out = _kvcache_out(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            cross_kv_boundary=cross_kv_boundary,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            return_softmax_lse=True,
        )
        _assert_close(out, out_ref)


@pytest.mark.parametrize("pack_gqa", [False, True])
def test_kvcache_cross_kv_boundary_append_updates_cache_when_new_kv_masked(pack_gqa):
    torch.manual_seed(1)
    batch, seqlen_q, cache_capacity, nheads_k, gqa, headdim = 1, 4, 160, 2, 2, 64
    nheads_q = nheads_k * gqa
    q = torch.randn(batch, seqlen_q, nheads_q, headdim, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(batch, cache_capacity, nheads_k, headdim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(batch, cache_capacity, nheads_k, headdim, device="cuda", dtype=torch.bfloat16)
    k_saved, v_saved = k_cache.clone(), v_cache.clone()
    k_new = torch.randn(batch, 4, nheads_k, headdim, device="cuda", dtype=torch.bfloat16)
    v_new = torch.randn(batch, 4, nheads_k, headdim, device="cuda", dtype=torch.bfloat16)
    cache_leftpad = torch.tensor([3], device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor([133], device="cuda", dtype=torch.int32)
    cross_kv_boundary = torch.tensor([[0, 32, 120, 64]], device="cuda", dtype=torch.int32)

    k_ref, v_ref = _append_logical_cache(k_saved, v_saved, k_new, v_new, cache_seqlens, cache_leftpad)
    out_ref = _boundary_attention_ref(q, k_ref, v_ref, cross_kv_boundary)
    metadata = _metadata_for(
        q,
        k_cache,
        cache_seqlens,
        cross_kv_boundary,
        cache_leftpad=cache_leftpad,
        max_seqlen_k_new=k_new.shape[1],
        num_splits=2,
        pack_gqa=pack_gqa,
    )

    for scheduler_metadata in [None, metadata]:
        k_cache.copy_(k_saved)
        v_cache.copy_(v_saved)
        out = _kvcache_out(
            q,
            k_cache,
            v_cache,
            k_new,
            v_new,
            cache_seqlens=cache_seqlens,
            cache_leftpad=cache_leftpad,
            cross_kv_boundary=cross_kv_boundary,
            scheduler_metadata=scheduler_metadata,
            num_splits=2,
            pack_gqa=pack_gqa,
            return_softmax_lse=True,
        )
        _assert_close(out, out_ref)
        start = int(cache_seqlens[0])
        torch.testing.assert_close(k_cache[0, start : start + k_new.shape[1]], k_new[0])
        torch.testing.assert_close(v_cache[0, start : start + v_new.shape[1]], v_new[0])


def test_kvcache_cross_kv_boundary_varlen_q_metadata():
    torch.manual_seed(2)
    q_lens = [3, 1, 4]
    batch, max_q, seqlen_k, nheads_k, gqa, headdim = len(q_lens), max(q_lens), 12, 2, 2, 64
    nheads_q = nheads_k * gqa
    q_padded = torch.randn(batch, max_q, nheads_q, headdim, device="cuda", dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, 3, 4, 8], device="cuda", dtype=torch.int32)
    q = torch.cat([q_padded[b, : q_lens[b]] for b in range(batch)], dim=0).contiguous()
    k_cache = torch.randn(batch, seqlen_k, nheads_k, headdim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(batch, seqlen_k, nheads_k, headdim, device="cuda", dtype=torch.bfloat16)
    cache_seqlens = torch.tensor([9, 7, 10], device="cuda", dtype=torch.int32)
    boundary_pieces = [
        torch.tensor([0, 5, 9], device="cuda", dtype=torch.int32),
        torch.tensor([7], device="cuda", dtype=torch.int32),
        torch.tensor([3, 10, 0, 6], device="cuda", dtype=torch.int32),
    ]
    cross_kv_boundary = torch.cat(boundary_pieces, dim=0).contiguous()

    k_ref, v_ref = _logical_cache(k_cache, v_cache, cache_seqlens)
    out_ref = torch.cat([
        _boundary_attention_ref(q_padded[b : b + 1, : q_lens[b]], k_ref[b : b + 1], v_ref[b : b + 1], boundary_pieces[b].view(1, -1))
        for b in range(batch)
    ], dim=1).squeeze(0).contiguous()
    metadata = _metadata_for(
        q,
        k_cache,
        cache_seqlens,
        cross_kv_boundary,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_q,
        num_splits=2,
    )

    for scheduler_metadata in [None, metadata]:
        out = _kvcache_out(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_q,
            cross_kv_boundary=cross_kv_boundary,
            scheduler_metadata=scheduler_metadata,
            num_splits=2,
            return_softmax_lse=True,
        )
        _assert_close(out, out_ref)


@pytest.mark.parametrize("pack_gqa", [False, True])
def test_kvcache_cross_kv_boundary_paged_kv_metadata(pack_gqa):
    torch.manual_seed(3)
    batch, seqlen_q, page_size, max_pages, nheads_k, gqa, headdim = 2, 4, 8, 3, 2, 2, 64
    nheads_q = nheads_k * gqa
    num_pages = batch * max_pages
    q = torch.randn(batch, seqlen_q, nheads_q, headdim, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(num_pages, page_size, nheads_k, headdim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(num_pages, page_size, nheads_k, headdim, device="cuda", dtype=torch.bfloat16)
    page_table = torch.tensor([[0, 2, 4], [1, 3, 5]], device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor([17, 10], device="cuda", dtype=torch.int32)
    cross_kv_boundary = torch.tensor([[1, 8, 17, 0], [2, 10, 4, 7]], device="cuda", dtype=torch.int32)

    k_ref, v_ref = _logical_cache(k_cache, v_cache, cache_seqlens, page_table=page_table)
    out_ref = _boundary_attention_ref(q, k_ref, v_ref, cross_kv_boundary)
    metadata = _metadata_for(
        q, k_cache, cache_seqlens, cross_kv_boundary,
        page_table=page_table,
        num_splits=2,
        pack_gqa=pack_gqa,
    )

    for scheduler_metadata in [None, metadata]:
        out = _kvcache_out(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            cross_kv_boundary=cross_kv_boundary,
            scheduler_metadata=scheduler_metadata,
            num_splits=2,
            pack_gqa=pack_gqa,
            return_softmax_lse=True,
        )
        _assert_close(out, out_ref)
