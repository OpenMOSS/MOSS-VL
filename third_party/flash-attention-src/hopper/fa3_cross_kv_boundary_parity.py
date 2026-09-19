#!/usr/bin/env python3
"""MOSS-VL FA3 cross_kv_boundary parity checks.

This is a lightweight, self-contained check for the exact path we care about:

1. Expand a frame-level cross_attention_mask to a dense token mask.
2. Rebuild cross_kv_boundary from the same frame-level mask.
3. Check that cross_kv_boundary implies exactly the same token mask.
4. Compare FA3(cross_kv_boundary) against a dense PyTorch reference for output,
   attention probabilities, and backward gradients.

Run from the same conda env used by training:

    cd hopper
    python fa3_cross_kv_boundary_parity.py
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch

from cross_kv_boundary_test_utils import recover_attention_probs


@dataclass
class SampleVisionInfo:
    repeats: List[int]
    pad_end: int


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_vision_infos(
    batch: int, frames: int, tokens_per_frame: int, pad_to: int
) -> List[SampleVisionInfo]:
    infos: List[SampleVisionInfo] = []
    for b in range(batch):
        # Use two media chunks per sample when possible. This mirrors the model
        # code path where repeats are derived media-by-media, while keeping the
        # final frame sequence regular enough to reason about.
        split = frames // 2 if frames >= 2 else frames
        repeats = [tokens_per_frame] * split + [tokens_per_frame] * (frames - split)
        total = sum(repeats)
        pad_end = int(math.ceil(total / pad_to) * pad_to) if pad_to > 1 else total
        infos.append(SampleVisionInfo(repeats=repeats, pad_end=pad_end))
    return infos


def make_frame_mask(
    batch: int, text_len: int, frames: int, pattern: str, device: torch.device
) -> torch.Tensor:
    """Return bool mask shaped (B, 1, T, F), True means masked."""
    cols = torch.arange(frames, device=device).view(1, 1, frames)
    rows = torch.arange(text_len, device=device)
    visible_counts = torch.zeros((batch, text_len), dtype=torch.int64, device=device)

    if pattern == "staircase":
        base = torch.div((rows + 1) * frames, text_len, rounding_mode="floor").clamp(
            0, frames
        )
        visible_counts[:] = base
    elif pattern == "delayed":
        visible_counts[:] = max(frames // 4, 1)
        visible_counts[:, : text_len // 8] = 0
        visible_counts[:, text_len // 2 :] = max(frames // 2, 1)
        visible_counts[:, (text_len * 3) // 4 :] = frames
    elif pattern == "random_prefix":
        g = torch.Generator(device=device)
        g.manual_seed(123)
        counts = torch.randint(
            0, frames + 1, (batch, text_len), generator=g, device=device
        )
        counts, _ = counts.sort(dim=1)
        visible_counts = counts
    elif pattern == "all_visible":
        visible_counts[:] = frames
    elif pattern == "zero_prefix":
        visible_counts[:] = frames
        visible_counts[:, : text_len // 4] = 0
    elif pattern == "bad_hole":
        # Intentionally not representable by cross_kv_boundary. Useful to prove the
        # exact assertion catches non-prefix masks.
        mask = torch.ones((batch, 1, text_len, frames), dtype=torch.bool, device=device)
        mask[..., ::2] = False
        if frames > 3:
            mask[:, :, text_len // 3 :, 1::3] = False
        return mask
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Prefix-visible frame mask: visible iff frame_idx < visible_counts[b, t].
    visible = cols < visible_counts.view(batch, text_len, 1)
    return (~visible).unsqueeze(1).contiguous()


def expand_dense_visible(
    frame_mask: torch.Tensor, infos: List[SampleVisionInfo]
) -> torch.Tensor:
    """Dense token visibility, shape (B, T, Vmax), bool True means visible."""
    batch, _, text_len, _ = frame_mask.shape
    device = frame_mask.device
    max_v = max(info.pad_end for info in infos) if infos else 0
    dense_visible = torch.zeros(
        (batch, text_len, max_v), dtype=torch.bool, device=device
    )
    frame_visible = (~frame_mask).squeeze(1)

    for b, info in enumerate(infos):
        repeats = info.repeats
        valid_frames = min(len(repeats), frame_visible.shape[-1])
        if valid_frames == 0:
            continue
        repeats_t = torch.tensor(
            repeats[:valid_frames], dtype=torch.int64, device=device
        )
        expanded = frame_visible[b, :, :valid_frames].repeat_interleave(
            repeats_t, dim=-1
        )
        dense_visible[b, :, : expanded.shape[-1]] = expanded[:, : info.pad_end]
    return dense_visible


def compute_cross_kv_boundary(
    frame_mask: torch.Tensor, infos: List[SampleVisionInfo]
) -> torch.Tensor:
    """Same semantics as MossVLModel.expand_cross_kv_boundary."""
    batch, _, text_len, _ = frame_mask.shape
    device = frame_mask.device
    frame_visible = (~frame_mask).squeeze(1)
    cross_kv_boundary = torch.zeros((batch, text_len), dtype=torch.int32, device=device)

    for b, info in enumerate(infos):
        repeats = info.repeats
        valid_frames = min(len(repeats), frame_visible.shape[-1])
        if valid_frames == 0:
            continue
        cumsum = torch.zeros(valid_frames + 1, dtype=torch.int32, device=device)
        cumsum[1:] = torch.tensor(
            repeats[:valid_frames], dtype=torch.int32, device=device
        ).cumsum(0)
        arange = torch.arange(1, valid_frames + 1, dtype=torch.int32, device=device)
        last_visible_plus_1 = (
            frame_visible[b, :, :valid_frames].to(torch.int32) * arange
        ).amax(dim=-1)
        cross_kv_boundary[b] = cumsum[last_visible_plus_1.long()].clamp(
            max=info.pad_end
        )
    return cross_kv_boundary.contiguous()


def boundary_visible(cross_kv_boundary: torch.Tensor, max_v: int) -> torch.Tensor:
    cols = torch.arange(max_v, dtype=torch.int32, device=cross_kv_boundary.device)
    return cols.view(1, 1, max_v) < cross_kv_boundary.view(
        cross_kv_boundary.shape[0], cross_kv_boundary.shape[1], 1
    )


def dense_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dense_visible: torch.Tensor,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h = q.shape[2]
    hk = k.shape[2]
    repeat = h // hk
    qf = q.float()
    kf = k.float().repeat_interleave(repeat, dim=2)
    vf = v.float().repeat_interleave(repeat, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", qf * softmax_scale, kf)
    scores = scores.masked_fill(~dense_visible.unsqueeze(1), float("-inf"))
    probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, vf)
    return out, probs


def check_nonpacked(args) -> None:
    from flash_attn_3 import flash_attn_func
    import flash_attn_3
    import flash_attn_3._C as C

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)

    infos = build_vision_infos(
        args.batch, args.frames, args.tokens_per_frame, args.pad_to
    )
    frame_mask = make_frame_mask(
        args.batch, args.text_len, args.frames, args.pattern, device
    )
    dense_visible = expand_dense_visible(frame_mask, infos)
    cross_kv_boundary = compute_cross_kv_boundary(frame_mask, infos)
    kv_visible = boundary_visible(cross_kv_boundary, dense_visible.shape[-1])

    mask_equal = torch.equal(dense_visible, kv_visible)
    mismatch = int((dense_visible ^ kv_visible).sum().item())

    print("=== import ===")
    print("flash_attn_3:", flash_attn_3.__file__)
    print("flash_attn_3._C:", C.__file__)
    print("=== mask reconstruction ===")
    print("pattern:", args.pattern)
    print("dense_visible:", tuple(dense_visible.shape))
    print(
        "cross_kv_boundary:",
        tuple(cross_kv_boundary.shape),
        "minmax:",
        (int(cross_kv_boundary.min()), int(cross_kv_boundary.max())),
    )
    print("mask_equal:", mask_equal, "mismatch:", mismatch)
    if not mask_equal:
        idx = (dense_visible ^ kv_visible).nonzero()[0].tolist()
        b, t, col = idx
        raise AssertionError(
            f"cross_kv_boundary cannot reconstruct dense mask exactly: first mismatch "
            f"(b,t,k)=({b},{t},{col}) dense={bool(dense_visible[b, t, col])} "
            f"kv={bool(kv_visible[b, t, col])} boundary={int(cross_kv_boundary[b, t])}"
        )

    b = args.batch
    t = args.text_len
    sk = dense_visible.shape[-1]
    h = args.heads
    hk = args.kv_heads
    d = args.headdim
    if h % hk != 0:
        raise ValueError("--heads must be divisible by --kv-heads")

    q = torch.randn((b, t, h, d), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((b, sk, hk, d), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((b, sk, hk, d), dtype=dtype, device=device, requires_grad=True)
    qr = q.detach().clone().float().requires_grad_(True)
    kr = k.detach().clone().float().requires_grad_(True)
    vr = v.detach().clone().float().requires_grad_(True)

    scale = args.softmax_scale if args.softmax_scale is not None else d**-0.5
    out_fa3 = flash_attn_func(
        q, k, v, softmax_scale=scale, causal=False, cross_kv_boundary=cross_kv_boundary
    )
    if isinstance(out_fa3, tuple):
        out_fa3 = out_fa3[0]
    out_ref, probs_ref = dense_attention_ref(qr, kr, vr, dense_visible, scale)

    grad = torch.randn_like(out_fa3)
    out_fa3.backward(grad)
    out_ref.backward(grad.float())

    print("=== forward/backward dense reference ===")
    out_abs = (out_fa3.float() - out_ref).detach().abs()
    dq_abs = (q.grad.float() - qr.grad.float()).detach().abs()
    dk_abs = (k.grad.float() - kr.grad.float()).detach().abs()
    dv_abs = (v.grad.float() - vr.grad.float()).detach().abs()
    print("out_max_abs:", float(out_abs.max()))
    print("out_mean_abs:", float(out_abs.mean()))
    print("dq_max_abs:", float(dq_abs.max()))
    print("dk_max_abs:", float(dk_abs.max()))
    print("dv_max_abs:", float(dv_abs.max()))

    if not args.skip_prob_recovery:
        probs_fa3 = recover_attention_probs(
            flash_attn_func,
            q.detach(),
            k.detach(),
            cross_kv_boundary,
            dtype=dtype,
        )
        masked = ~dense_visible.unsqueeze(1)
        row_has_visible = dense_visible.any(dim=-1).unsqueeze(1)
        prob_diff = (probs_fa3 - probs_ref).abs()
        print("=== recovered attention probabilities ===")
        print(
            "masked_max_leak:",
            float(probs_fa3.masked_select(masked).abs().max()) if masked.any() else 0.0,
        )
        print(
            "visible_prob_max_abs:",
            float(prob_diff.masked_select(dense_visible.unsqueeze(1)).max())
            if dense_visible.any()
            else 0.0,
        )
        print(
            "visible_prob_mean_abs:",
            float(prob_diff.masked_select(dense_visible.unsqueeze(1)).mean())
            if dense_visible.any()
            else 0.0,
        )
        print(
            "row_sum_max_abs:",
            float((probs_fa3.sum(dim=-1) - row_has_visible.float()).abs().max()),
        )

    max_out = float(out_abs.max())
    max_dq = float(dq_abs.max())
    max_dk = float(dk_abs.max())
    max_dv = float(dv_abs.max())
    max_all = max(max_out, max_dq, max_dk, max_dv)
    if max_all > args.max_abs_tol:
        raise AssertionError(
            f"FA3 vs dense max_abs {max_all:.6f} > tolerance {args.max_abs_tol}"
        )
    print("PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--text-len", type=int, default=576)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--tokens-per-frame", type=int, default=8)
    parser.add_argument("--pad-to", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--pattern",
        choices=[
            "staircase",
            "delayed",
            "random_prefix",
            "all_visible",
            "zero_prefix",
            "bad_hole",
        ],
        default="delayed",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--softmax-scale", type=float, default=None)
    parser.add_argument("--max-abs-tol", type=float, default=6e-2)
    parser.add_argument("--skip-prob-recovery", action="store_true")
    args = parser.parse_args()

    print("cwd:", os.getcwd())
    print(
        "torch:",
        torch.__version__,
        "cuda:",
        torch.version.cuda,
        "device:",
        torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    )
    check_nonpacked(args)


if __name__ == "__main__":
    main()
