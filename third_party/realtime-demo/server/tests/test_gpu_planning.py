"""Unit tests: topology parsing, attn-impl selection, placement, KV budget.

Run:  <repo>/.venv/bin/python -m server.tests.test_gpu_planning
"""
from __future__ import annotations

import dataclasses
import sys
from unittest.mock import patch

from server.config import Settings
from server.gpu.kv_budget import KvBudget, compute_kv_budget, kv_bytes_per_token
from server.gpu.placement import offline_gpu_count, plan_placement as _plan_placement, format_plan
from server.gpu.topology import GpuInfo, parse_smi_csv, select_attn_impl as _select_attn_impl


def plan_placement(topology, settings):
    # These fixtures describe CUDA hardware, independently of the test host.
    with patch("server.gpu.placement._device_str", new=lambda index: f"cuda:{index}"), \
         patch("server.gpu.topology.is_npu", return_value=False):
        return _plan_placement(topology, settings)


def select_attn_impl(*args):
    with patch("server.gpu.topology.is_npu", return_value=False):
        return _select_attn_impl(*args)

MIB = 2**20

# canned nvidia-smi output for the three deployment classes
SMI_H200_8X = "\n".join(
    f"{i}, GPU-{i:08x}, NVIDIA H200, 143771, 4, 9.0" for i in range(8))
SMI_RTX6000_2X = "\n".join(
    f"{i}, GPU-{i:08x}, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 12, 12.0"
    for i in range(2))
SMI_4090_1X = "0, GPU-deadbeef, NVIDIA GeForce RTX 4090, 24564, 350, 8.9"
SMI_MALFORMED = "garbage line\n0, GPU-x, NVIDIA H200, notanumber, 4, 9.0\n" + SMI_4090_1X


def make_settings(**overrides) -> Settings:
    # pin box-dependent dynamic defaults (tts/offline provider probe the venvs
    # on disk) so placement expectations are deterministic on any machine
    overrides.setdefault("tts_provider", "moss_tts_nano")
    overrides.setdefault("offline_provider", "none")
    return dataclasses.replace(Settings(), **overrides)


def test_parse() -> None:
    h200 = parse_smi_csv(SMI_H200_8X)
    assert len(h200) == 8
    assert h200[0].name == "NVIDIA H200" and h200[0].compute_cap == (9, 0)
    assert h200[7].mem_free_mib == 143771 - 4

    bw = parse_smi_csv(SMI_RTX6000_2X)
    assert len(bw) == 2 and bw[1].compute_cap == (12, 0)
    # name contains no comma-splitting hazards; 6 fields exactly
    assert bw[0].name.startswith("NVIDIA RTX PRO 6000")

    # malformed rows are skipped, valid rows survive
    mixed = parse_smi_csv(SMI_MALFORMED)
    assert len(mixed) == 1 and mixed[0].name.endswith("4090")
    assert parse_smi_csv("") == []
    print("topology parse: OK")


def test_attn_select() -> None:
    assert select_attn_impl((8, 9), "auto") == "flash_attention_2"   # 4090
    assert select_attn_impl((9, 0), "auto") == "flash_attention_2"   # H200
    assert select_attn_impl((12, 0), "auto") == "sdpa"               # Blackwell
    assert select_attn_impl((12, 1), "auto") == "sdpa"
    assert select_attn_impl((9, 0), "sdpa") == "sdpa"                # explicit wins
    assert select_attn_impl((12, 0), "flash_attention_2") == "flash_attention_2"
    assert select_attn_impl((0, 0), "") == "flash_attention_2"       # unknown cc, auto
    print("attn select: OK")


def test_placement_h200() -> None:
    # 8 cards → 2 offline (highest indices) + 6 online; ASR/TTS stay on the
    # online GPUs, off the dedicated sglang cards
    s = make_settings(asr_device="auto", tts_sidecar_gpu="", vlm_worker_gpus="",
                      offline_provider="sglang", offline_gpu_ratio=0.25,
                      offline_gpu_count=None)
    plan = plan_placement(parse_smi_csv(SMI_H200_8X), s)
    assert plan.capacity == 6
    assert [w.gpu_index for w in plan.workers] == list(range(6))
    assert [w.port for w in plan.workers] == list(range(9000, 9006))
    assert all(w.attn_impl == "flash_attention_2" for w in plan.workers)
    assert plan.offline_capacity == 2
    assert [o.gpu_index for o in plan.offline] == [6, 7]
    assert [o.port for o in plan.offline] == [30800, 30801]
    assert plan.asr_device == "cuda:5"
    assert len(plan.tts) == 4  # native provider: one per online GPU, capped at 4
    assert [t.gpu_index for t in plan.tts] == [5, 4, 3, 2]
    assert [t.port for t in plan.tts] == [18100, 18101, 18102, 18103]
    assert not plan.warnings
    format_plan(plan)  # must not raise
    print("placement 8xH200: OK")


def test_placement_blackwell() -> None:
    # 2 cards → 1:1 split
    s = make_settings(asr_device="auto", tts_sidecar_gpu="",
                      offline_provider="sglang", offline_gpu_count=None)
    plan = plan_placement(parse_smi_csv(SMI_RTX6000_2X), s)
    assert plan.capacity == 1 and plan.workers[0].gpu_index == 0
    assert all(w.attn_impl == "sdpa" for w in plan.workers)
    assert plan.offline_capacity == 1
    assert plan.offline[0].gpu_index == 1 and plan.offline[0].port == 30800
    assert plan.asr_device == "cuda:0"
    assert len(plan.tts) == 2 and all(t.gpu_index == 0 for t in plan.tts)
    print("placement 2xRTX6000: OK")


def test_placement_4090() -> None:
    # 1 card → online only, no offline plane
    s = make_settings(asr_device="auto", tts_sidecar_gpu="",
                      offline_provider="sglang", offline_gpu_count=None)
    plan = plan_placement(parse_smi_csv(SMI_4090_1X), s)
    assert plan.capacity == 1 and plan.workers[0].gpu_index == 0
    assert plan.workers[0].attn_impl == "flash_attention_2"
    assert plan.offline == ()
    assert plan.asr_device == "cuda:0"
    assert len(plan.tts) == 2 and all(t.gpu_index == 0 for t in plan.tts)
    assert [t.port for t in plan.tts] == [18100, 18101]
    print("placement 1x4090: OK")


def test_offline_split() -> None:
    def count(n: int, **overrides) -> int:
        kw = {"offline_provider": "sglang", "offline_gpu_ratio": 0.25,
              "offline_gpu_count": None, **overrides}
        return offline_gpu_count(n, make_settings(**kw))

    # spec table at the default 1:3 ratio — one formula, no special cases
    assert [count(n) for n in range(1, 9)] == [0, 1, 1, 1, 1, 2, 2, 2]
    # ratio override
    assert count(8, offline_gpu_ratio=0.5) == 4
    # explicit count wins; clamped to n-1 (never all GPUs) with a warning
    assert count(8, offline_gpu_count=3) == 3
    assert count(8, offline_gpu_count=0) == 0
    warnings: list = []
    n = offline_gpu_count(4, make_settings(
        offline_provider="sglang", offline_gpu_count=99), warnings)
    assert n == 3 and warnings
    # provider off → no offline plane regardless of count
    assert count(8, offline_provider="none") == 0
    assert count(2, offline_provider="none", offline_gpu_count=1) == 0

    # explicit VLM_WORKER_GPUS = hand-placed: no auto-carve …
    s = make_settings(vlm_worker_gpus="0,1", asr_device="auto",
                      offline_provider="sglang", offline_gpu_count=None)
    plan = plan_placement(parse_smi_csv(SMI_H200_8X), s)
    assert plan.offline == () and plan.capacity == 2
    # … unless OFFLINE_GPU_COUNT is explicit too (highest leftover GPUs)
    s = make_settings(vlm_worker_gpus="0,1", asr_device="auto",
                      offline_provider="sglang", offline_gpu_count=2)
    plan = plan_placement(parse_smi_csv(SMI_H200_8X), s)
    assert [o.gpu_index for o in plan.offline] == [6, 7]
    # ASR/TTS never land on offline GPUs
    assert plan.asr_device == "cuda:1"
    assert all(t.gpu_index in {0, 1} for t in plan.tts)
    print("offline split: OK")


def test_placement_overrides_and_fallbacks() -> None:
    # explicit worker GPUs with repeats (fake multi-GPU on one card)
    s = make_settings(vlm_worker_gpus="0,0", asr_device="auto", offline_gpu_count=None)
    plan = plan_placement(parse_smi_csv(SMI_4090_1X), s)
    assert [w.gpu_index for w in plan.workers] == [0, 0]
    assert [w.port for w in plan.workers] == [9000, 9001]
    assert plan.offline == ()  # hand-placed layout, no OFFLINE_GPU_COUNT

    # explicit ASR device wins over auto placement
    s = make_settings(asr_device="cuda:3")
    assert plan_placement(parse_smi_csv(SMI_H200_8X), s).asr_device == "cuda:3"

    # explicit TTS_GPU pins all sidecars
    s = make_settings(tts_sidecar_gpu="2", asr_device="auto")
    plan = plan_placement(parse_smi_csv(SMI_H200_8X), s)
    assert all(t.gpu_index == 2 for t in plan.tts)

    # busy GPUs are skipped; none eligible → single-worker fallback + warning
    busy = "0, GPU-a, NVIDIA GeForce RTX 4090, 24564, 20000, 8.9"
    s = make_settings(asr_device="auto", gpu_id=0)
    plan = plan_placement(parse_smi_csv(busy), s)
    assert plan.capacity == 1 and plan.workers[0].gpu_index == 0
    assert plan.warnings

    # empty topology (no nvidia-smi / CPU box) → degenerate single worker
    plan = plan_placement([], make_settings(asr_device="auto"))
    assert plan.capacity == 1 and plan.asr_device == "cuda:0"
    assert plan.tts[0].gpu_index is None  # inherit
    print("placement overrides/fallbacks: OK")


def test_kv_bytes_per_token() -> None:
    # Qwen2.5-7B-class shape (MOSS-VL 8B text tower): 28 layers, 4 kv heads, head_dim 128
    cfg = {"num_hidden_layers": 28, "num_attention_heads": 28,
           "num_key_value_heads": 4, "hidden_size": 3584}
    bpt = kv_bytes_per_token(cfg)
    assert bpt == 2 * 28 * 4 * 128 * 2 == 57344

    # nested text_config layout + explicit head_dim
    nested = {"model_type": "moss_vl", "text_config": {
        "num_hidden_layers": 36, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128}}
    assert kv_bytes_per_token(nested) == 2 * 36 * 8 * 128 * 2

    try:
        kv_bytes_per_token({})
    except ValueError:
        pass
    else:
        raise AssertionError("empty config must raise")
    print("kv bytes/token: OK")


def check_budget(b: KvBudget, *, floor_met: bool) -> None:
    assert b.floor_met is floor_met
    assert 0 <= b.budget_tokens <= b.cap_tokens


def test_kv_budget() -> None:
    bpt = 57344
    s = make_settings(kv_session_min_minutes=3.0, kv_memory_ratio=0.9,
                      kv_safety_margin_mib=2048, kv_tokens_per_second_est=270.0,
                      kv_enforce="auto")
    floor = int(3 * 60 * 270)  # 48600 tokens

    # H200-class: ~120 GB free → ratio dominates, floor easily met
    b = compute_kv_budget(120 * 1024 * MIB, bpt, s, gpu_total_mib=143771)
    check_budget(b, floor_met=True)
    assert b.budget_tokens > floor * 10
    assert abs(b.budget_tokens - 0.9 * b.cap_tokens) <= 1
    assert not b.enforce_hard  # auto → soft on a big box
    assert b.est_max_seconds > 3600

    # small-but-sufficient: floor > ratio×cap → floor dominates (clamped to cap)
    free = int(floor * bpt / 0.95) + 2048 * MIB  # cap ≈ floor/0.95 > floor
    b = compute_kv_budget(free, bpt, s, gpu_total_mib=24564)
    check_budget(b, floor_met=True)
    assert b.budget_tokens == floor  # max(floor, 0.9*cap) = floor here
    assert b.enforce_hard  # auto → hard on <32 GB

    # 4090 worst case: floor doesn't fit → budget = cap (nonzero), floor_met False
    b = compute_kv_budget(int(floor * bpt * 0.5) + 2048 * MIB, bpt, s, gpu_total_mib=24564)
    check_budget(b, floor_met=False)
    assert 0 < b.budget_tokens == b.cap_tokens < floor

    # explicit enforce beats auto
    b = compute_kv_budget(120 * 1024 * MIB, bpt,
                          dataclasses.replace(s, kv_enforce="hard"), gpu_total_mib=143771)
    assert b.enforce_hard
    b = compute_kv_budget(10 * 1024 * MIB, bpt,
                          dataclasses.replace(s, kv_enforce="soft"), gpu_total_mib=24564)
    assert not b.enforce_hard

    # free below the safety margin → zero budget, never negative
    b = compute_kv_budget(1024 * MIB, bpt, s, gpu_total_mib=24564)
    assert b.budget_tokens == 0 and b.cap_tokens == 0 and not b.floor_met
    print("kv budget: OK")


def main() -> int:
    test_parse()
    test_attn_select()
    test_placement_h200()
    test_placement_blackwell()
    test_placement_4090()
    test_offline_split()
    test_placement_overrides_and_fallbacks()
    test_kv_bytes_per_token()
    test_kv_budget()
    print("\nGPU PLANNING TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
