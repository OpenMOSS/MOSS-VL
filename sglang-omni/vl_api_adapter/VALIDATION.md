# VL API v2 Validation

**English** | [简体中文](./VALIDATION_zh.md)

[Usage](./README.md) | [API Reference](./API.md)

This page records the 2026-09-11 protocol validation. For the Ascend/main integration candidate, see [candidate validation](../deployment/npu/VALIDATION.md); the historical GPU results below are not a new hardware acceptance run.

## Environment

Date: 2026-09-11. Compared original main `68a0eef`, native v1 on the feature branch, and v2 on the feature branch.

| Item | Configuration |
| --- | --- |
| Hardware / precision | H200, one GPU per backend, TP=1; BF16 |
| Software | SGLang 0.5.16, Transformers 5.12.1 |
| Model | Local MOSS-VL-Realtime-SGLANG |
| Memory / capacity | `mem_fraction_static=0.5`; context 131072; four sessions |
| Original accuracy suite | HF eager, SGLang FlashInfer; window, pooling and async decode disabled |
| Protocol tests | Production profile, 60-second visual KV window; no memory/ASR/TTS |

## Accuracy and Protocol Alignment

**Result: PASS.** The original suite preserves prompts, complete PNG sequences, timestamps, events and task contracts, with `--strict-tokens`. Each event advances after silence; concurrent sessions within each backend share one GPU.

| Sessions | Main accuracy | Feature accuracy | Protocol text | Single/multi match |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4/4 | 4/4 | 4/4 | 4/4 |
| 2 | 4/4 | 4/4 | 4/4 | 4/4 |
| 4 | 4/4 | 4/4 | 4/4 | 4/4 |

- **Accuracy:** HF/SGLang task, raw-token and per-event text checks pass; each backend also matches its single-session output.
- **Protocol text:** 36 main/v1/v2 WebSocket sessions using the same manifest match in full and per-event text, including comparison with the original engine output. WebSocket does not expose raw token IDs.
- **Cross-branch comparison:** raw tokens and per-event text match in 12/12 HF and 12/12 SGLang runs.

## Lifecycle and Accounting

| Check | Result |
| --- | --- |
| CPU regressions | 499 passed, 2 skipped |
| Single session, continuation, interruption, window | Matching v1/v2 text and final usage |
| Same-turn responses | Distinct response IDs; repeated silence does not settle twice |
| Interrupted segments | No normal done; usage carries into later settlement |
| Input validation | Invalid fields, gaps and corrupt images rejected; corrected same-sequence retry succeeds |
| Abort and timeout | Idle, output and in-flight-input aborts and missing-binary timeout pass |
| Context limit | Oversized initial/extended input terminates correctly; uncommitted input not billed |
| Capacity and recovery | Shared v1/v2 capacity, excess rejection and post-disconnect recovery pass |
| Settlement and cleanup | Arithmetic, monotonicity, terminal uniqueness and ≤1 Hz telemetry pass; no remaining requests or child processes |

## Commands

Run from the repository root:

```bash
# Original HF/SGLang accuracy suite
bash deployment/moss_vl_realtime/test_accuracy.sh /path/to/model --strict-tokens

# CPU regressions
python -m pytest tests/unit_test/moss_vl_realtime \
  vl_api_adapter/tests \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/serve/test_video_realtime_lifecycle.py \
  vl_legacy_adapter/tests -q
```

These commands do not include WebSocket manifest comparisons. CPU skips require CUDA and an explicit model path respectively. Multi-GPU TP/NPU, full Demo integration, long-running stress and actual worker crashes were not tested. CPU tests cover failure branches such as unavailable final snapshots.
