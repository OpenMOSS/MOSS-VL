# Ascend Integration Validation

**English** | [简体中文](./VALIDATION_zh.md)

## Scope

Date: 2026-09-13. Main integration candidate: main `194bcc8` plus Ascend integration `790b96e`. The earlier Ascend integration combined `c0bacdd` and PR #1 `fd921c2`. Results below distinguish these two validation stages; neither constitutes end-to-end hardware acceptance.

The integration adopts the PR's explicit Torch cross-attention path and deployment-default centralization. It retains platform-specific startup locks, configuration error handling, instance-local version compatibility, fail-closed patch installation and owned-process cleanup. Mask naming, shape validation, empty-encoder handling and zero output for fully masked rows remain consistent across both patch sets. CUDA production defaults are unchanged.

## Ascend Branch Results

| Check | Result |
| --- | --- |
| ServerArgs, version compatibility, NPU locks, deployment defaults, installer and platform guards | 48 passed |
| Stage process environment, GPU memory helpers, execution bridge and legacy performance probe | 56 passed |
| 0.5.16 patch installation and repeat installation on private sources | Passed; repeat leaves files unchanged |
| 0.5.14 patch installation and repeat installation on pinned upstream private sources | Passed; repeat leaves files unchanged |
| Patched 0.5.16 attention on CPU | 9 passed |
| Patched 0.5.14 attention on CPU | 9 passed |

Attention checks cover mixed Q/KV lengths, grouped-query attention, FP32/BF16, softcapping, fully masked rows, empty encoder input, malformed masks, random-input reference comparison and unchanged self-attention dispatch. Cross-attention tests fail if fused SDPA is called. Installer tests cover explicit targets, API detection, idempotence and leaving originals unchanged on incompatible patches.

Source fixtures: installed SGLang 0.5.16 from the established backend environment, and the official SGLang `v0.5.14` source tag. Tests only patched private copies; installed model environments were not changed. These checks validate source integration and CPU arithmetic, not vendor NPU binaries.

## Main Candidate Results

| Check | Result |
| --- | --- |
| Merge with main | No textual conflicts |
| Related CPU regression matrix | 616 passed, 11 skipped |
| New history-prefill/v2 accounting contracts | Passed, included in the CPU matrix |
| Patched 0.5.16 / 0.5.14 attention, separately enabled on CPU | 9 / 9 passed |
| CUDA production configuration, dependency lock and original accuracy runner | Unchanged from main |
| History prefill, v1/v2 implementation and accounting files | Preserved from main; no additional production changes required |

The matrix covers realtime model/runtime tests, v1/v2 and legacy session tests, vendor compatibility, stage device mapping, startup locks, GPU memory helpers and the execution bridge. Eleven default skips are nine opt-in patched-attention cases, one CUDA model-step case and one processor case requiring a model path. The nine attention cases were separately run against each patched source version.

The previously timed-out cross-component run completed with writable caches and local socket access; the prior timeout's precise cause was not established. Reproduce the CPU matrix from the repository root:

```bash
python -m pytest tests/unit_test/moss_vl_realtime vl_api_adapter/tests \
  vl_legacy_adapter/tests tests/unit_test/vendor \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/serve/test_video_realtime_lifecycle.py \
  tests/unit_test/pipeline/test_stage_process_env.py \
  tests/unit_test/pipeline/test_npu_startup_lock.py \
  tests/unit_test/pipeline/test_gpu_memory.py \
  tests/unit_test/model_runner/test_sglang_execution.py -q
```

## Pending Acceptance

Real CUDA/NPU inference, CUDA TP startup, NPU HF alignment with original fixtures, mixed-length 1/2/4 sessions, visual-window transitions and device-memory recovery remain required. Run both supported NPU runtime variants. The GPU SSH tunnel was unavailable during this integration, and no NPU device was used.

Commands and environment requirements are in the [deployment guide](./README.md). Publish this candidate separately; update main only after hardware acceptance and another check for intervening main changes.
