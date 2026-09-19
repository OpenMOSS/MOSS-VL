# Ascend Integration

**English** | [简体中文](./README_zh.md)

Native MOSS-VL serving on Ascend, with the shared [legacy VL adapter](../../vl_legacy_adapter/README.md). The native backend serves `/v1/video/realtime`; the separate adapter serves `/v1/realtime`. There is no second legacy protocol implementation inside the backend.

## Environment

Target stack: Ascend 910B2C, CANN 9.0.0, PyTorch/torch_npu 2.11, SGLang 0.5.14 or 0.5.16 with Ascend support, Transformers 5.12.1, and Python 3.12 or 3.13. Use the NPU maintainer's matching environment and a local [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) checkpoint. Do not apply the CUDA dependency lock to this environment. Both SGLang versions are implementation targets and require separate runtime acceptance; vendor builds must provide the corresponding upstream APIs.

| SGLang API | Integration behavior |
| --- | --- |
| 0.5.16: `ModelRunner(ps=...)`, native `NextBatchPlan` and request range | Native APIs, no legacy runtime patches |
| 0.5.14: flat rank arguments, scheduler-owned batches, `fill_len`/`extend_input_len` | Instance-local scheduler bridge, synchronized request range, explicit forward overrides and KV profiling adapter |

Compatibility follows inspected API signatures rather than only a version string. Unknown constructors or unsupported forward arguments fail explicitly. The bridge does not replace methods on the upstream Scheduler class, disable memory allocation, or discard forward overrides. Source references: [0.5.14 scheduler](https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/managers/scheduler.py), [0.5.14 ModelRunner](https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/model_executor/model_runner.py).

Run from the repository root using that environment's Python:

```bash
python -m pip install --no-deps -e .
bash patches/npu/apply_npu_patches.sh
export ASCEND_USE_FA=false
export ASCEND_USE_FIA=false
```

The patch installer operates on the active interpreter's SGLang package; an optional argument selects its `site-packages` directory. It selects the 0.5.14 or 0.5.16 patch set from the installed ModelRunner signature. `--patch-set 0.5.14` or `--patch-set 0.5.16` selects a source layout explicitly for vendor backports; incompatible sources still fail. Stop serving processes before patching. All patches are staged before installation; already-applied patches are accepted, incompatible sources fail rather than being skipped, and modified files get `.moss-npu.bak` backups. Restart after installation. The environment must already contain application dependencies, including FastAPI, websockets, Pillow, psutil, pytest and pytest-asyncio.

Frame visibility uses per-request masks in the native extend path. Cross-attention uses explicit Torch operations rather than fused SDPA; self-attention reads the complete text prefix after the encoder slots. Both version-specific patch sets include the common `0004-use-torch-cross-attention.patch` capability marker. Decode retains its existing all-visible encoder behavior. Missing visibility or Torch-path support prevents startup. Use the native Ascend extend path with FA/FIA disabled; speculative and context-parallel execution require separate validation.

Rerun the installer to upgrade environments using the repository's previous patch sets. The 0.5.16 installer migrates the former `0002/0003/0004` attention state; the 0.5.14 installer applies its version-specific alignment supplement. Fresh and upgraded installations converge to the same source state, and repeated installation leaves sources unchanged. Unknown or partially modified patch states are rejected. The 0.5.16 vision patch includes `layers/attention/vision.py`; it is backed up and validated with the model and attention files.

## Start

```bash
MODEL_PATH=/path/to/model bash deploy.sh start
# Two TP2 groups, logical devices 0,1 and 2,3:
MODEL_PATH=/path/to/model bash deploy.sh start4
# Two TP4 groups, logical devices 0-3 and 4-7:
MODEL_PATH=/path/to/model bash deploy.sh start8
```

Each command runs in the foreground. Stop with Ctrl-C or SIGTERM from a service manager; cleanup is restricted to owned processes. The former process-name-based `stop`/`restart` commands are not used. Logs are kept in a printed, unique `/tmp/moss-npu-*` directory. Any child failure stops the supervised deployment.

| Setting | Default |
| --- | --- |
| Native backend ports | `PORT=8000`, then 8001 for the second group |
| Legacy adapter ports | `ADAPTER_PORT=18600`, then 18601 |
| Single group device | `GPU=0`, or `GPUS=0,1` for TP |
| Group override | `NPU_GROUPS='0,1;2,3'` |
| Backend session capacity | Device count in each group; override `MAX_RUNNING_REQUESTS` |
| Adapter capacity | `MAX_INFLIGHT=1`; set independently for concurrent callers |
| Context / memory fraction | TP2: 8192 / 0.70; otherwise 32768 / 0.70 |
| Optional frame chunking | `SGLANG_MOSS_VIT_CHUNK_FRAMES=0` (disabled); positive values limit frames per ViT call |
| Additional backend arguments | `EXTRA_SERVER_ARGS`, for example `--mm-attention-backend ascend_attn` |

NPU serving defaults are centralized in [config.py](./config.py) and used by the supervisor. `CONTEXT_LENGTH`, `MEM_FRACTION` and `HOST` override the defaults; CUDA configuration is unaffected. Device IDs are logical indices in the visible-device list. Verify HCCS planes on the actual host; the example grouping is not a portable hardware topology guarantee. Each adapter connects to its own backend, not to a load balancer. Capacity and memory defaults need NPU acceptance testing.

NPU startup locks resolve local IDs through `ASCEND_RT_VISIBLE_DEVICES` and use an NPU-specific lock namespace. CUDA retains its existing lock mapping. TP communication initialization receives the deployment-assigned rendezvous port through the standard model-runner path.

For the opt-in [VL API v2](../../vl_api_adapter/README.md), add `--vl-api-v2-port 18610` to `examples/run_moss_vl_realtime_server.py` in the prepared NPU environment. `deploy.sh` continues to start native v1 and the legacy adapter only. The `vl_api_adapter/start.sh` wrapper uses the CUDA deployment profile and is not the NPU launcher.

## Failure Handling

NPU serving uses a fail-fast policy for allocator OOM and runtime OOM errors: the worker exits and the deployment supervisor stops the affected deployment. Restart is managed externally. Scheduler-thread error notification waits at most 30 seconds before exit; parent-process health reflects failure after propagation, not instantaneously. This NPU policy does not change CUDA batch-error handling or enable automatic restart on CUDA.

## Acceptance

Run the following checks in each supported SGLang environment. CPU attention and installation tests do not replace NPU hardware validation. Earlier integration measurements are retained in [validation results](./VALIDATION.md).

```bash
python -m pytest tests/unit_test/vendor/test_sglang_server_args.py \
  tests/unit_test/vendor/test_sglang_versions.py \
  tests/unit_test/pipeline/test_stage_process_env.py \
  tests/unit_test/pipeline/test_npu_startup_lock.py \
  tests/unit_test/pipeline/test_resource_errors.py \
  tests/unit_test/moss_vl_realtime/test_npu_deployment.py \
  tests/unit_test/moss_vl_realtime/test_npu_patch_installer.py \
  tests/unit_test/moss_vl_realtime/test_npu_platform_contract.py \
  vl_legacy_adapter/tests/test_lifecycle.py \
  tests/unit_test/moss_vl_realtime/test_legacy_perf_probe.py -q
MOSSVL_TEST_NPU_PATCHES=1 NPU_TEST_DEVICE=cpu python -m pytest \
  tests/unit_test/moss_vl_realtime/test_npu_attention.py -q
MOSSVL_TEST_NPU_PATCHES=1 NPU_TEST_DEVICE=npu:0 python -m pytest \
  tests/unit_test/moss_vl_realtime/test_npu_attention.py -q
python vl_legacy_adapter/tests/smoke_client.py \
  --url ws://127.0.0.1:18600/v1/realtime \
  --testdata deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122
bash perf.sh
```

Attention tests normally read the installed SGLang sources. Set `NPU_TEST_SITE_PACKAGES` to test a patched private copy without modifying the serving environment.

`perf.sh` defaults to five rounds of four distinct repository frames at 160 tokens/s, with a real 10 s deadline per round. It requires all ACKs and visible text, preserves output interleaved with ACKs, and measures TTFT from connection start to the first visible text. `FRAMES_DIR`, `NFRAMES`, `ROUNDS`, `TOKEN_RATE`, `TIMEOUT_S` and `VL_MODEL_WS_URL` configure it. These are protocol/latency checks, not HF equivalence proofs.

Run the checks independently in both SGLang environments. Required before main: NPU HF/SGLang comparison for multi-frame extend, questions before/after frame boundaries, retained visual KV, visual-window transitions, no-visible-frame rows and mixed-length 1/2/4 sessions. Use matching inputs and sampling in the maintainer's HF harness; retain output and first-divergence evidence. Also cover delayed/missing binaries, malformed images, setup failure, cancellation/reconnect and memory reclamation, plus CUDA regressions. The existing CUDA accuracy launcher is not an NPU evaluation script.
