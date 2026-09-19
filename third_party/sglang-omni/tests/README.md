# Tests

**English** | [简体中文](./README_zh.md)

## MOSS-VL Realtime

After [installing the backend](../docs/get_started/installation.md), run from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_video_realtime*.py
```

These tests cover incremental requests, visual KV, batching, rate control, WebSocket lifecycle, process cleanup, and delivery scripts without loading checkpoint weights.

The three GPU tests for accuracy, single-session latency, and realtime concurrency are documented in the [delivery test guide](../deployment/moss_vl_realtime/README.md). Their reference results are separate from unit-test results.

## Test Layout

| Path | Scope |
| --- | --- |
| [unit_test/moss_vl_realtime](./unit_test/moss_vl_realtime/) | Realtime model and delivery contracts |
| [unit_test/serve](./unit_test/serve/) | API validation, streaming, and session lifecycle |
| [unit_test/pipeline](./unit_test/pipeline/) | Stage placement, IPC, routing, and ownership |
| [unit_test/scheduling](./unit_test/scheduling/) | Admission, queues, caches, and scheduling |
| [unit_test/router](./unit_test/router/) | Router control and data plane |
| [unit_test/client](./unit_test/client/) | Client events and completion handling |
| [test_model](./test_model/) | Model integration and GPU benchmarks |
| [test_ci](./test_ci/) | Deployment-specific CI scenarios |
| [data](./data/) | Small shared media fixtures |
| [utils](./utils/) | Shared test utilities |

## Broader Checks

```bash
python -m pytest tests/unit_test -q
python -m pytest tests/test_model -m benchmark -v -s
```

Some unit tests require accelerator or optional model dependencies. Integration tests can start real services, download models, and reserve substantial GPU memory; read their fixtures before running them on shared hardware. A skipped test is not a passing hardware validation.

Markers are registered in [pyproject.toml](../pyproject.toml). Model-specific CI flags are defined in [test_model/conftest.py](./test_model/conftest.py); use `python -m pytest tests/test_model --help` to inspect them.

## Adding Tests

- Test user-visible contracts and resource ownership, not incidental implementation details.
- Place tests in the component that owns the behavior; avoid new root-level `tests/test_*.py` files.
- Reuse existing fixtures and fakes. Keep GPU tests explicit about their hardware and model requirements.
- Cover cancellation, errors, and cleanup for stateful changes; clean up only resources created by the test.
- Keep fixtures small and deterministic. Store weights, downloaded datasets, and generated results outside the unit-test tree.
