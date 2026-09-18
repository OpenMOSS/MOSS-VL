# MOSS-VL Realtime Tests

**English** | [简体中文](./README_zh.md)

## Environment

Follow the [installation guide](../../docs/get_started/installation.md) to create an independent Python 3.12 environment. Prepare an idle GPU in the 80 GB memory class or above and a local [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) checkpoint.

Run commands from the repository root. The model path is the only required argument; test fixtures are included.

## Three Tests

| Entry | Purpose | Defaults |
| --- | --- | --- |
| [test_accuracy.sh](./test_accuracy.sh) | HF/SGLang task, token, and per-event alignment; single/multi-session consistency within each backend | 1, 2, 4 sessions; natural generation |
| [test_latency.sh](./test_latency.sh) | Unthrottled HF/SGLang prefill, frame processing, first token, TPOT, and throughput | Fixed inputs, 64 generated tokens, 5 timed rounds after warmup |
| [test_concurrency.sh](./test_concurrency.sh) | Realtime first-visible-text latency, token intervals, and per-session speed | 1, 2, 4, 8 sessions; 1 FPS and 10 tokens/s per session; 3 repeats |

Accuracy uses controlled input ordering. Each concurrency session independently sends scheduled frames and prompts and receives output, without waiting for other sessions or for an answer to finish. Concurrency uses natural generation and supports throttled and unthrottled settings.

Defaults: BF16, HF eager, SGLang FlashInfer with decode CUDA Graphs, and `mem_fraction_static=0.5` from `config.json`. Tests disable the visual window, pooling, and async decode. Timing is measured inside the backend process, excluding networking, gateways, and the Demo.

## Usage

```bash
bash deployment/moss_vl_realtime/test_accuracy.sh /path/to/model
bash deployment/moss_vl_realtime/test_latency.sh /path/to/model
bash deployment/moss_vl_realtime/test_concurrency.sh /path/to/model
```

One idle GPU is selected automatically. HF and SGLang run sequentially on that GPU; all concurrent sessions also share one GPU.

```bash
bash deployment/moss_vl_realtime/test_concurrency.sh /path/to/model \
  --gpus 2 --sessions 1 2 4 8 --fps 1 --token-rate 10 --repeats 3
```

- Common: `--gpus`, `--output-dir` for a new results directory, and `--dry-run`.
- Accuracy: `--strict-tokens` treats token/event mismatches as failures.
- Latency: `--repeats`; concurrency also accepts `--sessions`, `--fps`, and `--token-rate`.
- HF/SGLang comparisons can use `--gpus 2,3` for parallel execution. Prefer the same GPU for formal speed comparisons. Concurrency accepts only one GPU.

Use `--help` for other options and `PYTHON=/path/to/python` to select an interpreter. Tests use local weights, disable proxies, and do not stop existing services. Workers have dedicated process groups; cleanup removes remaining group members and reports shutdown timeouts.

## Outputs

Each run gets a separate directory under `results/accuracy/`, `results/latency/`, or `results/concurrency/`, containing a Markdown report, `summary.json`, raw outputs, parameters, and logs. NVML samples memory every 50 ms during loading and testing; `*_memory.json` records bytes and peaks. The 80 GB class is a capacity reference, not a hard pass threshold.

`PASS` means automated checks passed; `FAIL` indicates execution or validation failure; `REVIEW` requires examining output differences. Concurrency reports include per-session metrics, send lateness, pending events, and unanswered prompts.

## Reference Results

Single NVIDIA H200, BF16; PyTorch 2.11.0, Transformers 5.12.1, SGLang 0.5.16, FlashInfer 0.6.14, and `mem_fraction_static=0.5`. Times are milliseconds; TPS means Tokens Per Second. Realtime TTFT ends at the first visible text segment. TPOT and TPS cover continuous answer intervals, excluding silence.

### Accuracy Alignment

Result: **PASS**.

| Sessions | HF task pass | SGLang task pass | Cross-backend tokens/events match | Both match single-session baseline |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4/4 | 4/4 | 4/4 | 4/4 |
| 2 | 4/4 | 4/4 | 4/4 | 4/4 |
| 4 | 4/4 | 4/4 | 4/4 | 4/4 |

All 12 pairs pass task, token, per-event, and single/multi-session consistency checks.

### Single-Session Baseline (Unthrottled)

Result: **PASS**.

| Metric | HF mean / P95 | SGLang mean / P95 | Mean speedup |
| --- | --- | --- | ---: |
| Initial prefill | 45.58 / 49.30 | 42.62 / 55.71 | 1.07x |
| Frame extend | 106.99 / 218.37 | 88.95 / 129.57 | 1.20x |
| Prompt TTFT | 91.77 / 124.42 | 33.57 / 35.17 | 2.73x |
| TPOT | 52.05 / 61.73 | 11.08 / 26.97 | 4.70x |

Decode throughput: HF **19.21 tokens/s**, SGLang **90.26 tokens/s**.

### Realtime Concurrency (10 Tokens/s)

Result: **PASS**.

| Sessions | Frame processing P95 | TTFT mean / P95 | TPOT mean / P95 | Mean per-session TPS | Slowest session TPS |
| ---: | ---: | --- | --- | ---: | ---: |
| 1 | 142.56 | 518.70 / 910.41 | 99.21 / 128.02 | 10.08 | 10.08 |
| 2 | 222.93 | 532.85 / 1025.32 | 104.23 / 152.86 | 9.59 | 9.58 |
| 4 | 298.19 | 625.84 / 1135.57 | 114.13 / 235.76 | 8.76 | 8.65 |
| 8 | 385.91 | 794.75 / 1315.16 | 148.31 / 490.91 | 6.71 | 6.03 |

![Throttled concurrency: mean per-session TPS and TTFT](./assets/concurrency_10_tps.png)

Left axis: mean per-session TPS. Right axis: mean first-visible-text TTFT (ms).

With a 10 tokens/s target, four sessions average 8.76 tokens/s each and eight average 6.71 tokens/s. This natural-generation workload is separate from the unthrottled fixed-token baseline.

Memory peaks for the three tests follow. Whole-device figures include the driver and existing baseline allocation.

| Test | VLM process peak GiB | Device peak GiB | Device peak GB |
| --- | ---: | ---: | ---: |
| Accuracy alignment | 74.779 | 76.093 | 81.705 |
| Single-session latency | 71.014 | 72.328 | 77.661 |
| Independent-session latency | 72.879 | 74.193 | 79.664 |

`1 GiB=2^30 bytes`; `1 GB=10^9 bytes`. These measurements are approximately in the 80 GB memory class and do not change the test pass results.

### Realtime Concurrency (Unthrottled)

Result: **PASS**.

Pure SGLang-Omni backend performance at 1/2/4/8/16 sessions. Each session uses 1 FPS, natural generation, and `token_rate=86400`, without Demo, memory, ASR, TTS, or networking. Test capacity is 16; the deployment default remains four sessions.

```bash
bash deployment/moss_vl_realtime/test_concurrency.sh /path/to/model \
  --sessions 1 2 4 8 16 --fps 1 --token-rate 86400 --repeats 3
```

| Sessions | Frame processing P95 | TTFT mean / P95 | TPOT mean / P95 | Mean per-session TPS (tokens/s) | Slowest session TPS (tokens/s) |
| ---: | ---: | --- | --- | ---: | ---: |
| 1 | 157.50 | 512.53 / 974.75 | 16.95 / 27.45 | 59.00 | 59.00 |
| 2 | 233.59 | 543.75 / 1040.60 | 18.76 / 27.81 | 53.21 | 50.93 |
| 4 | 234.78 | 548.92 / 1026.21 | 21.43 / 27.92 | 47.82 | 45.26 |
| 8 | 401.91 | 728.11 / 1334.30 | 39.03 / 143.36 | 25.57 | 23.70 |
| 16 | 491.57 | 1684.67 / 3556.17 | 93.27 / 686.57 | 10.70 | 9.21 |

![Unthrottled concurrency: mean per-session TPS and TTFT](./assets/concurrency_unthrottled.png)

Left axis: mean per-session TPS. Right axis: mean first-visible-text TTFT (ms).

From one to 16 sessions, mean per-session speed decreases from 59.00 to 10.70 tokens/s, while mean TTFT increases from 512.53 to 1684.67 ms.

## Launch Inference

```bash
bash deployment/moss_vl_realtime/start.sh /path/to/model
```

Defaults: one GPU, four sessions, `http://127.0.0.1:18500`. Health: `GET /health`; realtime: `WS /v1/video/realtime`. Use `--gpus 2 --port 18510` to override placement and port. Other defaults are in [config.json](./config.json).
