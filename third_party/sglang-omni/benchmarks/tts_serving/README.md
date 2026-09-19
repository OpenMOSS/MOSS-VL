# TTS Serving Benchmark

**English** | [简体中文](./README_zh.md)

Tests OpenAI-compatible TTS API behavior under load: speech, raw PCM streaming, WebSocket, batch synthesis, malformed requests, and stateful voice management. SeedTTS quality evaluation is a [separate benchmark](../README.md).

## Run

Start the target service separately, then run from the repository root:

```bash
python -m benchmarks.eval.benchmark_tts_serving \
  --spec benchmarks/tts_serving/examples/stress.json \
  --out results/tts_serving/stress
```

Edit [stress.json](./examples/stress.json) for the target `base_url`, `model_name`, and optional `auth.api_key_env`. The included spec targets Higgs TTS and references audio under `docs/_static/audio`; the target server must allow that local path. Reference paths belong to the server's filesystem, not the benchmark client's.

Direct runs require FFmpeg for decoding compressed audio. Corpus-backed specs may download pinned SeedTTS metadata.

## Configuration

| Field | Purpose |
| --- | --- |
| `base_url` / `model_name` | Target service and request model ID |
| `test_type` / `run_id` | Artifact labels; types: `engine`, `e2e`, `external` |
| `seed` | Deterministic scenario order and arrivals; default 0 |
| `auth.api_key_env` | Environment variable holding the bearer token |
| `params.profile` | `stress` |
| `params.enabled_endpoints` | `speech`, `speech_stream`, `voices`, `batch`, `websocket` |
| `params.load_stages` | Staged load plan |
| `params.total_requests` / `max_concurrency` | Fallback load when stages are omitted |
| `params.timeout_s` | Per-request timeout |
| `params.speaker_max_uploaded` | Expected server-side voice limit |
| `params.voice_cache_pressure_voice_count` / `voice_speaker_cap_count` | Voice pressure and capacity budgets |
| `params.file_ref_audio` / `file_ref_text` | Server-readable reference clip and transcript |

See [spec.py](./spec.py) and [scenarios.py](./scenarios.py) for the complete schema and deterministic scenario matrix.

## Contracts

- Speech: response formats, reference audio, speed boundaries, SDK compatibility, nonzero decoded audio, and streaming.
- Errors: malformed HTTP requests return a structured JSON `error` with `message`, `type`, `param`, and `code`. Missing voice resources use 404; missing-voice deletion returns `success: false` and a nonempty `error` object or string.
- Batch: 1-32 items, per-item overrides, item-level results, and oversized-batch rejection.
- Voices: upload, list, metadata, overwrite, delete, reuse, races, and cache pressure. Created voices are deleted and cleanup is checked.
- WebSocket: configuration, incremental text, binary audio, ordering, disconnects, and malformed/missing-config errors.

Voice cache pressure requires observable `cache_stats`: entries, memory_bytes, max_bytes, eviction_count, hit_count, miss_count, and delete_invalidation_counter. Operations must move the relevant counters; eviction is required if traffic reaches the advertised budget. Enabled-but-missing contracts fail explicitly.

## Load Profile

The included spec has `mixed-production`, `voice-cache-pressure`, and `voice-speaker-cap` stages. The mixed stage runs 300 seconds and combines REST, streaming REST, WebSocket, batch-32, and long-prefill workloads. Six-workload cohorts start every 15 seconds, with background arrivals and required API coverage between cohorts. Coverage traffic is counted separately from workload percentiles.

Speaker-cap checks account for existing voices before creating enough voices to reach the configured cap, then require overflow rejection. Cache checks validate observable state, not just successful requests.

## Results

| Artifact | Contents |
| --- | --- |
| `results.json` | Pass/fail, coverage, load validity, latency, unsupported contracts |
| `manifest.json` | Parsed spec/scenario hashes and artifact metadata |
| `raw/*.jsonl` | Per-scenario records |
| `logs/harness.log` | Load-stage execution log |

**Check both the exit code and `overall.passed`.** Exit 0 means the harness completed and wrote artifacts; the service can still fail. Nonzero means infrastructure/runtime failure; when possible the report records `harness_status="error"`.

Inspect `overall.coverage_contract_valid`, `overall.load_generation_valid`, `overall.mixed_arrival_valid`, `metrics.by_stage_and_workload`, `unsupported_contracts`, and `coverage_failures` before interpreting performance.

## Standalone Container

This is the benchmark client image, not the inference backend:

```bash
docker build -f benchmarks/tts_serving/Dockerfile \
  -t sglang-omni-tts-serving-benchmark .
mkdir -p results/tts_serving/stress
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/benchmarks/tts_serving/examples/stress.json:/etc/benchmark/spec.json:ro" \
  -v "$PWD/results/tts_serving/stress:/var/benchmark/out" \
  sglang-omni-tts-serving-benchmark
```

Input: `/etc/benchmark/spec.json`; output: `/var/benchmark/out`. Use a target address reachable from the container. FFmpeg is included; reference audio must remain readable by the target service.
