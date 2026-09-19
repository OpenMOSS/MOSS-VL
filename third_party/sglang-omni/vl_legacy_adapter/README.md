# Legacy VL Adapter

**English** | [简体中文](./README_zh.md)

Translate the legacy VL WebSocket messages into the native sglang-omni video realtime protocol. Each client connection owns a fresh upstream session; model weights remain loaded in the backend.

## Run

For Ascend, use the [NPU deployment guide](../deployment/npu/README.md); it starts this same adapter beside the native backend.

Install the backend following the [installation guide](../docs/get_started/installation.md). Run these commands from the repository root, in separate terminals:

```bash
bash deployment/moss_vl_realtime/start.sh /path/to/MOSS-VL-Realtime-SGLANG --port 18500
```

```bash
cd vl_legacy_adapter
OMNI_WS_URL=ws://127.0.0.1:18500/v1/video/realtime ../../.venv-main/bin/python -m adapter
```

Connect the legacy client to `ws://<adapter-host>:18600/v1/realtime`. The adapter uses the backend environment's `websockets`; the smoke client also uses Pillow. Authentication is not provided: expose the endpoint only on a trusted network or behind an authenticated gateway.

## Protocol

```text
start -> ready -> frame metadata + JPEG binary (batch)
      -> frame_ack per accepted frame -> output text -> output <|im_end|>
      -> stop / disconnect
```

Send frames without waiting for each ACK. The adapter infers the final frame after a 300 ms quiet window, so the last ACK includes that wait. A frame whose metadata has arrived remains pending until its binary arrives or its receive deadline expires. Once final-frame submission begins, additional input is rejected. After the end marker, send `stop` and close the connection; open a new connection for the next round.

| Legacy field/event | Native mapping |
| --- | --- |
| `start.prompt` | Attached to the final `input.frame.prompt`, not `session.configure.prompt` |
| `frame_queue_size` | `input_queue_capacity`, clamped to 1-256 |
| `max_new_tokens` | `max_new_tokens`, at least 1 |
| `max_tokens_per_second` | `max_tokens_per_turn`, positive token pacing rate |
| `temperature` / `top_p` | Clamped to 0-2 / (0, 1]; `do_sample=false` forces temperature 0 |
| `top_k` / `repetition_penalty` | Not forwarded; unsupported by the native configure schema |
| `frame.timestamp` | Non-negative, non-decreasing timestamp |
| `frame_ack` | Sent after upstream `input.frame.accepted`, not after processing |
| `output.text` | Incremental `response.text.delta` |
| `<|im_end|>` | Adapter-generated end marker on session completion or the final-frame silence rule |
| `stop` / disconnect | Abort and close the upstream session |

Invalid numeric parameters, setup failures and missing frame binaries produce an `error` and close the round. A round with no visible text returns an error instead of a bare end marker. Capacity rejection contains `realtime session is already active`; rejected-client I/O does not hold the admission lock. Slot release does not wait for upstream teardown, so admission to a new adapter round is not a guarantee of immediate backend capacity.

## Configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `LISTEN_HOST` / `LISTEN_PORT` | `0.0.0.0` / `18600` | Adapter listener |
| `LISTEN_PATH` | `/v1/realtime` | Gateway path prefixes allowed; query parameters ignored |
| `OMNI_WS_URL` | `ws://127.0.0.1:18500/v1/video/realtime` | Backend endpoint |
| `MAX_INFLIGHT` | `1` | Adapter connection cap; 0 delegates capacity control to the backend |
| `FRAME_BUFFER_CAP` | `8` | Complete frames waiting in the adapter queue; each JPEG is limited to 10 MiB |
| `START_TIMEOUT_S` | `10` | Connection-to-start deadline |
| `SETUP_TIMEOUT_S` | `10` | Overall start-to-ready deadline |
| `READY_TIMEOUT_S` / `ACK_TIMEOUT_S` | `10` / `10` | Upstream readiness / per-handshake ACK deadlines |
| `FRAME_RECEIVE_TIMEOUT_S` | `10` | Metadata-to-complete-binary deadline, including the first frame |
| `FINALIZE_QUIET_MS` | `300` | Input quiet window when no binary is pending |
| `SILENCE_FINALIZE_MS` | `1000` | Silence threshold after final-frame submission |

These are individual deadlines, not a guarantee that the entire round finishes within 10 seconds. If the gap before the next frame metadata exceeds the quiet window, the current input can already be finalized; increase the window for slower senders. Silence-based completion is a heuristic, not an explicit model answer boundary.

## Tests

From `vl_legacy_adapter/`, run the CPU regression suite or the GPU-backed smoke client:

```bash
../../.venv-main/bin/python -m pytest tests/test_lifecycle.py -q
../../.venv-main/bin/python tests/smoke_client.py --url ws://127.0.0.1:18600/v1/realtime \
  --testdata ../deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122
```

The regression suite covers delayed/incomplete frames, input closure, startup failures and cancellation, capacity release and parameter mapping. Smoke cases cover batched input, consecutive rounds, invalid JPEG, busy rejection and disconnect recovery; `--tests vl01,vl02` selects a subset. Smoke output alone is not proof of semantic alignment or complete external-contract conformance.

The smoke client reconnects up to three times for busy rejection before `ready`, with 0.5/1/1.5-second backoff within the shared round deadline. Retry history is reported separately from terminal errors. VL-02 disables retries to check immediate capacity release between rounds. Test clients connect directly, including when environment proxies are configured.

### Reference Latency

Smoke measurements, 2026-09-10, warm GPU backend, current revision. Times are seconds; these are observations, not latency guarantees.

| Case | Ready | First ACK | Last ACK | First text | End marker |
| --- | ---: | ---: | ---: | ---: | ---: |
| VL-01, 4 frames | 0.065 | 0.077 | 0.389 | 0.602 | 1.423 |
| VL-01, 8 frames | 0.054 | 0.078 | 0.394 | 0.468 | 0.932 |
| VL-02, second round | 0.065 | 0.077 | 0.388 | 0.468 | 2.078 |
| VL-03, recovery round | 0.057 | 0.069 | 0.381 | 0.454 | 0.756 |

Each new session still performs initial prefill. The interface targets section 5 of `model-api-protocol-asr-tts-vl.md`. That external document and the `vision.go` caller are not bundled here; final integration acceptance requires the actual caller.
