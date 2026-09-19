# MOSS-VL Realtime Video Understanding

MOSS-VL Realtime serves one persistent video-understanding session over the
`/v1/video/realtime` WebSocket endpoint. A client appends timestamped image
frames and optional prompts while the same SGLang request remains alive. The
model can emit text, return `<|silence|>` and park, then wake when new input
arrives.

One instance can serve **multiple concurrent sessions**: the cap is
`--max-running-requests` (default 1), kept equal to the WebSocket session limit,
and parked sessions still occupy their slot. Once the cap is reached, a new
connection receives `session_capacity_exceeded` and is closed with code 1013.
The launch examples place all TP ranks on one host.

## Prerequisites

Install this repository following the [installation guide](../get_started/installation.md).
Use the [MOSS-VL-Realtime-SGLANG checkpoint](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG),
which includes Transformers 5.12.1-compatible configuration and processor code:

The installation guide is the single source for the hashed dependency lock,
model repository, and CUDA toolkit setup. Activate that environment and set
`MODEL_PATH` before running the examples below.

If checkpoint access returns 401/403, use an authorized Hugging Face account
(`hf auth login`). Keep the full downloaded directory together and pass its
local path with `--model-path`. The original
[MOSS-VL-Realtime](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime)
repository contains the 4.57-series reference implementation; this backend
uses the 5.12.1 compatibility package linked above.

## Start the server

The recommended single-GPU entry is `bash deployment/moss_vl_realtime/start.sh "$MODEL_PATH"`:
four sessions, 128K context, memory fraction 0.5, a 60-second raw visual window,
and port 18500. The lower-level launcher below exposes explicit overrides.

```bash
export REALTIME_FRAME_WINDOW_ENABLED=1
export REALTIME_FRAME_WINDOW_RAW_S=60
export REALTIME_FRAME_POOLING_ENABLED=0
export SGLANG_OMNI_STRICT_PORT=1

python examples/run_moss_vl_realtime_server.py \
  --model-path "$MODEL_PATH" \
  --gpu 0 \
  --host 127.0.0.1 --port 18500 \
  --context-length 131072 --mem-fraction-static 0.5 \
  --max-running-requests 4
```

For tensor parallel deployment, provide one distinct GPU per rank:

```bash
python examples/run_moss_vl_realtime_server.py \
  --model-path "$MODEL_PATH" \
  --tp-size 2 \
  --gpus 0,1 \
  --host 127.0.0.1 --port 18500 \
  --context-length 131072 --mem-fraction-static 0.5 \
  --max-running-requests 4
```

`--gpus` must contain exactly `--tp-size` unique GPU ids. Omni starts one
process per rank and exposes only that rank's GPU as local `cuda:0`. This
topology uses NCCL collectives; SGLang custom all-reduce is disabled because
its rendezvous requires distinct visible device ordinals in every rank.

Under TP>1, frame references (for example `shm://`) are resolved by rank 0
only, and the pixels are broadcast to every rank over the TP CPU group; the
single-consumer shared-memory frame lifecycle stays intact. A custom
`frame_resolver` (the scheduler's Python parameter) is likewise executed only
on rank 0 under TP.

### Data-parallel replicas (`--dp-size`)

`--dp-size N` replicates the realtime engine into N independent replicas
inside one process set. Replicas do not share KV or scheduling state: each
replica is its own TP group and owns `--tp-size` GPUs, so deployment requires
`--gpus` with exactly `--tp-size * --dp-size` distinct ids. Example,
two single-GPU replicas:

```bash
python examples/run_moss_vl_realtime_server.py \
  --model-path "$MODEL_PATH" \
  --dp-size 2 \
  --gpus 0,1 \
  --host 127.0.0.1 --port 18500 \
  --context-length 131072 --mem-fraction-static 0.5 \
  --max-running-requests 4
```

Replica semantics:

- A session is pinned to one replica for its whole lifetime (configure-time
  least-loaded admission); a parked session still holds its replica slot.
- `shm://` frame references stay single-consumer per replica: inputs of one
  session never reach another replica.
- Admission rejects with `session_capacity_exceeded` (WebSocket close 1013)
  only when every replica is full; the aggregate session cap is
  `--max-running-requests * --dp-size` (sessions per replica stay bounded by
  the scheduler's `max_running_requests`).
- Startup warmup exercises every replica; the port starts listening only
  after all replicas pass warmup.
- Two servers sharing one host must not collide on the NCCL port window;
  move this instance with `SGLANG_OMNI_NCCL_PORT_BASE` when needed.

For file/dotted configuration instead of the flag, set
`stages.moss_vl_realtime.parallelism.dp` and `stages.moss_vl_realtime.gpu`
together in one merge (config file or dotted CLI pairs). Combining `--set
stages.<stage>.gpu [...]` with `--dp-size` in the same command fails at merge
time because the GPU list only becomes valid once `dp` is applied.


The examples explicitly choose 128K context and a 0.5 memory fraction.
The launcher's code defaults are listed below. Size context and concurrency
against the available KV pool; startup reports an error if one full context
cannot fit.

| Setting | Default | Notes |
|---|---:|---|
| Context length | 262144 (256K) | Matches the checkpoint's `max_position_embeddings`; startup fails fast if the KV pool cannot hold one full session context (raise `--mem-fraction-static` or lower `--context-length`) |
| Static memory fraction | 0.40 | Adjust to available device memory together with `--context-length` |
| Concurrent sessions | 1 | `--max-running-requests N`; also the WebSocket session cap; parked sessions still occupy a slot |
| Decode CUDA Graph | On | FlashInfer decode; dynamic frame extend remains eager |
| KV page size | 1 | Fixed; realtime does not patch SGLang's paged allocator |
| Async decode | Off | Optional `--enable-async-decode` |
| Tensor parallelism | 1 | Use `--tp-size N --gpus g0,...,gN-1` |
| Data parallelism | 1 | Use `--dp-size N`; replicas pin sessions and scale the session cap by N |
| Overlap scheduling | Off | Not supported by the realtime update invariants |

Use `--disable-decode-cuda-graph` to run eager decode. When decode Graph is on,
the launcher only accepts the FlashInfer decode backend.

Before Uvicorn starts listening, the launcher opens an internal temporary
request and processes one generated 640 x 352 RGB frame through the normal
realtime update path. Startup only continues after `input.frame.processed`, so
the first external WebSocket does not pay the one-time vision/JIT setup cost.
The warmup request is then aborted and its shared-memory frame is cleaned up.
It is excluded from radix-prefix insertion and does not consume the single
user-session slot after startup, so its internal KV cannot cross into a real
streaming session.
Use `--disable-startup-warmup` only to isolate startup problems; warmup remains
enabled by default.

## Send timestamped frames

The example client defaults to **1 FPS**:

```bash
python examples/moss_vl_realtime_client.py \
  --url ws://127.0.0.1:18500/v1/video/realtime \
  --prompt "Describe relevant changes." \
  --frame /path/to/frame_000.png --timestamp 0.0 \
  --frame /path/to/frame_001.png --timestamp 1.0 \
  --frame /path/to/frame_002.png --timestamp 2.0
```

The example client schedules events with explicit timestamps relative to the
start of replay. `--fps` and `--frame-interval` are mutually exclusive and
provide fallback spacing for events without an explicit timestamp; explicit
timestamps take precedence. The example above therefore sends frames one
second apart. The existing semantic baseline uses 1 FPS.

The explicit timestamp is part of the model input. Each frame is lowered to the
same structure used by offline singleton video segments:

```text
<|vision_start|><|time_start|>1.0 seconds<|time_end|><|image|><|vision_end|>
```

## WebSocket protocol

This section describes the native endpoint, not the optional VL API v2 adapter
or the Demo browser protocol. The separate v2 listener has per-response
`response.done` semantics; native `response.done` terminates the persistent session.

### Restoring text history

`GET /v1/video/realtime/capabilities` advertises `prefill_messages: true`.
After probing support, a client may provide optional `prefill_messages` in
`session.configure`: 1-64 `{role, content}` objects, with roles `system`, `user`,
or `assistant` and string content totaling at most 131072 characters.
The configured model context limit still applies. The server restores assistant
openers; this is text prefill, not recovery of cached KV or past visual tensors.
When supplied, the history replaces the initial `prompt` input; include any active
task in that history. Without the field, the existing configure behavior remains.

### Initial configuration

`session.configure` accepts `prompt` (the initial user prompt), whose default
is an empty string, matching the Transformers realtime reference. Production
deployments are expected to pass the SFT prompt explicitly; the system prompt
default is byte-identical to the reference implementation.

`session.configure` accepts `max_tokens_per_turn`, matching the Transformers
realtime API. Despite its historical name, this is a generation-rate cap in
tokens per second, not a per-turn token-count limit. Its default `86400` is
effectively unlimited. `max_new_tokens` is the decode allowance re-anchored
after every input extend, rather than a lifetime session budget.

That allowance is capped by the remaining session context. Reaching the
generation limit ends the session with a length finish reason. Appends and
decode allocations also check historical token positions and physical KV
capacity, including pending/lookahead work, before writing the page table.
An input that cannot fit is rejected through the existing session-error path;
other sessions retain their own state and capacity.

Rate limiting happens in the scheduler before model execution. The request
keeps its KV allocation while ordinary decode waits, and the event loop
continues accepting frames and prompt interrupts. Frame/prompt extend always
takes priority over the decode-rate deadline, so a low output rate does not add
the same delay to user input handling. All TP ranks receive the same request
configuration and rank 0 broadcasts scheduler inputs to keep their forward
sequence aligned.

A normal frame follows this sequence:

1. Server sends `session.created`.
2. Client sends `session.configure`.
3. Server sends `session.configured`, then `session.ready` after initial prefill.
4. Client sends `input.frame` JSON metadata.
5. Server sends `input.frame.ready` when the bounded queue has capacity.
6. Client sends the binary image payload.
7. Server sends `input.frame.accepted` after reliable submission.
8. Server sends `input.frame.processed` after incremental forward and KV commit.
9. Text arrives through `response.text.delta`; terminal requests end with
   `response.done` and `session.done`.

`session.configured` also advertises two operational limits:
`max_frame_bytes` (the largest binary frame the server accepts; the transport
is configured to allow it end to end) and `parked_request_timeout_s` (the idle
timeout after which a silence-parked request is aborted and the session ends).

`session.created.configure_timeout_s` advertises the deadline for a valid
`session.configure` (180 seconds by default, starting when the connection is
admitted). Invalid messages do not reset it. Expiry returns
`configuration_timeout`, closes the connection, and releases its session slot.
The deadline does not limit model prefill after configuration. Applications
embedding the API can set `create_app(video_realtime_configure_timeout_s=...)`.

Control reception remains active while ordered input processing waits for
capacity, so `session.abort` and disconnects do not wait for a model input ACK.
The transport-side staging queue is also bounded: at most one metadata/binary
pair per configured input credit plus one additional pair. Exceeding that
queue returns `input_queue_full` and ends the session; oversized binary messages
return `frame_too_large` and end the session. Clients should respect the ready
and accepted acknowledgements instead of sending an unbounded input burst.

Frame metadata contains `seq_no`, video `timestamp`, optional `prompt`, `final`,
and `mime_type`. Prompt-only updates use `input.prompt` and the same ordered
sequence number space.

Event ordering is validated at the WebSocket edge: `seq_no` must be dense,
starting at 0 for the session, and frame timestamps must be
non-decreasing. A violating event is rejected with a per-event
`invalid_request` error and the session stays alive. Input submission failures
and engine failures terminate the session; explicit abort, disconnect and
parked timeout also end it.

### Context usage and completion

`session.created.capabilities` includes `session.usage`. Set `include_usage`
to `true` in `session.configure` to receive these resource updates:

| Field | Meaning |
| --- | --- |
| `decoder_tokens` | Decoder positions, including a sampled token awaiting its next forward |
| `encoder_tokens` | Historical encoder positions appended to the request |
| `encoder_kv_tokens` | Encoder positions currently retained in the KV cache |
| `token_space_used` | Historical encoder positions plus decoder positions |
| `context_limit` / `context_remaining` | Configured context and remaining positions |

`response.done` and `session.done` end the persistent session. A `turn_id` can
contain several text segments separated by `response.turn.silence`; it is not
a separate ID for every proactive answer. A final input closes the input
stream and lets generation finish. `session.abort` explicitly stops the whole
session. Handle errors and use a completion timeout in client applications.

### Text barge-in and turns

There is no separate turn-interrupt input. Any `input.prompt`, or an
`input.frame` carrying a non-empty `prompt`, atomically closes the current
assistant turn and opens the next user/assistant turn in the same persistent
request. The appended training-format text is:

```text
<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<|silence|>
```

The scheduler applies the update at the next safe token boundary, retains the
existing text and vision KV, and commits the prompt extend before confirming
the transition. It does not abort the request or recreate the WebSocket.
Already-running CUDA work is cooperative rather than forcibly cancelled.

All events queued at one scheduler step are drained into a single segment,
matching the Transformers reference drain: prompts are spliced first in
arrival order and every prompt is followed by `<|silence|>` (the trained
assistant-turn opener), including prompt-only updates; frames
follow, sorted by timestamp. Multiple prompts in one cycle therefore transition
through consecutive turn ids, and a frame that races with a prompt is appended
after the prompt text.

The observable event order is:

```text
input.prompt.accepted (or input.frame.accepted)
response.turn.interrupted  turn_id=N, next_turn_id=N+1
input.prompt.processed (or input.frame.processed)  turn_id=N+1
response.text.delta  turn_id=N+1
```

`session.created`, `session.ready`, every text delta, and `response.done` carry
a `turn_id`. Clients should stop rendering or speaking the old turn as soon as
they submit the new prompt, and use the server-confirmed
`response.turn.interrupted` boundary to reject any stale turn output.

### Backpressure

The input queue defaults to 4 events and is configurable from 1 to 256. When it
is full, the server delays `input.frame.ready`; a conforming client keeps the
next frame on the producer side until capacity is released. The server does not
silently drop frames and does not create an unbounded overflow queue.

If a live camera permanently produces faster than the model consumes, the
producer must slow down or provide its own bounded buffering/storage policy.
Finite memory, constant capture FPS, and unlimited no-drop retention cannot all
be guaranteed simultaneously.

## Concurrent sessions and KV pressure

With `--max-running-requests N` (N > 1) the instance serves N concurrent
sessions, each an independent persistent request; parked sessions still occupy
their slot, so the WebSocket session cap is kept equal to N. Sessions do not
share prefix KV (radix insertion is skipped for realtime requests), so each
session pays its own prefill.

Capacity ordering is:

- **Admission**: when all slots are busy, a new WebSocket receives
  `error(session_capacity_exceeded)` and is closed with code 1013.
- **Startup precheck**: boot fails fast when the KV pool cannot hold one full
  session context; oversubscribing sessions (`N * context_len > pool`) logs a
  warning, because the runtime then relies on the degradation below.
- **Runtime degradation**: when the KV pool cannot fit the next extend/decode,
  the scheduler aborts the currently heaviest live session first, including
  silence-parked sessions that retain KV (its client
  receives `error(response_failed)` and a closed connection) until the batch
  fits again. A session aborted this way must reconnect as a new session and
  re-push its stream from scratch.

Under concurrent load the decode-rate cap (`max_tokens_per_turn`) becomes a
soft target: a step runs as soon as any session in the batch is due, letting
not-yet-due sessions ride along.

Completed prefills are handed off to the running batch before the decode rate
gate. New prefills and frame updates are not delayed by that gate. See the
[capacity planning guide](moss_vl_realtime_capacity.md) for memory rollover,
deployment sizing and links to the standard performance results.

## Benchmark mode

`benchmark_ignore_eos` exists only to collect a fixed number of decode tokens.
Production servers reject it. A benchmark server must be started explicitly:

```bash
python examples/run_moss_vl_realtime_server.py \
  --model-path "$MODEL_PATH" \
  --context-length 131072 --mem-fraction-static 0.5 \
  --enable-benchmark-mode
```

Do not enable benchmark mode for production traffic.

## Vision KV sliding window (opt-in)

The lower-level launcher leaves the window disabled unless configured;
`deployment/moss_vl_realtime/start.sh` enables a 60-second raw window.
To configure it explicitly with the lower-level launcher, set:

```bash
export REALTIME_FRAME_WINDOW_ENABLED=1
export REALTIME_FRAME_WINDOW_RAW_S=60
export REALTIME_FRAME_POOLING_ENABLED=0
```

Pooling is a separate experimental option, also disabled by default. Setting
`REALTIME_FRAME_POOLING_ENABLED=1` allows full groups of aged raw frames to
become mean-pooled virtual frames. `REALTIME_FRAME_POOL_RATIO` controls group
size, and `REALTIME_FRAME_POOL_WINDOW_S` controls virtual-frame retention.

Raw/pool window durations and `--parked-request-timeout` must be finite,
positive numbers. NaN and infinity are rejected at initialization.

Window behavior:

- **Visual context:** The raw-only window removes aged visual context. Pooling additionally averages
  RoPE-rotated keys and values from different frames. Both change the visual
  history available to attention and can change the output.
- **Context accounting:** Token space keeps one pad placeholder per historical
  encoder slot. The context guard therefore still counts these positions after
  physical KV slots have been reclaimed.
- **Pooling under pressure:** If a pooled copy cannot be allocated, the round
  evicts aged frames directly and logs "degraded to eviction".

## Validation

Run the model-specific and serving tests:

```bash
python -m pytest -q \
  tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/client/test_control_event.py
```

Use GPU model-step and processor tests for reference comparisons, and fix the
model revision, prompts, timestamps and token boundaries when comparing eager,
CUDA Graph and async decode. Live input arrival can place new frames at
different generation boundaries.

Include `tests/unit_test/pipeline/test_async_decode.py` when validating async
decode. CUDA-dependent tests require a GPU; reference-model tests also require
the corresponding model path. Exercise single-device and TP deployments
separately.

Deployment validation should include mixed long/short-session workloads and
completion timeouts under accelerated input. Measure input processing and
session completion separately: processing every input does not by itself
guarantee that the session has finished. See the
[capacity reference](moss_vl_realtime_capacity.md) for workload conditions.
