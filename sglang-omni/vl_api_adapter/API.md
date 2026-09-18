# MOSS-VL Realtime WebSocket API v2

**English** | [简体中文](./API_zh.md)

Reference for the MOSS-VL Realtime `vl-api-v2` interface. [Deployment and usage](./README.md) | [Validation](./VALIDATION.md)

**Contents:** [Connection](#1-transport-and-connection) · [Lifecycle](#2-session-state-machine) · [Client events](#3-client-events) · [Server events](#4-server-events) · [Accounting](#5-usage-and-accounting) · [Limits](#6-limits-and-capacity) · [Errors](#7-termination-and-errors) · [Sequence](#8-complete-sequence-and-client-rules)

## 1. Transport and Connection

| Item | Definition |
| --- | --- |
| Default endpoint | `ws://127.0.0.1:18610/v1/video/realtime` |
| Text message | UTF-8 JSON; one control-event object per message |
| Binary message | Complete image file bytes |
| Session | One per admitted connection; no separate creation request |
| Protocol detection | `protocol_version` and `capabilities`, not the URL path |

Native v1 defaults to port 18500 and shares the model and capacity with v2. Admission failures are defined in section 7.4.

Every control event MUST contain `type`. Client events use strict validation (`extra="forbid"`): unknown fields reject the entire event. Values must have the correct types; NaN/Infinity are rejected, and strings cannot substitute for integers or booleans. The server may add output fields; clients should ignore unknown output fields.

When `VL_API_V2_API_KEY` is configured, the handshake MUST include `Authorization: Bearer <key>`. Otherwise there is no application-level authentication; use a trusted network or an authenticated gateway. Public TLS/WSS termination is a deployment responsibility. `GET /health` provides runtime status, not a session event or accounting API.

This interface does not provide REST session creation, `task_id` allocation, reset, or cross-connection memory merging. Those belong to the caller.

## 2. Session State Machine

| Phase | Events / Behavior |
| --- | --- |
| Configure | `session.created` → client `session.configure` → `session.configured` |
| Ready | Inputs are accepted after `session.ready` |
| Inference | Input acknowledgements, text and silence; `response.done` settles one response |
| Terminal | Connection closes after `session.done`; exceptions in section 7.2 |

Do not send frames or prompts until `session.ready`. Text and telemetry may interleave with input acknowledgements. Clients MUST keep receiving all events, not wait exclusively for a particular ACK. Each input's accepted event precedes its processed event.

**Response completion is not session completion.** Model-origin silence after visible text closes that response and produces one `response.done`. Repeated silence without new visible text does not create empty responses or duplicate settlements. A period without network messages is not a response boundary.

An orderly termination with successful final accounting ends with exactly one `session.done`. No text, usage or ACK events follow it. Exceptions for unavailable final accounting and transport failure are defined in section 7.2.

## 3. Client Events

### 3.1 `session.configure`

Send exactly once after connection, within `session.created.configure_timeout_s`.

```json
{
  "type": "session.configure",
  "prompt": "Observe the video and answer my questions.",
  "max_new_tokens": 128,
  "max_tokens_per_turn": 10,
  "include_usage": true
}
```

| Field | Type | Required | Default / Meaning |
| --- | --- | --- | --- |
| `type` | string | Yes | `session.configure` |
| `prompt` | string | No | `""`; initial user prompt, not the system prompt |
| `system_prompt` | string/null | No | `null`; omitted, null or empty string uses the backend default system prompt. A nonempty string supplies a separate system prompt |
| `prefill_messages` | array/null | No | `null`; optional text history, defined below |
| `max_new_tokens` | int | No | 4096; positive generation allowance, defined below |
| `max_tokens_per_turn` | number | No | 86400; positive generation rate limit in tokens/s, not a token total or guaranteed minimum speed |
| `temperature` | number | No | 0; range [0, 2] |
| `top_p` | number | No | 1; range (0, 1] |
| `input_queue_capacity` | int | No | 4; range [1, 256], defined in section 6 |
| `include_usage` | bool | No | false; controls only optional `session.usage` telemetry |
| `benchmark_ignore_eos` | bool | No | false; true is rejected unless the server explicitly permits benchmark mode |

`prefill_messages` contains 1–64 messages, each with only `role` (`system`, `user` or `assistant`) and string `content`, totaling at most 131072 characters. When provided, it replaces the history built from the initial `prompt`. The effective system prompt is prepended if the first message is not system; an empty user message is appended if the last message is not user. Historical assistant messages receive the model's opening silence marker when needed. Actual prefill positions count as text input and remain subject to context capacity.

`include_usage` does not affect `response.done.usage` or `session.done.usage`. Settlement usage is intrinsic to the protocol, regardless of telemetry preference.

**Generation allowance:** initial input processing and each extend processing new frames or prompts set the remaining allowance to `max_new_tokens`, subject to context capacity. A value of 128 permits up to 128 further generated tokens after each extend, including silence and control tokens.

This is not a per-response or session-total cap: continued input may allow larger cumulative output. `response.done` does not replenish the allowance. The model may choose silence or end the request when the allowance is exhausted or stop conditions are met.

### 3.2 `input.frame`

Send metadata, wait for `input.frame.ready`, then send the binary image.

```json
{
  "type": "input.frame",
  "seq_no": 0,
  "timestamp": 0,
  "mime_type": "image/jpeg"
}
```

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `type` | string | Yes | `input.frame` |
| `seq_no` | int | Yes | Non-negative integer; see section 3.4 |
| `timestamp` | number | Yes | Video time in seconds; finite, non-negative and not earlier than the previous accepted frame |
| `mime_type` | string | Yes | `image/jpeg`, `image/png` or `image/webp`; must match the encoding |
| `prompt` | string/null | No | Default null; a new question accompanying the frame. Whitespace-only text is treated as no question |
| `final` | bool | No | Default false; true identifies the last input |

Fields outside this table, including `size_bytes`, are rejected.

After ready, the client MUST send that frame's binary message next; it MUST NOT send the binary before ready. Send the complete image file bytes directly, not JSON, base64 or multipart.

Only one frame handshake may be awaiting binary on a connection. Clients should wait for accepted before sending the next input. Another frame's metadata or a prompt during an unfinished handshake is rejected. Binary delivery is due within 10 seconds after ready by default. Empty, oversized, corrupt or MIME-mismatched images are rejected; see section 7.3.

### 3.3 `input.prompt`

```json
{
  "type": "input.prompt",
  "seq_no": 1,
  "prompt": "What is in the image?",
  "final": false
}
```

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `type` | string | Yes | `input.prompt` |
| `seq_no` | int | Yes | See section 3.4 |
| `prompt` | string | Yes | Nonempty, non-whitespace question; the field is `prompt`, not `text` |
| `final` | bool | No | Default false; true identifies the last input |

A prompt inherits the most recent frame timestamp; it does not accept a separate `timestamp`. A non-final question may trigger speech or model silence while awaiting more frames. Immediate visible output is not guaranteed.

After `final=true` is accepted, no further input is allowed. The model completes remaining inference under its existing generation/stop conditions and ends the session; final does not close the connection at the first silence. Wait for `session.done` or explicitly abort.

### 3.4 Sequence Numbers

Frames and prompts share one contiguous `seq_no` sequence starting at zero. Accepted input MUST NOT reuse a sequence number or skip numbers.

Only explicitly rejected, unaccepted input may retry the same sequence number after correction. Rejection does not advance the expected number. A sequence gap is rejected without closing the session. If a disconnect leaves acceptance unknown, do not assume replay is idempotent; establish a new session.

### 3.5 `session.abort`

```json
{
  "type": "session.abort"
}
```

Requests termination. It consumes no sequence number, accepts no additional fields, and does not wait for frame handshakes or input credit. When final accounting succeeds, the server responds with `session.done`, reason `aborted`. An existing fatal cause is preserved rather than overwritten with aborted. Stopping and final accounting have bounded waits, not a zero-latency guarantee; see section 7.2.

## 4. Server Events

### 4.1 `session.created`

```json
{
  "type": "session.created",
  "session_id": "video_sess_...",
  "request_id": "video_req_...",
  "model": "moss-vl-realtime",
  "turn_id": 0,
  "configure_timeout_s": 180,
  "protocol_version": "vl-api-v2",
  "model_version": null,
  "model_version_source": "unknown",
  "capabilities": [
    "session.usage",
    "response.done.usage",
    "session.done.usage",
    "error.seq_no",
    "input.frame.rejected",
    "response.done.per_response",
    "response.id",
    "usage.text_input_output"
  ]
}
```

`session_id` and `request_id` are opaque backend identifiers, not gateway task/trace IDs. This interface does not return `task_id`.

`model_version` is the deployment label supplied through `VL_API_V2_MODEL_VERSION`. Its source is `deployment` when configured; otherwise the values are null/`unknown`. It is neither the protocol version nor an automatically computed weight checksum.

Clients MUST check capabilities. Per-response accounting requires `response.done.per_response` and `response.id`; `session.usage` or the model name alone is insufficient. The configure deadline starts at backend session creation and ends when valid configuration is accepted; model prefill is not part of that deadline.

### 4.2 `session.configured`

```json
{
  "type": "session.configured",
  "session_id": "video_sess_...",
  "request_id": "video_req_...",
  "max_frame_bytes": 33554432,
  "input_queue_capacity": 4,
  "max_tokens_per_turn": 10,
  "parked_request_timeout_s": 3600,
  "context_limit": 131072
}
```

Returns effective input credit, generation rate, frame byte limit, parked timeout and actual backend context capacity. This example uses repository deployment settings, not universal constants.

### 4.3 `session.ready`

```json
{
  "type": "session.ready",
  "session_id": "video_sess_...",
  "request_id": "video_req_...",
  "turn_id": 0
}
```

The model session is ready to receive input.

### 4.4 Input Acknowledgements and Rejection

| Event | Meaning | Fields besides type |
| --- | --- | --- |
| `input.frame.ready` | Awaiting this frame's binary; backpressure signal | `seq_no` |
| `input.frame.accepted` | Image passed reception validation and was submitted to the backend | `seq_no`, `timestamp`, `final`, `pending_events`, `interrupts_current_turn` |
| `input.prompt.accepted` | Prompt submitted to the backend | `seq_no`, `final`, `pending_events`, `interrupts_current_turn` |
| `input.frame.processed` | Model processed the frame input, not just received its bytes | `seq_no`, `timestamp`, `final`, `pending_events`; may include `turn_id`, `interrupted_turn_id` |
| `input.prompt.processed` | Model processed the prompt | Same processed fields |
| `input.frame.rejected` | This frame handshake was rejected | `seq_no`, `reason` |

`pending_events` counts outstanding inputs when the acknowledgement is sent, not global sessions. Accepted does not guarantee model processing or a completed answer. `interrupts_current_turn` indicates a new question; processed/interrupted events establish the actual turn transition.

Recoverable frame rejection sends both error and input.frame.rejected. The client MUST stop waiting for that attempt's ready/processed event. A missing or unusable sequence number cannot be invented; rejected may contain `seq_no: null`. A fatal session termination ends all remaining input waits.

### 4.5 `response.text.delta`

```json
{
  "type": "response.text.delta",
  "delta": "The image shows",
  "turn_id": 1,
  "response_id": "response_...",
  "response_seq": 1
}
```

Append delta text in receive order. The first non-whitespace visible text creates the response ID. Earlier whitespace-only deltas may have null response_id/response_seq and MUST NOT establish a settled response.

### 4.6 `response.turn.silence`

```json
{
  "type": "response.turn.silence",
  "turn_id": 1,
  "seq_no": 0,
  "timestamp": 0,
  "silence_seq": 1
}
```

The model enters silence. `seq_no`/`timestamp` can be null when no input is associated yet. `silence_seq` is the backend silence sequence, not the response sequence.

An active visible response is closed by silence, followed by its response.done. Silence without new visible text does not produce response.done; its usage carries into the next settlement or final residual.

### 4.7 `response.turn.interrupted`

```json
{
  "type": "response.turn.interrupted",
  "turn_id": 1,
  "next_turn_id": 2,
  "seq_no": 1,
  "response_id": "response_..."
}
```

A new question advances the turn and interrupts the old turn. response_id identifies the interrupted active segment, or is null if none exists. Interrupted segments do not receive a normal response.done. Repeatedly interrupted sessions may have no response.done at all; final session usage is the fallback.

### 4.8 `response.done`

Completes one response, not the session.

```json
{
  "type": "response.done",
  "response_id": "response_...",
  "response_seq": 1,
  "turn_id": 1,
  "finish_reason": "stop",
  "boundary": "silence",
  "usage": {
    "vision_tokens": 145,
    "text_input_tokens": 78,
    "text_output_tokens": 10,
    "text_tokens": 88,
    "total_tokens": 233,
    "cumulative": {
      "vision_tokens": 145,
      "text_input_tokens": 78,
      "text_output_tokens": 10,
      "text_tokens": 88,
      "total_tokens": 233
    }
  }
}
```

A model-silence boundary uses `boundary=silence`, `finish_reason=stop`. If the backend naturally finishes with an active segment, it uses `boundary=request_end` and the model finish reason (`stop` for normal stopping, `length` for exhausted allowance), followed by session.done. Already settled segments are not settled again. Abort/failure does not fabricate a normal response completion.

### 4.9 `session.usage`

Optional runtime telemetry, at most 1 Hz when `include_usage=true`. It is neither a periodic heartbeat nor a guarantee that the final snapshot will be sent. It is not the accounting source.

```json
{
  "type": "session.usage",
  "encoder_tokens": 3536,
  "decoder_tokens": 329,
  "encoder_kv_tokens": 1547,
  "token_space_used": 3865,
  "context_limit": 131072,
  "context_remaining": 127207
}
```

| Field | Meaning |
| --- | --- |
| `encoder_tokens` | Historical visual positions |
| `decoder_tokens` | Backend's currently observed text positions |
| `encoder_kv_tokens` | Resident visual KV positions; may decrease with sliding-window eviction |
| `token_space_used` | Historical context-position usage, not resident GPU memory or total resident KV |
| `context_limit` | Effective configured context capacity |
| `context_remaining` | `max(0, context_limit - token_space_used)` |

Telemetry may lag settlement watermarks and MUST NOT replace accurate response/session done usage. Remaining capacity does not guarantee admission of an arbitrarily sized next input.

### 4.10 `session.done`

```json
{
  "type": "session.done",
  "session_id": "video_sess_...",
  "request_id": "video_req_...",
  "reason": "completed",
  "aborted": false,
  "usage": {
    "vision_tokens": 145,
    "text_input_tokens": 78,
    "text_output_tokens": 10,
    "text_tokens": 88,
    "total_tokens": 233
  }
}
```

usage is the cumulative final session total on the same basis as response.done.usage.cumulative, including consumption after the last response.done.

| reason | Meaning |
| --- | --- |
| `completed` | Backend ended under its normal stop/length conditions, usually after final input |
| `aborted` | Client requested termination |
| `context_exhausted` | Context capacity exhausted |
| `error` | Timeout or backend failure, preceded by error |

aborted is true only when reason is aborted. Capacity rejection precedes session creation and uses the reduced terminal format in section 7.4.

### 4.11 `error`

```json
{
  "type": "error",
  "code": "invalid_request",
  "seq_no": 12,
  "message": "expected seq_no 1, received 12"
}
```

code and message are required. An input-triggered error includes seq_no when an integer sequence number is identifiable. Connection-level errors, configuration errors and missing/unparseable sequence numbers omit it. Clients MUST NOT drive behavior by matching message strings.

### 4.12 Identifiers and Turns

- Backend-assigned turn_id starts at zero and advances when a new question is processed, including a frame carrying a nonempty question. Frames alone, silence and same-turn speech continuation do not advance it. Accepted does not mean the model has already advanced the turn.
- One turn may contain multiple responses. turn_id MUST NOT be used as the unique response.done key, and not every turn necessarily has a done event.
- response_id is created on each segment's first visible text. Associate text, interruption and settlement using session identity plus response_id.
- response_seq starts at 1 and increases within the session. Interrupted segments consume a number, so consecutive done events may skip numbers. It is distinct from contiguous accepted input seq_no.
- New connections have new session, request and response identities. Cross-session deduplication, memory and usage aggregation belong to the caller.

## 5. Usage and Accounting

### 5.1 Token Definitions

| Field | Meaning |
| --- | --- |
| `vision_tokens` | Committed logical visual encoder positions, including visual separators; based on actual input, not a fixed 145 per frame |
| `text_input_tokens` | Committed initial user/system prompts, questions, frame-associated text and actual template positions |
| `text_output_tokens` | Confirmed generated tokens, including text, silence, EOS and control tokens |
| `text_tokens` | text_input_tokens + text_output_tokens |
| `total_tokens` | vision_tokens + text_tokens |

Accounting uses actual logical model positions, not tokenizing user strings alone, image bytes, FLOPs or resident memory. A generated token reused as input on the next step is not billed again as text input. Failed, uncommitted input and discarded lookahead steps are excluded. TP ranks do not multiply usage.

### 5.2 Response Increment

```text
Increment = current cumulative - previous response.done cumulative
Sum of increments = last response.done cumulative
```

Calculate each of the five fields independently; the first settlement subtracts zero. Initial templates, intervening frames, silence and interrupted segments enter the next settlement. Consumption after the last response becomes the final residual. Both increment and cumulative are returned; clients should verify the identity to detect missing or duplicate events.

### 5.3 Cumulative Usage and Residual

cumulative and session.done.usage use historical totals: committed positions remain counted after visual KV eviction.

```text
Final residual = session.done.usage - sum of received response.done increments
```

Do not add the full session.done total to the response increments; that double counts. With no response.done, the final total covers all consumption. Reset/rollover implemented through a new connection starts a separate ledger and does not inherit the previous one automatically.

Example numbers illustrate fields and arithmetic, not fixed per-frame costs or billing bounds.

## 6. Limits and Capacity

| Item | Current Default / Source |
| --- | --- |
| Configure deadline | 180 seconds; use session.created |
| Frame byte limit | Current backend: 32 MiB; use session.configured.max_frame_bytes. Deployment ingress may impose another limit |
| input_queue_capacity | Default 4; reserved inputs not yet processed per session, including a frame awaiting binary |
| Binary deadline | Default 10 seconds after ready; image validation has a separate default 10-second processing deadline |
| Parked timeout | Low-level default 300 seconds, repository launch setting 3600 seconds; use session.configured |
| context_limit | Actual backend configuration; repository launch setting 131072 |
| Session capacity | Repository launch setting 4, shared by v1/v2, not four each |
| Visual window | Repository launcher keeps 60 seconds of raw visual KV; this does not guarantee historical context-position reclamation |

input_queue_capacity is neither the number of simultaneous frame handshakes nor an inference batch size. When credit is exhausted, new input waits for credit. Clients MUST continue reading events and obey ready/accepted backpressure rather than accumulating messages.

The parked timeout applies to a model request in silence waiting for fresh input, not total session duration or WebSocket heartbeat. Visual-window eviction is not text-memory compression. This API does not create memory/ASR/TTS services or perform application rollover.

Visual positions depend on image resolution and actual preprocessing. Do not derive universal video-duration limits using a fixed 145 tokens/frame; plan input using actual observations and capacity.

## 7. Termination and Errors

### 7.1 Error Codes

| code | Meaning | Session Continues? |
| --- | --- | --- |
| `invalid_request` | Invalid client event, sequence or recoverable image validation failure | Yes |
| `context_exhausted` | Context exhausted; further input cannot be admitted | No |
| `session_timeout` | Configure, binary reception/validation or parked-idle timeout | No |
| `response_failed` | Backend failure, input queue overflow or final-accounting failure | No |
| `session_capacity_exceeded` | No session slot at connection admission | No model session created |

Clients MUST treat unknown codes as terminal to support future additions without synchronized releases.

### 7.2 Terminal Guarantee and Exceptions

On normal completion or orderly abort, the backend resolves in-flight work and freezes usage before sending one `session.done`. Fatal errors follow `error -> session.done -> close`; `session_timeout` maps to `reason=error`.

Final-accounting acknowledgement waits up to 20 seconds by default; network sends and cleanup have additional bounded waits. Twenty seconds is not a total shutdown SLA. Clients should stop sending input immediately but continue receiving the terminal event.

> **Final usage unavailable:** if the backend cannot confirm its final snapshot, it sends a best-effort `response_failed` and closes with 1011 without `session.done`. Network or process failure may also prevent terminal delivery. Clients MUST mark final usage unknown, not zero or fully settled.

### 7.3 Recoverable Rejection

Invalid client events receive error while the connection stays open. An identifiable sequence number is included. Rejected frames additionally receive input.frame.rejected to end the handshake wait.

Rejecting another metadata message does not clear a different valid pending frame. Recoverably rejected binary releases that frame's credit; retry by sending metadata with the same sequence number. Duplicate binary and excessive queued input can instead cause fatal termination under section 7.1.

### 7.4 Admission Failure

When capacity is exhausted, the server accepts the WebSocket, sends the following events and closes with 1013. There is no session.created, no model consumption and no allocated session identifiers:

```json
{
  "type": "error",
  "code": "session_capacity_exceeded",
  "message": "video realtime service has no free session slot (capacity 4)"
}
```

```json
{
  "type": "session.done",
  "reason": "error",
  "usage": {
    "vision_tokens": 0,
    "text_input_tokens": 0,
    "text_output_tokens": 0,
    "text_tokens": 0,
    "total_tokens": 0
  }
}
```

Authentication failure rejects before WebSocket acceptance, normally as handshake HTTP 403. WebSocket error/session.done delivery is not guaranteed in that case.

## 8. Complete Sequence and Client Rules

```text
client                                      server
  |-------- WebSocket Upgrade -------------->|
  |<------- session.created -----------------| capabilities / limits
  |-------- session.configure -------------->|
  |<------- session.configured --------------| context_limit
  |<------- session.ready -------------------|
  |-------- input.frame (seq 0) ------------>|
  |<------- input.frame.ready ---------------|
  |-------- binary image ------------------->|
  |<------- input.frame.accepted ------------|
  |<------- input.frame.processed -----------|
  |<------- response.turn.silence -----------| no text: no done
  |-------- input.prompt (seq 1) ----------->|
  |<------- input.prompt.accepted -----------|
  |<------- interrupted / processed ---------| turn advances
  |<------- response.text.delta x N ---------| response A
  |<------- response.turn.silence -----------|
  |<------- response.done -------------------| settle A; stay connected
  |-------- input.frame (seq 2) ... -------->|
  |<------- response.text.delta x N ---------| response B, same turn
  |<------- response.turn.silence -----------|
  |<------- response.done -------------------| settle B
  |-------- input.prompt (seq 3, final) ---->|
  |<------- accepted / processed / output ---|
  |<------- response.done -------------------| only if a segment completes
  |<------- session.done --------------------| frozen total, residual included
  |<------- WebSocket Close 1000 ------------|
```

Responses depend on the model; an input may produce only silence. Telemetry is omitted from the diagram. Keep reading all events, associate responses by response_id, stay connected after response.done, reconcile increments and final residuals, and end pending handshake waits on terminal events or abnormal disconnects. Native v1/Demo clients MUST NOT switch to this endpoint without adapting their done handling.
