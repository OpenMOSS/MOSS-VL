# `server/` — MOSS-Realtime backend

FastAPI gateway for the streaming video-understanding demo. Ports the MOSS-VL
realtime loop from the reference board backend; ASR/TTS/VLM sit behind pluggable
adapters. **Frontend serving is intentionally not wired yet** — this exposes the API
only.

## Architecture (after the overhaul)

One **server-orchestrated session** replaces the browser-stitched 3-socket design:

```
POST /api/sessions  ──▶  SessionManager ──▶ Orchestrator (async task per session)
                                     VAD/PTT → ASR → MOSS-VL realtime → TTS
WS /api/session/{sid}/ws  ◀── captions (response.text.delta) · TTS PCM (binary 0x11)
   mic PCM (0x01) · JPEG (0x02) · JSON control ──▶
```

- Wire protocol: `server/protocol.py` (frozen).
- The orchestrator owns barge-in (soft VLM interrupt that keeps the KV cache),
  control-token mapping, and the TTS back-pressure/drop-stale policy
  (`asr-tts_research.md` §1C hooks: `audio_queue_seconds` / `should_emit_next_unit`
  / `drop_stale`). Board-parity turn lifecycle: `<|silence|>` is the model's
  end-of-round signal — it finalizes the open response (one transcript bubble
  per round; the token itself is never surfaced) — and frames stream to the
  model continuously even mid-generation (backpressure = the frame queue's
  drop-oldest-pure-frame policy only).
- Reconnect: `?last_seq=N` replays missed events from a per-session ring buffer;
  a dropped socket keeps the session alive for `SESSION_GRACE_SECONDS` (45 s).
- The legacy 3-channel routers (`voice_ws`/`realtime_ws`) are **removed** — the
  frontend speaks the session plane (overhaul §5 step 4 complete).

## Context Commands

Send `{"type":"text.input","text":"/compact"}` or
`{"type":"text.input","text":"/clear"}` on the session WebSocket.
Only an exact, trimmed command is intercepted; it is not a model prompt.
The server emits `context.command` with `command` and `status`:
`running`, then `completed` or `failed` (with `message`). Events are replayable
and journaled. During execution, mic/video frames are dropped and other control
inputs are rejected with `context_busy` (ping/playback telemetry remain allowed).

`/compact` requires Memory and a summary provider; it summarizes and recreates
the model session while retaining retrieval Memory. `/clear` needs only session
recreation support: it removes KV, summary, session Memory and pending work,
preserving the system prompt, transport/settings and history archive. The old
initial user prompt and cached frame are not replayed on clear. Subsequent frames
resume vision input. On clear failure, retry `/clear` or create a new session;
cleared memory is not restored. No sglang-omni protocol change is required.

## Run

```bash
cd <repo>            # MOSS-VL-Realtime_Demo_App
# Models default to the repo-local, gitignored models/ zoo (grouped by kind):
#   models/asr/{SenseVoiceSmall,fsmn-vad}
#   models/tts/MOSS-TTS-Nano-100M
#   models/vlms/{hf_mossvl_streaming_processor (online), offline}
# Populate models/ (or point MODELS_DIR at a zoo laid out the same way), then boot:
GPU_ID=0 AUTOLOAD_VLM=1 \
.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
# Override the online checkpoint explicitly with MODEL_PATH=… if it lives elsewhere.
```

Without `AUTOLOAD_VLM=1` the app boots with no model; load one later via
`POST /api/models/load {model_path, gpu_id, hf_mode}` (legacy alias
`/api/load_model`; refused with 409 while sessions are live).

For the manual/legacy entry below, TTS runs as a separate **sidecar/engine** on `http://127.0.0.1:18100+`
(spawned + health-gated by the backend lifespan), or set `TTS_ENABLED=0`
(sessions then stream captions without audio). `TTS_PROVIDER` picks the model
(per-model adapters under `server/adapters/tts/`): `moss_tts_nano` (pytorch
sidecar) · `vllm_omni` (Nano on vLLM-Omni) · `cosyvoice3` /
`moss_tts_realtime` (Fun-CosyVoice3-0.5B / MOSS-TTS-Realtime on vLLM-Omni,
24 kHz mono) · `cosyvoice3_native` / `moss_tts_realtime_native` (vendored
fallback stacks in `.venv-cosy` / `.venv-mossrt`).

The recommended `scripts/repro/run.py` deployment instead uses TTS port 18505,
API 18501 and browser 18502 by default; `--base-port` moves the whole group and
updates companion URLs together. It does not read `.env.deploy`.
For manual deployment, `PORT` drives the frontend proxy unless
`VITE_BACKEND_ORIGIN` is explicitly set. `PI_PORT` drives the default
`MEMORY_PI_URL`; standalone pi-agent and Demo both default to 38082.

## Online vs offline serving (two models, split GPU fleet)

Two checkpoints serve two surfaces (paths default in `config.py`, under the
grouped `models/vlms/` zoo):

- **ONLINE** (`MODEL_PATH`, default `models/vlms/hf_mossvl_streaming_processor` —
  the streaming HF bundle, weights + processor): realtime streaming sessions on
  the replica-per-GPU HF worker fleet (`adapters/vlm/moss_vl_hf/online_pool.py`).
- **OFFLINE** (`OFFLINE_MODEL_PATH`, default `models/vlms/offline`): the chat page
  (`POST/WS /api/chat/stream`) on dedicated `sglang.launch_server` sidecars
  (`adapters/vlm/moss_vl_sglang/adapter.py`, ports 30800+) — the fnlp-vision sglang
  fork from `.venv-sglang` (`scripts/build_venv_sglang.sh`; stock sglang cannot
  serve `moss_vl`).

`gpu/placement.py` carves the eligible GPUs ≈1:3 offline:online (8→2+6,
4→1+3, 2→1+1, 1→online only; `OFFLINE_GPU_RATIO`/`OFFLINE_GPU_COUNT`), giving
offline the highest indices; ASR/TTS colocate with online workers only. When
the plane is absent or down (no `.venv-sglang`, sidecar crash, 1-GPU box),
`routers/chat.py` falls back to the online pool — pre-split behavior —
and `/api/status` shows `vlm_offline.loaded: false`.

## Tests

```bash
V=.venv/bin/python
$V -m server.tests.test_protocol         # B1 wire-format round-trips
$V -m server.tests.test_session_manager  # B2 lifecycle + grace GC + supersede
$V -m server.tests.smoke_orchestrator    # B3 scripted turns, barge-in, §1C hooks
$V -m server.tests.test_session_ws       # B4/B5/B7 live uvicorn + websockets client
$V -m server.tests.test_asr_partials     # SenseVoice partial-decode worker (stub engine)
$V -m server.tests.test_persistence      # journal+index recording, CAS, delete, rebuild
$V -m server.tests.smoke_media           # upload/dedup/reject + ETag/Range + history HTTP
$V -m server.tests.test_gpu_planning     # topology, attn select, online/offline GPU split
$V -m server.tests.test_vlm_online_pool  # online replica pool state machine
$V -m server.tests.test_tts_providers    # TTS provider registry/sizing/wire protocols
$V -m server.tests.test_offline_sglang   # sglang adapter: SSE deltas, media prep, failover
$V -m server.tests.test_vlm_workers      # gateway + 4 fake worker processes E2E
ASR_ENABLED=0 TTS_ENABLED=0 AUTOLOAD_VLM=0 $V -m server.tests.smoke_import
```

All of the above run **without a GPU** (fake engines). Against a live backend
(real model on Box 2):

```bash
$V scripts/e2e_session.py --text "画面里是什么？" --frames 3         # typed turn
$V scripts/e2e_session.py --wav question_16k.wav --frames 3        # voice turn
# prints captions + asr/ttft/ttfa latencies, saves the reply to e2e_reply.wav
```

## Endpoints

| surface | path | purpose |
|---|---|---|
| HTTP| `POST /api/sessions` · `GET/DELETE /api/sessions/{sid}` | session lifecycle (409 = model occupied) |
| WS  | `/api/session/{sid}/ws?last_seq=N` | **the** realtime socket: mic/video/control in, captions/PCM/events out |
| HTTP| `POST /api/models/load` | load/replace the VLM (409 while sessions live) |
| WS/SSE | `/api/chat/stream` | offline multimodal chat (separate from realtime by design); **multi-turn**: `messages` carries the whole conversation, with per-turn images as `{"type":"image","media":"sha256:…"}` parts (data-URLs also accepted) and per-turn videos as `{"type":"video","media":"sha256:…"}` parts (CAS handles ONLY — the adapter resolves the blob path and torchcodec samples frames, fps/min/max_frames tunable via `params`); legacy top-level `images`/`videos` still attach to the last user turn; `conversation_id` opts the turn pair into history |
| HTTP| `POST /api/media` · `GET /api/media/{hash}[/info\|/thumb]` | CAS upload (multipart image/video) + immutable serving (ETag, Range/206); `/info` = descriptor probe so a client that hashed its video locally can skip a duplicate transfer (404 = upload it) |
| HTTP| `POST /api/asr` (raw PCM16 mono 16 kHz) · `POST /api/tts` ({text, voice?} → WAV) | one-shot speech for the CHAT page: rolling dictation + bubble read-aloud (the realtime plane keeps its own ASR/TTS on the session socket) |
| HTTP| `GET/DELETE /api/history[/{cid}]` (`?q=` = FTS search, `?kind=chat\|realtime` = facet) | recorded conversations: sidebar list, transcript, delete |
| HTTP| `/api/status` (incl. voice + live sessions) · `/api/health` | ops |

Wire-message details: `server/protocol.py` docstring.

## Config (env)

Config resolves in **four layers**, highest priority first (the docker-compose
CLI > environment > env_file > image-ENV model; full contract in the
`server/config.py` docstring):

1. **command env** — `MODEL_PATH=x demo.sh up` or `demo.sh up MODEL_PATH=x`
   (trailing `KEY=VALUE` args; args beat the prefix)
2. **startup-script pins** — the marked layer-2 blocks in
   `scripts/deploy/run_*.sh` (`: "${VAR:=value}"; export VAR` idiom)
3. **`.env.deploy`** — plain KEY=VALUE at the repo root (gitignored; complete
   generated reference in `.env.deploy.example`). Applied with *setdefault*
   semantics by both `get_settings()` and `scripts/deploy/env_lib.sh` — a set
   env var always wins. `ENV_DEPLOY_FILE` overrides the path; empty disables
   (tests pin it empty via `server/tests/__init__.py`).
4. **code defaults** — `server/config.py` (the single source of defaults;
   deploy scripts are orchestration-only)

After adding/renaming a setting run `scripts/dev/check_env.py --write` — it
regenerates `.env.deploy.example` and `scripts/deploy/env_manifest.sh` (the
vars demo.sh forwards into its tmux windows); `--check` lints drift and is
covered by `server/tests/test_config_layering.py` territory.

All knobs live in `server/config.py`. Key ones: `MODEL_PATH`, `GPU_ID`, `HF_MODE`,
`MOSSVL_STREAMING_PROCESSOR_PATH`, `ATTN_IMPL` (+`ATTN_IMPL_FALLBACK=sdpa`),
`ASR_PROVIDER`/`ASR_DEVICE`/`ASR_FP16`, `TTS_PROVIDER`/`MOSS_TTS_NANO_BASE_URL`,
`CAPTURE_MODE` (`ptt`|`auto`), `VOICE_ASR_AUTO_*` thresholds.
`ASR_PARTIAL_INTERVAL_MS` (default 800, 0 = off) drives realtime captions:
SenseVoice has no native streaming, so the stream re-decodes its buffer every
interval and emits `input.transcription.delta` through the standard
`on_partial` adapter hook — a genuinely streaming engine later is a pure
adapter swap.

Session-plane knobs: `SESSION_GRACE_SECONDS`, `SESSION_REPLAY_BUFFER`,
`STATUS_INTERVAL_S`, `FRAME_MAX_AGE_S`; TTS pacing policy:
`AUDIO_BUFFER_HIGH_S`/`AUDIO_BUFFER_LOW_S`, `TTS_MAX_PENDING_UNITS`,
`TTS_UNIT_MAX_AGE_S`, `TTS_FIRST_CLAUSE_CHARS`, `TTS_SEGMENT_MIN/SOFT/MAX_CHARS`
(now 12/75/150 per `asr-tts_research.md` §1A T4).

Persistence knobs: `HISTORY_ENABLED`/`MEDIA_ENABLED` (both default 1), `DATA_DIR`
(default `<repo>/data` — the shared FS survives pod restarts; /tmp does not),
`HISTORY_DB_PATH`, `UPLOAD_MAX_BYTES` (512 MB), `MEDIA_MIME_ALLOW`,
`MEDIA_THUMB_MAX_EDGE`, `HISTORY_KEEP_FRAMES` (0 — camera frames are never stored).

## History + media store (`server/persistence/`)

Runtime retrieval memory has a separate lifecycle from these archives. On final
session close (explicit close or reconnect-grace expiry), the gateway rejects
new memory work, waits for admitted worker operations, and deletes that session's
rows, vectors, index keys, and RAM caches from `memory.db`. Detaching a socket
within its reconnect grace period does not clear memory. Cleanup is idempotent;
failures are logged and raised rather than reported as successful cleanup.

The initial system prompt remains the model default or the user's original
prompt; memory does not append instructions to it. Candidate verification considers recent corrections before context
deduplication; acknowledged user input and completed assistant output are also
tracked as already present in the current context. Rollover resets that tracking
to the content actually retained. A selected still image remains valid until the
source changes; camera/screen frames retain their normal freshness limit.

With the default system prompt and no initial task, an explicit history question
with no prior records produces `memory.notice {code: "no_history", message: "..."}`.
A bounded search or candidate-verification miss does not prove that the complete
history or native model context lacks evidence. Such misses are logged and do not
veto the model turn. The client displays and archives absence-of-history notices
as system messages, not model answers. Each notice includes a `response_id` and
is followed by `response.created`, empty `response.text.done`, and `response.done`
with `source: "system"`. Ordinary questions still go to the model.
Deploy the updated frontend and API together.

Earliest/first/latest/last event questions use a bounded chronological text scan
in addition to semantic search, not a fixed first-minute window. Batches are
verified in the requested time order before context deduplication. Whole-record
timestamps and original within-record order are preserved; precise event times
are not invented. Repeated identical observations retain first/last observation
times without duplicate vectors; historical QA retellings are excluded from
event-order scans. Pre-upgrade rows lack those annotations and cannot reconstruct
lost repeat times. The default scan is 128 records / 128 KiB under the existing
8-second decision budget. Missing timestamps, truncated text, scan exhaustion and
selector errors are not proof of absence. Diagnostics include candidate IDs,
timestamps, lengths and stage outcomes, not additional history bodies.

Long records are split into budgeted original-text spans before temporal evidence
selection; no generated summary replaces the source. Injection coverage is tracked
per span so retaining one excerpt does not suppress a different excerpt from the
same record. The frontend continues to show the original source record.

### Resource Budgets

Limits apply to retrieval memory, not model weights or total process RSS. SQLite,
NumPy and Python may retain reusable allocations after a session closes.

| Resource | Default | Configuration |
| --- | --- | --- |
| Stored items, session / total | 5000 / 20000 | `MEMORY_SESSION_MAX_ITEMS`, `MEMORY_TOTAL_MAX_ITEMS` |
| Text/vector/key bytes, session / total | 64 / 256 MiB | `MEMORY_SESSION_MAX_BYTES`, `MEMORY_TOTAL_MAX_BYTES` |
| Retained vector cache | 128 MiB | `MEMORY_CACHE_MAX_BYTES` |
| Queued + in-flight payload, total / session | 32 / 8 MiB | `MEMORY_QUEUE_MAX_BYTES`, `MEMORY_QUEUE_SESSION_BYTES` |
| Temporary frames, session / total | 128 / 512 MiB | `MEMORY_SESSION_FRAME_BYTES`, `MEMORY_TOTAL_FRAME_BYTES` |
| One frame / text record | 2 MiB / 8192 characters | `MEMORY_FRAME_MAX_BYTES`, `MEMORY_ITEM_MAX_CHARS` |
| WAL pressure threshold | 128 MiB | `MEMORY_MAX_WAL_BYTES` |
| Tracked active/pending-cleanup sessions | 128 | `MEMORY_TRACKED_SESSIONS` |
| Fact tasks / interrupted-text buffer per session | 8 / 64 KiB | `MEMORY_BACKGROUND_TASKS`, `MEMORY_UNCOMMITTED_BYTES` |
| Injection per turn / session lifetime | 180 / 1200 estimated tokens | `MEMORY_INJECT_MAX_TOKENS`, `MEMORY_INJECT_SESSION_MAX_TOKENS` |

Over-budget writes are rejected, existing evidence is retained, and resource
status/counters (`/api/status.memory` and `/api/status.memory_writer`) expose
degradation, cached bytes and pending payloads. Cache eviction removes only recomputable
RAM indexes, not stored evidence. The status byte count excludes Python object
overhead and temporary embedding/search workspace; it is not an RSS hard quota.
Database/free-disk/WAL checks remain periodic soft admission controls. Online WAL
checkpoint/truncation does not wait for old readers; persistent readers may keep
admission paused until the next maintenance pass can reclaim the WAL.

History recording has separate limits: `HISTORY_QUEUE_BYTES` (16 MiB),
`HISTORY_QUEUE_ITEMS` (2048), and `HISTORY_PENDING_SESSIONS` (128). A session whose
archive overflows stops accepting further archive events, logs the failure and
is finalized with an `archive_incomplete` marker. `/api/status.history` exposes
queue use and errors. Finalization has bounded reserved bookkeeping; shutdown
waits for accepted writes instead of closing the index under an active writer.

On context rollover, the gateway probes `/v1/video/realtime/capabilities` and uses
role-preserving `prefill_messages` when supported. Update the realtime backend
alongside the gateway to enable this path. Older backends receive the legacy
text prompt without unsupported fields; they cannot preserve native history roles.

New memory keyframes live in session-private directories beside the memory DB
(`memory.db.frames/`), not in the history CAS. Closing a session removes its
temporary frames; history journals, uploads and shared model weights remain.
The gateway holds an exclusive process lock on its memory DB. API instances
must use separate memory DB paths; do not remove lock files to bypass ownership.

Startup recovers owned sessions left by a previous process. Failed cleanup stays
in a persistent retry queue and is retried every `MEMORY_MAINTENANCE_INTERVAL_S`
(default 30 seconds). Pre-upgrade records without ownership metadata require
explicit offline migration and are not automatically deleted.

`MEMORY_MIN_FREE_BYTES` (default 1 GiB) and `MEMORY_MAX_DB_BYTES` (default 2 GiB
of occupied SQLite pages) pause new memory writes under storage pressure; normal
inference continues. The DB limit is a soft admission bound, not a filesystem
quota or a cap on media size. Admission resumes after the next successful check.
Filesystem capacity checks run in periodic maintenance, outside the retrieval
database lock; individual writes use the cached admission state. Online
maintenance checkpoints the WAL without a full vacuum. Freed DB pages
are reusable; use offline vacuum when physical file shrinkage is required.

Preview runtime-memory cleanup (read-only by default):

```bash
.venv/bin/python scripts/memory_maint.py --db data/memory.db
```

After stopping every API that uses the database, migrate old records and reclaim
space. This creates a database backup before deletion; it does not back up or
delete shared CAS uploads. The backup itself needs disk space and should be
retained or removed according to your operational retention policy.

```bash
.venv/bin/python scripts/memory_maint.py --db data/memory.db --apply --offline --include-legacy --vacuum
```

Legacy shared-media cleanup remains a separate offline operation. The pruner
checks both history references and remaining memory references; failed file
deletion does not remove the corresponding media index row.

```bash
.venv/bin/python scripts/history_prune.py --unreferenced --dry-run
.venv/bin/python scripts/history_prune.py --unreferenced --offline --older-than 1
```

### Archive Retention

Retention is opt-in and offline. `scripts/history_retention.py` previews only by
default; `--apply --offline` plus an explicit policy is required for deletion.
It selects closed conversations oldest first, protects remaining history/memory
media references, and refuses to run while an updated API holds its archive lock.
Age is based on `ended_at`; unclosed records are preserved. `--max-bytes` targets
journal/media payload, not allocated SQLite pages, caches, backups or log files.
An unmet target (protected content or batch limit) is reported rather than hidden.
No automatic archive-deletion schedule is installed.

```bash
.venv/bin/python scripts/history_retention.py --days 90
# Only after approving that policy and stopping all APIs:
.venv/bin/python scripts/history_retention.py --days 90 --apply --offline
```

Two durable archives under `DATA_DIR` — nothing auto-evicts:

- **History**: append-only JSONL journal (`journal/YYYY/MM/<cid>.jsonl`, the source
  of truth) + a derived SQLite index (`index.db`: conversations/turns/FTS5-trigram).
  Realtime sessions record automatically (the recorder taps `SessionState.emit`;
  turn-level events only — text deltas and PCM are never journaled). Chat threads
  record when the client sends a `conversation_id`. Live (`kind=realtime`) and
  offline (`kind=chat`) histories are separate lists (`GET /api/history?kind=`);
  live replays are read-only by design. File-streaming sessions additionally
  record the source video: the client uploads it to the CAS and sends
  `video.attach {media}` on the session WS (→ journaled `input.video.attached`,
  a media-only turn), and every turn carries `media_ts` — the video position it
  happened at — in its metrics, so the replay UI can seek the stored video. The
  session's `video_source` config field (`camera`/`file`) tags the sidebar entry.
- **Media**: content-addressed blobs (`media/blobs/sha256/ab/cd/<hash>`) for uploads —
  dedup for free, immutable HTTP caching. Ingest is hardened per OWASP: magic-byte
  sniffing (client MIME is never trusted), MIME allowlist, streaming size cap, and
  images are **re-encoded** (EXIF/GPS/polyglot strip) before hashing. Video metadata
  and poster frames come from torchcodec (needs the FFmpeg libs on
  `LD_LIBRARY_PATH`, see `run_backend.sh`; degrades gracefully without).

Writes never touch the event loop: one daemon writer thread owns the SQLite write
connection + journal appends (producers only enqueue); media ingest runs via
`asyncio.to_thread`. Maintenance (backend stopped):

```bash
.venv/bin/python scripts/history_prune.py --unreferenced --offline [--older-than 30]  # GC unref'd blobs
.venv/bin/python scripts/history_prune.py --drop-conversations --older-than 90 --offline
.venv/bin/python scripts/history_prune.py --rebuild --offline   # index.db from journals + blob scan
```

## Layout

```
app.py · config.py · deps.py · schemas.py · protocol.py · logging_conf.py
session/     state · manager · orchestrator      ← the overhaul (§B1–B7)
persistence/ store · recorder · media · ingest   ← history journal+index, CAS media
routers/     sessions · session_ws · chat · history · media · ops
adapters/    base · registry · asr/ · tts/ · vlm/ (per-model subdirs)
realtime/    session · mossvl_patches
voice/       segmenter · tts_session · vad
tests/       test_protocol · test_session_manager · smoke_orchestrator ·
             test_session_ws · test_asr_partials · test_persistence · smoke_media · smoke_import ·
             smoke_asr · smoke_realtime · test_tts_providers · fakes
```

## Frontend counterpart

`src/lib/{sessionSocket,micWorklet,frameSampler,pcmPlayer}.ts` +
`src/hooks/useSession.ts` + `public/worklets/pcm-worklet.js` implement the
client half (overhaul §F1–F5); `src/App.tsx` binds them (§F6). Vite proxies
`/api` (HTTP + WS) to `:8000` on both `dev` and `preview`, so the one gateway
forward link `…/proxy/5173/` serves page and backend together (§8 decision 1).
Restart `npm run preview` after pulling this change — the proxy config loads
at server start.
