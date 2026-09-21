# MOSS-VL Realtime Demo

English | [简体中文](./README_zh.md)

A browser application for realtime video and voice interaction with MOSS-VL, with optional speech recognition, speech synthesis, and long-term memory.

Memory is scoped to the active session: reconnects within the grace period retain
it; final session closure deletes retrieval records, vectors and temporary frames.
Saved conversation archives have a separate lifecycle.

Temporal recall handles first/last events using chronological candidates; a
bounded retrieval miss does not suppress the model's existing context. Memory
data, vector caches, pending payloads and temporary frames have separate budgets.
Overload is reported; archive deletion remains opt-in. See [memory and retention
operations](./server/README.md#resource-budgets) for limits and maintenance commands.

## Related Projects

| Project | Role |
| --- | --- |
| [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) | Model weights, configuration, tokenizer, and processor |
| [sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime) | Stateful streaming inference and GPU scheduling |
| This repository | Browser UI, ASR/TTS, memory, and REST/WebSocket gateway |

Start here for the complete application. For a standalone model service, use the [backend README](https://github.com/fnlp-vision/sglang-omni-realtime#readme).

## Features

- Camera, screen, image, and video-file input.
- Streaming captions, text and voice questions, and response interruption.
- Optional SenseVoice ASR, local MOSS-TTS-Nano, and text/image memory.
- Multi-session backend access, context monitoring, and memory rollover.
- REST session management and a WebSocket gateway for external clients.

## Quick Start

Prerequisites: Linux x86_64, Python 3.12, a CUDA 13-compatible NVIDIA driver, Git, a C/C++ compiler, CMake, and FFmpeg. The installer manages application dependencies, not system packages or drivers.

From the repository root:

```bash
bash bootstrap.sh --doctor-only
bash bootstrap.sh --with-memory --with-asr --with-tts
.venv/bin/python scripts/repro/run.py up --main-gpu 0 --memory-gpu 1
```

The installer prepares Python/Node tools, separate Demo and backend environments, npm dependencies, the backend source, and models. The full configuration uses two separate GPUs. Omit all `--with-*` options for video/text only on one GPU. Existing unmanaged `.venv` directories are not overwritten.

For a source bundle containing both `demo/` and `backend/`, replace the installation command with:

```bash
bash bootstrap.sh --backend-source ../backend --with-memory --with-asr --with-tts
```

Open **http://localhost:18502**, switch to video-call mode, choose a media source, and connect. Voice input defaults to push-to-talk; select VAD for automatic capture. Remote camera/microphone access requires HTTPS. This configuration does not enable offline chat or require cloud TTS credentials.

```bash
.venv/bin/python scripts/repro/run.py status
.venv/bin/python scripts/repro/smoke.py
.venv/bin/python scripts/repro/run.py down
```

Wait for model loading and kernel compilation on first startup. Use `--base-port 19500` to move the port group; the browser port becomes `19502`.

## Context Commands

In a realtime session, send `/compact` or `/clear` as the entire text message.
Both commands interrupt generation and speech; inputs received while running
are not submitted to the model. Wait for completion before continuing.

- `/compact`: generate a summary and rebuild context, keeping session Memory
  and recent turns. Requires Memory and a configured summary service (normally
  pi_agent). Summary failure reports an error without replacing the context.
- `/clear`: discard model context, summaries, session retrieval Memory, and
  pending work. Preserve system instructions, connection/voice settings and
  chat archives; resume with newly received frames. No summary service needed.

These commands require backend session recreation support. They reduce or reset
context; they do not guarantee correction of visual hallucinations.

## Configuration

| Component | Default |
| --- | --- |
| Main VLM | GPU 0; 4 sessions; 131072 context; memory fraction 0.5 |
| Memory 4B | GPU 1; 4 concurrent requests; 16384 context; 65536 KV tokens; memory fraction 0.2 |
| ASR / local TTS / retrieval models | CPU |
| Backend / API / browser | 18500 / 18501 / 18502 |
| pi-agent / 4B / TTS | 18503 / 18504 / 18505 |

Adjust memory and concurrency for your hardware. See the [installation guide](./deployment/repro/README.md).

## Compatibility and Updates

The source manifest names repositories, not fixed backend/model commits. New
installs fetch the backend's default branch and current model repositories.
Rerunning bootstrap resumes the installed versions; `--update` explicitly refreshes
the backend and enabled models. Runtime dependency locks remain separate: Demo
uses its CPU environment, while the backend installs its own dependency lock.

| Environment | Runtime |
| --- | --- |
| Demo | Python 3.12; Torch 2.8.0 CPU; Transformers 4.57.1; Node 22.12.0 |
| CUDA backend | Python 3.12; Torch 2.11.0; Transformers 5.12.1; SGLang 0.5.16; FlashInfer 0.6.14 |
| CUDA toolkit | Compiler/CRT/NVVM 13.0.88 with CUDA 13.0 runtime |

Do not mix the original Transformers 4.57 custom model files into the SGLANG
checkpoint. Model context (256K) differs from configured service context (131072).
ASR/TTS and cross-context memory rollover belong to the Demo, not the weights.
Ascend uses the backend's separate NPU installation instructions, not these locks.

Stop the managed deployment before updating; commit or preserve local changes
before pulling. Run from this repository root:

```bash
.venv/bin/python scripts/repro/run.py down
git pull --ff-only
bash bootstrap.sh --update
.venv/bin/python scripts/repro/run.py up --main-gpu 0 --memory-gpu 1
.venv/bin/python scripts/repro/smoke.py
```

Enabled profiles are retained. Local backend edits are never overwritten.
Actual installed revisions are recorded in `.repro/managed-install.json` and
`.repro/models-verified.json` for diagnostics; these records do not pin future
updates. Source bundles use the supplied backend and are not updated from Git.
Previously downloaded snapshots are retained; budget disk space for updates.

## API and Deployment

```text
Browser -> /api/session/{id}/ws -> orchestration -> backend /v1/video/realtime
External client -> gateway /v1/realtime -> backend /v1/video/realtime
```

The browser and backend protocols are different. The web server proxies `/api` only; external gateway clients use the API port or an explicitly configured `/v1` reverse proxy. Browser and gateway pools do not share a global admission counter.

For the recommended installation (default ports):

```bash
curl --fail http://127.0.0.1:18501/api/status
curl --fail http://127.0.0.1:18501/v1/realtime/health
curl --fail -X POST http://127.0.0.1:18501/v1/realtime/sessions
```

The last command creates a thin-gateway session, not a Demo memory session;
close it after use with `DELETE /v1/realtime/sessions/{session_id}`.
Realtime frames and prompts use WebSocket, not a chat-completions curl request.
Startup prints the actual browser/API addresses and health-check commands.
For manual deployment, set `PORT` for the API and `PI_PORT` for pi-agent;
the frontend proxy and `MEMORY_PI_URL` follow those ports automatically when
their explicit URL overrides are unset. An existing `VITE_BACKEND_ORIGIN` or
`MEMORY_PI_URL` remains authoritative; remove it to restore automatic linking.
Standalone pi-agent and Demo both default to 38082. Recommended `run.py` deployment
continues to derive every component's port and URL from `--base-port`.
The backend's optional VL API v2 uses a separate listener (default 18610);
it does not replace the Demo/native endpoint and has different response-completion
semantics. Do not switch the Demo's backend URL to v2.

Services bind to loopback by default. Public deployments must add authentication, session ownership checks, TLS, and rate limits. A one-time WebSocket token is not a complete authentication system.

## Documentation

- [Gateway protocol](./docs/gateway_contract.md) and [operations / legacy environment](./docs/ops_runbook.md).
- [Memory service](./services/pi_agent/README.md) and [capacity planning](./docs/vlm_memory_capacity.md).
- [Manual/legacy deployment](./docs/manual_deployment.md) and [deployment operations](./docs/deployment_operations.md). Keep legacy HF/NPU environments separate from the recommended setup.

## Development

Install `requirements-dev.txt` in a separate development environment, then run:

```bash
python scripts/run_tests.py
python scripts/dev/check_env.py --check
npm run build
```

The test runner supports both script-based suites and pytest. Use it rather than collecting every suite with a single pytest invocation.

After building the frontend and installing Playwright's Chromium, run browser recovery checks with `node scripts/tests/realtime_recovery.cjs`. These checks use simulated media and a mocked backend.

## License

The model, backend, and vendored components retain their respective licenses. See [OpenMOSS/MOSS-VL](https://github.com/OpenMOSS/MOSS-VL) for the model project and [SGLang-Omni](https://github.com/sgl-project/sglang-omni) for the upstream inference framework.
