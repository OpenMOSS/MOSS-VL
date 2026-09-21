# Third-Party Code

Snapshots of the third-party projects MOSS-VL builds on, included in this repository with their original licenses and attribution.

## Directory Structure

```
third_party/
├── flash-attention-src/   # FlashAttention-3 backend specialized for MOSS-VL cross-attention
├── sglang/                # SGLang snapshot and usage notes for offline serving
├── sglang-omni/           # SGLang-Omni fork specialized for realtime multi-stream serving
└── realtime-demo/         # Full-stack realtime demo: browser frontend, service backend, memory, ASR/TTS
```

## flash-attention-src/

The FlashAttention-3 backend used by MOSS-VL cross-attention. It adds the `cross_kv_boundary` interface, which represents the visible KV prefix of each query row with one `int32` value instead of materializing a dense attention mask. This is a MOSS-VL specific version, not a general FlashAttention release — see [`flash-attention-src/README.md`](./flash-attention-src/README.md) for build instructions and source lineage.

## sglang/

SGLang usage notes and code snapshot for serving MOSS-VL offline. Upstream [SGLang](https://github.com/sgl-project/sglang) already supports MOSS-VL, so for new deployments you can simply pull the latest upstream — see [`sglang/README.md`](./sglang/README.md).

## sglang-omni/

A fork of [SGLang-Omni](https://github.com/sgl-project/sglang-omni) specialized for MOSS-VL realtime serving: dynamic multi-session scheduling, tensor-parallel and data-parallel (`--dp-size`) replicas, decode CUDA Graphs, and a sliding visual KV window enable concurrent multi-stream realtime inference with substantially higher throughput. It requires the SGLang-format checkpoint [`OpenMOSS-Team/MOSS-VL-Realtime-SGLANG`](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG). Copied from [fnlp-vision/sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime) — see [`sglang-omni/UPSTREAM.md`](./sglang-omni/UPSTREAM.md) for the exact snapshot.

## realtime-demo/

A complete full-stack application for realtime interaction with MOSS-VL-Realtime: a browser frontend (camera and screen sharing, streaming captions) plus a service backend (REST session API and WebSocket gateway) with a built-in memory system for long conversations. It runs against the SGLang-Omni inference backend in [`sglang-omni/`](./sglang-omni/) — together they form the complete realtime serving path from browser to GPU. Voice is covered end to end: ASR (SenseVoice) for speech input, and TTS for speech output either deployed locally (MOSS-TTS-Nano, CosyVoice) or via cloud API keys (ElevenLabs, MiniMax). See [`realtime-demo/README.md`](./realtime-demo/README.md) for the installer and deployment guide (its docs refer to the backend by its GitHub name [`fnlp-vision/sglang-omni-realtime`](https://github.com/fnlp-vision/sglang-omni-realtime) — the identical snapshot is [`sglang-omni/`](./sglang-omni/) here). Copied from [fnlp-vision/MOSS-VL-Realtime_Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo); see [`realtime-demo/UPSTREAM.md`](./realtime-demo/UPSTREAM.md) for the exact snapshot.
