# Third-Party Components

Vendored third-party code used by MOSS-VL. Each directory is a full snapshot of its upstream project, bundled with the original license and attribution.

## Contents

- [`flash-attention-src/`](./flash-attention-src/) — the FlashAttention-3 backend used by MOSS-VL cross-attention. It adds the `cross_kv_boundary` interface, which represents the visible KV prefix of each query row with a single `int32` value instead of materializing a dense attention mask. This is a MOSS-VL specific version, not a general FlashAttention release; see [`flash-attention-src/README.md`](./flash-attention-src/README.md) for build instructions and source lineage.
- [`sglang/`](./sglang/) — SGLang usage notes and code snapshot for serving MOSS-VL offline. Upstream [SGLang](https://github.com/sgl-project/sglang) already supports MOSS-VL as a first-class model; for new deployments, pulling the latest upstream is recommended.
- [`sglang-omni/`](./sglang-omni/) — SGLang-Omni fork specialized for MOSS-VL realtime serving. Dynamic multi-session scheduling, tensor-parallel and data-parallel (`--dp-size`) replicas, decode CUDA Graphs, and a sliding visual KV window enable concurrent multi-stream realtime inference with substantially higher throughput. Requires the SGLang-format checkpoint [`OpenMOSS-Team/MOSS-VL-Realtime-SGLANG`](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG). Vendored from [fnlp-vision/sglang-omni-realtime](https://github.com/fnlp-vision/sglang-omni-realtime); see [`sglang-omni/UPSTREAM.md`](./sglang-omni/UPSTREAM.md) for provenance.
