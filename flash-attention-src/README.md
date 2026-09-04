# MOSS-VL FlashAttention-3 Cross-Attention Backend

This source directory is bundled with [OpenMOSS/MOSS-VL](../) and is a
specialized derivative of
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). It
extends the FlashAttention-3 (FA3) implementation with a compact mask interface
for prefix-structured cross-attention used by MOSS-VL training.

> [!IMPORTANT]
> This bundled backend is not a general-purpose FlashAttention release and is
> not intended to replace the upstream package. Use the upstream repository
> unless your model needs the `cross_kv_boundary` contract described below.

## Purpose

MOSS-VL uses cross-attention between text queries and visual key/value tokens.
The model produces a `cross_attention_mask`, but passing a dense mask to an
attention kernel is expensive in both memory and execution overhead.

For masks where each query can see a prefix of the KV sequence, the dense mask
can be represented by one integer per query row:

```text
cross_kv_boundary[b, q] = number of visible KV tokens for query row q
visible KV positions     = [0, cross_kv_boundary[b, q])
```

This version passes that compact representation through the FA3 Python API,
operator schema, scheduler, and CUDA forward/backward paths. It allows FA3 to be
used as the attention backend for the MOSS-VL cross-attention modules without
materializing the full `cross_attention_mask`.

## Source Lineage

This directory retains the upstream FlashAttention source tree and license.
The cross-attention extension is based on the following exact source revision:

| Item | Source |
| --- | --- |
| Upstream project | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| Upstream FA3 source | [`hopper/`](https://github.com/Dao-AILab/flash-attention/tree/d7f60e6639250626589c528a943aaaba6fe5955a/hopper) |
| Base revision | [`d7f60e6639250626589c528a943aaaba6fe5955a`](https://github.com/Dao-AILab/flash-attention/commit/d7f60e6639250626589c528a943aaaba6fe5955a) |
| Bundled project | [OpenMOSS/MOSS-VL](https://github.com/OpenMOSS/MOSS-VL) |
| Bundled location | [`flash-attention-src/`](https://github.com/OpenMOSS/MOSS-VL/tree/main/flash-attention-src) |
| Vendored CUTLASS revision | [`7127592069c2fe01b041e174ba4345ef9b279671`](https://github.com/NVIDIA/cutlass/commit/7127592069c2fe01b041e174ba4345ef9b279671) |
| License | [BSD 3-Clause](LICENSE) |

The main extension is implemented under `hopper/`. Files outside that subtree
are retained primarily because this directory follows the upstream source
layout. CUTLASS is vendored under `csrc/cutlass/`, so the backend can live in
the MOSS-VL repository without a nested Git repository or submodule setup.
The exact CUTLASS tree is retained for reproducibility. Its
`csrc/cutlass/python/CuTeDSL/` subtree is not used to build this FA3 backend and
is governed by the separate NVIDIA Software License Agreement included at
[`csrc/cutlass/python/CuTeDSL/EULA.txt`](csrc/cutlass/python/CuTeDSL/EULA.txt).

## Mask Contract

`cross_kv_boundary` represents a per-query visible KV prefix. For a boundary
value `n`, KV positions `0` through `n - 1` are visible and positions `n` onward
are masked.

Properties:

- The tensor dtype must be `torch.int32`.
- The tensor must be on the same CUDA device as the attention inputs.
- Boundary values are interpreted independently for each query row.
- Boundaries may be non-monotonic, including rows that drop back to zero for
  right padding.
- Values below zero behave as zero; values above the logical KV length behave
  as the logical KV length.
- A zero boundary produces a fully masked query row and a zero output row.
- `cross_kv_boundary` cannot be combined with `causal=True`.
- Calls without `cross_kv_boundary` retain the existing FA3 behavior.

Required shapes:

| Execution layout | Shape | Coordinate system |
| --- | --- | --- |
| Dense or QKV-packed | `(batch_size, seqlen_q)` | KV positions within each batch item |
| Variable length | `(total_q,)` | KV positions within the matching `cu_seqlens_k` segment |
| KV-cache, dense query | `(batch_size, seqlen_q)` | Logical cache after left padding is removed and new KV is appended |
| KV-cache, variable-length query | `(total_q,)` | Logical cache coordinates flattened by `cu_seqlens_q` |

### Representable masks

This backend supports masks whose visible positions form a KV prefix for every
query row. It does not represent arbitrary dense masks containing holes or
multiple disjoint visible regions.

For a boolean mask where `True` means visible, the contract is:

```python
expected_visible = (
    torch.arange(seqlen_k, device=boundary.device)
    < boundary[..., None]
)
assert torch.equal(cross_attention_mask, expected_visible)
```

Model integrations are responsible for converting their mask convention to
this representation. In particular, some frameworks use `True` to mean masked
rather than visible; that convention must be normalized before computing the
boundary.

## Supported FA3 Paths

The extension is wired through:

- dense forward and backward;
- QKV-packed forward and backward;
- variable-length forward and backward;
- MQA/GQA and pack-GQA;
- deterministic backward and split-KV execution;
- KV-cache forward, append, paged cache, and scheduler metadata;
- top-level and `flash_attn_3` package import layouts.

KV-cache follows the upstream FA3 contract and does not provide a backward
pass.

## Python API

`cross_kv_boundary` is appended as the final optional argument to preserve
existing positional call compatibility. New code should always pass it by
keyword.

```python
import torch
from flash_attn_interface import flash_attn_func

batch_size = 2
seqlen_q = 512
seqlen_k = 1024

# One logical KV prefix length for each query row.
cross_kv_boundary = torch.full(
    (batch_size, seqlen_q),
    seqlen_k,
    dtype=torch.int32,
    device=query.device,
)

output = flash_attn_func(
    query,
    key,
    value,
    causal=False,
    cross_kv_boundary=cross_kv_boundary,
)
```

The argument is available on:

- `flash_attn_qkvpacked_func`
- `flash_attn_func`
- `flash_attn_varlen_func`
- `flash_attn_with_kvcache`
- `get_scheduler_metadata`

## MOSS-VL Integration

Within the MOSS-VL repository, this directory is the source distribution for
the specialized FA3 backend described in the main
[`README.md`](../README.md#specialized-flashattention-3-backend).

The intended integration flow is:

```text
MOSS-VL cross_attention_mask
        -> validate prefix-visible structure
        -> build int32 cross_kv_boundary
        -> call FA3 cross-attention backend
        -> run fused forward/backward attention
```

The boundary should be passed by keyword at the model/backend boundary. This is
important because FA3 private operator interfaces may contain other optional
arguments such as `sm_margin`.

This directory provides the FA3 backend implementation. MOSS-VL model code,
data processing, training configuration, and checkpoints belong to the
corresponding MOSS-VL integration project and are not documented as general FA3
features here.

## Requirements

This specialized version is developed for MOSS-VL training, with its operators
validated on NVIDIA Ampere and Hopper GPUs.

- NVIDIA A100 or A800 (SM80), or H100, H800, or H200 (SM90)
- CUDA 12.3 or newer
- CUDA 12.8 recommended
- PyTorch with CUDA support
- `packaging`, `psutil`, `ninja`, and `einops`
- Linux

SM80 supports FP16 and BF16 with matching Q/K/V head dimensions. FP8, `q_v`,
and different Q/K and V head dimensions remain Hopper-only.

The operator validation environments include H200 with CUDA 12.8 and PyTorch
2.8, and A800 80GB with CUDA 13.0 and PyTorch 2.11. PyTorch versions below 2.9
build `flash_api.cpp`; PyTorch 2.9 or newer selects `flash_api_stable.cpp`.

## Build and Installation

Run all FA3 commands from the `hopper/` directory.

Build the extension in place for development:

```bash
cd hopper
MAX_JOBS=8 python setup.py build_ext --inplace
```

An architecture-specific build can omit kernels for the other GPU family:

```bash
# Ampere/Ada only. This also disables Hopper-only FP8 kernels.
FLASH_ATTENTION_DISABLE_SM90=TRUE MAX_JOBS=8 python setup.py build_ext --inplace

# Hopper only.
FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=8 python setup.py build_ext --inplace
```

`FLASH_ATTENTION_DISABLE_SM80` and `FLASH_ATTENTION_DISABLE_SM90` cannot both
be enabled.

Or install the specialized FA3 package into the active environment:

```bash
cd hopper
MAX_JOBS=8 pip install --no-build-isolation .
```

Building in place does not update an already installed FA3 package. Set
`PYTHONPATH` explicitly when testing the local extension:

```bash
export PYTHONPATH="$PWD"
python -c "import flash_attn_interface; print(flash_attn_interface.__file__)"
```

Always verify the printed import path before training. Accidentally importing a
different installed FA3 build will not provide this extension's
`cross_kv_boundary` interface.

## Validation

After building the extension in place, run the focused test suite from
`hopper/`:

```bash
pytest -q \
  test_cross_kv_boundary_api.py \
  test_staircase_mask.py \
  test_mask_reconstruction.py \
  test_kvcache_cross_kv_boundary.py
```

The focused suite covers API compatibility, forward/backward numerical parity,
non-monotonic boundaries, fully masked rows, variable-length inputs, packed QKV,
GQA, KV-cache execution, scheduler metadata, package exports, and mask
reconstruction. The current suite contains 265 tests. On SM80, 9
SM90-specific regression tests are skipped.

Run the MOSS-VL-shaped parity utility with:

```bash
cd hopper
python fa3_cross_kv_boundary_parity.py
```

## Relevant Files

| Path | Purpose |
| --- | --- |
| `hopper/flash_attn_interface.py` | Public Python API and autograd integration |
| `hopper/flash_api.cpp` | Standard PyTorch C++ operator binding |
| `hopper/flash_api_stable.cpp` | PyTorch stable ABI operator binding |
| `hopper/flash.h` | Forward/backward parameter structures |
| `hopper/block.h` | Tile-level boundary pruning |
| `hopper/mask.h` | Per-element boundary masking |
| `hopper/mainloop_fwd_*.hpp` | Forward kernel integration |
| `hopper/mainloop_bwd_*.hpp` | Backward kernel integration |
| `hopper/test_cross_kv_boundary_api.py` | API and package compatibility tests |
| `hopper/test_staircase_mask.py` | Forward/backward and layout coverage |
| `hopper/test_mask_reconstruction.py` | Mask-only reconstruction tests |
| `hopper/test_kvcache_cross_kv_boundary.py` | KV-cache coverage |
| `hopper/fa3_cross_kv_boundary_parity.py` | MOSS-VL-shaped end-to-end parity check |

## Non-Goals and Support Boundary

This repository does not claim to provide:

- a generic arbitrary `cross_attention_mask` kernel;
- support for dense masks with holes or disjoint visible regions;
- a drop-in replacement for every upstream FlashAttention configuration;
- general support for GPUs or software stacks outside the validated target;
- model code, training recipes, or checkpoints for MOSS-VL itself.

Extension-specific issues should be reported to the MOSS-VL integration
project. Please report an issue to upstream FlashAttention only after reproducing
it on an unmodified upstream revision.

## Attribution and License

This is a modified derivative of FlashAttention. The upstream source is
copyright its respective contributors and is distributed under the
[BSD 3-Clause License](LICENSE). Redistribution must preserve the upstream
copyright notice, license conditions, and disclaimer.

The vendored CUTLASS sources remain under NVIDIA's
[BSD 3-Clause License](csrc/cutlass/LICENSE.txt), except for components that
carry separate terms. In particular, `csrc/cutlass/python/CuTeDSL/` is governed
by its bundled [NVIDIA Software License Agreement](csrc/cutlass/python/CuTeDSL/EULA.txt).
These third-party licenses continue to apply within the top-level Apache-2.0
MOSS-VL repository.

When redistributing this specialized version:

1. Identify it as a modified derivative, not an official upstream FA3 release.
2. Link to the exact upstream project and base revision listed above.
3. Retain `LICENSE`, `AUTHORS`, and existing source-file copyright notices.
4. Describe local modifications separately from upstream functionality.
5. Do not imply endorsement by the upstream FlashAttention authors.

## Citation

If this backend contributes to research results, cite the applicable
FlashAttention papers. Work using it for MOSS-VL should also cite the MOSS-VL
project or its current technical report.

```bibtex
@inproceedings{dao2022flashattention,
  title     = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author    = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022}
}

@inproceedings{dao2023flashattention2,
  title     = {Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author    = {Dao, Tri},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}

@article{shah2024flashattention3,
  title   = {FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision},
  author  = {Shah, Jay and Bikshandi, Ganesh and Zhang, Ying and Thakkar, Vijay and Ramani, Pradeep and Dao, Tri},
  journal = {arXiv preprint arXiv:2407.08608},
  year    = {2024}
}

@misc{moss_vl_2026,
  title        = {{MOSS-VL Technical Report}},
  author       = {{OpenMOSS Team}},
  year         = {2026},
  howpublished = {\url{https://github.com/OpenMOSS/MOSS-VL}},
  note         = {GitHub repository}
}
```

FlashAttention-3 paper: <https://arxiv.org/abs/2407.08608>
