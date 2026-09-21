#!/usr/bin/env python3
"""Idempotent patch: fix vLLM-Omni's MOSS-Audio-Tokenizer codec loader so it
serves MOSS-TTS-Realtime (and any base/v2 tokenizer) instead of crashing at
weight load with `MOSS Audio Tokenizer weights were not fully loaded:
loaded=1600/1606 missing=6`.

Root cause (verified against the checkpoint): the real MOSS-Audio-Tokenizer
ships 6 projections (encoder.3/5/7.input_proj, decoder.0/2/4.output_proj) as
`nn.Identity()` because at those positions input/output dim == d_model — its own
`modeling_moss_audio_tokenizer.py` builds them conditionally and saves NO weight
tensor (confirmed in model.safetensors.index.json). vLLM-Omni's port builds them
as UNCONDITIONAL `nn.Linear`, expecting 1606 params vs the checkpoint's 1600.
The in-code comment ("upstream always materializes a learned Linear ...
substituting Identity ... silently drops a trained projection") is factually
wrong for this checkpoint.

Fix: build in/out projections as `nn.Identity()` when the dims match — mirroring
the checkpoint's true architecture (so the forward pass is an exact passthrough,
NOT a dropped-then-randomly-initialized projection). Patches both the v1
(`audio_tokenizer.py`, `in_proj`/`out_proj`) and v2 (`audio_tokenizer_v2.py`,
`input_proj`/`output_proj`) `_ProjectedTransformer` constructors; the loader
tries v2 first and falls back to v1, so patching both covers either path.

Usage:  <venv>/bin/python scripts/patch_vllm_omni_moss_codec.py
        (defaults to sys.executable's own site-packages — run it with the
         .venv-vllm python). Re-run safely; already-patched files are skipped.

Upstream bug — worth reporting to github.com/vllm-project/vllm-omni.
"""
from __future__ import annotations
import importlib.util
import os
import sys

# (single-line unconditional construction) -> (conditional Identity when dims match)
_REPLACEMENTS = [
    ("self.in_proj = nn.Linear(input_dimension, d_model, bias=False)",
     "self.in_proj = nn.Linear(input_dimension, d_model, bias=False) if input_dimension != d_model else nn.Identity()"),
    ("self.out_proj = nn.Linear(d_model, output_dimension, bias=False)",
     "self.out_proj = nn.Linear(d_model, output_dimension, bias=False) if d_model != output_dimension else nn.Identity()"),
    ("self.input_proj = nn.Linear(input_dimension, d_model, bias=False)",
     "self.input_proj = nn.Linear(input_dimension, d_model, bias=False) if input_dimension != d_model else nn.Identity()"),
    ("self.output_proj = nn.Linear(d_model, output_dimension, bias=False)",
     "self.output_proj = nn.Linear(d_model, output_dimension, bias=False) if d_model != output_dimension else nn.Identity()"),
]

_FILES = ("audio_tokenizer.py", "audio_tokenizer_v2.py")


def _codec_dir() -> str:
    spec = importlib.util.find_spec("vllm_omni")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("vllm_omni not importable in this interpreter — run with the .venv-vllm python")
    return os.path.join(spec.submodule_search_locations[0],
                        "model_executor", "models", "moss_tts")


def main() -> int:
    base = _codec_dir()
    changed = False
    for fname in _FILES:
        path = os.path.join(base, fname)
        if not os.path.isfile(path):
            print(f"skip (absent): {path}")
            continue
        src = open(path, encoding="utf-8").read()
        out = src
        applied = []
        for old, new in _REPLACEMENTS:
            if new in out:            # already patched
                continue
            if old in out:
                assert out.count(old) == 1, f"{old!r} not unique in {fname} — vllm-omni layout changed, re-check the patch"
                out = out.replace(old, new)
                applied.append(old.split(" = ")[0].strip())
        if out != src:
            open(path, "w", encoding="utf-8").write(out)
            print(f"patched {fname}: {', '.join(applied)}")
            changed = True
        else:
            print(f"already patched / nothing to do: {fname}")
    print("DONE — vllm-omni MOSS codec projections are now conditional-Identity" if changed
          else "DONE — no changes (already patched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
