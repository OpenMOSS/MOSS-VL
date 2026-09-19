# MiniMax Music 3

**English** | [简体中文](./README_zh.md)

Text-to-music with a Qwen3 backbone, eight-codebook RVQ frames, a flow-matching DIT, and a DAC decoder. Output is 32 kHz stereo.

## Launch

Install the repository in a dedicated environment. The DIT uses `sglang.multimodal_gen`; its dependencies are declared in [pyproject.toml](../../../pyproject.toml). An incompatible `flashinfer-cubin` package must not override the pinned FlashInfer version.

```bash
# Choose one layout.
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000
```

One visible GPU colocates both stages; with two or more, DIT/DAV runs on the second. Both layouts use FP32 for the acoustic stage. Default optimizations include backbone and RVQ-depth decode CUDA Graphs, compiled DIT/DAV, and batched seeded sampling.

## Request

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Verse]\nHello from SGLang Omni",
    "instructions": "A bright piano pop song with a warm female vocal at 100 BPM",
    "seed": 7,
    "max_new_tokens": 250
  }' \
  --output song.wav
```

`input` carries lyrics; `instructions` carries the caption. Put structure tags on separate lines: normalization discards text after a tag on the same line. `max_new_tokens` caps audio frames at 25 per second, up to 9,000; generation may stop earlier.

The [cookbook](../../../docs/cookbook/minimax_music3.md) describes supported and rejected parameters and provides more examples.

## Tests

```bash
python -m pytest tests/unit_test/minimax_music3 -q
```

These tests cover model and request contracts. Acoustic quality and GPU throughput require separate model execution; unit-test results are not audio-quality measurements.
