# Inference

This directory contains ready-to-run offline inference examples for MOSS-VL.
The script supports full-modality offline inference through `model.offline_generate(...)`, including:

- pure text
- single image
- multiple images
- single video
- multiple videos
- interleaved image-video inputs in the `messages` format

## Supported checkpoint versions

`run_inference.py` is compatible with both releases of the MOSS-VL checkpoint:

- `transformers==4.57.1` — uses `MossVLImageProcessorFast` and exposes a `vision_chunked_length` knob inside the vision tower.
- `transformers==5.5.4` — uses the slow PIL-based `MossVLImageProcessor` and processes the entire vision input in a single forward pass.

The `offline_generate` / `offline_batch_generate` / `offline_image_generate` / `offline_video_generate` API is identical on both versions, so the script requires no changes when switching checkpoints. Empirically the two checkpoints produce **token-identical outputs under `do_sample=false`** on the example queries in this directory.

### Behaviour differences
- **`vision_chunked_length` is accepted but has no effect on `MOSS-VL-Instruct`.** The new modeling file runs the visual tower over all media in one pass, so the value is silently ignored when using `transformers==5.5.4`. You can keep it in `generate_kwargs` for backward compatibility; it will still shard prefill on the legacy `transformers==4.57.1` checkpoint.

## Run

Image examples:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/dummy-checkpoint \
  --mode offline \
  --input inference/image_queries.json
```

Video example:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/dummy-checkpoint \
  --mode offline \
  --input inference/video_queries.json
```

Text-only example:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/dummy-checkpoint \
  --mode offline \
  --input inference/batch_queries.json
```

SFT / validation-set example in training `jsonl` format:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/dummy-checkpoint \
  --mode offline \
  --input /path/to/valid.jsonl
```

If `--output` is omitted, the script writes results to `<input_stem>_results.json`.

Cross-attention keeps its existing default unless an override is provided. To
use the repository's specialized FA3 kernel for cross-attention only, first
[install the kernel](../flash-attention-src/README.md#build-and-installation),
then add:

```bash
--cross-attention-implementation flash_attention_3
```

## Input Format

The input file can be either:

- a JSON list, where each item is one query
- a JSONL file, where each line is one sample

Each query can use either of the following formats:

- `messages`
- `prompt` with optional `images` and `videos`

Optional fields such as `media_kwargs`, `generate_kwargs`, and `system_prompt` are also supported.

For JSONL inputs, the script also accepts the standard SFT training formats documented in `finetune/README.md`:

- `messages` or `conversations` with top-level `images` / `videos`
- `prompt` / `response` with optional `images` / `videos`

When a training sample includes assistant targets, the loader automatically trims trailing assistant turns at inference time and keeps the remaining context up to the last user turn. For conversation-style samples that use `<|image|>` or `<|video|>` placeholders in text, the loader also reconstructs structured multimodal `messages` content before calling `offline_generate`.

The provided examples use the `messages` format. Example:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "image", "image": "assets/images/bill.png" },
          { "type": "text", "text": "Describe this image." }
        ]
      }
    ],
    "media_kwargs": {},
    "generate_kwargs": {
      "max_new_tokens": 256,
      "do_sample": false,
      "vision_chunked_length": 64
    }
  }
]
```

The `prompt` format is also supported. Example:

```json
[
  {
    "prompt": "Describe this image.",
    "images": ["assets/images/bill.png"],
    "videos": [],
    "media_kwargs": {},
    "generate_kwargs": {
      "max_new_tokens": 256,
      "do_sample": false,
      "vision_chunked_length": 64
    }
  }
]
```

Relative media paths are resolved relative to the JSON file location.

## Files

- `image_queries.json`: image and multi-image examples
- `video_queries.json`: video example
- `batch_queries.json`: text-only example
- `assets/images`: demo images
- `assets/videos`: demo videos
