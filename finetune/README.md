# MOSS-VL Fine-Tuning

Supervised fine-tuning framework for MOSS-VL, built on HuggingFace `transformers.Trainer`.

## Recommended Training Frameworks

The code in this directory is a **minimal training reference**: it covers basic full-parameter and LoRA SFT with a simple data pipeline, and is mainly intended to demonstrate the training contract of MOSS-VL (data format, label masking, module freezing).

For production or domain-specific (vertical) fine-tuning, we recommend using one of the following mature frameworks, both of which support MOSS-VL as a first-class model:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — unified fine-tuning framework for 100+ LLMs/VLMs. MOSS-VL is supported out of the box with LoRA, freeze, and full-parameter workflows, validated checkpoint save/resume/merge, and ready-made example configs (see [PR #10708](https://github.com/hiyouga/LLaMA-Factory/pull/10708)). Step-by-step guides: [English tutorial](https://blog.llamafactory.net/en/posts/moss_vl_finetuning/) | [Chinese tutorial](https://blog.llamafactory.net/posts/moss_vl_finetuning/).
- [ms-swift](https://github.com/modelscope/ms-swift) — ModelScope's training and inference framework. MOSS-VL works with `swift sft` (LoRA or full-parameter) and `swift infer` for image/video, including multi-image, multi-video, mixed-media, and multi-turn data, with fine-grained `freeze_vit` / `freeze_aligner` / `freeze_llm` control (see [PR #9944](https://github.com/modelscope/ms-swift/pull/9944)).

Both frameworks handle MOSS-VL's cross-attention masks, media ordering, and module boundaries natively, and have been validated with real training runs (checkpoint resume, adapter merging, standalone reloading).

## Directory Structure

```
finetune/
├── train.py          # Training entry point
├── data.py           # Dataset and data collator
├── arguments.py      # Argument dataclasses
├── scripts/
│   ├── run_sft.sh        # Full-parameter SFT launch script
│   └── run_sft_lora.sh   # LoRA SFT launch script
└── demo/
    └── sft_data.json     # Example training data
```

## Environment

Use the same environment as the model checkpoint:

```bash
conda create -n moss_vl python=3.12 pip -y
conda activate moss_vl
pip install -i https://pypi.org/simple --no-build-isolation -r requirements.txt
```

For LoRA training, additionally install:

```bash
pip install peft
```

## Data Format

Training data must be a JSON list. Each item should use exactly one of the following formats.

### Format 1: Prompt / Response (compatible with inference queries)

```json
[
  {
    "prompt": "Describe this image.",
    "response": "A beautiful landscape with mountains and a sunset.",
    "images": ["path/to/image.jpg"],
    "videos": [],
    "system_prompt": "You are a helpful assistant."
  }
]
```

Use this format for single-turn supervised fine-tuning data.

- `prompt`: user input text.
- `response`: target assistant output text.
- `images`: optional list of image paths.
- `videos`: optional list of video entries. See [Video Entries](#video-entries) for the supported formats.
- `system_prompt`: optional system instruction.

**Automatic Media Placement**

Media placeholders (`<|image|>` and `<|video|>`) are automatically **prepended** to the user message according to the following rules:

* **Images:** Each image consumes a single `<|image|>` placeholder.
* **Videos:**
    * **Plain Paths:** One `<|video|>` placeholder per video.
    * **Segmented Videos:** One `<|video|>` placeholder **per segment** when using the dictionary format:
        ```json
        {"video_path": "...", "segments": [...]}
        ```

### Format 2: Conversations (multi-turn, explicit placeholders)

```json
[
  {
    "conversations": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "<|image|>\nDescribe this image."},
      {"role": "assistant", "content": "A beautiful landscape."},
      {"role": "user", "content": "What is the dominant color?"},
      {"role": "assistant", "content": "Green."}
    ],
    "images": ["path/to/image.jpg"],
    "videos": []
  }
]
```

Use this format for multi-turn chat data.

- `conversations`: required list of chat messages.
- Each message should be an object like `{"role": "...", "content": "..."}`.
- `images`: optional list of image paths.
- `videos`: optional list of video entries. See [Video Entries](#video-entries) for the supported formats.

Multimodal Placeholder Rules

When using `conversations`, you must explicitly include `<|image|>` or `<|video|>` placeholders in the message content:

- Images: each image requires exactly one `<|image|>` placeholder.

- Videos: each plain video path consumes one `<|video|>` placeholder.

- Segmented videos: each segment within a video dictionary consumes one `<|video|>` placeholder.

> [!NOTE]
> If a sample provides fewer `<|video|>` placeholders than the actual number of video segments, the loader will expand them during preprocessing.
>
> **After this expansion, the final placement of `<|video|>` placeholders may not exactly match the user's original expectation.**

### Path Resolution

Relative media paths in the JSON are resolved relative to the JSON file's parent directory (or the `--data_dir` argument if provided).

### Video Entries

Each item in `videos` can use one of the following formats.

**1. Plain video path**

```json
{
  "videos": [
    "path/to/video.mp4"
  ]
}
```

This represents one full video and consumes one `<|video|>` placeholder.

**2. Segmented videos**

```json
{
  "videos": [
    {
      "video_path": "path/to/video_1.mp4",
      "segments": [[0, 10]]
    },
    {
      "video_path": "path/to/video_2.mp4",
      "segments": [[20, 30]]
    }
  ]
}
```

In the segmented format:

- `video_path` is the path to the source video file.
- `segments` is a list of time segments in seconds.
- Each segment is written as `[start, end]`, using a left-closed, right-open interval: `[start, end)`.
- Each segment expands to one video unit and therefore consumes one `<|video|>` placeholder during training text construction.

In the example above, there are two segmented video entries and each entry has one segment, so the sample expands to two video units and needs two `<|video|>` placeholders.

## Usage

> [!NOTE]
> Run from the repository root.

### Full-Parameter SFT

```bash
bash finetune/scripts/run_sft.sh
```

### LoRA SFT

```bash
bash finetune/scripts/run_sft_lora.sh
```

### Single-GPU Quick Test

```bash
python finetune/train.py \
  --model_name_or_path /path/to/checkpoint \
  --data_path finetune/demo/sft_data.json \
  --output_dir ./checkpoints/test \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --dataloader_num_workers 0 \
  --gradient_checkpointing True \
  --report_to none
```

Cross-attention keeps its existing default unless an override is provided. To
use the repository's specialized FA3 kernel for cross-attention only, first
[install the kernel](../flash-attention-src/README.md#build-and-installation),
then add `--cross_attention_implementation flash_attention_3` to the training
command.

## Key Arguments

### ModelArguments

| Argument | Default | Description |
|---|---|---|
| `--model_name_or_path` | (required) | Path to the MOSS-VL checkpoint |
| `--cross_attention_implementation` | `None` | Optional cross-attention backend override |
| `--tune_vision` | `False` | Train the vision encoder |
| `--tune_language` | `True` | Train the language model layers |
| `--tune_lm_head` | `True` | Train the LM head projection |

### DataArguments

| Argument | Default | Description |
|---|---|---|
| `--data_path` | (required) | Path to the training data JSON file |
| `--data_dir` | auto | Base directory for relative media paths |
| `--max_length` | `4096` | Maximum token sequence length |

### TrainingArguments (extends HF TrainingArguments)

| Argument | Default | Description |
|---|---|---|
| `--vision_chunked_length` | `64` | Chunk size for vision encoding (saves VRAM) |
| `--lora_enable` | `False` | Enable LoRA training |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `128` | LoRA alpha |
| `--lora_dropout` | `0.0` | LoRA dropout |
| `--lora_target_modules` | `q_proj,k_proj,v_proj,o_proj` | Comma-separated LoRA target modules |

Plus all standard HuggingFace `TrainingArguments` (`--learning_rate`, `--num_train_epochs`, `--deepspeed`, etc.).

## Module Freeze Control

By default the vision encoder is frozen while the language model and LM head are trained:

```
tune_vision=False   →  vision encoder frozen
tune_language=True  →  all decoder layers trained
tune_lm_head=True   →  output projection trained
```

When LoRA is enabled (`--lora_enable True`), all base parameters are frozen and only the LoRA adapters are trained.

## DeepSpeed

Pass a DeepSpeed config via `--deepspeed`:

```bash
torchrun --nproc_per_node=8 finetune/train.py \
  ... \
  --deepspeed ds_config_zero2.json
```

## Label Masking

To ensure the model learns effectively, we apply a specific masking strategy to our training tokens:

- Training Targets: Only the Assistant's responses are used as active training labels.

- Masked Content: System prompts, user queries, and all vision-related tokens (e.g., <|image_pad|>) are assigned an ignore_index=-100 to exclude them from loss calculation.

- EOS Learning: The trailing <|im_end|> token at the end of each Assistant turn is explicitly included in the labels, ensuring the model learns when to stop generating.
