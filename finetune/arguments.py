from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to the MOSS-VL checkpoint directory."},
    )
    cross_attention_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Optional cross-attention backend override."},
    )
    tune_vision: bool = field(
        default=False,
        metadata={"help": "Whether to train the vision encoder."},
    )
    tune_language: bool = field(
        default=True,
        metadata={"help": "Whether to train the language model layers."},
    )
    tune_lm_head: bool = field(
        default=True,
        metadata={"help": "Whether to train the LM head projection."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to the training data JSON file."},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Base directory for resolving relative media paths. "
                  "Defaults to the parent directory of data_path."},
    )
    max_length: int = field(
        default=4096,
        metadata={"help": "Maximum token sequence length (truncation boundary)."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    vision_chunked_length: int = field(
        default=64,
        metadata={"help": "Chunk size for vision encoder forward pass (saves VRAM)."},
    )

    # LoRA
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated list of module names to apply LoRA to."},
    )
