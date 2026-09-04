"""MOSS-VL supervised fine-tuning entry point.

Usage::

    torchrun --nproc_per_node=8 finetune/train.py \\
        --model_name_or_path /path/to/checkpoint \\
        --data_path finetune/demo/sft_data.json \\
        --output_dir ./checkpoints \\
        --bf16 True \\
        --per_device_train_batch_size 1 \\
        --gradient_accumulation_steps 8
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer

from arguments import ModelArguments, DataArguments, TrainingArguments
from data import MossVLSFTDataset, MossVLDataCollator

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Model save helper (handles DeepSpeed ZeRO-3 state-dict gathering)
# ------------------------------------------------------------------

def safe_save_model_for_hf_trainer(
    trainer: Trainer,
    output_dir: str,
) -> None:
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict: Dict[str, torch.Tensor] = {
            k: v.cpu() for k, v in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


# ------------------------------------------------------------------
# Module-level freeze / unfreeze
# ------------------------------------------------------------------

def configure_trainable_parameters(model, model_args: ModelArguments) -> None:
    if not model_args.tune_vision:
        for p in model.visual.parameters():
            p.requires_grad = False

    if not model_args.tune_language:
        for p in model.language_model.parameters():
            p.requires_grad = False

    if not model_args.tune_lm_head:
        model.lm_head.weight.requires_grad = False


def log_trainable_summary(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}",
        100.0 * trainable / total if total else 0,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def train() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments),
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARN,
    )

    # ---- Processor ---------------------------------------------------
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "right"

    # ---- Model -------------------------------------------------------
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }
    if model_args.cross_attention_implementation is not None:
        model_kwargs["cross_attention_implementation"] = (
            model_args.cross_attention_implementation
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        # Cross-attention models share vision features across multiple
        # checkpointed decoder layers.  The default reentrant implementation
        # frees the graph after the first backward; when a later layer is
        # recomputed it tries to re-use those freed tensors and crashes.
        # use_reentrant=False avoids this by never running backward inside
        # the recomputed forward.
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _make_inputs_require_grad(_module, _input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(
                _make_inputs_require_grad,
            )

    # ---- LoRA or full-param -----------------------------------------
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType

        for p in model.parameters():
            p.requires_grad = False

        target_modules = [
            m.strip()
            for m in training_args.lora_target_modules.split(",")
        ]
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        configure_trainable_parameters(model, model_args)

    log_trainable_summary(model)

    # ---- Data --------------------------------------------------------
    dataset = MossVLSFTDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_dir=data_args.data_dir,
        max_length=data_args.max_length,
    )
    collator = MossVLDataCollator(
        processor=processor,
        max_length=data_args.max_length,
        vision_chunked_length=training_args.vision_chunked_length,
    )

    logger.info("Training samples: %d", len(dataset))

    # ---- Trainer -----------------------------------------------------
    # The dataset returns custom keys (text, image_paths, …) that are not
    # parameters of model.forward().  Prevent the Trainer from stripping
    # them before they reach our collator.
    training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor.tokenizer,
    )

    # ---- Train -------------------------------------------------------
    checkpoint_dirs = list(
        pathlib.Path(training_args.output_dir).glob("checkpoint-*"),
    )
    if checkpoint_dirs:
        logger.info("Resuming from checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info("Model and processor saved to %s", training_args.output_dir)


if __name__ == "__main__":
    train()
