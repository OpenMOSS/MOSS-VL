from __future__ import annotations

import os
import queue
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.models.moss_vl_realtime.model_step import (
    MossVLRealtimeStepper,
    compute_realtime_mrope_for_segment,
)


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run the MOSS-VL model-step golden test")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{name} does not exist: {path}")
    return path


def test_compute_realtime_mrope_for_one_frame() -> None:
    torch = pytest.importorskip("torch")
    input_ids = torch.tensor([[10, 99, 11]])
    grid_thw = torch.tensor([[1, 4, 6]])

    text_positions, vision_positions, next_position = (
        compute_realtime_mrope_for_segment(
            new_input_ids=input_ids,
            new_grid_thw=grid_thw,
            start_position=7,
            image_token_id=99,
            merge_size=2,
        )
    )

    assert text_positions.tolist() == [
        [[7, 11, 12]],
        [[7, 11, 12]],
        [[7, 11, 12]],
    ]
    assert vision_positions is not None
    assert vision_positions.shape == (3, 1, 7)
    assert vision_positions[:, 0, -1].tolist() == [11, 11, 11]
    assert next_position == 13


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
@pytest.mark.gpu
def test_split_decode_then_extend_matches_tf512_packed_step() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torchvision_functional = pytest.importorskip("torchvision.transforms.functional")

    model_path = _required_path("MOSS_VL_STREAMING_MODEL_PATH")
    video_path = _required_path("MOSS_VL_TEST_VIDEO")
    timestamp = float(os.environ.get("MOSS_VL_TEST_TIMESTAMP", "5.0"))

    assert transformers.__version__ == "5.12.1"

    processor = transformers.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    ).eval()
    stepper = MossVLRealtimeStepper(model, processor)

    messages = [
        {"role": "system", "content": "You are a helpful visual assistant."},
        {"role": "user", "content": ""},
    ]
    initial_inputs = processor.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    initial_input_ids = (
        initial_inputs["input_ids"]
        if hasattr(initial_inputs, "keys")
        else initial_inputs
    ).to(model.device)
    initial_attention_mask = torch.ones_like(initial_input_ids)

    split_state = stepper.initial_prefill(
        initial_input_ids.clone(), initial_attention_mask.clone()
    )
    rotary = model.model.language_model.rotary_emb
    rotary.inv_freq.fill_(-1.468446e34)
    rotary.original_inv_freq.zero_()
    reference_state = stepper.initial_prefill(
        initial_input_ids.clone(), initial_attention_mask.clone()
    )
    torch.testing.assert_close(
        split_state.next_token_logits,
        reference_state.next_token_logits,
        rtol=0,
        atol=0,
    )

    forced_token_id = int(split_state.next_token_logits.argmax(dim=-1).item())
    stepper.sample_next_token(split_state, forced_token_id=forced_token_id)
    assert split_state.pending_text_length == 1

    frame_tensor, _ = processor.video_processor._fetch_video_segment(
        str(video_path), [timestamp]
    )
    frame = torchvision_functional.to_pil_image(frame_tensor[0])
    prompt = "What changed?"
    stepper.apply_event_and_extend(
        split_state,
        prompt=prompt,
        frames=[(frame, timestamp)],
    )
    assert split_state.pending_text_length == 0

    reference_input_ids = torch.cat(
        [
            reference_state.input_ids,
            torch.tensor([[forced_token_id]], dtype=torch.long, device=model.device),
        ],
        dim=1,
    )
    frame_queue: queue.Queue = queue.Queue()
    prompt_queue: queue.Queue = queue.Queue()
    frame_queue.put((frame, timestamp))
    prompt_queue.put(prompt)
    reference_kwargs = {
        "attention_mask": reference_state.attention_mask,
        "position_ids": reference_state.position_ids,
        "past_key_values": reference_state.past_key_values,
        "cache_position": torch.arange(
            reference_state.text_cache_position,
            dtype=torch.long,
            device=model.device,
        ),
        "realtime_next_position": reference_state.next_mrope_position,
        "full_vision_token_info": reference_state.full_vision_token_info,
        "cross_attention_mask": reference_state.cross_attention_mask,
        "use_cache": True,
        "logits_to_keep": 1,
    }
    reference_input_ids, reference_kwargs = (
        model._update_model_kwargs_for_real_time_generation(
            outputs=SimpleNamespace(past_key_values=reference_state.past_key_values),
            input_ids=reference_input_ids,
            model_kwargs=reference_kwargs,
            should_wait_for_new_input=False,
            new_video_frames=frame_queue,
            new_prompts=prompt_queue,
            output_text_queue=None,
            token_buffer=deque(),
            processor=processor,
        )
    )
    reference_inputs = model.prepare_inputs_for_real_time_generation(
        reference_input_ids, **reference_kwargs
    )
    with torch.no_grad():
        reference_outputs = model(**reference_inputs, return_dict=True)

    assert torch.equal(split_state.input_ids, reference_input_ids)
    assert torch.equal(split_state.attention_mask, reference_kwargs["attention_mask"])
    assert torch.equal(split_state.position_ids, reference_kwargs["position_ids"])
    assert torch.equal(
        split_state.cross_attention_mask,
        reference_kwargs["cross_attention_mask"],
    )
    assert (
        split_state.full_vision_token_info == reference_kwargs["full_vision_token_info"]
    )
    assert split_state.text_cache_position == split_state.input_ids.shape[1]
    assert (
        split_state.visible_vision_length
        == reference_kwargs["full_vision_token_info"][0]["total_length"]
    )
    assert (
        split_state.vision_cache_position
        == reference_kwargs["full_vision_token_info"][0]["pad_end"]
    )

    reference_logits = reference_outputs.logits[:, -1, :].float()
    torch.testing.assert_close(
        split_state.next_token_logits,
        reference_logits,
        rtol=1e-2,
        atol=2e-2,
    )
    assert split_state.next_token_logits.argmax(dim=-1).item() == (
        reference_logits.argmax(dim=-1).item()
    )
