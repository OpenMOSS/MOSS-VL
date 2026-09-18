"""Transformers reference on the same event/token schedules as semantic_engine."""

import gc
import time


class Reference:
    def __init__(self, path, *, attention="eager"):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        from sglang_omni.models.moss_vl_realtime.model_step import MossVLRealtimeStepper

        self.torch = torch
        torch.set_num_threads(4)
        self.processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={
                "": (
                    "npu:0"
                    if getattr(torch, "npu", None) and torch.npu.is_available()
                    else "cuda:0"
                )
            },
            attn_implementation=attention,
        ).eval()
        self.device = (
            "npu"
            if getattr(torch, "npu", None) and torch.npu.is_available()
            else "cuda"
        )
        self.stepper = MossVLRealtimeStepper(self.model, self.processor)
        self.tokenizer = self.processor.tokenizer
        self.silence = self.tokenizer.convert_tokens_to_ids("<|silence|>")
        self.word = self.tokenizer.encode("the", add_special_tokens=False)[0]

    def _empty_cache(self):
        if self.device == "npu":
            self.torch.npu.empty_cache()
        else:
            self.torch.cuda.empty_cache()

    def _sync(self):
        if self.device == "npu":
            self.torch.npu.synchronize()
        else:
            self.torch.cuda.synchronize()

    def initial(self, case):
        encoded = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": case["system_prompt"]},
                {"role": "user", "content": case["initial_prompt"]},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        ids = (encoded["input_ids"] if hasattr(encoded, "keys") else encoded).to(
            self.device
        )
        return self.stepper.initial_prefill(ids)

    def extend(self, state, event):
        from PIL import Image

        frames = []
        if event["type"] == "frame":
            with Image.open(event["frame_path"]) as image:
                frames = [(image.convert("RGB"), event["timestamp"])]
        self.stepper.apply_event_and_extend(
            state, prompt=event.get("prompt"), frames=frames
        )

    def semantic(self, cases, count, cap=256):
        from semantic_checks import assess

        states = [self.initial(case) for case in cases]
        rows = [
            dict(
                backend="TF",
                sessions=count,
                lane=i,
                case_id=c["case_id"],
                chunks=[],
                token_ids=[],
                error=None,
                capped=False,
                input_ok=True,
            )
            for i, c in enumerate(cases)
        ]
        try:
            for phase in range(-1, max(len(c["events"]) for c in cases)):
                for case, state, row in zip(cases, states, rows):
                    if row["error"] or phase >= len(case["events"]):
                        continue
                    try:
                        if phase >= 0:
                            self.extend(state, case["events"][phase])
                        tokens = []
                        for offset in range(cap):
                            token = int(self.stepper.sample_next_token(state).item())
                            tokens.append(token)
                            if token == self.silence:
                                break
                            if offset + 1 < cap:
                                self.stepper.commit_pending_tokens(state)
                        else:
                            row["capped"] = True
                            row["error"] = "event token cap"
                        row["token_ids"].extend(tokens)
                        row["chunks"].append(
                            self.tokenizer.decode(
                                tokens,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                        )
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        row["input_ok"] = False
            for case, row in zip(cases, rows):
                row["text"] = self.tokenizer.decode(
                    row["token_ids"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                row["task_check"] = assess(
                    row["text"],
                    case["contract"],
                    error=row["error"],
                    capped=row["capped"],
                )
            return rows
        finally:
            states.clear()
            gc.collect()
            self._empty_cache()

    def performance(self, case, repeats=3):
        rows = []
        frames = [e for e in case["events"] if e["type"] == "frame"][:11]
        assert len(frames) == 11
        for _ in range(repeats + 1):
            self._sync()
            begin = time.perf_counter()
            state = self.initial(case)
            self.stepper.sample_next_token(state, forced_token_id=self.silence)
            self._sync()
            prefill = time.perf_counter() - begin
            durations = []
            for index, event in enumerate(frames):
                self._sync()
                begin = time.perf_counter()
                self.extend(state, event)
                self.stepper.sample_next_token(state, forced_token_id=self.silence)
                self._sync()
                if index >= 3:
                    durations.append(time.perf_counter() - begin)
            self._sync()
            begin = time.perf_counter()
            self.stepper.apply_event_and_extend(
                state,
                prompt="Describe everything that happened in the video so far, in detail.",
                frames=[],
            )
            signature = dict(
                forced_token=self.word,
                frames=[e["sha256"] for e in frames],
                encoder_tokens=state.vision_cache_position,
                decoder_prefix_tokens=state.text_cache_position,
            )
            self.stepper.sample_next_token(state, forced_token_id=self.word)
            self._sync()
            prompt_ttft = time.perf_counter() - begin
            intervals = []
            for _ in range(63):
                begin = time.perf_counter()
                self.stepper.commit_pending_tokens(state)
                self.stepper.sample_next_token(state, forced_token_id=self.word)
                self._sync()
                intervals.append(time.perf_counter() - begin)
            rows.append(
                dict(
                    prefill_seconds=prefill,
                    frame_seconds=durations,
                    decode_tokens=64,
                    prompt_ttft_seconds=prompt_ttft,
                    decode_intervals=intervals,
                    signature=signature,
                )
            )
            del state
            gc.collect()
        return rows[1:]

    def close(self):
        del self.stepper, self.model
        gc.collect()
        self._empty_cache()
