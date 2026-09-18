"""Controlled event boundaries and raw-token timing for the SGLang engine."""

import queue
import threading
import time
import uuid

from common import CONFIG


class Engine:
    def __init__(
        self,
        model_path,
        *,
        disable_cuda_graph=False,
        server_args_overrides=None,
        max_running_requests=4,
        frame_window_enabled=False,
        device=None,
        context_length=131072,
    ):
        import torch

        from sglang_omni.models.moss_vl_realtime.stages import (
            create_sglang_moss_vl_realtime_executor,
        )

        self.torch = torch
        self.scheduler = create_sglang_moss_vl_realtime_executor(
            str(model_path),
            device=device,
            gpu_id=0,
            max_running_requests=max_running_requests,
            max_new_tokens=4096,
            context_length=context_length,
            mem_fraction_static=CONFIG["mem_fraction_static"],
            disable_cuda_graph=disable_cuda_graph,
            enable_async_decode=False,
            server_args_overrides=server_args_overrides,
            realtime_frame_window_enabled=frame_window_enabled,
            realtime_frame_window_raw_s=CONFIG["raw_window_seconds"],
            realtime_frame_pooling_enabled=False,
        )
        self.tokenizer = self.scheduler.segment_builder.processor.tokenizer
        self.silence = self.tokenizer.convert_tokens_to_ids("<|silence|>")
        self.word = self.tokenizer.encode("the", add_special_tokens=False)[0]
        self.streams = {}
        runner = self.scheduler._model_runner
        original = runner._sample_next_token_ids

        def sample(logits, forward_batch, schedule_batch, requests):
            token_ids = original(logits, forward_batch, schedule_batch, requests)
            for i, req in enumerate(schedule_batch.reqs):
                stream = self.streams.get(req.rid)
                if stream is not None and stream["force"] is not None:
                    token_ids[i] = stream["force"]
            torch.cuda.synchronize()
            arrived = time.perf_counter()
            for i, req in enumerate(schedule_batch.reqs):
                stream = self.streams.get(req.rid)
                if stream is None:
                    continue
                state = req._moss_vl_realtime_state
                stream["records"].append(
                    dict(
                        phase=stream["phase"],
                        token=int(token_ids[i].item()),
                        time=arrived,
                        encoder=state.encoder_length,
                        decoder=state.decoder_length,
                        batch_size=len(schedule_batch.reqs),
                        turn_id=state.turn_id,
                    )
                )
            return token_ids

        runner._sample_next_token_ids = sample
        self.thread_error = None

        def serve():
            try:
                self.scheduler.start()
            except BaseException as exc:
                self.thread_error = repr(exc)

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()

    def new(
        self, case, *, force=None, allowance=4096, benchmark=False, token_rate=86400.0
    ):
        from sglang_omni.proto import OmniRequest, StagePayload
        from sglang_omni.scheduling.messages import IncomingMessage

        rid = "semantic-" + uuid.uuid4().hex
        self.streams[rid] = dict(phase=-1, force=force, records=[], case=case)
        payload = StagePayload(
            request_id=rid,
            request=OmniRequest(
                inputs=dict(
                    initial_prompt=case["initial_prompt"],
                    system_prompt=case["system_prompt"],
                    session_id=rid,
                ),
                params=dict(
                    stream=True,
                    temperature=0.0,
                    max_new_tokens=allowance,
                    max_tokens_per_turn=token_rate,
                    benchmark_ignore_eos=benchmark,
                ),
            ),
            data={},
        )
        self.scheduler.inbox.put(IncomingMessage(rid, "new_request", payload))
        return rid

    def update(self, rid, event, phase, *, final=False):
        from sglang_omni.models.moss_vl_realtime.payload_types import FramePromptEvent
        from sglang_omni.scheduling.messages import IncomingMessage

        self.streams[rid]["phase"] = phase
        update = FramePromptEvent(
            request_id=rid,
            session_id=rid,
            seq_no=event["seq_no"],
            timestamp=event["timestamp"],
            frame_ref=event.get("frame_path"),
            prompt=event.get("prompt"),
            final=final,
        )
        self.scheduler.inbox.put(
            IncomingMessage(rid, "request_update", update.to_dict())
        )

    def message(self, deadline):
        while time.monotonic() < deadline:
            if not self.thread.is_alive():
                raise RuntimeError("Scheduler stopped: " + str(self.thread_error))
            try:
                return self.scheduler.outbox.get(
                    timeout=min(0.2, max(0.01, deadline - time.monotonic()))
                )
            except queue.Empty:
                pass
        raise TimeoutError("No event boundary before deadline")

    def wait_silence(self, rid):
        deadline = time.monotonic() + 120
        while True:
            message = self.message(deadline)
            if message.request_id != rid:
                continue
            if message.type in ("error", "result"):
                raise RuntimeError(
                    f"Unexpected terminal {message.type}: {message.data}"
                )
            if (
                message.type == "stream"
                and message.data.get("event") == "response.turn.silence"
            ):
                return time.perf_counter()

    def recover(self, rids):
        for rid in rids:
            self.scheduler.abort(rid)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            pool = self.scheduler.token_to_kv_pool_allocator
            if (
                not self.scheduler.realtime_sessions._sessions
                and pool.available_size() == pool.size
            ):
                for rid in rids:
                    self.streams.pop(rid, None)
                return True
            time.sleep(0.05)
        return False

    def semantic(self, cases, count, cap=256):
        from semantic_checks import assess

        active = {}
        completed = []
        rids = []
        for lane, case in enumerate(cases):
            rid = self.new(case)
            rids.append(rid)
            active[rid] = dict(
                case=case,
                lane=lane,
                processed=[],
                error=None,
                capped=False,
                deadline=time.monotonic() + 120,
            )

        def finish(rid, item):
            records = list(self.streams[rid]["records"])
            ids = [r["token"] for r in records]
            text = self.tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            chunks = [
                self.tokenizer.decode(
                    [r["token"] for r in records if r["phase"] == p],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for p in range(-1, len(item["case"]["events"]))
            ]
            expected = [e["seq_no"] for e in item["case"]["events"]]
            transport = item["processed"] == expected and not item["error"]
            completed.append(
                dict(
                    backend="SGLang",
                    sessions=count,
                    lane=item["lane"],
                    case_id=item["case"]["case_id"],
                    text=text,
                    token_ids=ids,
                    chunks=chunks,
                    processed_events=item["processed"],
                    input_ok=transport,
                    error=item["error"],
                    capped=item["capped"],
                    task_check=assess(
                        text,
                        item["case"]["contract"],
                        error=item["error"],
                        capped=item["capped"],
                    ),
                )
            )
            self.scheduler.abort(rid)
            del active[rid]

        try:
            while active:
                for rid, item in list(active.items()):
                    stream = self.streams[rid]
                    n = sum(r["phase"] == stream["phase"] for r in stream["records"])
                    if n > cap or time.monotonic() > item["deadline"]:
                        item["capped"] = n > cap
                        item["error"] = (
                            "event token cap" if n > cap else "event timeout"
                        )
                        finish(rid, item)
                if not active:
                    break
                try:
                    message = self.scheduler.outbox.get(timeout=0.2)
                except queue.Empty:
                    if not self.thread.is_alive():
                        raise RuntimeError(self.thread_error or "scheduler stopped")
                    continue
                rid = message.request_id
                if rid not in active:
                    continue
                item = active[rid]
                if message.type in ("error", "result"):
                    item["error"] = f"unexpected terminal: {message.type}"
                    finish(rid, item)
                elif message.type == "stream":
                    event = message.data.get("event")
                    if event in ("input.frame.processed", "input.prompt.processed"):
                        item["processed"].append(message.data["seq_no"])
                    if event == "response.turn.silence":
                        next_phase = self.streams[rid]["phase"] + 1
                        if next_phase == len(item["case"]["events"]):
                            finish(rid, item)
                        else:
                            self.update(
                                rid, item["case"]["events"][next_phase], next_phase
                            )
                            item["deadline"] = time.monotonic() + 120
        finally:
            clean = self.recover(rids)
        for row in completed:
            row["kv_recovered"] = clean
        return sorted(completed, key=lambda r: r["lane"])

    def performance(self, case, repeats=3):
        rows = []
        frames = [e for e in case["events"] if e["type"] == "frame"][:11]
        assert len(frames) == 11
        for _ in range(repeats + 1):
            begin = time.perf_counter()
            rid = self.new(case, force=self.silence, allowance=64, benchmark=True)
            try:
                prefill = self.wait_silence(rid) - begin
                latencies = []
                for index, event in enumerate(frames):
                    begin = time.perf_counter()
                    self.update(rid, event, index)
                    duration = self.wait_silence(rid) - begin
                    if index >= 3:
                        latencies.append(duration)
                prompt = dict(
                    seq_no=11,
                    type="prompt",
                    timestamp=frames[-1]["timestamp"],
                    prompt="Describe everything that happened in the video so far, in detail.",
                )
                self.streams[rid]["force"] = self.word
                prompt_begin = time.perf_counter()
                self.update(rid, prompt, 11, final=True)
                deadline = time.monotonic() + 120
                while True:
                    message = self.message(deadline)
                    if message.request_id != rid:
                        continue
                    if message.type == "error":
                        raise RuntimeError(str(message.data))
                    if message.type == "result":
                        break
                records = [r for r in self.streams[rid]["records"] if r["phase"] == 11]
                if len(records) != 64:
                    raise RuntimeError(f"Expected 64 raw tokens, got {len(records)}")
                rows.append(
                    dict(
                        prefill_seconds=prefill,
                        frame_seconds=latencies,
                        decode_tokens=len(records),
                        prompt_ttft_seconds=records[0]["time"] - prompt_begin,
                        decode_intervals=[
                            b["time"] - a["time"] for a, b in zip(records, records[1:])
                        ],
                        signature=dict(
                            forced_token=self.word,
                            frames=[e["sha256"] for e in frames],
                            encoder_tokens=records[0]["encoder"],
                            decoder_prefix_tokens=records[0]["decoder"],
                        ),
                    )
                )
            finally:
                if not self.recover([rid]):
                    raise RuntimeError("Performance request KV not recovered")
        return rows[1:]

    def close(self):
        self.scheduler.stop()
        self.thread.join(30)
        if self.thread.is_alive():
            raise RuntimeError("Scheduler thread did not stop")
        if self.torch.distributed.is_initialized():
            self.torch.distributed.destroy_process_group()
