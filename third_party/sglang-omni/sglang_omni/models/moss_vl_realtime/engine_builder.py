"""SGLang engine builder for MOSS-VL realtime."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoProcessor

from sglang_omni.models.moss_vl_realtime import request_builders
from sglang_omni.models.moss_vl_realtime.model_runner import MossVLRealtimeModelRunner
from sglang_omni.models.moss_vl_realtime.payload_types import SILENCE_TOKEN
from sglang_omni.models.moss_vl_realtime.platform_compat import (
    preferred_attention_backend,
    relax_mossvl_flashinfer_guard,
)
from sglang_omni.models.moss_vl_realtime.scheduler import MossVLRealtimeScheduler
from sglang_omni.models.moss_vl_realtime.segment import MossVLRealtimeSegmentBuilder
from sglang_omni.scheduling.engine_factory import SGLangGenerationEngineBuilder

# Representative encoder length used as the CUDA-graph capture fill value;
# only needs to be non-zero so cross-attention kernels are captured.
DECODE_GRAPH_ENCODER_LEN_FILL_VALUE = 4096


class MossVLRealtimeEngineBuilder(SGLangGenerationEngineBuilder):
    model_name = "MOSS-VL-Realtime"
    model_arch_override = "MossVLRealtimeForConditionalGeneration"

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_new_tokens: int,
        context_length: int,
        mem_fraction_static: float | None,
        frame_resolver: Any = None,
        parked_request_timeout_s: float = 300.0,
        disable_cuda_graph: bool = False,
        page_size: int = 1,
        enable_async_decode: bool = False,
        frame_window_config: Any | None = None,
    ) -> None:
        self.max_running_requests = int(max_running_requests)
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests must be at least 1")
        self.max_new_tokens = int(max_new_tokens)
        self.context_length = int(context_length)
        self.mem_fraction_static = mem_fraction_static
        self.frame_resolver = frame_resolver
        self.parked_request_timeout_s = float(parked_request_timeout_s)
        self.disable_cuda_graph = bool(disable_cuda_graph)
        page_size = int(page_size)
        if page_size != 1:
            raise ValueError("MOSS-VL realtime requires page_size == 1")
        self.page_size = page_size
        self.enable_async_decode = bool(enable_async_decode)
        if frame_window_config is not None and not getattr(
            frame_window_config, "enabled", False
        ):
            frame_window_config = None
        self.frame_window_config = frame_window_config
        self.model_vocab_size: int | None = None
        self.processor: Any = None
        self.segment_builder: MossVLRealtimeSegmentBuilder | None = None
        self.silence_token_ids: tuple[int, ...] = ()

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        # Upstream pins Moss-VL to the FlashInfer prefill backend in
        # ServerArgs validation; on NPU the ascend backend takes over, so
        # relax that guard before any ServerArgs is constructed.
        relax_mossvl_flashinfer_guard()
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
        )
        config = self.processor.image_processor
        merge_size = int(
            getattr(config, "merge_size", None)
            or getattr(config, "spatial_merge_size", 2)
        )
        image_token_id = int(
            getattr(self.processor, "image_token_id", None)
            or self.processor.tokenizer.convert_tokens_to_ids("<|image|>")
        )
        self.segment_builder = MossVLRealtimeSegmentBuilder(
            self.processor,
            image_token_id=image_token_id,
            merge_size=merge_size,
        )
        silence_token_id = self.processor.tokenizer.convert_tokens_to_ids(SILENCE_TOKEN)
        if silence_token_id is None:
            silence_token_ids = self.processor.tokenizer.encode(
                SILENCE_TOKEN,
                add_special_tokens=False,
            )
        else:
            silence_token_ids = [silence_token_id]
        self.silence_token_ids = tuple(int(token_id) for token_id in silence_token_ids)
        if not self.silence_token_ids:
            raise ValueError("tokenizer cannot encode the silence marker")
        hf_config = AutoConfig.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
        )
        model_vocab_size = getattr(hf_config, "vocab_size", None)
        text_config = getattr(hf_config, "text_config", None)
        if model_vocab_size is None and text_config is not None:
            model_vocab_size = getattr(text_config, "vocab_size", None)
        self.model_vocab_size = (
            None if model_vocab_size is None else int(model_vocab_size)
        )

    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        defaults = {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": self.disable_cuda_graph,
            "disable_overlap_schedule": True,
            "enable_torch_compile": False,
            "mem_fraction_static": self.mem_fraction_static,
            "max_prefill_tokens": 4096,
            "chunked_prefill_size": 4096,
            "sampling_backend": "pytorch",
            # Incremental encoder insertion currently requires token-granular
            # allocation. Do not patch SGLang's process-global paged allocator.
            "page_size": self.page_size,
            "dtype": dtype,
        }
        attention_backend = preferred_attention_backend()
        # FlashInfer is the project's decode backend on CUDA. (It is also
        # required for decode CUDA graphs: fa3 decode-graph replay indexes
        # req_to_token rows with encoder_lens + arange(max_context_len)
        # when seq_lens_cpu is unavailable, which overflows the row for
        # our encoder-prefix KV layout; FlashInfer re-plans from device
        # buffers each replay and has no such issue.) NPU serves both
        # dimensions through the ascend backend instead.
        defaults["decode_attention_backend"] = attention_backend
        # Setting any single backend dimension stops the upstream MossVL
        # override from injecting its prefill default; pin it explicitly.
        defaults["prefill_attention_backend"] = attention_backend
        if not self.disable_cuda_graph:
            # Only stable-shape decode steps enter the graph; the dynamic
            # multimodal frame extend stays on the eager path.
            defaults["disable_prefill_cuda_graph"] = True
        return defaults

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        if int(overrides.get("tp_size", 1)) > 1:
            # Omni maps every TP rank to one visible local cuda:0. SGLang's
            # custom all-reduce rendezvous requires distinct visible device
            # ordinals, so this topology must use NCCL collectives.
            overrides["disable_custom_all_reduce"] = True

    def setup_model(
        self,
        *,
        model_worker: Any,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: Any,
    ) -> None:
        del checkpoint_dir, device, gpu_id, server_args
        if self.disable_cuda_graph:
            return
        # The MOSS-VL HF config defines no max_source_positions, so SGLang
        # would capture decode graphs with encoder_len fill value 0 and skip
        # the cross-attention kernels. Provide a representative non-zero
        # encoder length for capture; replay re-plans with the real lengths.
        hf_config = model_worker.model_runner.model_config.hf_config
        hf_config.max_source_positions = DECODE_GRAPH_ENCODER_LEN_FILL_VALUE

    def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
        return MossVLRealtimeModelRunner(model_worker, output_proc)

    def make_adapters(self, model: Any) -> tuple[Any, Any]:
        del model
        return request_builders.make_moss_vl_realtime_scheduler_adapters(
            tokenizer=self.processor.tokenizer,
            max_new_tokens=self.max_new_tokens,
            vocab_size=self.model_vocab_size,
        )

    def extra_scheduler_kwargs(self) -> dict[str, Any]:
        return {
            "stream_output_builder": request_builders.make_moss_vl_realtime_stream_output_builder(
                tokenizer=self.processor.tokenizer,
                silence_token_ids=self.silence_token_ids,
                context_length=self.context_length,
            ),
            "enable_overlap": False,
            "enable_async_decode": self.enable_async_decode,
            # bs=1 lookahead is opt-in here: the MOSS workload is dominated by
            # single-stream sessions, and the perf A/B decides whether the
            # flag stays worthwhile.
            "async_decode_min_batch_size": 1,
        }

    def _make_scheduler(self, **kwargs: Any) -> Any:
        extra = kwargs.pop("extra_scheduler_kwargs")
        scheduler_kwargs = {
            "tp_worker": kwargs["model_worker"],
            "tree_cache": kwargs["tree_cache"],
            "req_to_token_pool": kwargs["req_to_token_pool"],
            "token_to_kv_pool_allocator": kwargs["token_to_kv_pool_allocator"],
            "server_args": kwargs["server_args"],
            "model_config": kwargs["model_config"],
            "prefill_manager": kwargs["prefill_manager"],
            "decode_manager": kwargs["decode_manager"],
            "model_runner": kwargs["model_runner"],
            "request_builder": kwargs["request_builder"],
            "result_adapter": kwargs["result_adapter"],
            "abort_callback": self.make_abort_callback(),
            "request_finished_callback": self.make_request_finished_callback(),
            "dp_rank": self.dp_rank,
            "segment_builder": self.segment_builder,
            "frame_resolver": self.frame_resolver,
            "silence_token_ids": self.silence_token_ids,
            "parked_request_timeout_s": self.parked_request_timeout_s,
            "frame_window_config": self.frame_window_config,
        }
        scheduler_kwargs.update(extra)
        return MossVLRealtimeScheduler(**scheduler_kwargs)
