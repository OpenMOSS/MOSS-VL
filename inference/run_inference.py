from __future__ import annotations

import argparse
import json
import queue
import re
import time
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


IMAGE_MEDIA_DEFAULTS: Dict[str, Any] = {
    "min_pixels": 4096,
    "max_pixels": 16777216,
    "multi_image_max_pixels": 201326592,
    "patch_size": 16,
    "temporal_patch_size": 1,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}

VIDEO_MEDIA_DEFAULTS: Dict[str, Any] = {
    "min_pixels": 4096,
    "max_pixels": 16777216,
    "video_max_pixels": 201326592,
    "patch_size": 16,
    "temporal_patch_size": 1,
    "merge_size": 2,
    "video_fps": 1.0,
    "min_frames": 1,
    "max_frames": 256,
    "num_extract_threads": 4,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}

GENERATE_DEFAULTS: Dict[str, Any] = {
    "max_new_tokens": 256,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "do_sample": False,
    "vision_chunked_length": 64,
}

LEGACY_MODE_ALIASES = ("offline", "image", "video", "batch")
IMAGE_PLACEHOLDER = "<|image|>"
VIDEO_PLACEHOLDER = "<|video|>"
MEDIA_PLACEHOLDER_PATTERN = re.compile(
    f"({re.escape(IMAGE_PLACEHOLDER)}|{re.escape(VIDEO_PLACEHOLDER)})"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MOSS-VL inference via the generic offline_generate API.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint directory, for example /path/to/dummy-checkpoint.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=LEGACY_MODE_ALIASES,
        help="Legacy label kept for compatibility. All modes now route through offline_generate.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON/JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save output JSON. Defaults to <input>_results.json.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout per query while waiting for offline_generate to finish.",
    )
    parser.add_argument(
        "--cross-attention-implementation",
        default=None,
        help="Optional cross-attention backend override provided by the checkpoint.",
    )
    return parser.parse_args()


def load_model(
    checkpoint: str,
    cross_attention_implementation: str | None = None,
):
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        frame_extract_num_threads=1,
    )
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }
    if cross_attention_implementation is not None:
        model_kwargs["cross_attention_implementation"] = cross_attention_implementation
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)
    return model, processor


def count_video_units(video_entry: Any) -> int:
    if isinstance(video_entry, dict) and "segments" in video_entry:
        return len(video_entry.get("segments") or [])
    return 1


def expand_segment_video_entry(video_entry: Any) -> List[Any]:
    if isinstance(video_entry, dict) and "segments" in video_entry:
        segments = video_entry.get("segments") or []
        return [{**video_entry, "segments": [segment]} for segment in segments]
    return [video_entry]


def count_video_placeholders_in_content(content: Any) -> int:
    if isinstance(content, str):
        return content.count(VIDEO_PLACEHOLDER)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, str):
                total += item.count(VIDEO_PLACEHOLDER)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                total += item["text"].count(VIDEO_PLACEHOLDER)
        return total
    return 0


def expand_video_placeholders_in_text(
    text: str,
    video_units: List[int],
    video_index: int,
) -> tuple[str, int]:
    if VIDEO_PLACEHOLDER not in text:
        return text, video_index

    pieces = text.split(VIDEO_PLACEHOLDER)
    rebuilt = [pieces[0]]
    next_index = video_index
    for suffix in pieces[1:]:
        if next_index < len(video_units):
            rebuilt.append(VIDEO_PLACEHOLDER * video_units[next_index])
            next_index += 1
        else:
            rebuilt.append(VIDEO_PLACEHOLDER)
        rebuilt.append(suffix)
    return "".join(rebuilt), next_index


def expand_video_placeholders_in_content(
    content: Any,
    video_units: List[int],
    video_index: int,
) -> tuple[Any, int]:
    if isinstance(content, str):
        return expand_video_placeholders_in_text(content, video_units, video_index)
    if isinstance(content, list):
        expanded_content = []
        next_index = video_index
        for item in content:
            if isinstance(item, str):
                new_item, next_index = expand_video_placeholders_in_text(
                    item,
                    video_units,
                    next_index,
                )
                expanded_content.append(new_item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                new_item = dict(item)
                new_item["text"], next_index = expand_video_placeholders_in_text(
                    item["text"],
                    video_units,
                    next_index,
                )
                expanded_content.append(new_item)
            else:
                expanded_content.append(item)
        return expanded_content, next_index
    return content, video_index


def expand_segment_video_placeholders(
    messages: List[Dict[str, Any]],
    videos: List[Any],
) -> List[Dict[str, Any]]:
    video_units = [count_video_units(video) for video in videos]
    if not video_units:
        return [dict(message) for message in messages if isinstance(message, dict)]

    raw_video_count = len(video_units)
    expanded_video_count = sum(video_units)
    if expanded_video_count == raw_video_count:
        return [dict(message) for message in messages if isinstance(message, dict)]

    explicit_placeholder_count = sum(
        count_video_placeholders_in_content(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
    )
    if explicit_placeholder_count != raw_video_count:
        return [dict(message) for message in messages if isinstance(message, dict)]

    expanded_messages: List[Dict[str, Any]] = []
    next_index = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        message_copy = dict(message)
        message_copy["content"], next_index = expand_video_placeholders_in_content(
            message_copy.get("content", ""),
            video_units,
            next_index,
        )
        expanded_messages.append(message_copy)
    return expanded_messages


def trim_messages_for_inference(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trimmed = [dict(message) for message in messages if isinstance(message, dict)]
    while trimmed and trimmed[-1].get("role") == "assistant":
        trimmed.pop()
    return trimmed


def append_text_item(items: List[Dict[str, Any]], text: str) -> None:
    if not text:
        return
    if items and items[-1].get("type") == "text":
        items[-1]["text"] += text
    else:
        items.append({"type": "text", "text": text})


def materialize_text_with_media(
    text: str,
    images: List[str],
    videos: List[Any],
    image_index: int,
    video_index: int,
) -> tuple[List[Dict[str, Any]], int, int]:
    items: List[Dict[str, Any]] = []
    for part in MEDIA_PLACEHOLDER_PATTERN.split(text):
        if not part:
            continue
        if part == IMAGE_PLACEHOLDER:
            if image_index >= len(images):
                raise ValueError("Encountered `<|image|>` placeholder without a matching image.")
            items.append({"type": "image", "image": images[image_index]})
            image_index += 1
        elif part == VIDEO_PLACEHOLDER:
            if video_index >= len(videos):
                raise ValueError("Encountered `<|video|>` placeholder without a matching video.")
            items.append({"type": "video", "video": videos[video_index]})
            video_index += 1
        else:
            append_text_item(items, part)
    return items, image_index, video_index


def materialize_message_content(
    content: Any,
    images: List[str],
    videos: List[Any],
    image_index: int,
    video_index: int,
) -> tuple[Any, int, int]:
    if isinstance(content, str):
        items, image_index, video_index = materialize_text_with_media(
            content,
            images,
            videos,
            image_index,
            video_index,
        )
        return items or [{"type": "text", "text": ""}], image_index, video_index

    if isinstance(content, list):
        normalized_items: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                text_items, image_index, video_index = materialize_text_with_media(
                    item,
                    images,
                    videos,
                    image_index,
                    video_index,
                )
                normalized_items.extend(text_items)
            elif isinstance(item, dict):
                if item.get("type") == "image" or "image" in item or "image_url" in item:
                    normalized_items.append(dict(item))
                elif item.get("type") == "video" or "video" in item:
                    normalized_items.append(dict(item))
                elif "text" in item:
                    text_items, image_index, video_index = materialize_text_with_media(
                        str(item.get("text", "")),
                        images,
                        videos,
                        image_index,
                        video_index,
                    )
                    normalized_items.extend(text_items)
                else:
                    normalized_items.append(dict(item))
            else:
                append_text_item(normalized_items, str(item))
        return normalized_items or [{"type": "text", "text": ""}], image_index, video_index

    return content, image_index, video_index


def convert_messages_sample_to_query(sample: Dict[str, Any], raw_messages: Any) -> Dict[str, Any]:
    if not isinstance(raw_messages, list):
        raise ValueError("Expected `messages`/`conversations` to be a list.")

    base_query: Dict[str, Any] = dict(sample)
    base_query.pop("response", None)
    base_query["media_kwargs"] = dict(sample.get("media_kwargs") or {})
    base_query["generate_kwargs"] = dict(sample.get("generate_kwargs") or {})

    images = list(sample.get("images") or [])
    videos = list(sample.get("videos") or [])
    expanded_videos = [unit for video in videos for unit in expand_segment_video_entry(video)]

    messages = expand_segment_video_placeholders(raw_messages, videos)
    messages = trim_messages_for_inference(messages)
    if not messages:
        raise ValueError("No user-facing messages remain after trimming assistant turns.")

    normalized_messages = []
    image_index = 0
    video_index = 0
    for message in messages:
        message_copy = dict(message)
        message_copy["content"], image_index, video_index = materialize_message_content(
            message_copy.get("content", ""),
            images,
            expanded_videos,
            image_index,
            video_index,
        )
        normalized_messages.append(message_copy)

    if image_index != len(images):
        raise ValueError(
            f"Not all images were consumed by placeholders: used {image_index}, total {len(images)}."
        )
    if video_index != len(expanded_videos):
        raise ValueError(
            f"Not all videos were consumed by placeholders: used {video_index}, total {len(expanded_videos)}."
        )

    base_query["messages"] = normalized_messages
    return base_query


def convert_prompt_sample_to_query(sample: Dict[str, Any]) -> Dict[str, Any]:
    base_query: Dict[str, Any] = dict(sample)
    base_query.pop("response", None)
    base_query["media_kwargs"] = dict(sample.get("media_kwargs") or {})
    base_query["generate_kwargs"] = dict(sample.get("generate_kwargs") or {})

    content: List[Dict[str, Any]] = []
    for image in sample.get("images") or []:
        content.append({"type": "image", "image": image})
    for video in sample.get("videos") or []:
        for video_unit in expand_segment_video_entry(video):
            content.append({"type": "video", "video": video_unit})

    prompt = str(sample.get("prompt", ""))
    if prompt or not content:
        content.append({"type": "text", "text": prompt})

    messages: List[Dict[str, Any]] = []
    if sample.get("system_prompt") is not None:
        messages.append({"role": "system", "content": str(sample["system_prompt"])})
    messages.append({"role": "user", "content": content})
    base_query["messages"] = messages
    return base_query


def normalize_loaded_query(sample: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sample, dict):
        raise ValueError(f"Each query must be a JSON object, but got {type(sample).__name__}.")

    if sample.get("messages") is not None:
        return convert_messages_sample_to_query(sample, sample.get("messages"))
    if sample.get("conversations") is not None:
        return convert_messages_sample_to_query(sample, sample.get("conversations"))
    if sample.get("prompt") is not None:
        return convert_prompt_sample_to_query(sample)
    raise ValueError(
        "Unsupported query format. Expected one of: `messages`, `conversations`, or `prompt`."
    )


def load_queries(input_path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Failed to parse JSONL line {line_number} in {input_path}: {exc}"
                    ) from exc
                samples.append(normalize_loaded_query(sample))
        return samples

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {input_path}, but got {type(data).__name__}.")
    return [normalize_loaded_query(sample) for sample in data]


def resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def resolve_image_reference(base_dir: Path, value: str) -> str:
    if urlparse(value).scheme.lower() in {"http", "https"}:
        return value
    return resolve_path(base_dir, value)


def resolve_video_entry(base_dir: Path, value: Any) -> Any:
    if isinstance(value, str):
        return resolve_path(base_dir, value)
    if isinstance(value, dict):
        resolved = dict(value)
        if "video_path" not in resolved:
            raise ValueError(f"Video dict is missing `video_path`: {value}")
        resolved["video_path"] = resolve_path(base_dir, resolved["video_path"])
        return resolved
    raise ValueError(f"Unsupported video entry type: {type(value).__name__}")


def resolve_message_content_media_paths(content: Any, base_dir: Path) -> Any:
    if not isinstance(content, list):
        return content

    resolved_content = []
    for item in content:
        if not isinstance(item, dict):
            resolved_content.append(item)
            continue

        item_copy = dict(item)
        if item_copy.get("type") == "image_url":
            item_copy["type"] = "image"
        if item_copy.get("type") == "image" or "image" in item_copy or "image_url" in item_copy:
            if item_copy.get("image") is not None:
                item_copy["image"] = resolve_image_reference(base_dir, item_copy["image"])
            if item_copy.get("image_url") is not None:
                item_copy["image_url"] = resolve_image_reference(base_dir, item_copy["image_url"])
        elif item_copy.get("type") == "video" or "video" in item_copy:
            if item_copy.get("video") is not None:
                item_copy["video"] = resolve_video_entry(base_dir, item_copy["video"])
        resolved_content.append(item_copy)
    return resolved_content


def resolve_query_media_paths(query: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    resolved = dict(query)
    resolved["images"] = [
        resolve_image_reference(base_dir, image) for image in query.get("images", [])
    ]
    resolved["videos"] = [resolve_video_entry(base_dir, video) for video in query.get("videos", [])]
    resolved["media_kwargs"] = dict(query.get("media_kwargs") or {})
    resolved["generate_kwargs"] = dict(query.get("generate_kwargs") or {})

    if query.get("messages") is not None:
        resolved_messages = []
        for message in query.get("messages") or []:
            if not isinstance(message, dict):
                continue
            message_copy = dict(message)
            message_copy["content"] = resolve_message_content_media_paths(message.get("content", ""), base_dir)
            resolved_messages.append(message_copy)
        resolved["messages"] = resolved_messages

    return resolved


def iter_message_media_items(messages: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict):
                yield item


def query_has_images(query: Dict[str, Any]) -> bool:
    if query.get("images"):
        return True
    return any(
        item.get("type") == "image" or "image" in item or "image_url" in item
        for item in iter_message_media_items(query.get("messages") or [])
    )


def query_has_videos(query: Dict[str, Any]) -> bool:
    if query.get("videos"):
        return True
    return any(
        item.get("type") == "video" or "video" in item
        for item in iter_message_media_items(query.get("messages") or [])
    )


def merge_defaults(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(overrides or {})
    return merged


def normalize_query_for_offline_generate(query: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    resolved = resolve_query_media_paths(query, base_dir)

    default_media_kwargs: Dict[str, Any] = {}
    if query_has_images(resolved):
        default_media_kwargs.update(IMAGE_MEDIA_DEFAULTS)
    if query_has_videos(resolved):
        default_media_kwargs.update(VIDEO_MEDIA_DEFAULTS)

    normalized = dict(resolved)
    normalized["media_kwargs"] = merge_defaults(default_media_kwargs, resolved["media_kwargs"])
    normalized["generate_kwargs"] = merge_defaults(GENERATE_DEFAULTS, resolved["generate_kwargs"])
    return normalized


def collect_offline_generate_output(
    output_queue: "queue.Queue[str]",
    timeout_seconds: float,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    chunks: List[str] = []
    events: List[str] = []
    error_text = None

    while time.time() < deadline:
        remaining = max(0.1, min(1.0, deadline - time.time()))
        try:
            item = output_queue.get(timeout=remaining)
        except queue.Empty:
            continue

        events.append(item)
        if item == "<|round_start|>":
            continue
        if item == "<|round_end|>":
            return {
                "text": "".join(chunks),
                "events": events,
                "error": error_text,
            }
        if item.startswith("[ERROR] "):
            error_text = item
            continue
        chunks.append(item)

    raise TimeoutError("Timed out waiting for offline_generate to finish the current round.")


def run_offline_generate_query(
    model,
    processor,
    query: Dict[str, Any],
    timeout_seconds: float,
) -> Dict[str, Any]:
    input_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    output_queue: "queue.Queue[str]" = queue.Queue()
    worker = Thread(
        target=model.offline_generate,
        args=(processor, input_queue, output_queue),
        kwargs={"vision_chunked_length": query["generate_kwargs"].get("vision_chunked_length", 64)},
        daemon=True,
    )
    worker.start()

    try:
        input_queue.put(dict(query))
        result = collect_offline_generate_output(output_queue, timeout_seconds)
    finally:
        input_queue.put({"stop_offline_generate": True})
        worker.join(timeout=30.0)

    if result["error"] is not None:
        raise RuntimeError(result["error"])
    return result


def build_single_result(
    index: int,
    original_query: Dict[str, Any],
    response_text: str,
    elapsed_seconds: float,
    events: List[str],
) -> Dict[str, Any]:
    return {
        "index": index,
        "prompt": original_query.get("prompt", ""),
        "messages": original_query.get("messages"),
        "images": original_query.get("images", []),
        "videos": original_query.get("videos", []),
        "media_kwargs": original_query.get("media_kwargs", {}),
        "generate_kwargs": original_query.get("generate_kwargs", {}),
        "text": response_text,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "event_count": len(events),
    }


def run_queries_via_offline_generate(
    model,
    processor,
    original_queries: List[Dict[str, Any]],
    input_dir: Path,
    timeout_seconds: float,
) -> Dict[str, Any]:
    results = []

    overall_start = time.time()
    for index, original_query in enumerate(original_queries):
        resolved_query = normalize_query_for_offline_generate(original_query, input_dir)

        start_time = time.time()
        run_result = run_offline_generate_query(
            model,
            processor,
            resolved_query,
            timeout_seconds=timeout_seconds,
        )
        elapsed_seconds = time.time() - start_time
        results.append(
            build_single_result(
                index=index,
                original_query=original_query,
                response_text=run_result["text"],
                elapsed_seconds=elapsed_seconds,
                events=run_result["events"],
            )
        )

    return {
        "results": results,
        "elapsed_seconds": round(time.time() - overall_start, 3),
    }


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_results.json")


def save_json(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else default_output_path(input_path)
    )

    queries = load_queries(input_path)
    model, processor = load_model(
        checkpoint,
        cross_attention_implementation=args.cross_attention_implementation,
    )
    output = run_queries_via_offline_generate(
        model,
        processor,
        queries,
        input_path.parent,
        timeout_seconds=args.timeout_seconds,
    )

    payload: Dict[str, Any] = {
        "mode": "offline_generate",
        "mode_alias": args.mode,
        "checkpoint": checkpoint,
        "input": str(input_path),
        **output,
    }

    save_json(output_path, payload)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
