from __future__ import annotations

import pytest

from sglang_omni.models.moss_vl_realtime.benchmark_cases import compile_training_record


def _compile(messages, segments):
    return compile_training_record(
        {
            "messages": messages,
            "videos": [{"video_path": "/video.mp4", "segments": segments}],
        },
        case_id="case",
        source_tag="source",
        task_shape="shape",
        source_path="/source.jsonl",
        source_line=7,
    )


def test_compile_training_record_preserves_singleton_segments_and_outputs() -> None:
    case = _compile(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<|silence|><|video|>answer<|video|><|silence|>",
            },
        ],
        [[3], [4]],
    )

    assert case["system_prompt"] == "system"
    assert case["initial_prompt"] == "question"
    assert case["initial_expected"] == "<|silence|>"
    assert case["frame_interval_seconds"] == 1.0
    assert case["events"] == [
        {
            "type": "frame",
            "frame_index": 0,
            "seq_no": 0,
            "timestamp": 3.0,
            "segment": [3.0],
            "expected_before": "<|silence|>",
            "expected_after": "answer",
            "final": False,
        },
        {
            "type": "frame",
            "frame_index": 1,
            "seq_no": 1,
            "timestamp": 4.0,
            "segment": [4.0],
            "expected_before": "answer",
            "expected_after": "<|silence|>",
            "final": True,
        },
    ]


def test_compile_training_record_attaches_injected_text_to_next_frame() -> None:
    case = _compile(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<|silence|><|video|><|silence|>"},
            {"role": "user", "content": "Subtitle: hello"},
            {
                "role": "assistant",
                "content": "<|silence|><|video|>updated answer",
            },
        ],
        [[0], [1]],
    )

    assert "prompt" not in case["events"][0]
    assert case["events"][1]["prompt"] == "Subtitle: hello"
    assert case["events"][1]["expected_after"] == "updated answer"


def test_compile_training_record_preserves_text_only_probe() -> None:
    case = _compile(
        [
            {"role": "user", "content": "watch"},
            {"role": "assistant", "content": "<|silence|><|video|><|silence|>"},
            {"role": "user", "content": "How many?"},
            {"role": "assistant", "content": "<|response|>3<|silence|>"},
        ],
        [[0]],
    )

    assert case["frame_count"] == 1
    assert case["events"][1] == {
        "type": "prompt",
        "prompt": "How many?",
        "expected_after": "<|response|>3<|silence|>",
        "seq_no": 1,
        "final": True,
    }
    assert case["events"][0]["final"] is False


def test_compile_training_record_rejects_non_singleton_segments() -> None:
    with pytest.raises(ValueError, match="one frame timestamp"):
        _compile(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<|silence|><|video|>"},
            ],
            [[0, 1]],
        )
