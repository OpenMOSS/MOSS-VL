from sglang_omni.client.client import Client
from sglang_omni.proto import StreamMessage


def test_default_stream_builder_preserves_control_event() -> None:
    message = StreamMessage(
        request_id="req-1",
        from_stage="moss_vl_realtime",
        chunk={
            "event": "input.frame.processed",
            "seq_no": 2,
            "timestamp": 4.0,
            "final": False,
            "modality": "control",
        },
        modality="control",
    )

    chunk = Client._default_stream_builder("req-1", message)

    assert chunk.control_event == "input.frame.processed"
    assert chunk.control_data["seq_no"] == 2
    assert chunk.modality == "control"


def test_default_stream_builder_does_not_reclassify_data_events() -> None:
    message = StreamMessage(
        request_id="req-1",
        from_stage="event-producer",
        chunk={"event": "domain.event", "text": "payload"},
        modality="text",
    )

    chunk = Client._default_stream_builder("req-1", message)

    assert chunk.control_event is None
    assert chunk.control_data is None
    assert chunk.text == "payload"


def test_default_stream_builder_preserves_text_turn_id() -> None:
    message = StreamMessage(
        request_id="req-1",
        from_stage="moss_vl_realtime",
        chunk={"text": "answer", "turn_id": 3, "modality": "text"},
        modality="text",
    )

    chunk = Client._default_stream_builder("req-1", message)

    assert chunk.text == "answer"
    assert chunk.turn_id == 3
