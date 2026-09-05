import asyncio
import importlib.util
import json
import sys
import types
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

realtime_dir = Path(__file__).resolve().parents[1]
module_path = realtime_dir / "run_online_inference.py"
module_spec = importlib.util.spec_from_file_location("run_online_inference", module_path)
run_online_inference = importlib.util.module_from_spec(module_spec)
pil = types.ModuleType("PIL")
pil.Image = types.SimpleNamespace()
video_sources = types.ModuleType("video_sources")
video_sources.VideoFrame = object
for function_name in (
    "iter_camera_frames",
    "iter_screen_frames",
    "iter_synthetic_frames",
    "iter_video_file",
    "iter_video_timestamps",
    "pace_frames",
):
    setattr(video_sources, function_name, lambda *_args, **_kwargs: ())
with patch.dict(sys.modules, {"PIL": pil, "video_sources": video_sources}):
    module_spec.loader.exec_module(run_online_inference)


class FakeFastAPI:
    def __init__(self, **_kwargs):
        self.websocket_endpoints = {}

    def get(self, _path):
        return lambda endpoint: endpoint

    def websocket(self, path):
        def register(endpoint):
            self.websocket_endpoints[path] = endpoint
            return endpoint

        return register


class FakeWebSocketDisconnect(Exception):
    pass


class FakeWebSocket:
    def __init__(self):
        self.messages = []
        self.closed = False
        self._receive_blocker = asyncio.Event()

    async def accept(self):
        pass

    async def receive_text(self):
        return json.dumps({"type": "start"})

    async def receive(self):
        await self._receive_blocker.wait()

    async def send_text(self, payload):
        self.messages.append(json.loads(payload))

    async def send_json(self, payload):
        self.messages.append(payload)

    async def close(self, code=None):
        self.closed = True


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = deque(outcomes)
        self.active = False
        self.pending_frames = 0

    def start(self):
        pass

    def poll_output(self, _timeout):
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self, _timeout):
        pass


class FakeModel:
    def __init__(self, session):
        self.session = session

    def create_realtime_session(self, *_args, **_kwargs):
        return self.session


def service_args():
    return SimpleNamespace(
        decode_batch_size=8,
        do_sample=False,
        frame_queue_size=256,
        max_frame_bytes=20 * 1024 * 1024,
        max_new_tokens=4096,
        max_tokens_per_second=12,
        prompt="Describe the video.",
        repetition_penalty=1.0,
        sample_fps=1.0,
        system_prompt=None,
        temperature=0.7,
        top_k=20,
        top_p=0.8,
    )


def build_endpoint(session):
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FakeFastAPI
    fastapi.WebSocketDisconnect = FakeWebSocketDisconnect
    with patch.dict(sys.modules, {"fastapi": fastapi}):
        app = run_online_inference.create_service_app(
            FakeModel(session),
            processor=object(),
            default_args=service_args(),
        )
    return app.websocket_endpoints["/v1/realtime"]


class RealtimeServiceOutputTest(unittest.IsolatedAsyncioTestCase):
    async def test_drains_output_before_session_end(self):
        endpoint = build_endpoint(FakeSession(["first chunk", "final chunk", None]))
        websocket = FakeWebSocket()

        await endpoint(websocket)

        self.assertEqual(
            websocket.messages,
            [
                {"type": "ready"},
                {"type": "output", "text": "first chunk"},
                {"type": "output", "text": "final chunk"},
                {"type": "session_end"},
            ],
        )

    async def test_drains_output_before_reporting_session_error(self):
        endpoint = build_endpoint(
            FakeSession(
                [
                    "first chunk",
                    "final chunk",
                    RuntimeError("inference failed"),
                ]
            )
        )
        websocket = FakeWebSocket()

        await endpoint(websocket)

        self.assertEqual(
            websocket.messages,
            [
                {"type": "ready"},
                {"type": "output", "text": "first chunk"},
                {"type": "output", "text": "final chunk"},
                {"type": "error", "message": "inference failed"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
