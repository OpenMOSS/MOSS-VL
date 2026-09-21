"""MOSS-VL-Realtime_Demo_App backend (server/).

FastAPI gateway that fronts three protocol surfaces used by the demo frontend:
  - WS /api/voice/ws            microphone ASR in, streamed TTS PCM out
  - WS /api/realtime/{sid}/ws   JSON control + binary JPEG frames in, text chunks out
  - WS /api/chat/stream         offline (non-realtime) multimodal chat

The heavy engines sit behind pluggable adapters (server/adapters). The MOSS-VL
realtime loop is a faithful port of the reference board backend at
/inspire/hdd/project/video-understanding/public/personal/train/board.
"""

__version__ = "0.1.0"
