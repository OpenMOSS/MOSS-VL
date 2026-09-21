"""Exercise the shipped client, gateway, optional memory, ASR and TTS."""
import json
import asyncio
import io
import os
from pathlib import Path
import subprocess
import sys
import time
import wave
import struct
from urllib.parse import urljoin

import requests
import websockets
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / ".repro"
state = json.loads((HOME / "run-state.json").read_text())
ports, profiles = state["ports"], state["profiles"]
api = f"http://127.0.0.1:{ports['api']}"
http = requests.Session()
http.trust_env = False
results = {}


def json_request(method, path, **kwargs):
    response = http.request(method, api + path, timeout=180, **kwargs)
    response.raise_for_status()
    return response.json()


status = json_request("GET", "/api/status")
assert status["vlm"]["loaded"] and status["vlm"]["capacity"] == 4
results["gateway"] = True
web = http.get(f"http://127.0.0.1:{ports['web']}/api/status", timeout=15)
web.raise_for_status()
assert web.json()["vlm"]["loaded"]
results["web_proxy"] = True

picture = HOME / "smoke-frame.png"
frame = Image.new("RGB", (448, 448), "white")
ImageDraw.Draw(frame).rectangle((80, 80, 368, 368), fill="red")
frame.save(picture)
session = json_request("POST", "/v1/realtime/sessions", json={})
url = urljoin(api.replace("http://", "ws://") + "/", session["ws_url"])
url += ("&" if "?" in url else "?") + "ws_token=" + session["ws_token"]
output = HOME / "smoke-vlm.json"
try:
    subprocess.run([str(ROOT / ".venv/bin/python"), str(HOME / "sglang-omni-main/examples/moss_vl_realtime_client.py"),
        "--url", url, "--frame", str(picture), "--timestamp", "0", "--prompt", "Describe the main color in this image.",
        "--max-new-tokens", "32", "--output", str(output)], check=True, timeout=180)
    reply = json.loads(output.read_text())
    assert reply["normalized_text"].replace("<|silence|>", "").strip(), reply
    results["vlm_reply"] = reply["normalized_text"]
finally:
    http.delete(api + "/v1/realtime/sessions/" + session["session_id"], timeout=15)

if "memory" in profiles:
    pi = f"http://127.0.0.1:{ports['pi']}"
    decision = http.post(pi + "/decide", json={"pending_user_text": "我之前说的相机型号是什么？"}, timeout=15)
    decision.raise_for_status()
    assert decision.json()["retrieve"] is True
    compact = http.post(pi + "/compact", json={"journal": "用户: 预算是3000元。\n助手: 好的。\n用户: 纠正，预算改为2000元。"}, timeout=90)
    compact.raise_for_status()
    assert "2000" in json.dumps(compact.json())
    results["memory"] = compact.json()

if "asr" in profiles:
    assert status["voice"]["asr"]["ready"]
    source = HOME / "models/asr/SenseVoiceSmall/example/zh.mp3"
    pcm = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(source), "-ar", "16000", "-ac", "1", "-f", "s16le", "-"])
    recognized = json_request("POST", "/api/asr", data=pcm, headers={"Content-Type": "application/octet-stream"})
    assert recognized.get("text", "").strip(), recognized
    results["asr_text"] = recognized["text"]

    async def voice_session():
        voice_pcm = pcm
        if "tts" in profiles:
            question = http.post(api + "/api/tts", json={"text": "请告诉我图片里的主要颜色是什么？", "voice": "Junhao"}, timeout=180)
            question.raise_for_status()
            voice_pcm = subprocess.check_output(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ar", "16000",
                "-ac", "1", "-f", "s16le", "-"], input=question.content)
        created = json_request("POST", "/api/sessions", json={"config": {"capture_mode": "ptt", "tts_voice": "mute"}})
        transcribed, answered = False, False
        try:
            async with websockets.connect(urljoin(api.replace("http://", "ws://") + "/", created["ws_url"]),
                                          max_size=64 * 1024 * 1024) as ws:
                welcome = json.loads(await asyncio.wait_for(ws.recv(), 15))
                assert welcome["type"] == "session.created", welcome
                jpeg = io.BytesIO()
                frame.save(jpeg, format="JPEG")
                await ws.send(b"\x02" + struct.pack(">I", 0) + jpeg.getvalue())
                await ws.send(json.dumps({"v": 1, "type": "input.audio.start"}))
                for offset in range(0, len(voice_pcm), 5120):
                    await ws.send(b"\x01" + voice_pcm[offset:offset + 5120])
                    await asyncio.sleep(.16)
                await ws.send(json.dumps({"v": 1, "type": "input.audio.commit"}))
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    event = await asyncio.wait_for(ws.recv(), max(.1, deadline - time.monotonic()))
                    if not isinstance(event, str):
                        continue
                    event = json.loads(event)
                    if event["type"] in {"error", "session.error"}:
                        raise RuntimeError(event)
                    if event["type"] == "input.transcription.done":
                        transcribed = bool(event.get("text", "").strip())
                        results["demo_voice_transcription"] = event.get("text")
                        if "tts" not in profiles:
                            await ws.send(json.dumps({"v": 1, "type": "text.input",
                                "text": "What is the main color in the image? Reply briefly."}))
                    if event["type"] == "response.text.delta" and event.get("delta", "").strip():
                        answered = True
                    if transcribed and answered:
                        break
                assert transcribed and answered, "Demo voice turn did not complete transcription and generate text"
        finally:
            json_request("DELETE", "/api/sessions/" + created["session_id"])
        return True

    results["demo_voice_turn" if "tts" in profiles else "demo_asr_and_text_turn"] = asyncio.run(voice_session())

if "tts" in profiles:
    assert status["voice"]["tts"]["ready"]
    audio = http.post(api + "/api/tts", json={"text": "你好，这是语音测试。", "voice": "Junhao"}, timeout=180)
    audio.raise_for_status()
    assert len(audio.content) > 1000 and audio.content[:4] == b"RIFF"
    with wave.open(io.BytesIO(audio.content)) as wav:
        assert wav.getsampwidth() == 2
        samples = np.frombuffer(wav.readframes(min(wav.getnframes(), 2_000_000)), dtype="<i2")
        assert samples.size and np.max(np.abs(samples.astype(np.int32))) > 32, "TTS produced silent audio"
    (HOME / "smoke-tts.wav").write_bytes(audio.content)
    results["tts_audio_bytes"] = len(audio.content)

(HOME / "smoke-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(json.dumps(results, ensure_ascii=False), flush=True)
