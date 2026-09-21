"""M1.0 smoke: the app imports and boots without a GPU/model.

Run:  <repo>/.venv/bin/python -m server.tests.smoke_import

Verifies every module imports, the app object builds, and (via TestClient) the
lifespan runs with AUTOLOAD_VLM off so no checkpoint/GPU is required. ASR/TTS
startup is allowed to report "not ready" (models may be absent in a bare env);
the point is the wiring holds.
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("AUTOLOAD_VLM", "0")
    os.environ.setdefault("ASR_ENABLED", "0")   # skip model load for a bare-import smoke
    os.environ.setdefault("TTS_ENABLED", "0")
    # bare-import smoke boots the in-process VLM path; the workers plane has its
    # own integration test (server/tests/test_vlm_workers.py)
    os.environ.setdefault("VLM_DEPLOY", "inproc")

    # import surface
    from server import app as app_module  # noqa: F401
    from server import protocol  # noqa: F401
    from server.adapters import base, registry  # noqa: F401
    from server.adapters.asr.funasr_sensevoice import adapter as asr_funasr  # noqa: F401
    from server.adapters.tts import providers as tts_providers  # noqa: F401
    from server.adapters.tts.common import nano_protocol, openai_speech, pool  # noqa: F401
    from server.adapters.tts.moss_tts_nano import adapter as tts_moss_nano  # noqa: F401
    from server.adapters.tts.vllm_omni import adapter as tts_vllm_omni  # noqa: F401
    from server.adapters.tts.cosyvoice3 import adapter as tts_cosyvoice3  # noqa: F401
    from server.adapters.tts.moss_tts_realtime import adapter as tts_mossrt  # noqa: F401
    from server.adapters.vlm.moss_vl_hf import adapter as vlm_hf  # noqa: F401
    from server.adapters.vlm.moss_vl_hf import online_pool as vlm_online_pool  # noqa: F401
    from server.adapters.vlm.moss_vl_sglang import adapter as vlm_offline_sglang  # noqa: F401
    from server.realtime import session, mossvl_patches  # noqa: F401
    from server.session import manager, orchestrator, state  # noqa: F401
    from server.voice import segmenter, tts_session, vad  # noqa: F401
    from server.routers import ops, chat, session_ws, sessions  # noqa: F401
    print("imports: OK")

    # segmenter behaviour
    seg = segmenter.Segmenter(min_chars=8, soft_chars=16, max_chars=32)
    out = seg.feed("你好，这是一个测试。继续")
    assert any("。" in s or len(s) >= 8 for s in out) or seg.buffer, f"segmenter unexpected: {out} buf={seg.buffer!r}"
    print("segmenter: OK", out, "buffer=", repr(seg.buffer))

    # vad meter
    from server.voice.vad import ActivityMeter, measure_pcm
    rms, dur = measure_pcm(b"\x10\x00" * 1600, 16000)
    m = ActivityMeter(rms_threshold=1, min_speech_ms=10, silence_ms=50)
    m.update(rms, dur)
    assert m.has_enough_speech, "meter should register speech"
    print("vad: OK rms=", rms, "dur_ms=", round(dur, 1))

    # routes registered? (check the routers directly — this starlette keeps included
    # routers nested rather than flattening them into app.routes)
    from server.routers import ops as ops_r, chat as c_r, session_ws as sw_r, sessions as ss_r
    paths = set()
    for mod in (ops_r, c_r, sw_r, ss_r):
        paths |= {getattr(r, "path", None) for r in mod.router.routes}
    for expected in (
        "/api/health", "/api/status", "/api/chat/stream", "/api/sessions",
        "/api/sessions/{sid}", "/api/session/{sid}/ws", "/api/models/load",
    ):
        assert expected in paths, f"route missing: {expected} (have {sorted(p for p in paths if p)})"
    print("routes: OK")

    # drive the lifespan directly (builds Runtime, starts voice with models disabled) — no httpx needed
    import asyncio
    from server.deps import get_runtime
    from server.routers import ops as ops_router

    async def _boot():
        async with app_module.app.router.lifespan_context(app_module.app):
            rt = get_runtime()
            assert rt is not None
            h = await ops_router.health()
            assert h.get("ok") is True, h
            st = await ops_router.status(rt)
            assert st.get("ok") is True, st
            return st

    st = asyncio.run(_boot())
    print("app lifespan boot + status: OK ->", {"vlm_loaded": st["vlm"]["loaded"], "capture_mode": st["capture_mode"]})

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
