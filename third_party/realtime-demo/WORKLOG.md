# WORKLOG

- **Repo**: /inspire/hdd/project/video-understanding/public/personal/yxchen/mossvl_realtime_inference
- **Note**: 本日志条目继承自原始 Demo 仓库（chwang/live/MOSS-VL-Realtime_Demo_App），迁移时随代码一并带入；其中的工作状态描述以源仓库当时为准。
- **Date range**: 2026-08-17 ~ 2026-08-17
- **Total logs**: 1
- **Last updated**: 2026-08-17

---

## 2026-08-17 · MiniMax Cloud TTS Provider + ElevenLabs Prosody Fixes

1. **Added MiniMax cloud TTS as a second external TTS lane**
   1. New `server/adapters/tts/minimax/adapter.py` (~280 lines) implementing the T2A v2 streaming API: one SSE POST per segment to `/v1/t2a_v2`, decoding hex PCM16 chunks into the shared `synthesize_pcm` contract; includes a curated Chinese-first preset voice registry and boot-time health probe (1-char synth + `/v1/get_voice`).
   2. Wired through the stack: provider aliases/`EXTERNAL_PROVIDERS` in `providers.py`, factory in `registry.py`, full `MINIMAX_*` settings block in `config.py` (key, base URL, group ID, model, voice, sample rate, proxy, warmup), optional boot lane in `deps.py` (`tts_minimax`, exposed in `voice_status()`), and session engine pick in `routers/sessions.py` (lazy probe-retry, fallback to local pool with warning). Motivation: speech-02 is the strongest Chinese cloud TTS and the mainland endpoint is directly reachable from the box.
2. **ElevenLabs voice-consistency fixes**
   1. Pinned `voice_settings.stability` (new `ELEVENLABS_STABILITY`, default 0.65) per request so timbre no longer drifts with the voice's stored defaults across segments.
   2. Added per-session `previous_text` (thread-local, committed only after a clean read, capped at 512 chars) for cross-segment prosody continuity, per the ElevenLabs docs.
3. **Frontend generalized from one cloud lane to N cloud TTS lanes**
   1. `App.tsx` now probes both `tts_elevenlabs` and `tts_minimax` from `/api/status` into a `cloudTts` map; the Engine select lists MiniMax ("中文最佳") and/or ElevenLabs only when the lane is ready, and the voice select re-seats to that lane's `voice_id` registry.
   2. A lane whose boot probe failed (bad key, unreachable) no longer surfaces as a selectable-but-broken option; `useSession.ts` `ttsEngine` type extended to `'local' | 'elevenlabs' | 'minimax'`.
4. **Deploy docs**
   1. `.env.deploy.example` gained a commented MiniMax config block (key, documented host vs. mirror gotcha, model/voice/sample-rate, proxy, warmup) plus the new `ELEVENLABS_STABILITY` knob.

（原日志称当日工作未提交、last commit 为 `afab88d`——那是源仓库 2026-08-17 时的状态；在本仓库中，上述工作已包含于 Initial commit `7682620`。）
