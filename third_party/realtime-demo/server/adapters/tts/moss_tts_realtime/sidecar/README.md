# Vendored MOSS-TTS realtime stack (the `moss_tts_realtime_native` fallback provider)

Vendored 2026-07-14 from `github.com/OpenMOSS/MOSS-TTS` (main):

- `third_party/MOSS-TTS/moss_tts_realtime/` ← upstream dir excluding
  `finetuning/` (the `mossttsrealtime/` package, `fast_api.py`, `inferencer.py`,
  examples, prompt `audio/`)
- `third_party/MOSS-TTS/LICENSE` (Apache-2.0)

Unlike the other sidecars there is NO backend/ wrapper of ours: the spawner
(server/sidecars.py) runs the UPSTREAM `fast_api.py` session server directly
(`uvicorn fast_api:app`, cwd = `third_party/MOSS-TTS/moss_tts_realtime`),
configured via its own env surface:

- `MOSS_TTS_MODEL_PATH` / `MOSS_TTS_TOKENIZER_PATH` ← `MOSSRT_MODEL`
  (`MODELS_DIR/MOSS-TTS-Realtime`)
- `MOSS_TTS_CODEC_MODEL_PATH` ← `MOSSRT_CODEC` (`MODELS_DIR/MOSS-Audio-Tokenizer`,
  the FULL tokenizer — not the Nano one)
- `MOSS_TTS_DEVICE=cuda:0` (CUDA_VISIBLE_DEVICES picks the physical card),
  `MOSS_TTS_ATTN_IMPL=sdpa` (`MOSSRT_ATTN_IMPL` overrides)

The model loads inside fast_api's FastAPI lifespan, so `/health` answers only
once the model is up — the standard spawn health-gate just works
(slow_first_boot budget 600 s).

Client: `MossRtNativeEngine` (../adapter.py) drives the session protocol —
`POST /tts/session/start {session_id, new_turn, prompt_audio}` →
`POST /tts/session/push {text, is_final:true}` →
`GET /tts/session/{sid}/audio` (PCM16LE stream, X-Audio-* headers) →
`POST /tts/session/close`. One session per synthesized segment today
(batch-1 upstream ⇒ `tts_sessions_per_sidecar=1` sizing).

STREAMING (item 4 — IMPLEMENTED): `MossRtNativeEngine.open_stream()` +
`MossRtStream` map one TTS turn onto ONE session — `start` once, `push_text`
per segment as it arrives, audio streams from `/audio` continuously, `is_final`
then EOF → `tts_turn_end`; a barge-in `close()`s it. `TtsSession` drives this
whenever the engine advertises `supports_streaming` (adapter caps
`token_streaming_input=True`); other providers keep the per-segment path. This
removes the per-segment cold-start gaps and prosody resets (the stutter).

FOLLOW-UP (still open): the same session protocol is the surface for the
model's context-aware MULTI-TURN mode (turn-0 KV reset, turn-1+ reuse, 32K ctx
≈ 40 min) — mapping a whole CONVERSATION onto one long-lived session (not just
one turn) would add cross-turn voice/prosody coherence. Needs per-session
engine affinity in the pool.

venv: `.venv-mossrt` — `scripts/build_venv_mossrt.sh` /
`requirements-mossrt.txt` (transformers 5.0, torch 2.9.1+cu128; conflicts
with `.venv` and `.venv-vllm` are the reason for the separate venv).
