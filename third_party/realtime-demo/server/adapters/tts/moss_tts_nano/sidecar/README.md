# Vendored pytorch TTS sidecar (the `moss_tts_nano` provider)

Copied 2026-07-11 from the board so the demo no longer depends on
`BOARD_ROOT` existing:

- `backend/moss_tts_nano_sidecar.py`
  ← `train/board/backend/moss_tts_nano_sidecar.py` (verbatim)
- `third_party/MOSS-TTS-Nano/`
  ← `train/board/third_party/MOSS-TTS-Nano/` excluding `models/` (730 MB ONNX
  copy — unused, we run the pytorch backend), `finetuning/`, `examples/`,
  `assets/images|videos`, `__pycache__`. The `assets/audio/` voice-clone
  prompts (5.8 MB) ARE vendored — both TTS providers read them.

Layout is preserved on purpose: the sidecar resolves its roots RELATIVE to its
own file (`BOARD_ROOT = parents[1]`, `third_party/MOSS-TTS-Nano` beside it),
so the verbatim copy just works and future re-syncs are a plain re-copy.

The one thing the relative layout can NOT find here is the model checkpoints
(`third_party/models/` is not vendored): the spawner (`server/sidecars.py`)
passes `MOSS_TTS_NANO_CHECKPOINT` / `MOSS_TTS_NANO_AUDIO_TOKENIZER` env vars
pointing at `MODELS_DIR` (server/config.py) — without them the sidecar would
fall back to HF hub ids and die offline.

`backend/uploads/` is runtime synth output — gitignored, safe to delete.
Upstream license: `third_party/MOSS-TTS-Nano/LICENSE` (Apache-2.0).
