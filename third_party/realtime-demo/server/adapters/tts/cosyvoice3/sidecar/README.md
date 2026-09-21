# Vendored CosyVoice stack (the `cosyvoice3_native` fallback provider)

Vendored 2026-07-14 from `github.com/FunAudioLLM/CosyVoice` @ `074ca6d`
(2026-05-26) with its Matcha-TTS submodule @ `dd9105b`:

- `third_party/CosyVoice/cosyvoice/` ← upstream `cosyvoice/` (verbatim)
- `third_party/CosyVoice/third_party/Matcha-TTS/` ← submodule, excluding
  `.git*`, `notebooks/`, `data/`, `synthetic_data/`
- `third_party/CosyVoice/{requirements.txt,LICENSE,README.md}` for provenance
- EXCLUDED: `runtime/` (docker/Triton/gRPC serving — the box has no docker;
  the vLLM path is the primary `cosyvoice3` provider instead), `examples/`,
  `docker/`, `asset/`, `tools/`, `webui.py`, training data dirs.

`backend/cosyvoice3_sidecar.py` is OURS: a nano-protocol FastAPI wrapper
(same endpoints/headers as `../..//moss_tts_nano/sidecar/backend/`) around
`cosyvoice.cli.cosyvoice.AutoModel`. It adds `cosyvoice/` and
`third_party/Matcha-TTS/` to sys.path relative to its own file, so the layout
must be preserved on re-syncs.

Checkpoint: the ModelScope-layout dir (`llm.pt/flow.pt/hift.pt` +
`cosyvoice3.yaml` + `CosyVoice-BlankEN/` tokenizer), zoo dir
`Fun-CosyVoice3-0.5B` — NOT the HF-format `-2512` dir that `vllm serve` uses.
Passed via `COSY3_MODEL_DIR` (server/sidecars.py). `spk2info.pt` is absent
from the release — fine, zero-shot cloning computes speaker embeddings from
the prompt WAV.

Acceleration (env, read by the sidecar): `COSY3_FP16=1`, `COSY3_TRT=1`
(TensorRT flow-matching estimator; FIRST boot compiles
`flow.decoder.estimator.fp32.mygpu.plan` INTO the model dir — minutes, needs
a writable model dir; the spawn gate allows 600 s), `COSY3_VLLM=0` (optional
in-process vLLM for the token LM; needs a vllm-enabled venv + a `{model}/vllm`
export — see upstream `vllm_example.py`).

Voices: CosyVoice3 has no presets. Every request zero-shot-clones from the
Nano voice-prompt WAVs (`TTS_VOICE_PROMPT_DIR`) using the transcript from
`assets/demo.jsonl` as prompt text. Only voices whose prompt file AND
transcript both exist are served (health lists them).

venv: `.venv-cosy` — `scripts/build_venv_cosyvoice.sh` /
`requirements-cosyvoice.txt` (plain python3.12; `wetext` replaces the old
pynini text-norm stack).
Upstream license: `third_party/CosyVoice/LICENSE` (Apache-2.0).
