#!/usr/bin/env python3
"""Config-surface generator + drift lint (stdlib only).

Single source of truth for "which env vars exist" is the code itself
(server/config.py `_env*` calls + the known direct os.environ readers). This
script derives the two artifacts that must stay in sync with it:

  scripts/deploy/env_manifest.sh   MOSS_ENV_VARS=(…) — the vars demo.sh
                                   forwards across the tmux boundary
  .env.deploy.example              curated preamble + complete generated
                                   reference of every settings var

Usage:
  scripts/dev/check_env.py --write   regenerate both files in place
  scripts/dev/check_env.py --check   regenerate in memory, diff, exit 1 on drift
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PY = os.path.join(REPO, "server", "config.py")
MANIFEST = os.path.join(REPO, "scripts", "deploy", "env_manifest.sh")
EXAMPLE = os.path.join(REPO, ".env.deploy.example")

# Files (besides config.py) that read env vars directly, bypassing Settings.
DIRECT_READERS = [
    "server/logging_conf.py",
    "server/sidecars.py",
    "server/adapters/asr/funasr_sensevoice/adapter.py",
    "server/adapters/tts/common/__init__.py",
    "server/adapters/tts/common/nano_protocol.py",
    "server/adapters/tts/common/openai_speech.py",
    "server/adapters/tts/moss_tts_nano/adapter.py",
    "server/adapters/tts/cosyvoice3/adapter.py",
    "server/adapters/tts/cosyvoice3/sidecar/backend/cosyvoice3_sidecar.py",
    "server/adapters/tts/moss_tts_realtime/adapter.py",
    "server/adapters/tts/moss_tts_nano/sidecar/backend/moss_tts_nano_sidecar.py",
    # vendored MOSS-TTS-Realtime session server (native provider sidecar) —
    # its MOSS_TTS_* env (incl. MOSS_TTS_STREAM_EMIT) is deploy-relevant
    "server/adapters/tts/moss_tts_realtime/sidecar/third_party/MOSS-TTS/moss_tts_realtime/fast_api.py",
]

# Shell-layer-only vars (never reach python Settings) with one-line docs.
SHELL_ONLY = {
    "WEB_PORT": "vite preview port (the ONE port to expose; default 20941)",
    "PORT": "backend api port (loopback only; default 8000)",
    "PI_PORT": "local pi-agent port; MEMORY_PI_URL defaults to this port (default 38082)",
    "LOG_ROOT": "in-repo log tree root (default <repo>/logs)",
    "PYBIN": "API/rotating_tee interpreter (default <repo>/.venv/bin/python)",
    "FFMPEG_LIBS": "vendored FFmpeg .so dir for torchcodec (default <repo>/.venv/lib/ffmpeg)",
    "BUILD_LOG": "vite build-watcher log (default <repo>/logs/stdout/web/build.log)",
    "API_WAIT_S": "demo.sh api health-gate budget (default 600)",
    "VLM_WAIT_S": "demo.sh VLM-loaded wait budget (default 480)",
    "DEMO_SESSION": "tmux session name (default moss)",
    "DEMO_SKIP_GPU": "1 = CPU-only bring-up (frontend work; VLM won't load)",
    "TTS_PORT": "demo.sh down: first TTS sidecar port to sweep (default 18100)",
    "UVICORN_LOOP": "uvicorn event loop (default uvloop)",
    "WS_MAX_SIZE": "WebSocket transport limit in bytes (default twice GATEWAY_MAX_FRAME_BYTES)",
    "WS_PING_INTERVAL": "WebSocket ping interval in seconds (default 20)",
    "WS_PING_TIMEOUT": "WebSocket ping timeout in seconds (default 20)",
    "WATCH_POLL": "1 = polling vite build watcher (load-bearing on shared FS)",
    "WEB_NVM_NODE": "machine-dependent nvm version/alias for the web window (run_web.sh); set on boxes whose system Node is too old for Vite (Blackwell: 22); unset = no-op",
    "VITE_BACKEND_ORIGIN": "explicit vite /api proxy URL; otherwise derived from PORT (default http://127.0.0.1:8000)",
    "GPU": "standalone worker: CUDA_VISIBLE_DEVICES value",
    "WORKER_ID": "standalone worker: worker id",
    "ENV_DEPLOY_FILE": "path override for THIS file; empty string disables layer 3",
}

# Per-process internals each process resolves for itself — never forwarded.
INTERNAL = {
    "CUDA_VISIBLE_DEVICES", "VLM_WORKER_ATTN", "PYTORCH_CUDA_ALLOC_CONF",
    "LD_LIBRARY_PATH", "PYTHONUNBUFFERED", "ENV_DEPLOY_FILE",
    # read by the vendored sidecar's gradio demos only, not deploy config
    "VSCODE_PROXY_URI",
    # offline-mode switches the SIDECAR processes set for themselves
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_LOGGING_CONFIG_PATH",
}

_ENV_CALL = re.compile(r"""_env(?:_int|_float|_flag|_opt_int|_opt_float)?\(\s*["']([A-Z][A-Z0-9_]+)["']""")
_GETENV = re.compile(r"""os\.(?:getenv|environ\.get)\(\s*["']([A-Z][A-Z0-9_]+)["']""")
_IN_ENVIRON = re.compile(r"""["']([A-Z][A-Z0-9_]+)["']\s+(?:not\s+)?in\s+os\.environ""")
_SECTION = re.compile(r"#\s*----\s*(.+?)\s*----")


def _env_matches(text: str) -> "list[tuple[int, str]]":
    """(position, env_name) for every env read in `text` — full-text regexes,
    so multi-line calls (`_env(\\n "NAME", …)`) are captured too."""
    out = []
    for rx in (_ENV_CALL, _GETENV):
        out.extend((m.start(), m.group(1)) for m in rx.finditer(text))
    out.sort()
    return out


def scan_config() -> "list[tuple[str, str]]":
    """(section, env_name) pairs from server/config.py, in source order.

    A name can appear twice (e.g. SGLANG_PYTHON in the provider helper AND in
    its field definition) — the LAST occurrence's section wins (fields come
    after the module helpers, and the field's section is the meaningful one),
    while first-seen order is kept for stable output.
    """
    text = open(CONFIG_PY, encoding="utf-8").read()
    headers = [(m.start(), m.group(1)) for m in _SECTION.finditer(text)]

    def section_at(pos: int) -> str:
        current = "module roots / providers"
        for hpos, title in headers:
            if hpos > pos:
                break
            current = title
        return current

    order: "list[str]" = []
    sections: "dict[str, str]" = {}
    for pos, name in _env_matches(text):
        if name not in sections:
            order.append(name)
        sections[name] = section_at(pos)
    return [(sections[n], n) for n in order]


def scan_direct_readers() -> "list[tuple[str, str]]":
    out, seen = [], set()
    for rel in DIRECT_READERS:
        path = os.path.join(REPO, rel)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            print(f"WARNING: direct-reader missing: {rel}", file=sys.stderr)
            continue
        for rx in (_GETENV, _IN_ENVIRON, _ENV_CALL):
            for name in rx.findall(text):
                if name not in seen:
                    seen.add(name)
                    out.append((rel, name))
    return out


def resolved_defaults(names: "set[str]") -> "dict[str, str]":
    """env name → default value string, from a scrubbed-env Settings()."""
    sys.path.insert(0, REPO)
    saved = {n: os.environ.pop(n) for n in list(names) if n in os.environ}
    os.environ["ENV_DEPLOY_FILE"] = ""
    try:
        import server.config as config  # noqa: PLC0415

        settings = config.Settings()
        # env name → field name: attribute each match to the nearest preceding
        # dataclass-field definition; last occurrence wins (field defs follow
        # the module helpers)
        text = open(CONFIG_PY, encoding="utf-8").read()
        field_defs = [(m.start(), m.group(1))
                      for m in re.finditer(r"^    (\w+)\s*:[^=\n]+=\s*field\(", text, re.M)]
        env_to_field: "dict[str, str]" = {}
        for pos, name in _env_matches(text):
            owner = None
            for fpos, fname in field_defs:
                if fpos > pos:
                    break
                owner = fname
            if owner:
                env_to_field[name] = owner
        env_to_field.setdefault("TTS_PROVIDER", "tts_provider")
        env_to_field.setdefault("OFFLINE_PROVIDER", "offline_provider")
        out = {}
        for name, fieldname in env_to_field.items():
            if not hasattr(settings, fieldname):
                continue
            v = getattr(settings, fieldname)
            if isinstance(v, bool):
                out[name] = "1" if v else "0"
            elif v is None:
                out[name] = ""
            else:
                out[name] = str(v)
        out["MODELS_DIR"] = config._models_dir()  # module-level, no field
        return out
    finally:
        os.environ.pop("ENV_DEPLOY_FILE", None)
        os.environ.update(saved)


def gen_manifest(all_names: "list[str]") -> str:
    names = sorted(set(all_names) - INTERNAL)
    lines = [
        "# shellcheck shell=bash",
        "# DO NOT EDIT — generated by scripts/dev/check_env.py (--write).",
        "# Every settings var the stack consumes (config.py + direct readers +",
        "# shell layer), minus per-process internals. demo.sh forwards these",
        "# across the tmux boundary when set in its environment.",
        "MOSS_ENV_VARS=(",
    ]
    for i in range(0, len(names), 4):
        lines.append("  " + " ".join(names[i:i + 4]))
    lines.append(")")
    return "\n".join(lines) + "\n"


PREAMBLE = """\
# .env.deploy — box-local deployment overrides (config layer 3).
#   cp .env.deploy.example .env.deploy   # then uncomment + edit what differs
#
# Config resolves in four layers, highest first (server/config.py docstring):
#   1. command env        VAR=x demo.sh up   /   demo.sh up VAR=x
#   2. startup-script pins  the layer-2 blocks in scripts/deploy/run_*.sh
#   3. THIS FILE          plain KEY=VALUE (a set env var always wins — this
#                         file only fills gaps). NOT bash-sourced: no $VAR
#                         expansion; `export ` prefixes tolerated; quotes
#                         stripped; ` # trailing comments` allowed.
#                         ENV_DEPLOY_FILE overrides the path; empty disables.
#   4. code defaults      server/config.py (single source of defaults)
#
# .env.deploy itself is gitignored — it describes ONE box, not the project.
# Paths shown with <repo> are documentation placeholders, not shell variables.
# Replace them with absolute paths when enabling an override.
#
# ============ the roots almost everything hangs off ============
#
# 1) MODELS_DIR — the model zoo (ASR + TTS checkpoints). Copy these dirs and
#    point MODELS_DIR at their parent (each is ALSO individually overridable):
#      SenseVoiceSmall/  fsmn-vad/                     (ASR + VAD)
#      MOSS-TTS-Nano-100M/  MOSS-Audio-Tokenizer-Nano/ (TTS LM + codec, vllm engine)
#
# 2) The VLM checkpoints: MODEL_PATH (online realtime) / OFFLINE_MODEL_PATH
#    (chat page, sglang plane).
#
# 3) The repo checkout itself — everything repo-relative travels with it:
#    logs/, data/, models/ (vllm shadow ckpt, auto-rebuilt at spawn), dist/.
#    The venvs do NOT survive a path move — REBUILD them on the new box:
#      .venv        pip install -r requirements.txt      (+ flash-attn note there)
#      .venv-vllm   see .env.deploy.example history / requirements-vllm.txt
#      .venv-sglang scripts/build_venv_sglang.sh         (offline chat plane)
#
# Everything below is the COMPLETE generated reference (scripts/dev/
# check_env.py --write): uncomment a line to pin it for this box. Values shown
# are the code defaults resolved on the generating box. Full semantics live on
# each field in server/config.py.
"""


# Defaults that depend on what is INSTALLED on the generating box — shown as a
# doc note instead of a resolved value, so --check is stable across boxes.
DYNAMIC = {
    "TTS_PROVIDER": "auto: vllm_omni iff .venv-vllm/bin/vllm is executable, else moss_tts_nano",
    "OFFLINE_PROVIDER": "auto: sglang iff .venv-sglang/bin/python is executable, else none",
}


def gen_example(config_pairs, reader_pairs, defaults) -> str:
    chunks = [PREAMBLE]
    by_section: "dict[str, list[str]]" = {}
    for section, name in config_pairs:
        by_section.setdefault(section, []).append(name)
    for section, names in by_section.items():
        chunks.append(f"\n# ============ {section} ============\n")
        for name in names:
            if name in DYNAMIC:
                chunks.append(f"#export {name}=        # {DYNAMIC[name]}\n")
            else:
                value = defaults.get(name, '').replace(REPO, '<repo>')
                chunks.append(f"#export {name}={value}\n")
    chunks.append("\n# ============ direct readers (bypass Settings; see the file) ============\n")
    for rel, name in reader_pairs:
        if name in INTERNAL or any(name == n for _, n in config_pairs):
            continue
        chunks.append(f"#export {name}=        # {rel}\n")
    chunks.append("\n# ============ shell / deploy layer (demo.sh + run_*.sh) ============\n")
    for name, doc in SHELL_ONLY.items():
        if name == "ENV_DEPLOY_FILE":
            continue  # meta: configuring the file from the file makes no sense
        chunks.append(f"#export {name}=        # {doc}\n")
    return "".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()

    config_pairs = scan_config()
    reader_pairs = scan_direct_readers()
    all_names = ([n for _, n in config_pairs] + [n for _, n in reader_pairs]
                 + list(SHELL_ONLY))
    defaults = resolved_defaults(set(all_names))

    manifest = gen_manifest(all_names)
    example = gen_example(config_pairs, reader_pairs, defaults)

    ok = True
    for path, content in ((MANIFEST, manifest), (EXAMPLE, example)):
        rel = os.path.relpath(path, REPO)
        try:
            on_disk = open(path, encoding="utf-8").read()
        except OSError:
            on_disk = None
        if args.write:
            if on_disk != content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"wrote {rel}")
            else:
                print(f"unchanged {rel}")
        elif on_disk != content:
            ok = False
            print(f"DRIFT: {rel} (regenerate with scripts/dev/check_env.py --write)")
            diff = difflib.unified_diff((on_disk or "").splitlines(), content.splitlines(),
                                        rel, rel + " (generated)", lineterm="", n=1)
            for i, line in enumerate(diff):
                if i > 40:
                    print("  …")
                    break
                print(f"  {line}")
    if args.check and ok:
        print(f"env surface in sync ({len(set(all_names) - INTERNAL)} vars)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
