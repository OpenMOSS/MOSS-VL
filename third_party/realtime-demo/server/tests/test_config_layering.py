"""Unit tests: 4-layer config resolution (command env > script pins >
.env.deploy > config.py defaults).

Run:  <repo>/.venv/bin/python -m server.tests.test_config_layering

Covers the .env.deploy parser (python + shell — the two implementations must
stay line-for-line equivalent), the setdefault precedence contract, the
one-shot load cache, bare-Settings() purity, and the defaults that migrated
out of run_backend.sh.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile

from server import config
from server.config import Settings, _parse_env_file, get_settings

REPO = config.REPO_ROOT
ENV_LIB = os.path.join(REPO, "scripts", "deploy", "env_lib.sh")


@contextlib.contextmanager
def scrubbed_env(**extra: str):
    """Snapshot os.environ, drop the vars under test, apply `extra`, restore."""
    saved = dict(os.environ)
    for key in ("PORT", "CORS_ORIGINS", "KV_ENFORCE", "MODELS_DIR",
                "SENSEVOICE_MODEL", "VISION_SEQ_PAD_MULTIPLE", "GEN_TOP_K",
                "ENV_DEPLOY_FILE", "TTS_PROVIDER", "OFFLINE_PROVIDER"):
        os.environ.pop(key, None)
    os.environ.update(extra)
    config._reset_settings_for_tests()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        config._reset_settings_for_tests()


def write_file(text: str) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
    f.write(text)
    f.close()
    return f.name


# ------------------------------------------------------------------ parser

def test_parser() -> None:
    cases = [
        ("PORT=1234", {"PORT": "1234"}),                          # P1 plain
        ("export PORT=1234", {"PORT": "1234"}),                   # P2 export
        ("export\tPORT=1234", {"PORT": "1234"}),                  # P2 tab
        ('A="x = y"', {"A": "x = y"}),                            # P3 quotes keep =
        ("A='x # y'", {"A": "x # y"}),                            # P3 quotes keep #
        ('B="a,b"  # cmt', {"B": "a,b"}),                         # P3 comment after quote
        ("A=v=w=z", {"A": "v=w=z"}),                              # P4 value keeps =
        ("A=hello  # comment", {"A": "hello"}),                   # P5 trailing comment
        ("A=1\r\nB=2\r\n", {"A": "1", "B": "2"}),                 # P6 CRLF
        ("\n# note\nno_equals_line\n", {}),                       # P7 skips
        ("1BAD=x\nA-B=x\nif [ -x /y ]; then\n", {}),              # P8 invalid keys
        ("A=", {"A": ""}),                                        # P9 empty value
        ("A=1\nA=2", {"A": "2"}),                                 # P10 last wins
        ('A="unterminated', {"A": '"unterminated'}),              # P11 raw kept
        ("KEY = spaced", {"KEY": "spaced"}),                      # spaces around =
    ]
    for text, expected in cases:
        path = write_file(text)
        try:
            got = _parse_env_file(path)
        finally:
            os.unlink(path)
        assert got == expected, f"parser({text!r}) = {got!r}, want {expected!r}"
    print("parser matrix: OK")


# ------------------------------------------------------------------ precedence

def test_precedence() -> None:
    dep = write_file("PORT=1234\nKV_ENFORCE=hard\nMODELS_DIR=/zoo\n")
    try:
        # L1 layer 4 only
        with scrubbed_env(ENV_DEPLOY_FILE=""):
            assert get_settings().port == 8000

        # L2 file > default
        with scrubbed_env(ENV_DEPLOY_FILE=dep):
            assert get_settings().port == 1234

        # L3 env > file
        with scrubbed_env(ENV_DEPLOY_FILE=dep, PORT="9999"):
            assert get_settings().port == 9999

        # L4 set-but-empty env beats the file
        with scrubbed_env(ENV_DEPLOY_FILE=dep, CORS_ORIGINS=""):
            s = get_settings()
            assert s.cors_origins == "" and s.port == 1234

        # L5 kill-switch
        with scrubbed_env(ENV_DEPLOY_FILE=""):
            assert get_settings().kv_enforce == "auto"

        # L6 bare Settings() purity: no file read, no os.environ mutation
        with scrubbed_env(ENV_DEPLOY_FILE=dep):
            s = Settings()
            assert s.port == 8000 and "KV_ENFORCE" not in os.environ

        # L7 one-shot: a file change after the first get_settings() is not seen
        with scrubbed_env(ENV_DEPLOY_FILE=dep):
            assert get_settings().port == 1234
            with open(dep, "a", encoding="utf-8") as f:
                f.write("PORT=4321\n")
            assert get_settings().port == 1234
        with open(dep, "w", encoding="utf-8") as f:  # restore
            f.write("PORT=1234\nKV_ENFORCE=hard\nMODELS_DIR=/zoo\n")

        # L8 propagation contract: the file lands in os.environ (workers and
        # sidecars copy it; direct-env readers see it)
        with scrubbed_env(ENV_DEPLOY_FILE=dep):
            get_settings()
            assert os.environ["KV_ENFORCE"] == "hard"

        # L9 lazy MODELS_DIR: file-level zoo root reaches derived defaults
        # (grouped by kind — models/asr, models/tts, models/vlms)
        with scrubbed_env(ENV_DEPLOY_FILE=dep):
            assert get_settings().sensevoice_model == "/zoo/asr/SenseVoiceSmall"

        # L10 VISION_SEQ_PAD_MULTIPLE: unset -> 1, "" -> None (escape hatch), "8" -> 8
        with scrubbed_env(ENV_DEPLOY_FILE=""):
            assert Settings().vision_seq_pad_multiple_override == 1
        with scrubbed_env(ENV_DEPLOY_FILE="", VISION_SEQ_PAD_MULTIPLE=""):
            assert Settings().vision_seq_pad_multiple_override is None
        with scrubbed_env(ENV_DEPLOY_FILE="", VISION_SEQ_PAD_MULTIPLE="8"):
            assert Settings().vision_seq_pad_multiple_override == 8
    finally:
        os.unlink(dep)
    print("precedence matrix: OK")


# ------------------------------------------------------------------ shell loader

def _shell(dep_file: str, script: str, env: dict) -> str:
    full = f'source {ENV_LIB!r}\nexport ENV_DEPLOY_FILE={dep_file!r}\n{script}'
    proc = subprocess.run(["bash", "-euo", "pipefail", "-c", full],
                          capture_output=True, text=True,
                          env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **env})
    assert proc.returncode == 0, f"shell failed: {proc.stderr}"
    return proc.stdout.strip()


def test_shell_loader() -> None:
    dep = write_file('WEB_PORT=1111\nexport GEN_TOP_K="7"  # cmt\r\njunk line\n1B=2\n')
    try:
        # S1 file applies when caller env unset
        out = _shell(dep, 'load_env_deploy /x; echo "$WEB_PORT:$GEN_TOP_K"', {})
        assert out == "1111:7", out
        # S2 caller env wins
        out = _shell(dep, 'load_env_deploy /x; echo "$WEB_PORT"', {"WEB_PORT": "2222"})
        assert out == "2222", out
        # S3 caller set-but-empty wins
        out = _shell(dep, 'load_env_deploy /x; echo "[$WEB_PORT]"', {"WEB_PORT": ""})
        assert out == "[]", out
        # S5 kill-switch
        out = _shell("", 'load_env_deploy /x; echo "[${WEB_PORT-}]"', {})
        assert out == "[]", out
        # S6 junk lines are skipped, set -u safe, ENV_DEPLOY_KEYS collected
        out = _shell(dep, 'load_env_deploy /x; echo "${ENV_DEPLOY_KEYS[*]}"', {})
        assert out == "WEB_PORT GEN_TOP_K", out
    finally:
        os.unlink(dep)
    print("shell loader matrix: OK")


# ------------------------------------------------------------------ migrated defaults

def test_migrated_defaults() -> None:
    with scrubbed_env(ENV_DEPLOY_FILE=""):
        for key in ("AUTOLOAD_VLM", "GEN_DO_SAMPLE", "GEN_MAX_TOKENS_PER_TURN",
                    "MODEL_PATH", "HF_MODE"):
            os.environ.pop(key, None)
        s = Settings()
        assert s.autoload_vlm is True
        assert s.do_sample is True
        assert s.max_tokens_per_turn == 86400  # uncapped default; UI knob sends an explicit tokens/s cap
        assert s.vision_seq_pad_multiple_override == 1
        assert s.model_path and s.hf_mode == "online_streaming"
        # default: moss_tts_realtime when vllm + its ckpt exist, else vllm_omni,
        # else the pytorch sidecar (mirrors config._default_tts_provider)
        vllm = os.access(os.path.join(REPO, ".venv-vllm", "bin", "vllm"), os.X_OK)
        if vllm:
            from server.config import _tts_dir
            mossrt = os.getenv("MOSSRT_MODEL", os.path.join(_tts_dir(), "MOSS-TTS-Realtime"))
            expected_tts = "moss_tts_realtime" if os.path.isdir(mossrt) else "vllm_omni"
        else:
            expected_tts = "moss_tts_nano"
        assert s.tts_provider == expected_tts, (s.tts_provider, expected_tts)
        sgl = os.access(os.path.join(REPO, ".venv-sglang", "bin", "python"), os.X_OK)
        assert s.offline_provider == ("sglang" if sgl else "none")
    print("migrated defaults: OK")


def main() -> int:
    test_parser()
    test_precedence()
    test_shell_loader()
    test_migrated_defaults()
    print("\nCONFIG LAYERING TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
