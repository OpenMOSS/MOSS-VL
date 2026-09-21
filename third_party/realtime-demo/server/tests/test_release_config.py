"""Deployment contract after merging NPU startup and realtime gateway changes."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from server.config import _parse_env_file


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("custom_interpreter", [False, True])
def test_startup_preserves_interpreter_selection_and_websocket_limits(tmp_path, custom_interpreter):
    deploy = tmp_path / "scripts/deploy"
    deploy.mkdir(parents=True)
    for name in ["run_backend.sh", "env_lib.sh"]:
        shutil.copyfile(ROOT / "scripts/deploy" / name, deploy / name)
    interpreter = tmp_path / ("custom-python" if custom_interpreter else ".venv/bin/python")
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    interpreter.chmod(0o755)
    env = dict(os.environ)
    env.pop("PYBIN", None)
    env.update(ENV_DEPLOY_FILE="", GATEWAY_MAX_FRAME_BYTES="33554432", WS_MAX_SIZE="67108864",
               WS_PING_INTERVAL="17", WS_PING_TIMEOUT="19", PORT="18999", UVICORN_LOOP="asyncio")
    if custom_interpreter:
        env["PYBIN"] = str(interpreter)
    result = subprocess.run(["bash", str(deploy / "run_backend.sh")], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[:3] == ["-m", "uvicorn", "server.app:app"]
    for option, expected in [("--port", "18999"), ("--ws-max-size", "67108864"),
                             ("--ws-ping-interval", "17"), ("--ws-ping-timeout", "19")]:
        assert args[args.index(option) + 1] == expected


def test_realtime_example_and_generated_env_surface():
    values = _parse_env_file(str(ROOT / ".env.deploy.sglang-omni.example"))
    assert values["VLM_DEPLOY"] == "sglang_omni"
    assert values["SGLANG_OMNI_SESSIONS_PER_REPLICA"] == "1"
    assert int(values["WS_MAX_SIZE"]) > int(values["GATEWAY_MAX_FRAME_BYTES"])
    assert values["ASR_ENABLED"] == values["TTS_ENABLED"] == values["TTS_SPAWN"] == "0"
    manifest = (ROOT / "scripts/deploy/env_manifest.sh").read_text().split()
    for name in ["PYBIN", "OFFLINE_GPUS", "WS_MAX_SIZE", "WS_PING_INTERVAL", "WS_PING_TIMEOUT",
                 "SGLANG_OMNI_CONTEXT_LENGTH", "GATEWAY_CREATE_TIMEOUT_S"]:
        assert name in manifest
    assert "/inspire/" not in (ROOT / ".env.deploy.example").read_text()
