#!/usr/bin/env bash
# Build the MOSS-TTS-Realtime NATIVE-sidecar venv (.venv-mossrt) ON THE GPU BOX.
# Usage (on the box):  nohup bash scripts/build_venv_mossrt.sh > .venv-mossrt-build.log 2>&1 &
#
# Only needed for TTS_PROVIDER=moss_tts_realtime_native (the upstream
# fast_api.py session server on transformers 5.0 — which conflicts with both
# .venv and .venv-vllm, hence its own venv). The PRIMARY moss_tts_realtime
# provider rides the existing .venv-vllm engine and needs no extra venv.
# NOTE: torch==2.9.1+cu128 comes from the pytorch extra index inside
# requirements-mossrt.txt; if the box can't reach it, drop the +cu128 pin to
# the plain PyPI wheel or point at the wheelhouse.
set -uo pipefail
WS=/inspire/hdd/project/video-understanding/public/personal/chwang/live/MOSS-VL-Realtime_Demo_App
export PIP_CONFIG_FILE=/dev/null            # skip /etc/pip.conf (its tuna extra-index times out on the box)
unset PIP_CONSTRAINT                        # NGC containers pin torch==2.8.0a0+…nvXX via /etc/pip/constraint.txt;
                                            # our isolated venv wants its OWN torch, so drop the container constraint

# OFFLINE-FIRST: a prebuilt wheelhouse (populated on a networked box via
#   pip download -r requirements-mossrt.txt pip setuptools wheel packaging -d <dir>
# ) lets the air-gapped GPU box install with NO network. Falls back to the
# Nexus mirror when no wheelhouse is present (dev/networked box).
WHEELHOUSE="${WHEELHOUSE:-$WS/../wheelhouse/mossrt}"
if ls "$WHEELHOUSE"/*.whl >/dev/null 2>&1; then
  echo "=== offline wheelhouse: $WHEELHOUSE ($(ls "$WHEELHOUSE"/*.whl | wc -l) wheels) ==="
  PIP_ARGS=(--no-index --find-links "$WHEELHOUSE")   # --no-index also ignores the requirements' extra-index-url
else
  MIRROR="${MIRROR:-http://nexus.sii.shaipower.online/repository/pypi/simple}"
  echo "=== no wheelhouse — using mirror: $MIRROR ==="
  PIP_ARGS=(--index-url "$MIRROR" --trusted-host "$(echo "$MIRROR" | sed -E 's#^https?://([^/]+)/.*#\1#')")
fi

cd "$WS" || exit 1

echo "=== [$(date)] venv create ==="
/usr/bin/python3 -m venv .venv-mossrt || exit 1
. .venv-mossrt/bin/activate

echo "=== [$(date)] bootstrap pip ==="
pip install "${PIP_ARGS[@]}" -U pip setuptools wheel packaging || exit 1

echo "=== [$(date)] install transformers-5.0 stack (~10 min) ==="
pip install "${PIP_ARGS[@]}" -r requirements-mossrt.txt || exit 1

echo "=== [$(date)] validate: vendored realtime imports (CPU-safe, no model load) ==="
python - <<'PYEOF' || exit 1
import sys, pathlib
repo = pathlib.Path("server/adapters/tts/moss_tts_realtime/sidecar/third_party/MOSS-TTS/moss_tts_realtime").resolve()
sys.path.insert(0, str(repo))
from mossttsrealtime.streaming_mossttsrealtime import (  # noqa: F401
    AudioStreamDecoder, MossTTSRealtimeInference, MossTTSRealtimeStreamingSession)
import transformers
assert transformers.__version__.startswith("5."), transformers.__version__
import fastapi, soxr  # noqa: F401
print("moss-tts-realtime imports: OK (transformers", transformers.__version__ + ")")
PYEOF

echo "=== [$(date)] .venv-mossrt READY ==="
