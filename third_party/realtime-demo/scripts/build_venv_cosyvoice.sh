#!/usr/bin/env bash
# Build the Fun-CosyVoice3 NATIVE-sidecar venv (.venv-cosy) ON THE GPU BOX.
# Usage (on the box):  nohup bash scripts/build_venv_cosyvoice.sh > .venv-cosy-build.log 2>&1 &
#
# Only needed for TTS_PROVIDER=cosyvoice3_native (the vendored CosyVoice repo
# stack with fp16 + TensorRT flow estimator). The PRIMARY cosyvoice3 provider
# rides the existing .venv-vllm engine and needs no extra venv.
set -uo pipefail
WS=/inspire/hdd/project/video-understanding/public/personal/chwang/live/MOSS-VL-Realtime_Demo_App
export PIP_CONFIG_FILE=/dev/null            # skip /etc/pip.conf (its tuna extra-index times out on the box)
unset PIP_CONSTRAINT                        # NGC containers pin torch==2.8.0a0+…nvXX via /etc/pip/constraint.txt;
                                            # our isolated venv wants its OWN torch, so drop the container constraint

# OFFLINE-FIRST: a prebuilt wheelhouse (populated on a networked box via
#   pip download -r requirements-cosyvoice.txt pip setuptools wheel packaging -d <dir>
# ) lets the air-gapped GPU box install with NO network. Falls back to the
# Nexus mirror when no wheelhouse is present (dev/networked box).
WHEELHOUSE="${WHEELHOUSE:-$WS/../wheelhouse/cosy}"
if ls "$WHEELHOUSE"/*.whl >/dev/null 2>&1; then
  echo "=== offline wheelhouse: $WHEELHOUSE ($(ls "$WHEELHOUSE"/*.whl | wc -l) wheels) ==="
  PIP_ARGS=(--no-index --find-links "$WHEELHOUSE")
else
  MIRROR="${MIRROR:-http://nexus.sii.shaipower.online/repository/pypi/simple}"
  echo "=== no wheelhouse — using mirror: $MIRROR ==="
  PIP_ARGS=(--index-url "$MIRROR" --trusted-host "$(echo "$MIRROR" | sed -E 's#^https?://([^/]+)/.*#\1#')")
fi

cd "$WS" || exit 1

echo "=== [$(date)] venv create ==="
/usr/bin/python3 -m venv .venv-cosy || exit 1
. .venv-cosy/bin/activate

echo "=== [$(date)] bootstrap pip ==="
pip install "${PIP_ARGS[@]}" -U pip setuptools wheel packaging || exit 1

echo "=== [$(date)] install inference stack (torch 2.3.1 + TRT wheels, ~10 min) ==="
pip install "${PIP_ARGS[@]}" -r requirements-cosyvoice.txt || exit 1

echo "=== [$(date)] validate: vendored CosyVoice3 imports (CPU-safe, no model load) ==="
python - <<'PYEOF' || exit 1
import sys, pathlib
repo = pathlib.Path("server/adapters/tts/cosyvoice3/sidecar/third_party/CosyVoice").resolve()
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))
from cosyvoice.cli.cosyvoice import AutoModel, CosyVoice3  # noqa: F401
from cosyvoice.utils.file_utils import load_wav  # noqa: F401
import wetext, whisper, onnxruntime  # noqa: F401
print("cosyvoice3 imports: OK")
PYEOF

echo "=== [$(date)] .venv-cosy READY ==="
