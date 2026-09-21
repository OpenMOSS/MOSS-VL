#!/usr/bin/env bash
# Build the backend .venv ON THE GPU BOX (H200, internal Nexus mirror, CUDA 12.9).
# Usage (on the box):  nohup bash scripts/build_venv.sh > .venv-build.log 2>&1 &
# The workspace lives on shared /inspire storage, so the log is readable from
# the dev container too.
set -uo pipefail
WS=/inspire/hdd/project/video-understanding/public/personal/chwang/live/MOSS-VL-Realtime_Demo_App
NEXUS=http://nexus.sii.shaipower.online/repository/pypi/simple
PIP_ARGS=(--index-url "$NEXUS" --trusted-host nexus.sii.shaipower.online)
export PIP_CONFIG_FILE=/dev/null            # skip /etc/pip.conf (its tuna extra-index times out on the box)
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

cd "$WS" || exit 1
echo "=== [$(date)] venv create ==="
/usr/bin/python3 -m venv .venv || exit 1
. .venv/bin/activate

echo "=== [$(date)] bootstrap pip ==="
pip install "${PIP_ARGS[@]}" -U pip setuptools wheel ninja packaging || exit 1

echo "=== [$(date)] install requirements.txt (torch stack ~5GB, be patient) ==="
pip install "${PIP_ARGS[@]}" -r requirements.txt || exit 1

echo "=== [$(date)] build flash-attn 2.8.1 (matches board; source build, needs nvcc; ~10-30 min) ==="
nvcc --version | tail -1
MAX_JOBS=48 pip install "${PIP_ARGS[@]}" flash-attn==2.8.1 --no-build-isolation \
  || echo "WARN: flash-attn build failed — backend must use attn_implementation=sdpa"

echo "=== [$(date)] validate ==="
python - <<'EOF'
import torch, transformers, torchcodec, funasr, fastapi, torchaudio, torchvision
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("torchcodec", getattr(torchcodec, "__version__", "?"), "torchaudio", torchaudio.__version__, "torchvision", torchvision.__version__)
print("funasr", funasr.__version__, "fastapi", fastapi.__version__)
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn MISSING ->", e)
import soundfile, sentencepiece, onnxruntime, websockets, aiohttp, multipart
print("audio/server deps OK")
EOF
echo "=== [$(date)] DONE ==="
