#!/usr/bin/env bash
# Build the OFFLINE-CHAT engine venv (.venv-sglang) ON THE GPU BOX.
# Usage (on the box):  nohup bash scripts/build_venv_sglang.sh > .venv-sglang-build.log 2>&1 &
#
# Installs the fnlp-vision sglang FORK from its /inspire checkout (native
# moss_vl support — see requirements-sglang.txt for why stock sglang cannot
# serve this model). Non-editable install: the fork's code is copied into the
# venv, so fork-repo churn can't silently change what we serve; re-run this
# script to pick up fork updates.
set -uo pipefail
WS=/inspire/hdd/project/video-understanding/public/personal/chwang/live/MOSS-VL-Realtime_Demo_App
FORK=/inspire/hdd/project/video-understanding/public/personal/train/sglang/python
# The venv lands on shared /inspire storage, so it can be built from ANY box
# that mounts it with a matching /usr/bin/python3 (e.g. the dev container,
# whose mirror reachability differs — box 2 sees Nexus, the dev box sees
# TUNA). The GPU box then needs no network at all. Override via MIRROR=…
MIRROR="${MIRROR:-http://nexus.sii.shaipower.online/repository/pypi/simple}"
PIP_ARGS=(--index-url "$MIRROR" --trusted-host "$(echo "$MIRROR" | sed -E 's#^https?://([^/]+)/.*#\1#')")
export PIP_CONFIG_FILE=/dev/null            # skip /etc/pip.conf (its tuna extra-index times out on the box)

cd "$WS" || exit 1
[ -d "$FORK" ] || { echo "fork not found at $FORK"; exit 1; }

echo "=== [$(date)] venv create ==="
/usr/bin/python3 -m venv .venv-sglang || exit 1
. .venv-sglang/bin/activate

echo "=== [$(date)] bootstrap pip ==="
pip install "${PIP_ARGS[@]}" -U pip setuptools wheel ninja packaging || exit 1

echo "=== [$(date)] install sglang fork [all] (torch 2.8 stack + flashinfer, ~10-20 min) ==="
pip install "${PIP_ARGS[@]}" "${FORK}[all]" || exit 1

echo "=== [$(date)] ckpt remote-code deps (transformers check_imports vets the MOSS-VL"
echo "    processing/video files at load time: torchcodec pairs 0.7.0<->torch 2.8) ==="
pip install "${PIP_ARGS[@]}" joblib torchcodec==0.7.0 || exit 1

echo "=== [$(date)] fork fixup: MultimodalInputs.release_features ==="
# Fork commit 2de0278 ("Drop heavy per-request vision tensors") added a
# moss_vl.py CALL to mm_input.release_features() but never defined the method
# on MultimodalInputs — every multimodal request crashes with AttributeError
# at fork HEAD (text-only warmup passes, so the boot gate won't catch it).
# Insert the intended method until the fork ships it; idempotent.
python - <<'PYEOF' || exit 1
import sglang, os
path = os.path.join(os.path.dirname(sglang.__file__), "srt", "managers", "schedule_batch.py")
src = open(path).read()
if "def release_features" in src:
    print("release_features already present —", path)
    raise SystemExit(0)
anchor = "    visible_frame_counts: Optional[torch.Tensor] = None\n"
assert src.count(anchor) == 1, f"anchor not found/unique in {path} — fork layout changed, re-check the fixup"
method = anchor + '''
    def release_features(self) -> None:
        """Drop the heavy per-request vision tensors (raw pixel features /
        precomputed embeddings) once the encoder KV is cached — moss_vl.py
        calls this at the end of prefill so they don't stay pinned across the
        whole decode phase. MISSING in fork commit 2de0278 (which added the
        caller only); patched in by scripts/build_venv_sglang.sh."""
        for item in self.mm_items:
            item.feature = None
            item.precomputed_embeddings = None
'''
open(path, "w").write(src.replace(anchor, method))
print("patched", path)
PYEOF

echo "=== [$(date)] validate ==="
python - <<'EOF'
import sglang, torch
print("sglang", sglang.__version__, "(want 0.5.5.post3)")
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "devices", torch.cuda.device_count())
import flashinfer
print("flashinfer", flashinfer.__version__)
import sgl_kernel  # noqa: F401
print("sgl_kernel OK")
from sglang.srt.models import moss_vl
print("moss_vl model registered:", moss_vl.EntryClass.__name__)
from sglang.srt.multimodal.processors import moss_vl as moss_vl_proc  # noqa: F401
print("moss_vl multimodal processor OK")
EOF
echo "=== [$(date)] DONE ==="
