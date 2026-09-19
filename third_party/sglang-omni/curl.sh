#!/usr/bin/env bash
# Smoke tests for the MOSS-VL realtime service.
#
# Usage:
#   bash curl.sh                    # test 127.0.0.1:8000
#   HOST=<ip> PORT=<port> bash curl.sh
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE="http://${HOST}:${PORT}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_DIR}"

FAILURES=0
check() {
    local name="$1" ok="$2"
    if [ "$ok" = "true" ]; then
        echo "PASS  ${name}"
    else
        echo "FAIL  ${name}"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "== 1. health =="
HEALTH="$(curl -sf --max-time 5 "${BASE}/health" || true)"
[ -n "$HEALTH" ] && check "GET /health" true || check "GET /health" false
echo "      ${HEALTH}"

echo "== 2. model list =="
MODELS="$(curl -sf --max-time 5 "${BASE}/v1/models" || true)"
echo "$MODELS" | grep -q "moss-vl-realtime" \
    && check "GET /v1/models (moss-vl-realtime)" true \
    || check "GET /v1/models (moss-vl-realtime)" false

echo "== 3. realtime vision E2E (WebSocket, 1 frame + prompt) =="
FRAME="$(mktemp /tmp/moss_frame_XXXX.jpg)"
python - "$FRAME" <<'EOF'
import sys
from PIL import Image
img = Image.new("RGB", (448, 448))
for x in range(0, 448, 56):
    for y in range(0, 448, 56):
        c = ((x // 56) * 40 % 256, (y // 56) * 40 % 256, 128)
        for dx in range(56):
            for dy in range(56):
                img.putpixel((x + dx, y + dy), c)
img.save(sys.argv[1], format="JPEG")
EOF
E2E_OUT="$(python "$REPO_DIR/examples/moss_vl_realtime_client.py" \
    --url "ws://${HOST}:${PORT}/v1/video/realtime" \
    --prompt "Describe this image in one short sentence." \
    --frame "$FRAME" --timestamp 0.0 2>&1 || true)"
rm -f "$FRAME"
echo "$E2E_OUT" | sed 's/^/      /'
echo "$E2E_OUT" | grep -qE "\[response.done: (stop|length)\]" \
    && check "realtime session completes" true \
    || check "realtime session completes" false
# Non-empty assistant text means the model answered instead of staying silent.
echo "$E2E_OUT" | grep -qE "^[A-Z].{10,}" \
    && check "model produced text answer" true \
    || check "model produced text answer" false

echo
if [ "$FAILURES" -eq 0 ]; then
    echo "ALL CHECKS PASSED"
else
    echo "${FAILURES} CHECK(S) FAILED" >&2
    exit 1
fi
