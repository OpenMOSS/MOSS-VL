#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=(--url "${VL_MODEL_WS_URL:-ws://${HOST:-127.0.0.1}:${ADAPTER_PORT:-18600}/v1/realtime}"
      --rounds "${ROUNDS:-5}" --frames "${NFRAMES:-4}"
      --token-rate "${TOKEN_RATE:-160}" --timeout-s "${TIMEOUT_S:-10}"
      --max-new-tokens "${MAX_NEW_TOKENS:-512}" --json)
if [[ -n "${FRAMES_DIR:-}" ]]; then
    ARGS+=(--frames-dir "$FRAMES_DIR")
fi
if [[ -n "${PROMPT:-}" ]]; then
    ARGS+=(--prompt "$PROMPT")
fi
exec "${PYTHON:-python}" "$HERE/perf_probe.py" "${ARGS[@]}" "$@"
