#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
if [[ -z "${PYTHON:-}" ]]; then
  PYTHON=python3
  if [[ -z "${VIRTUAL_ENV:-}" && -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  fi
fi
exec "$PYTHON" "$HERE/evaluation.py" latency "$@"
