#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-start}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    start|start4|start8)
        exec "${PYTHON:-python}" "$HERE/deployment/npu/serve.py" --mode "$MODE" "$@"
        ;;
    *)
        echo "Usage: $0 {start|start4|start8} [--model-path PATH]" >&2
        echo "Runs in the foreground. Use Ctrl-C or your service manager to stop/restart." >&2
        exit 2
        ;;
esac
