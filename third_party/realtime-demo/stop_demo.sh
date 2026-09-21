#!/usr/bin/env bash
# Stop only this checkout's API and explicitly owned component processes.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$REPO/scripts/deploy/demo.sh" down
"$REPO/.venv/bin/python" "$REPO/scripts/deploy/stop_backend.py" --repo "$REPO" \
  --role omni --role pi --role memory-llm --role ssh-socks --role ssh-browser
echo "Shutdown requested for owned services. Unmarked legacy/adopted services were left running."
