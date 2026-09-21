#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${PYTHON_BOOTSTRAP:-python3}" "$ROOT/scripts/repro/bootstrap.py" "$@"
