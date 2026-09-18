#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../deployment/moss_vl_realtime/start.sh" "$@" \
  --vl-api-v2-port "${VL_API_V2_PORT:-18610}"
