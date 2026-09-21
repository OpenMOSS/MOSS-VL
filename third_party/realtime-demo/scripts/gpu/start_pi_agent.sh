#!/usr/bin/env bash
# Reconcile the owned memory backend and pi-agent with the requested config.
set -euo pipefail
REPO=${MOSS_DEPLOY_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}
. "$REPO/scripts/deploy/env_lib.sh"
load_env_deploy "$REPO"
[ ! -f "$REPO/deploy.conf" ] || . "$REPO/deploy.conf"
resolve_service_endpoints
KEYS=(OMNI_ROOT PI_AGENT_DIR PI_AGENT_NODE PI_PORT START_4B FORCE_PI FORCE_4B LOG_DIR AIGW_AUTH_MODE
  DECIDE_LLM_MODEL DECIDE_LLM_GPU DECIDE_LLM_PORT DECIDE_LLM_MEM_FRAC
  DECIDE_LLM_CONTEXT_LENGTH DECIDE_LLM_MAX_RUNNING_REQUESTS DECIDE_LLM_MAX_TOTAL_TOKENS
  MEMORY_BACKEND_WAIT_S AIGW_API_KEY AIGW_KEY_FILE AIGW_DECIDE_BASE_URL AIGW_DECIDE_MODEL
  AIGW_COMPACT_BASE_URL AIGW_COMPACT_MODEL AIGW_LOCAL_NO_REASONING
  PI_CONTEXT_TOKENS PI_COMPACT_CHUNK_TOKENS PI_COMPACT_TIMEOUT_MS PI_DECIDE_TIMEOUT_MS
  PI_AGENT_TIMEOUT_MS PI_AGENT_MODE BOARD_MEMORY_URL)
if [ "${_PI_ON_GPU:-0}" != 1 ]; then
  # Transfer overrides over stdin, not in SSH argv (which is visible in ps).
  {
    printf 'export _PI_ON_GPU=1 MOSS_DEPLOY_REPO=%q\n' "$REPO"
    for key in "${KEYS[@]}"; do
      [ -z "${!key+x}" ] || printf 'export %s=%q\n' "$key" "${!key}"
    done
    printf 'exec bash %q\n' "$REPO/scripts/gpu/start_pi_agent.sh"
  } | ${GPU_SSH:-ssh -p 10008 -o BatchMode=yes -o ConnectTimeout=10 root@127.0.0.1} bash -s
  exit $?
fi
for key in "${KEYS[@]}"; do
  [ -z "${!key+x}" ] || export "$key"
done
OMNI_ROOT=${OMNI_ROOT:-/inspire/qb-ilm/project/video-understanding/public/train/moss_vl_streaming/8B/final_release/MOSS-VL-Realtime-sglang}
export OMNI_ROOT
exec "$OMNI_ROOT/.venv-main/bin/python" "$REPO/scripts/deploy/memory_backend.py" --repo "$REPO"
