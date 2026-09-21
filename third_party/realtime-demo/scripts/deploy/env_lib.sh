# shellcheck shell=bash
# Config layer 3 loader — sourced by demo.sh / run_backend.sh / run_web.sh.
# Mirrors server/config.py:_parse_env_file line for line —
# change BOTH together (test: server/tests/test_config_layering.py).
#
# load_env_deploy <repo_root>
#   Reads .env.deploy (plain KEY=VALUE; optional `export ` prefix; no $VAR
#   expansion; one matching surrounding quote pair stripped; unquoted values
#   lose a trailing ` # comment`; CRLF ok) and exports each key ONLY IF UNSET
#   (setdefault: caller env — layers 1-2 — always wins; set-but-empty counts
#   as set). ENV_DEPLOY_FILE overrides the path; empty value disables.
#   Populates ENV_DEPLOY_KEYS with every key seen (set or not) so demo.sh can
#   forward file-provided vars across the tmux boundary. `set -euo pipefail`
#   safe; always returns 0.

ENV_DEPLOY_KEYS=()

# Resolve local companion addresses after configuration is loaded. Explicit
# URLs may point to another machine and must never be rewritten from a port.
resolve_service_endpoints() {
  export VITE_BACKEND_ORIGIN="${VITE_BACKEND_ORIGIN:-http://127.0.0.1:${PORT:-8000}}"
  export PI_PORT="${PI_PORT:-38082}"
  export MEMORY_PI_URL="${MEMORY_PI_URL-http://127.0.0.1:$PI_PORT}"
  case "$VITE_BACKEND_ORIGIN" in
    http://127.0.0.1:*|http://localhost:*)
      if [ "$VITE_BACKEND_ORIGIN" != "http://127.0.0.1:${PORT:-8000}" ] &&
         [ "$VITE_BACKEND_ORIGIN" != "http://localhost:${PORT:-8000}" ]; then
        echo "[config] explicit VITE_BACKEND_ORIGIN=$VITE_BACKEND_ORIGIN differs from local API port ${PORT:-8000}; preserved. Unset it to follow PORT." >&2
      fi ;;
  esac
}

load_env_deploy() {
  local f
  if [ "${ENV_DEPLOY_FILE+x}" = "x" ]; then
    f="$ENV_DEPLOY_FILE"                       # explicit path ("" = kill-switch)
  else
    f="${1:?load_env_deploy needs the repo root}/.env.deploy"
  fi
  [ -n "$f" ] && [ -f "$f" ] || return 0

  local line key value quote close rest
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    # trim surrounding whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    # strip an `export` prefix (space/tab separated)
    case "$line" in
      export' '*|export$'\t'*)
        line="${line#export}"
        line="${line#"${line%%[![:space:]]*}"}" ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    # trim key whitespace (mirrors the python parser's key.strip()), then
    # require a valid identifier
    key="${key%"${key##*[![:space:]]}"}"
    case "$key" in ''|[0-9]*|*[!A-Za-z0-9_]*) continue ;; esac
    # trim value whitespace
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    case "$value" in
      \"*|\'*)
        quote="${value:0:1}"
        rest="${value:1}"
        close="${rest%%"$quote"*}"
        if [ "$close" != "$rest" ]; then
          value="$close"                       # quoted: inner text, trailer dropped
        fi ;;                                  # unterminated: keep raw text
      *)
        # unquoted: strip a trailing ` # comment`
        case "$value" in
          *[[:space:]]'#'*)
            value="${value%%[[:space:]]#*}"
            value="${value%"${value##*[![:space:]]}"}" ;;
        esac ;;
    esac
    ENV_DEPLOY_KEYS+=("$key")
    if [ -z "${!key+x}" ]; then                # setdefault: caller env wins
      export "$key=$value"
    fi
  done < "$f"
  return 0
}
