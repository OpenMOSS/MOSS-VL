#!/usr/bin/env bash
# Frontend, ONE process for the manifest (demo.sh window "web"): the
# build-on-save watcher in the background + vite preview serving dist/ in
# the foreground. WATCH_POLL is load-bearing: edits come from OTHER
# shared-FS clients (login node / VSCode pods) whose inotify events never
# reach this box, so the watcher must poll (see vite.config.ts). Killing
# this shell (tmux window kill / Ctrl-C) reaps the watcher via the trap.
# NOTE: watch mode skips `tsc -b` — run a full `npm run build` before commits.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# .env.deploy (config layer 3) — WEB_PORT / BUILD_LOG / PYBIN / VITE_* on
# standalone runs; caller env (layers 1-2) wins
. "$REPO/scripts/deploy/env_lib.sh"
load_env_deploy "$REPO"
resolve_service_endpoints
echo "[web] /api proxy: $VITE_BACKEND_ORIGIN"

BUILD_LOG="${BUILD_LOG:-$REPO/logs/stdout/web/build.log}"
PYBIN="${PYBIN:-$REPO/.venv/bin/python}"
[ -x "$PYBIN" ] || PYBIN="$(command -v python3 || echo python3)"

# Node selection is machine-dependent (config layer 3). A box whose default-PATH
# Node is too old for Vite (the Blackwell box ships system Node 18; Vite needs
# >=20.19, else `ReferenceError: CustomEvent is not defined`) sets WEB_NVM_NODE
# in its .env.deploy to an nvm version/alias (e.g. 22). Boxes with a fine system
# Node (4090 / H200) leave it unset and are untouched. Guarded on the var so this
# is a no-op wherever it isn't explicitly opted in — no cross-machine divergence.
if [ -n "${WEB_NVM_NODE:-}" ]; then
  NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [ -s "$NVM_DIR/nvm.sh" ]; then
    # shellcheck disable=SC1091
    . "$NVM_DIR/nvm.sh"
    nvm use "$WEB_NVM_NODE" >/dev/null 2>&1 \
      || echo "[web] nvm use $WEB_NVM_NODE failed — using $(command -v node) $(node -v 2>/dev/null)" >&2
  else
    echo "[web] WEB_NVM_NODE=$WEB_NVM_NODE set but nvm not found at $NVM_DIR" >&2
  fi
  echo "[web] node $(node -v 2>/dev/null || echo '?') (WEB_NVM_NODE=$WEB_NVM_NODE)"
fi

# process substitution, NOT a pipe: $! must stay vite's pid for the trap
# (a pipe would make it the tee's pid and the EXIT trap would orphan vite);
# the tee rotates the file for real and exits on its own at EOF
WATCH_POLL=1 npx vite build > >("$PYBIN" "$REPO/scripts/deploy/rotating_tee.py" --quiet "$BUILD_LOG") 2>&1 &
WATCH_PID=$!
trap 'kill "$WATCH_PID" 2>/dev/null || true' EXIT INT TERM

# foreground (NOT exec — the trap above must outlive vite to reap the watcher)
npx vite preview --host --port "${WEB_PORT:-20941}" --strictPort
