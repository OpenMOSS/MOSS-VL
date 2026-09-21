#!/usr/bin/env bash
# MOSS-Realtime backend (server/) — pure ORCHESTRATION. No manual sidecar step:
# the backend spawns, health-gates and terminates its own TTS/sglang sidecars in
# the FastAPI lifespan (server/sidecars.py, server/gpu/supervisor.py) — this one
# script IS the whole voice stack.
#
# Config comes in four layers (see server/config.py docstring): command env >
# the layer-2 block below > .env.deploy > config.py defaults. Every model pin
# and serving knob that used to live here is a config.py default now.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---- startup-script overrides (config layer 2) ----
# Explicit deploy pins for THIS entrypoint: they yield to command env but beat
# .env.deploy (they run before load_env_deploy). Idiom — never plain `export
# VAR=value` (that would clobber the command line):
#   : "${GEN_MAX_TOKENS_PER_TURN:=20}"; export GEN_MAX_TOKENS_PER_TURN
# (currently empty — defaults live in server/config.py)

# ---- .env.deploy (config layer 3; box-local, gitignored) ----
. "$REPO/scripts/deploy/env_lib.sh"
load_env_deploy "$REPO"
resolve_service_endpoints

# torchcodec dlopens the FFmpeg shared libs (libavutil & co), which fresh pods
# don't ship (apt state dies with the container). They are vendored INSIDE the
# venv on the shared FS (.venv/lib/ffmpeg — the .so closure of a conda-forge
# "ffmpeg=6.1" env), so they survive pod restarts with everything else. To
# rebuild: conda create -p /tmp/ffenv -c conda-forge ffmpeg=6.1 && cp -a
# /tmp/ffenv/lib/*.so* "$REPO/.venv/lib/ffmpeg/".
FFMPEG_LIBS="${FFMPEG_LIBS:-$REPO/.venv/lib/ffmpeg}"
if [ -d "$FFMPEG_LIBS" ]; then
  export LD_LIBRARY_PATH="$FFMPEG_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif ! ldconfig -p 2>/dev/null | grep -q libavutil; then
  echo "WARNING: no FFmpeg libs found (system or $FFMPEG_LIBS) — VLM autoload will fail on torchcodec" >&2
fi

# gateway python log: rotating file under the in-repo logs/handler/ tree
# (shared FS — survives pod restarts, readable from the CPU box);
# logging_conf.py adds a RotatingFileHandler next to stdout when this is set,
# and routes uvicorn's access/error lines through it too. Workers rotate under
# logs/handler/backend/workers/, TTS engines under logs/handler/sidecar/
# (config.py defaults). This script is the ONE place that defaults
# MOSS_LOG_FILE — unset (tests, ad-hoc runs) stays stdout-only.
export MOSS_LOG_FILE="${MOSS_LOG_FILE:-$REPO/logs/handler/backend/backend.log}"
# stdout feeds a pipe (demo.sh rotating tee), not a tty — keep it line-live;
# children (workers, sidecars) inherit this too
export PYTHONUNBUFFERED=1

cd "$REPO"
# Leave room above the application limit to emit an error before close 1009.
frame_limit=${GATEWAY_MAX_FRAME_BYTES:-33554432}
[[ "$frame_limit" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid GATEWAY_MAX_FRAME_BYTES" >&2; exit 1; }
ws_limit=${WS_MAX_SIZE:-$((frame_limit * 2))}
[[ "$ws_limit" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid WS_MAX_SIZE" >&2; exit 1; }
(( ws_limit > frame_limit )) || { echo "WS_MAX_SIZE must exceed GATEWAY_MAX_FRAME_BYTES" >&2; exit 1; }
# uvloop (ships with uvicorn[standard]) keeps the WS plane snappy under load
# interpreter: .venv if built, else PYBIN from env (e.g. mamba python on NPU boxes)
PYBIN="${PYBIN:-$REPO/.venv/bin/python}"
[ -x "$PYBIN" ] || PYBIN="$(command -v python3 || echo python3)"
exec "$PYBIN" -m uvicorn server.app:app --host 127.0.0.1 --port "${PORT:-8000}" \
  --loop "${UVICORN_LOOP:-uvloop}" \
  --ws-max-size "$ws_limit" \
  --ws-ping-interval "${WS_PING_INTERVAL:-20}" --ws-ping-timeout "${WS_PING_TIMEOUT:-20}"
