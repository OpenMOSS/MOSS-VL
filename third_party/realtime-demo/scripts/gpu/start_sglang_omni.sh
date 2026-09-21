#!/usr/bin/env bash
# start_sglang_omni.sh — 在 GPU 节点拉起 sglang-omni MOSS-VL realtime 推理实例。
#
# 从 CPU 节点直接执行即可（自动经反向隧道 ssh 到 GPU 节点）；
# 也可以拷贝到 GPU 节点上设 _OMNI_ON_GPU=1 直接跑。
#
# 实例布局：GPUS 按 TP_SIZE 分组，每组一个实例，端口 = PORT_BASE + 实例序号。
#   默认：GPUS=0,1,2,3,4,5,6,7 TP_SIZE=1 → 8 实例，18500..18507
#   TP=2：GPUS=0,1,2,3 TP_SIZE=2       → 2 实例，18500(0,1) 18501(2,3)
#
# 常用 env 覆盖：
#   GPUS TP_SIZE PORT_BASE MODEL_PATH MEM_FRACTION CONTEXT_LENGTH PARKED_TIMEOUT MAX_RUNNING_REQUESTS
set -euo pipefail

GPU_SSH=${GPU_SSH:-"ssh -p 10008 -o BatchMode=yes -o ConnectTimeout=10 root@127.0.0.1"}
REPO=${MOSS_DEPLOY_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

if [ "${_OMNI_ON_GPU:-0}" != "1" ]; then
  # CPU 节点入口：把所有配置透传到 GPU 节点执行本脚本
  exec $GPU_SSH "_OMNI_ON_GPU=1 MOSS_DEPLOY_REPO='$REPO' \
    OMNI_ROOT='${OMNI_ROOT:-}' OMNI_PYTHON='${OMNI_PYTHON:-}' \
    MODEL_PATH='${MODEL_PATH:-}' GPUS='${GPUS:-}' TP_SIZE='${TP_SIZE:-}' \
    PORT_BASE='${PORT_BASE:-}' MEM_FRACTION='${MEM_FRACTION:-}' \
    CONTEXT_LENGTH='${CONTEXT_LENGTH:-}' \
    PARKED_TIMEOUT='${PARKED_TIMEOUT:-}' MAX_RUNNING_REQUESTS='${MAX_RUNNING_REQUESTS:-}' \
    LOG_DIR='${LOG_DIR:-}' WAIT='${WAIT:-}' bash -s" < "$0"
fi

# ---------------- 以下在 GPU 节点执行 ----------------
OMNI_ROOT=${OMNI_ROOT:-/inspire/qb-ilm/project/video-understanding/public/train/moss_vl_streaming/8B/final_release/MOSS-VL-Realtime-sglang}
OMNI_PYTHON=${OMNI_PYTHON:-$OMNI_ROOT/.venv-main/bin/python}
MODEL_PATH=${MODEL_PATH:-/inspire/qb-ilm/project/video-understanding/public/train/moss_vl_streaming/8B/final_release/mossvl_streaming_tf_5.12.1}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TP_SIZE=${TP_SIZE:-1}
PORT_BASE=${PORT_BASE:-18500}
MEM_FRACTION=${MEM_FRACTION:-0.5}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-131072}
PARKED_TIMEOUT=${PARKED_TIMEOUT:-3600}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-1}
LOG_DIR=${LOG_DIR:-/inspire/hdd/project/video-understanding/public/personal/yxchen/mossvl_realtime_inference/logs/sglang_omni}
WAIT=${WAIT:-1}

# 帧窗口：60s 原生滑窗，超龄淘汰，不做池化（对齐 deployment/moss_vl_realtime/config.json）
export REALTIME_FRAME_WINDOW_ENABLED=${REALTIME_FRAME_WINDOW_ENABLED:-1}
export REALTIME_FRAME_WINDOW_RAW_S=${REALTIME_FRAME_WINDOW_RAW_S:-60}
export REALTIME_FRAME_POOLING_ENABLED=${REALTIME_FRAME_POOLING_ENABLED:-0}

# 对齐参考启动器（sglang-omni-main/deployment/moss_vl_realtime/common.py environment()）：
# 严格端口、离线模式、剥离代理变量，避免继承 shell 的 http_proxy 等污染服务进程
export SGLANG_OMNI_STRICT_PORT=${SGLANG_OMNI_STRICT_PORT:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
for v in $(compgen -v | grep -i '_proxy$'); do unset "$v"; done
export NO_PROXY='*' no_proxy='*'

mkdir -p "$LOG_DIR"
[ -x "$OMNI_PYTHON" ] || { echo "FATAL: python not found: $OMNI_PYTHON" >&2; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "FATAL: model not found: $MODEL_PATH" >&2; exit 1; }

# 参考实现的卡占用护栏：used ≤ 2GiB 且 free ≥ 65GiB 才允许落实例
check_gpu_idle() { # $1 = 物理卡号
  local used free
  read -r used free < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" \
    | awk -F', *' '{print $1, $2-$1}')
  [ -n "$used" ] || { echo "FATAL: 读不到 GPU $1 的显存状态" >&2; return 1; }
  if (( used > 2048 )); then
    echo "FATAL: GPU $1 已被占用（used ${used}MiB > 2048MiB）；先释放或换卡" >&2; return 1
  fi
  if (( free < 65 * 1024 )); then
    echo "FATAL: GPU $1 空闲显存不足（free $((free/1024))GiB < 65GiB，128K profile 需要）" >&2; return 1
  fi
}

IFS=',' read -ra GPU_ARR <<< "$GPUS"
n=${#GPU_ARR[@]}
if (( n % TP_SIZE != 0 )); then
  echo "FATAL: |GPUS|=$n not divisible by TP_SIZE=$TP_SIZE" >&2; exit 1
fi
instances=$(( n / TP_SIZE ))
echo "Plan: $instances instance(s), TP_SIZE=$TP_SIZE, ports $PORT_BASE..$((PORT_BASE+instances-1))"

for (( i=0; i<instances; i++ )); do
  port=$((PORT_BASE + i))
  if (( TP_SIZE == 1 )); then
    gpu_args="--gpu ${GPU_ARR[$i]}"
    tag="gpu${GPU_ARR[$i]}"
  else
    group=()
    for (( k=0; k<TP_SIZE; k++ )); do group+=("${GPU_ARR[$((i*TP_SIZE+k))]}"); done
    gpu_args="--tp-size $TP_SIZE --gpus $(IFS=,; echo "${group[*]}")"
    tag="gpu$(IFS=-; echo "${group[*]}")"
  fi

  "$OMNI_PYTHON" "$REPO/scripts/deploy/stop_backend.py" --repo "$REPO" \
    --role omni --port "$port" --require-free-port "$port"

  # 同端口旧实例已停，再按参考实现的护栏确认目标卡空闲
  if (( TP_SIZE == 1 )); then
    check_gpu_idle "${GPU_ARR[$i]}"
  else
    for (( k=0; k<TP_SIZE; k++ )); do check_gpu_idle "${GPU_ARR[$((i*TP_SIZE+k))]}"; done
  fi

  log="$LOG_DIR/omni_${tag}_p${port}.log"
  echo "[$tag] starting on :$port  (log: $log)"
  cd "$OMNI_ROOT/sglang-omni-main"
  MOSS_DEPLOY_REPO="$REPO" MOSS_DEPLOY_ROLE=omni MOSS_DEPLOY_PORT="$port" \
    PATH="$(dirname "$OMNI_PYTHON"):$PATH" nohup "$OMNI_PYTHON" \
    examples/run_moss_vl_realtime_server.py \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 --port "$port" \
    $gpu_args \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length "$CONTEXT_LENGTH" \
    --parked-request-timeout "$PARKED_TIMEOUT" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    >> "$log" 2>&1 &
  echo $! > "$LOG_DIR/omni_${tag}_p${port}.pid"
done

if [ "$WAIT" = "1" ]; then
  echo "Waiting for /health (flashinfer JIT 可能要 10-20min，Ctrl-C 只中断等待不影响实例)..."
  for (( i=0; i<instances; i++ )); do
    port=$((PORT_BASE + i))
    for (( t=0; t<180; t++ )); do
      if curl -s --max-time 3 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"status": *"healthy"'; then
        echo "[:$port] healthy"; break
      fi
      sleep 10
    done
  done
fi
echo "Done."
