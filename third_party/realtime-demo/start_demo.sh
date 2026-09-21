#!/usr/bin/env bash
# start_demo.sh — 一键启动 MOSS-VL realtime demo（sglang-omni 后端 + 新版 memory）。
#
# 【在 GPU 节点直接执行】：所有进程本地拉起，
# 只经 rtunnel 回连通道 (127.0.0.1:2222) 向 CPU 节点推两条转发，
# 不依赖 CPU→GPU 的 10008 反向隧道（那个挂在 rtunnel 上，rtunnel 一断就死）。
#
# 启动顺序（MiniMax 出口在 gateway 之前，保证 minimax lane 探测就绪）：
#   1. sglang-omni 推理实例   (本地, 布局读 deploy.conf, 默认 OMNI_GPUS=0 → 1 实例 :18500, 全部健康则跳过)
#   2. pi_agent + 4B 后端     (本地, :38082 / :38090, 已健康则跳过)
#   3. MiniMax 出口        (ssh -fN -D 17890, 先于 gateway, 保证 minimax lane 探测就绪)
#   4. demo.sh up             (本地, gateway :8100 + TTS sidecar + web :20941)
#   5. 入口转发（经 2222 回连 CPU）:
#        ssh -fN -R 20941   浏览器入口: CPU 127.0.0.1:20941 → GPU 20941 (nat2 /proxy/20941/)
#      最后从 rtunnel 进程里自动抠出公网 URL 打印
#
# 常用覆盖（env 前缀即可；持久布局改 deploy.conf，别改脚本）：
#   OMNI_GPUS=0,1,2,3 ./start_demo.sh   多铺实例（自动同步 .env.deploy 的 SGLANG_OMNI_URLS）
#   OMNI_GPUS=0,1 OMNI_TP_SIZE=2 ./start_demo.sh
#   DECIDE_LLM_GPU=2 ./start_demo.sh    memory 4B 换卡
#   FORCE_OMNI=1 / FORCE_PI=1           即使健康也强制重启对应组件
#   CPU_PORT=xxxxx ./start_demo.sh      换 CPU 侧入口端口（默认 20941，与 GPU 侧同号）
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$REPO/scripts/deploy/env_lib.sh"
load_env_deploy "$REPO"
WEB_PORT=${WEB_PORT:-20941}          # GPU 侧 web preview 端口
CPU_PORT=${CPU_PORT:-$WEB_PORT}      # 暴露到 CPU 节点的端口（nat2 /proxy/<CPU_PORT>/）
CPU_SSH=${CPU_SSH:-"ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=10 root@127.0.0.1"}
MM_SOCKS_PORT=${MM_SOCKS_PORT:-17890}

# ---- 必须在 GPU 节点上跑 ----
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: 这里没有 GPU（无 nvidia-smi）——请在 GPU 节点上执行本脚本。" >&2
  echo "       （CPU 节点只作为转发入口，不承担任何服务进程）" >&2
  exit 1
fi

# ---- 部署布局配置（deploy.conf，env 优先）----
[ -f "$REPO/deploy.conf" ] && . "$REPO/deploy.conf"
resolve_service_endpoints

GPUS=${OMNI_GPUS:-0,1}
TP_SIZE=${OMNI_TP_SIZE:-1}
port_base=${OMNI_PORT_BASE:-18500}
IFS=',' read -ra _gpu_arr <<< "$GPUS"
if (( ${#_gpu_arr[@]} % TP_SIZE != 0 )); then
  echo "FATAL: |GPUS|=${#_gpu_arr[@]} 不能被 TP_SIZE=$TP_SIZE 整除" >&2; exit 1
fi
instances=$(( ${#_gpu_arr[@]} / TP_SIZE ))
port_last=$((port_base + instances - 1))

# 实例数与 .env.deploy 的 SGLANG_OMNI_URLS 自动对齐（gateway 容量=实际实例数）
omni_urls=""
for (( i=0; i<instances; i++ )); do
  omni_urls+="${omni_urls:+,}http://127.0.0.1:$((port_base + i))"
done
export SGLANG_OMNI_URLS="$omni_urls"
if [ -f "$REPO/.env.deploy" ] && grep -q "^SGLANG_OMNI_URLS=" "$REPO/.env.deploy"; then
  sed -i "s|^SGLANG_OMNI_URLS=.*|SGLANG_OMNI_URLS=$omni_urls|" "$REPO/.env.deploy"
else
  echo "SGLANG_OMNI_URLS=$omni_urls" >> "$REPO/.env.deploy"
fi

healthy() { curl -sf --noproxy '*' --max-time 3 "http://127.0.0.1:$1/health" 2>/dev/null | grep -q '"status": *"healthy"\|"ok": *true'; }

echo "==> [1/5] sglang-omni 推理实例 ($instances 实例: :$port_base..:$port_last, TP_SIZE=$TP_SIZE)"
all_healthy=1
[ "${FORCE_OMNI:-0}" = "1" ] && all_healthy=0
if [ "$all_healthy" = "1" ]; then
  for (( i=0; i<instances; i++ )); do
    port=$((port_base + i))
    if healthy "$port"; then echo "      :$port healthy"; else echo "      :$port 未就绪"; all_healthy=0; fi
  done
fi
if [ "$all_healthy" = "1" ]; then
  echo "      全部 healthy，跳过（FORCE_OMNI=1 可强制重启）"
else
  _OMNI_ON_GPU=1 GPUS=$GPUS TP_SIZE=$TP_SIZE bash "$REPO/scripts/gpu/start_sglang_omni.sh"
fi

echo "==> [2/5] pi_agent + 4B decide/compact 后端 (4B → GPU ${DECIDE_LLM_GPU:-1})"
_PI_ON_GPU=1 START_4B=${START_4B:-1} FORCE_PI=${FORCE_PI:-0} \
  bash "$REPO/scripts/gpu/start_pi_agent.sh"

# 转发重建：按命令行里的转发特征（方向+端口）找到占坑的旧 ssh 客户端杀掉，
# 不依赖 MOSS_DEPLOY_ROLE 标记——无标记的历史/手动进程占着端口也必须能换掉。
# 只匹配 comm=ssh，不会误伤包含同样文本的其他进程（如本脚本自身）。
kill_stale_ssh() { # $1=方向特征（"-D"/"-R"） $2=端口特征（如 "127.0.0.1:17890"）
  local pids pid
  pids="$(ps -ww -eo pid=,comm=,args= | awk -v s1="$1" -v s2="$2" \
    '$2 == "ssh" && index($0, s1) && index($0, s2) {print $1}')"
  for pid in $pids; do
    kill "$pid" 2>/dev/null && echo "      清掉旧转发 pid $pid" || true
  done
  [ -n "$pids" ] && sleep 1
  return 0
}

echo "==> [3/5] MiniMax 云 TTS 出口 (GPU 本地 SOCKS5 :$MM_SOCKS_PORT，经 CPU 直连)"
# 必须先于 demo.sh up：gateway 启动时用 MINIMAX_PROXY 探测 minimax  lane，
# 转发不在则 lane 卡在 not-ready，前端选 minimax 会静默回落本地 nano
kill_stale_ssh "-D" "127.0.0.1:$MM_SOCKS_PORT"
if MOSS_DEPLOY_REPO="$REPO" MOSS_DEPLOY_ROLE=ssh-socks \
  ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -D "127.0.0.1:$MM_SOCKS_PORT" root@127.0.0.1 -p 2222; then
  # 注意不能加 --noproxy '*'：它会把 -x 指定的代理也禁掉，导致探测走直连误报不通
  mm_code=$(curl -s --max-time 8 -x "socks5h://127.0.0.1:$MM_SOCKS_PORT" \
    -o /dev/null -w '%{http_code}' https://api.minimaxi.com/v1/t2a_v2 2>/dev/null || true)
  case "$mm_code" in
    40*) echo "      MiniMax 链路 OK (HTTP $mm_code)" ;;
    *)   echo "      WARNING: MiniMax 链路不通（TTS 选/默认 minimax 会回落 nano）" ;;
  esac
else
  echo "      WARNING: MiniMax SOCKS5 转发建立失败（2222 不通？）"
fi

echo "==> [4/5] gateway + TTS sidecar + web (demo.sh up)"
cd "$REPO" && bash scripts/deploy/demo.sh up

echo "==> [5/5] 浏览器入口转发（经 rtunnel 回连 127.0.0.1:2222）"
# CPU 127.0.0.1:CPU_PORT → GPU 127.0.0.1:WEB_PORT
# （转发失败不致命——端口被占只影响入口，服务本体已就绪，不能拖垮整个脚本）
kill_stale_ssh "-R" "127.0.0.1:$CPU_PORT:"
if MOSS_DEPLOY_REPO="$REPO" MOSS_DEPLOY_ROLE=ssh-browser \
  ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R "127.0.0.1:$CPU_PORT:127.0.0.1:$WEB_PORT" root@127.0.0.1 -p 2222; then
  echo "      入口转发 OK: CPU :$CPU_PORT → GPU :$WEB_PORT"
else
  echo "      WARNING: 入口转发失败（CPU 侧 $CPU_PORT 被占或 2222 不通）——浏览器入口不可用，服务本体正常"
fi

# 4c. 从 rtunnel 进程命令行抠公网 URL，换端口后缀打印
RAW_URL="$(ps -ww -ef | sed -nE 's#.*(https://[^[:space:]]+/proxy/[0-9]+/).*#\1#p' | head -n 1 || true)"
echo
echo "全部就绪。浏览器打开："
echo
if [ -n "$RAW_URL" ]; then
  echo "  $(printf '%s' "$RAW_URL" | sed -E "s#/proxy/[0-9]+/?\$#/proxy/$CPU_PORT/#")"
else
  echo "  <nat2 网关前缀>/proxy/$CPU_PORT/   （没检测到 rtunnel 进程则手动拼前缀）"
fi
echo
echo "（若提示端口未转发，在 VS Code 端口面板手动添加 $CPU_PORT）"
