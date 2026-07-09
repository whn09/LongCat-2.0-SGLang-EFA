#!/bin/bash
# SGLang 2P2D 启动脚本 — 在推理 Pod 容器内执行
# 参考: efa-validation/customer_glm_5.1 docker-compose + sglang-2p2d-ucclep-nixl.skill
#
# 用法:
#   launch-2p2d.sh prefill <node_rank 0|1> <prefill_master_ip>
#   launch-2p2d.sh decode  <node_rank 0|1> <decode_master_ip>
#   PREFILL_ADDR=<ip> DECODE_ADDR=<ip> launch-2p2d.sh router 0 0
#
# 必改项:
#   MODEL_PATH        模型本地路径 (/models/<MODEL>)
#   KV_IB_DEVICES     Mooncake KV 用的 EFA 设备 (ls /sys/class/infiniband/ 枚举后填)
#   EP_IB_DEVICES     UCCL-EP 用的 EFA 设备 (与 KV 分半, 避免争抢)
set -e
export PYTHONUNBUFFERED=1

ROLE=${1:?usage: $0 <prefill|decode|router> <node_rank> <master_ip>}
NODE_RANK=${2:?node_rank required}
MASTER_IP=${3:?master_ip required}
MODEL_PATH="${MODEL_PATH:-/models/MODEL_NAME}"

# ============ EFA rail 切分 (按实机枚举结果替换!) ============
# 全部 16 设备可: ls /sys/class/infiniband/ | paste -sd, -
ALL_IB=$(ls /sys/class/infiniband/ | paste -sd, -)
KV_IB_DEVICES="${KV_IB_DEVICES:-$ALL_IB}"           # Mooncake KV (可全给或低 8)
EP_IB_DEVICES="${EP_IB_DEVICES:-}"                  # UCCL-EP 高 8 (留空 = 不隔离)

# ============ 通用环境变量 ============
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_HUGE_PAGE=0
export FI_MR_CACHE_MONITOR=disabled

export NCCL_SOCKET_IFNAME=enp
export NCCL_PROTO=Simple
export NCCL_CROSS_NIC=1
export NCCL_NVLS_ENABLE=1

export UCCL_SOCKET_IFNAME=enp
export UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=128
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
if [ -n "$EP_IB_DEVICES" ]; then
  export UCCL_IB_HCA="$EP_IB_DEVICES"
  export FI_EFA_IFACE="$EP_IB_DEVICES"
fi

export SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=10
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=300
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=300
export SGLANG_SET_CPU_AFFINITY=1

# JIT 缓存持久化 (热启动 5min vs 冷启动 30min)
export SGLANG_DG_CACHE_DIR=/data/cache/deep_gemm
export TRITON_CACHE_DIR=/data/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor
mkdir -p "$SGLANG_DG_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --disaggregation-transfer-backend mooncake
  --disaggregation-ib-device "$KV_IB_DEVICES"
  --moe-a2a-backend deepep
  --ep-dispatch-algorithm dynamic
  --eplb-algorithm deepseek
  --page-size 64
  --host 0.0.0.0
)

case "$ROLE" in
  prefill)
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --disaggregation-mode prefill \
      --port 30000 \
      --tp-size 8 --pp-size 2 --dp-size 1 \
      --nnodes 2 --node-rank "$NODE_RANK" \
      --dist-init-addr "${MASTER_IP}:5757" \
      --deepep-mode normal \
      --chunked-prefill-size 16384 \
      --mem-fraction-static 0.85 \
      --watchdog-timeout 3600
    ;;
  decode)
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --disaggregation-mode decode \
      --port 30001 \
      --tp-size 16 --dp-size 16 --enable-dp-attention \
      --nnodes 2 --node-rank "$NODE_RANK" \
      --dist-init-addr "${MASTER_IP}:5757" \
      --deepep-mode low_latency \
      --mem-fraction-static 0.74 \
      --cuda-graph-max-bs 16 \
      --max-running-requests 256 \
      --prefill-round-robin-balance \
      --watchdog-timeout 1200
    ;;
  router)
    : "${PREFILL_ADDR:?router 需要 PREFILL_ADDR}"
    : "${DECODE_ADDR:?router 需要 DECODE_ADDR}"
    exec python3 -m sglang_router.launch_router \
      --pd-disaggregation \
      --prefill "http://${PREFILL_ADDR}:30000" \
      --decode "http://${DECODE_ADDR}:30001" \
      --host 0.0.0.0 --port 8000
    ;;
  *)
    echo "unknown role: $ROLE"; exit 2 ;;
esac
