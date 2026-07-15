#!/usr/bin/env bash
# Single NON-disaggregated LongCat-2.0-FP8 instance (tp16/ep16 across 2 nodes).
#
# PD-disaggregation needs 4 nodes; with only 2 you can't run PD. This brings up one
# plain serve you can curl directly (no router, no Mooncake KV transfer) to check
# correctness / concurrency. All real-EP fixes are baked into the image, so there is
# no runtime patching here.
#
# Run ONE command per node (both nodes form one tp16/ep16 world):
#   head : bash serve-tp16.sh 0 <head_ip>
#   node1: bash serve-tp16.sh 1 <head_ip>
# Then smoke-test from the head:  bash serve-tp16.sh smoke
#
# Env:
#   MOE_A2A=deepep     real EP dispatch over UCCL-EP (default). none = EP-over-TP.
#   DEEPEP_MODE=normal|low_latency  (default normal; LL keeps CUDA graph on for decode
#                                    but needs small CHUNK / DDT — see below).
#   DDT                LL per-rank dispatch token cap (default 128; only used by LL).
#   IMG, MODEL_DIR, CTX_LEN, MEM_FRAC, CHUNK, KV_DTYPE, DSA_BACKEND, MAXRUN, PORT, DIST_PORT
set -euo pipefail

# --- smoke subcommand: hit the local server and print outputs for eyeballing ---
if [ "${1:-}" = "smoke" ]; then
  PORT="${PORT:-8000}"
  for pair in \
    "The capital of France is|32" \
    "请用一句话介绍北京：|48" \
    "12 * 8 =|16"; do
    txt="${pair%|*}"; n="${pair#*|}"
    echo "=== prompt: ${txt} ==="
    curl -s "http://localhost:${PORT}/generate" -H 'Content-Type: application/json' \
      -d "{\"text\": $(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$txt"), \
           \"sampling_params\": {\"temperature\": 0, \"max_new_tokens\": ${n}}}" \
      | python3 -c 'import sys,json; print(repr(json.load(sys.stdin).get("text","<none>")))'
  done
  exit 0
fi

RANK="${1:?usage: serve-tp16.sh <node-rank 0|1> <head-ip>   (or: serve-tp16.sh smoke)}"
HEADIP="${2:?usage: serve-tp16.sh <node-rank 0|1> <head-ip>}"

CONTAINER="${CONTAINER:-sglang-tp16}"
PORT="${PORT:-8000}"
DIST_PORT="${DIST_PORT:-20000}"
IMG="${IMG:-ucclep-sglang-efa:latest}"
MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/LongCat-2.0-FP8}"
IFACE="${IFACE:-$(ip -o -4 addr show | awk '$2!="lo" && $2!~"docker"{print $2; exit}')}"
CTX_LEN="${CTX_LEN:-8192}"
MEM_FRAC="${MEM_FRAC:-0.85}"
CHUNK="${CHUNK:-8192}"
KV_DTYPE="${KV_DTYPE:-bfloat16}"
DSA_BACKEND="${DSA_BACKEND:-fa3}"
MAXRUN="${MAXRUN:-128}"
# UCCL/DeepEP low-latency per-rank dispatch token cap (only used when DEEPEP_MODE=low_latency).
# Must be >= the per-rank decode token count (~MAXRUN); 256 gives headroom over MAXRUN=128
# without OOMing CUDA-graph capture (512 does OOM on LongCat). Ignored in normal mode.
DDT="${DDT:-256}"

# MoE all-to-all backend. Default deepep (real EP); set none for the EP-over-TP control.
MOE_A2A="${MOE_A2A:-deepep}"
DEEPEP_MODE="${DEEPEP_MODE:-normal}"
case "$MOE_A2A" in
  ""|none)  MOE_A2A_ARGS="" ;;
  deepep)   MOE_A2A_ARGS="--moe-a2a-backend deepep --deepep-mode ${DEEPEP_MODE}" ;;
  *)        MOE_A2A_ARGS="--moe-a2a-backend ${MOE_A2A}" ;;
esac

sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
sudo mkdir -p /opt/dlami/nvme/jit-cache

sudo docker run -d --name "$CONTAINER" --gpus all --network host --shm-size 64g --ipc host \
  --privileged -v /dev/infiniband:/dev/infiniband \
  -v "${MODEL_DIR}:/models/LongCat-2.0-FP8" \
  -v /opt/dlami/nvme/jit-cache:/root/.cache \
  --entrypoint bash "$IMG" -c "
    export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1 FI_MR_CACHE_MONITOR=disabled;
    export FI_HMEM_CUDA_USE_DMABUF=0;
    export NCCL_SOCKET_IFNAME=${IFACE} GLOO_SOCKET_IFNAME=${IFACE};
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1;
    export UCCL_EP_CPU_TIMEOUT_SECS=${UCCL_EP_CPU_TIMEOUT_SECS:-600};
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${DDT};
    python3 -m sglang.launch_server \
      --model-path /models/LongCat-2.0-FP8 --trust-remote-code \
      --tp 16 --ep 16 --nnodes 2 --node-rank ${RANK} --dist-init-addr ${HEADIP}:${DIST_PORT} \
      --context-length ${CTX_LEN} \
      --max-running-requests ${MAXRUN} --mem-fraction-static ${MEM_FRAC} \
      --chunked-prefill-size ${CHUNK} --nsa-prefill-backend ${DSA_BACKEND} --kv-cache-dtype ${KV_DTYPE} \
      ${MOE_A2A_ARGS} \
      --disable-radix-cache \
      --host 0.0.0.0 --port ${PORT}"

echo "launched rank=${RANK} head=${HEADIP} iface=${IFACE} port=${PORT} MOE_A2A=${MOE_A2A} DEEPEP_MODE=${DEEPEP_MODE} (container: ${CONTAINER})"
echo "watch load: sudo docker logs -f ${CONTAINER} 2>&1 | grep -iE 'fired up|error'"
echo "when ready: bash $(basename "$0") smoke"
