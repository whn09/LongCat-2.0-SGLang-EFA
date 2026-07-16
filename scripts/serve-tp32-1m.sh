#!/usr/bin/env bash
# Standalone (non-disagg) LongCat-2.0-FP8 across ALL 4 nodes = 32 GPU, aiming at 1M context.
#   tp32 / ep32  -> moe_ep_size=32 (=tp) so experts fully EP-sharded: 1.6T/32 ≈ 50GB/GPU weight,
#                   leaving ~90GB/GPU for KV+activation.
#   NSA prefill context parallel (attn-cp) shards the 1M-token KV/activation across ranks.
# Run one per node (rank 0..3), head = node 0 (P5EN-1 172.31.4.150):
#   bash serve-tp32-1m.sh <node-rank> <head-ip>
set -e

RANK="${1:?usage: serve-tp32-1m.sh <node-rank 0..3> <head-ip>}"
HEADIP="${2:?usage}"

IMG="${IMG:-ucclep-sglang-efa:latest}"
MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/LongCat-2.0-FP8}"
IFACE="${IFACE:-$(ip -o -4 addr show | awk '$2!="lo" && $2!~"docker"{print $2; exit}')}"
DIST_PORT="${DIST_PORT:-20000}"
TP="${TP:-32}"
EP="${EP:-32}"
NNODES="${NNODES:-4}"
CTX_LEN="${CTX_LEN:-1048576}"          # 1M
# chunk=8192 beats 16384 by ~40% at 1M: flashmla_kv exceeds GPU shared-memory at chunk
# 16384 and falls back to a slow low-smem kernel; 8192 avoids that fallback.
CHUNK="${CHUNK:-8192}"
# 0.88 (not 0.85) so the fp8 KV pool reaches ~1.23M tokens = fits 1M. Do NOT raise to 0.93
# (some nodes OOM loading the fp8 weights, leaving a broken world).
MEM_FRAC="${MEM_FRAC:-0.88}"
CP="${CP:-0}"                           # attn context parallel — MUST be 0 at tp32 (CP needs tp<=8)
PORT="${PORT:-30000}"
MAXRUN="${MAXRUN:-1}"                   # lower => more KV pool for a single long request (1 = one 1M req)
# Defaults target the full 1M context, which REQUIRES fp8 KV: bf16 KV pool only reaches
# ~752K tokens (can't fit 1M); fp8_e4m3 halves per-token KV -> ~1.23M-token pool. fp8 in
# turn REQUIRES flashmla_kv (FlashMLA, fp8-native) — fa3 crashes on fp8 ("query and key
# must have the same dtype"). For ≤512K, bf16 + fa3 is faster: KV_DTYPE=bfloat16 DSA_BACKEND=fa3.
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
DSA_BACKEND="${DSA_BACKEND:-flashmla_kv}"  # DSA/NSA prefill kernel: fa3 | flashmla_kv | flashmla_auto | ...

# CP args (NSA prefill context parallel — LongCat uses NSA). Set CP=0 to disable.
CP_ARGS=""
if [ "${CP:-0}" != "0" ]; then
  CP_ARGS="--attn-cp-size ${CP} --enable-nsa-prefill-context-parallel --nsa-prefill-cp-mode round-robin-split"
fi

sudo docker rm -f sglang-tp32 mscclpp-efa >/dev/null 2>&1 || true
sudo mkdir -p /opt/dlami/nvme/jit-cache

sudo docker run -d --name sglang-tp32 --gpus all --network host --shm-size 64g --ipc host \
  --privileged -v /dev/infiniband:/dev/infiniband \
  -v "${MODEL_DIR}:/models/LongCat-2.0-FP8" \
  -v /opt/dlami/nvme/jit-cache:/root/.cache \
  --entrypoint bash "$IMG" -c "
    export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_HMEM_CUDA_USE_DMABUF=0;
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
    export NCCL_SOCKET_IFNAME=${IFACE} GLOO_SOCKET_IFNAME=${IFACE};
    export NCCL_NET_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so;
    export NCCL_TIMEOUT=7200 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200;
    # LongCat config derives max context 262144 (256K); 1M requires YARN extrapolation override.
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1;
    export UCCL_EP_CPU_TIMEOUT_SECS=${UCCL_EP_CPU_TIMEOUT_SECS:-600};
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${DDT:-128};
    python3 -m sglang.launch_server \
      --model-path /models/LongCat-2.0-FP8 --trust-remote-code \
      --tp ${TP} --ep ${EP} \
      ${CP_ARGS} \
      --nnodes ${NNODES} --node-rank ${RANK} --dist-init-addr ${HEADIP}:${DIST_PORT} \
      --context-length ${CTX_LEN} \
      --max-running-requests ${MAXRUN} --mem-fraction-static ${MEM_FRAC} \
      --chunked-prefill-size ${CHUNK} --nsa-prefill-backend ${DSA_BACKEND} --kv-cache-dtype ${KV_DTYPE} \
      --moe-a2a-backend deepep --deepep-mode normal \
      --disable-radix-cache --allow-auto-truncate \
      --watchdog-timeout 1000000 --dist-timeout 7200 \
      --host 0.0.0.0 --port ${PORT}"

echo "launched tp32 rank=${RANK}/${NNODES} TP=${TP} EP=${EP} CP=${CP} ctx=${CTX_LEN} head=${HEADIP} (sglang-tp32)"
