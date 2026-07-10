#!/usr/bin/env bash
# PD-disaggregated serve of LongCat-2.0-FP8 over Mooncake + EFA.
#
# Topology (LongCat-2.0 is ~2TB and needs 16 GPUs => each instance spans 2 nodes):
#   Prefill instance = tp16/ep16 across 2 nodes (nnodes=2), disaggregation-mode=prefill
#   Decode  instance = tp16/ep16 across 2 nodes (nnodes=2), disaggregation-mode=decode
#   => "2 nodes prefill + 2 nodes decode" == 1 Prefill instance + 1 Decode instance (1P1D).
#
# Uses the UCCL-EP + Mooncake-EFA image built by scripts/build-and-push-ecr.sh.
# KV cache is transferred prefill->decode over EFA via Mooncake (MOONCAKE_PROTOCOL=efa).
#
# Run ONE command per node:
#   Prefill head : bash serve-pd.sh prefill 0 <prefill_head_ip>
#   Prefill node1: bash serve-pd.sh prefill 1 <prefill_head_ip>
#   Decode  head : bash serve-pd.sh decode  0 <decode_head_ip>
#   Decode  node1: bash serve-pd.sh decode  1 <decode_head_ip>
#
# Then start the router on the prefill head (see serve-router.sh).
set -euo pipefail

ROLE="${1:?usage: serve-pd.sh <prefill|decode> <node-rank 0|1> <group-head-ip>}"
RANK="${2:?usage: serve-pd.sh <prefill|decode> <node-rank 0|1> <group-head-ip>}"
HEADIP="${3:?usage: serve-pd.sh <prefill|decode> <node-rank 0|1> <group-head-ip>}"

case "$ROLE" in
  # MoE all-to-all over UCCL-EP (DeepEP-compatible wrapper, EFA — no NVSHMEM):
  #   prefill -> normal mode (throughput); decode -> low_latency (keeps CUDA graph on).
  prefill) PORT="${PORT:-8010}"; CONTAINER=sglang-prefill; DEEPEP_MODE="${DEEPEP_MODE:-normal}" ;;
  decode)  PORT="${PORT:-8020}"; CONTAINER=sglang-decode;  DEEPEP_MODE="${DEEPEP_MODE:-low_latency}" ;;
  *) echo "ROLE must be 'prefill' or 'decode'"; exit 1 ;;
esac

# v3 image = EFA installer 1.49 + NCCL_NET_PLUGIN baked in (fixes NCCL->TCP fallback).
IMG="${IMG:-579019700964.dkr.ecr.us-east-2.amazonaws.com/ep-benchmarks-efa/uccl-ep-efa:ucclep-longcat2-efa149-v3}"
MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/LongCat-2.0-FP8}"
IFACE="${IFACE:-$(ip -o -4 addr show | awk '$2!="lo" && $2!~"docker"{print $2; exit}')}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
# Separate torch-distributed rendezvous port per group so the two 2-node worlds don't clash.
DIST_PORT="${DIST_PORT:-20000}"
# chunked-prefill-size: 16384 gives lower TTFT but needs mem-fraction 0.85 to avoid OOM
# (0.92 OOMs at chunk 16384). Default here = 16384/0.85 for the prefill-focused sweep.
CHUNK="${CHUNK:-16384}"
MEM_FRAC="${MEM_FRAC:-0.85}"
# context-length: MUST cap this. LongCat-2.0's native context is 256K; if left unset,
# sglang sizes the KV pool for 256K and — after the ~107GB/GPU weights on tp16 — the KV
# cache allocation fails, so the server hangs AFTER "Load weight end" and never reaches
# "KV Cache is allocated"/"fired up" (looks like a load hang, but it's KV-pool OOM).
# 131072 (128K) sizes the KV pool to ~62784 tokens (~5.7GB/rank) and starts cleanly.
# Lower it further (e.g. 32768) if you need more KV headroom or higher concurrency.
CTX_LEN="${CTX_LEN:-131072}"

sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
sudo mkdir -p /opt/dlami/nvme/jit-cache

sudo docker run -d --name "$CONTAINER" --gpus all --network host --shm-size 64g --ipc host \
  --privileged -v /dev/infiniband:/dev/infiniband \
  -v "${MODEL_DIR}:/models/LongCat-2.0-FP8" \
  -v /opt/dlami/nvme/jit-cache:/root/.cache \
  --entrypoint bash "$IMG" -c "
    export MOONCAKE_PROTOCOL=efa;
    export FI_PROVIDER=efa FI_EFA_USE_DEVICE_RDMA=1 FI_EFA_FORK_SAFE=1 FI_MR_CACHE_MONITOR=disabled;
    # EFA installer 1.49's default CUDA memory-registration path (dmabuf) fails to
    # register one GPU rail (cuda:1, fi_mr_regattr 'Operation not supported', ret -202)
    # on this host (kernel 6.17-aws, no /dev/dmabuf_reg), which kills Mooncake KV-over-EFA.
    # Force the legacy nvidia-peermem path so all 8 GPUs register. (1.48 defaulted to this.)
    export FI_HMEM_CUDA_USE_DMABUF=0;
    export NCCL_SOCKET_IFNAME=${IFACE} GLOO_SOCKET_IFNAME=${IFACE};
    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=300;
    python3 -m sglang.launch_server \
      --model-path /models/LongCat-2.0-FP8 --trust-remote-code \
      --tp 16 --ep 16 --nnodes 2 --node-rank ${RANK} --dist-init-addr ${HEADIP}:${DIST_PORT} \
      --context-length ${CTX_LEN} \
      --max-running-requests 64 --mem-fraction-static ${MEM_FRAC} \
      --chunked-prefill-size ${CHUNK} --nsa-prefill-backend fa3 --kv-cache-dtype bfloat16 \
      --moe-a2a-backend deepep --deepep-mode ${DEEPEP_MODE} \
      --disable-radix-cache \
      --disaggregation-mode ${ROLE} \
      --disaggregation-transfer-backend mooncake \
      --disaggregation-bootstrap-port ${BOOTSTRAP_PORT} \
      --host 0.0.0.0 --port ${PORT}"

echo "launched role=${ROLE} rank=${RANK} iface=${IFACE} head=${HEADIP} port=${PORT} (container: ${CONTAINER})"
echo "verify EFA KV path: sudo docker logs ${CONTAINER} 2>&1 | grep -i 'EFA transport installed successfully'"
