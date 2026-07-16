#!/usr/bin/env bash
# PD-disaggregation router. Run on the PREFILL head node.
#
# IMPORTANT (from the Mooncake PD notes):
#   * <prefill_ip> / <decode_ip> must be the nodes' REACHABLE IPs, never 127.0.0.1 —
#     the router forwards the prefill address to the decode node for the bootstrap_room
#     handshake; localhost stalls traffic at 0/N with "Connection refused".
#   * The trailing bootstrap port after --prefill MUST match the prefill's
#     --disaggregation-bootstrap-port (default 8998).
#
# Usage:
#   bash serve-router.sh <prefill_ip> <decode_ip>
# Scale out (4P4D etc.) by repeating --prefill/--decode pairs.
set -euo pipefail

PREFILL_IP="${1:?usage: serve-router.sh <prefill_ip> <decode_ip>}"
DECODE_IP="${2:?usage: serve-router.sh <prefill_ip> <decode_ip>}"

# Same image as the serve scripts (any image with sglang works — the router only runs
# sglang_router.launch_router and loads no model — but keep it consistent).
IMG="${IMG:-ucclep-sglang-efa:latest}"
PREFILL_PORT="${PREFILL_PORT:-8010}"
DECODE_PORT="${DECODE_PORT:-8020}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

sudo docker rm -f sglang-router >/dev/null 2>&1 || true

sudo docker run -d --name sglang-router --network host --entrypoint bash "$IMG" -c "
  pip install -q sglang-router 2>/dev/null || true;
  python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://${PREFILL_IP}:${PREFILL_PORT} ${BOOTSTRAP_PORT} \
    --decode  http://${DECODE_IP}:${DECODE_PORT} \
    --policy round_robin \
    --host 0.0.0.0 --port ${ROUTER_PORT}"

echo "router up on :${ROUTER_PORT}  prefill=${PREFILL_IP}:${PREFILL_PORT}(bootstrap ${BOOTSTRAP_PORT})  decode=${DECODE_IP}:${DECODE_PORT}"
echo "test: bash smoke-test.sh localhost:${ROUTER_PORT}"
