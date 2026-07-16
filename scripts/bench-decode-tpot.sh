#!/usr/bin/env bash
# Decode TPOT benchmark against the PD router.
#   ISL = 1024, OSL = 1024, concurrency = 1,2,4,8,16,32,64.
#   Reports TPOT (time-per-output-token) — the decode-bound metric.
#
# STRICT lengths: --random-range-ratio 1.0 => every request exactly
# --random-input-len / --random-output-len.
# EOS handling: REQUIRED to ignore EOS here — LongCat-2.0 is a reasoning model and
# would emit EOS before 1024 tokens, making OSL (and thus TPOT) unrepresentative.
# This bench_serving build IGNORES EOS BY DEFAULT (only --disable-ignore-eos turns
# it off), so no flag is needed — every request generates the full 1024 tokens.
# Do NOT pass --disable-ignore-eos.
#
# Run ON the router host (P5EN-1).
#   PD-disaggregated (serve-pd.sh + serve-router.sh):  bash bench-decode-tpot.sh
#   Single non-disagg instance (serve-tp16.sh):        CONTAINER=sglang-tp16 bash bench-decode-tpot.sh
# (serve-tp16.sh and the PD router both listen on PORT 8000; only the container name differs.)
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"                 # PD router
MODEL="${MODEL:-/models/LongCat-2.0-FP8}"
CONTAINER="${CONTAINER:-sglang-prefill}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32 64}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
OUT_DIR="${OUT_DIR:-/opt/dlami/nvme/bench/decode}"
STAMP="${STAMP:-run}"

sudo mkdir -p "$OUT_DIR" && sudo chown -R "$(id -un)":"$(id -gn)" "$OUT_DIR"
echo "=== Decode TPOT sweep: ISL=$ISL OSL=$OSL conc=[$CONCURRENCIES] -> $OUT_DIR ==="

for C in $CONCURRENCIES; do
  # num-prompts = 2x concurrency (min 8): each request is long (1024 out), keep it bounded
  NP=$(( C * 2 )); [ "$NP" -lt 8 ] && NP=8
  OUT="$OUT_DIR/tpot_isl${ISL}_osl${OSL}_c${C}_${STAMP}.log"
  JSONL_C="/root/.cache/bench_tpot_isl${ISL}_osl${OSL}_c${C}_${STAMP}.jsonl"
  JSONL_H="/opt/dlami/nvme/jit-cache/bench_tpot_isl${ISL}_osl${OSL}_c${C}_${STAMP}.jsonl"
  echo ">>> conc=$C num_prompts=$NP"
  # warmup = one concurrency wave to trigger decode CUDA-graph capture for this
  # batch size; --flush-cache clears prefix cache. Save BOTH full stdout ($OUT)
  # and machine-readable JSONL (--output-file via mounted jit-cache).
  sudo docker exec "$CONTAINER" python3 -m sglang.bench_serving \
    --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --max-concurrency "$C" --num-prompts "$NP" \
    --warmup-requests "$C" --flush-cache \
    --output-file "$JSONL_C" 2>&1 | grep -viE 'it/s|%\|' \
    | tee "$OUT" \
    | grep -iE 'Successful|Concurrency|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|Output token throughput' || true
  cp -f "$JSONL_H" "$OUT_DIR/" 2>/dev/null || true
done
echo "=== done. per-run logs (.log) + JSONL in $OUT_DIR ==="
