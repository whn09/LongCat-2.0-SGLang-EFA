#!/usr/bin/env bash
# Prefill TTFT benchmark against the PD router.
#   ISL = 1024 and 8192, OSL = 1, concurrency = 1,2,4,8,16,32,64.
#   Reports TTFT (time-to-first-token) — the prefill-bound metric.
#
# STRICT lengths: --random-range-ratio 1.0 makes every request exactly
# --random-input-len / --random-output-len (no random shrink; see datasets/common.py
# compute_random_lens: randint(int(full*ratio), full+1) => full when ratio=1.0).
# EOS handling: this bench_serving build IGNORES EOS BY DEFAULT (only
# --disable-ignore-eos exists to turn it off), so no flag is needed — every
# request generates the full OSL. (At OSL=1 this is moot anyway.)
#
# Run ON the router host (P5EN-1) inside a served container that has sglang.bench_serving.
#   PD-disaggregated (serve-pd.sh + serve-router.sh):  bash bench-prefill-ttft.sh
#   Single non-disagg instance (serve-tp16.sh):        CONTAINER=sglang-tp16 bash bench-prefill-ttft.sh
# (serve-tp16.sh and the PD router both listen on PORT 8000; only the container name differs.)
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"                 # PD router
MODEL="${MODEL:-/models/LongCat-2.0-FP8}"
CONTAINER="${CONTAINER:-sglang-prefill}"   # any container with sglang.bench_serving
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32 64}"
INPUT_LENS="${INPUT_LENS:-1024 8192}"
OUT_DIR="${OUT_DIR:-/opt/dlami/nvme/bench/prefill}"
STAMP="${STAMP:-run}"

sudo mkdir -p "$OUT_DIR" && sudo chown -R "$(id -un)":"$(id -gn)" "$OUT_DIR"
echo "=== Prefill TTFT sweep: ISL=[$INPUT_LENS] OSL=1 conc=[$CONCURRENCIES] -> $OUT_DIR ==="

for ISL in $INPUT_LENS; do
  for C in $CONCURRENCIES; do
    # num-prompts = 4x concurrency (min 16) for a stable steady state
    NP=$(( C * 4 )); [ "$NP" -lt 16 ] && NP=16
    OUT="$OUT_DIR/ttft_isl${ISL}_c${C}_${STAMP}.log"
    # bench_serving --output-file writes INSIDE the container; jit-cache is mounted
    # to /opt/dlami/nvme/jit-cache, so the JSONL lands on the host too.
    JSONL_C="/root/.cache/bench_ttft_isl${ISL}_c${C}_${STAMP}.jsonl"
    JSONL_H="/opt/dlami/nvme/jit-cache/bench_ttft_isl${ISL}_c${C}_${STAMP}.jsonl"
    echo ">>> ISL=$ISL conc=$C num_prompts=$NP"
    # warmup = one full concurrency wave to trigger JIT/CUDA-graph for this batch size;
    # --flush-cache clears radix/prefix cache so repeated random prompts don't get
    # a prefix-cache hit that would skip prefill and fake a low TTFT (server also
    # runs --disable-radix-cache, so this is belt-and-suspenders).
    # Save BOTH: full human-readable stdout ($OUT) and machine-readable JSONL
    # (--output-file, every metric + per-request detail).
    sudo docker exec "$CONTAINER" python3 -m sglang.bench_serving \
      --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
      --dataset-name random \
      --random-input-len "$ISL" --random-output-len 1 \
      --random-range-ratio 1.0 \
      --max-concurrency "$C" --num-prompts "$NP" \
      --warmup-requests "$C" --flush-cache \
      --output-file "$JSONL_C" 2>&1 | grep -viE 'it/s|%\|' \
      | tee "$OUT" \
      | grep -iE 'Successful|Concurrency|Mean TTFT|Median TTFT|P99 TTFT|Input token throughput' || true
    cp -f "$JSONL_H" "$OUT_DIR/" 2>/dev/null || true
  done
done
echo "=== done. per-run logs (.log) + JSONL in $OUT_DIR ==="
