#!/bin/bash
# Patch: wire LongCat-2.0 n-gram embedding into the PD-disaggregated PREFILL and
# DECODE scheduler loops.
#
# Bug: LongCat-2.0 uses an "outer/n-gram embedding" (config.use_ngram_embedding=True).
# The standard scheduler loop calls Scheduler._maybe_prepare_ngram_embedding(batch)
# (scheduler.py get_next_batch_to_run) to fill batch.ne_token_table before run_batch.
# The disaggregation loops do NOT, so:
#   * prefill: batch.ne_token_table stays None -> ForwardBatch._init_ngram_embedding_info
#     feeds None as arg #7 to compute_n_gram_ids:
#       TypeError: Mismatched type on argument #7 ... Expected `DLTensor*` but got `None`
#   * decode: same missing prep -> update_token_table arg #1 (token table) is None:
#       TypeError: Mismatched type on argument #1 ... Expected `DLTensor*` but got `None`
# Every disagg TP rank crashes at first forward. (Non-disaggregated serving is fine.)
#
# Fix: after each `self.cur_batch = batch` in the four disagg loops
# (event_loop_{normal,overlap}_disagg_{prefill,decode}), call
# self._maybe_prepare_ngram_embedding(batch) — parity with the standard loop;
# no-op unless model_config.use_ngram_embedding.
#
# Idempotent. Usage: patch_longcat2_disagg_ngram.sh [sglang_srt_dir]
#   default sglang_srt_dir = /sgl-workspace/sglang/python/sglang/srt
set -e
SRT="${1:-/sgl-workspace/sglang/python/sglang/srt}"

patch_file() {
    local FILE="$1"
    [ -f "$FILE" ] || { echo "ERROR: $FILE not found"; exit 1; }
    if grep -q "_maybe_prepare_ngram_embedding" "$FILE"; then
        echo "[PATCH] already applied to $FILE"; return 0
    fi
    python3 - "$FILE" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
needle = "            self.cur_batch = batch\n"
inject = (
    "            self.cur_batch = batch\n"
    "            # LongCat-2.0 n-gram embedding: fill batch.ne_token_table before forward\n"
    "            # (parity with the standard scheduler loop; no-op unless use_ngram_embedding).\n"
    "            batch = self._maybe_prepare_ngram_embedding(batch)\n"
    "            self.cur_batch = batch\n"
)
n = src.count(needle)
assert n >= 2, f"{path}: expected >=2 cur_batch assignments, found {n}"
src = src.replace(needle, inject)
open(path, "w").write(src)
print(f"[PATCH] injected ngram prep at {n} loop site(s) in {path}")
PY
}

patch_file "${SRT}/disaggregation/prefill.py"
patch_file "${SRT}/disaggregation/decode.py"
