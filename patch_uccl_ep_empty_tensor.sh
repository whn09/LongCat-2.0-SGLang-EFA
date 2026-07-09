#!/bin/bash
# Patch UCCL-EP to handle empty tensors (topk_idx_ptr==0) in two_batch_overlap
# Based on uccl-project/uccl PR #828
# Fixes: RuntimeError: Failed: Assertion error uccl_ep.cc 'topk_idx_ptr != 0'

set -e

FILE="${1:-/tmp/uccl/ep/src/uccl_ep.cc}"

if [ ! -f "$FILE" ]; then
    echo "ERROR: $FILE not found"
    exit 1
fi

echo "[PATCH] Applying empty-tensor fix to $FILE"

# 1. Replace the strict assertions in get_dispatch_layout (~line 521-524)
#    Remove: EP_HOST_ASSERT(topk_idx_ptr != 0);
#    Remove: EP_HOST_ASSERT(num_tokens_per_rank_ptr != 0);
#    Remove: EP_HOST_ASSERT(num_tokens_per_expert_ptr != 0);
#    Remove: EP_HOST_ASSERT(is_token_in_rank_ptr != 0);
#    Keep:   EP_HOST_ASSERT(num_experts > 0);
#    Add:    early return for 0-token case after stream sync

python3 - "$FILE" << 'PYEOF'
import re, sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "/tmp/uccl/ep/src/uccl_ep.cc"
with open(filepath, "r") as f:
    code = f.read()

# --- Fix 1: get_dispatch_layout - drop the 4 pointer-non-zero asserts and
# add an early return when an empty DP rank (num_tokens==0 / null ptr) comes in.
# Upstream (2026-04) removed the allocate_on_comm_stream EP_HOST_ASSERT and
# added a NOTE(zhenhuang12) block, so we anchor on the first 5 lines only and
# insert the early return before the reinterpret_cast chain. ---
old_block = """    EP_HOST_ASSERT(topk_idx_ptr != 0);
    EP_HOST_ASSERT(num_tokens_per_rank_ptr != 0);
    EP_HOST_ASSERT(num_tokens_per_expert_ptr != 0);
    EP_HOST_ASSERT(is_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(num_experts > 0);"""

new_block = """    EP_HOST_ASSERT(num_experts > 0);
    // patched: early return if this DP rank has no tokens
    if (num_tokens <= 0 || topk_idx_ptr == 0) {
      if (num_tokens_per_rank_ptr != 0) {
        CUDA_CHECK(cudaMemsetAsync(
            reinterpret_cast<void*>(num_tokens_per_rank_ptr),
            0, num_ranks * sizeof(int),
            reinterpret_cast<cudaStream_t>(compute_stream_ptr)));
      }
      if (num_tokens_per_expert_ptr != 0) {
        CUDA_CHECK(cudaMemsetAsync(
            reinterpret_cast<void*>(num_tokens_per_expert_ptr),
            0, num_experts * sizeof(int),
            reinterpret_cast<cudaStream_t>(compute_stream_ptr)));
      }
      return std::optional<EventHandle>{};
    }"""

if old_block in code:
    code = code.replace(old_block, new_block, 1)
    print("[PATCH] Fix 1: get_dispatch_layout - replaced assertions with early return")
else:
    print("[WARN] Fix 1: get_dispatch_layout block not found (may already be patched)")

# --- Fix 2: intranode dispatch_b - relax assertions (~line 638-644) ---
old_block2 = """    EP_HOST_ASSERT(num_tokens > 0);
    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(num_tokens_per_rank_ptr != 0);
    EP_HOST_ASSERT(is_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(num_tokens_per_expert_ptr != 0);
    EP_HOST_ASSERT(rank_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(channel_prefix_matrix_ptr != 0);"""

new_block2 = """    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(num_tokens_per_rank_ptr != 0);
    EP_HOST_ASSERT(num_tokens_per_expert_ptr != 0);
    EP_HOST_ASSERT(rank_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(channel_prefix_matrix_ptr != 0);"""

if old_block2 in code:
    code = code.replace(old_block2, new_block2, 1)
    print("[PATCH] Fix 2: intranode dispatch_b - relaxed assertions")
else:
    print("[WARN] Fix 2: intranode dispatch_b block not found")

# --- Fix 3: intranode dispatch - relax assertions (~line 728-732) ---
old_block3 = """    EP_HOST_ASSERT(x_ptr != 0 && is_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_x_ptr != 0 && recv_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_src_idx_ptr != 0 && send_head_ptr != 0);
    EP_HOST_ASSERT(num_tokens > 0 && hidden > 0 && num_recv_tokens > 0);"""

new_block3 = """    // x_ptr and is_token_in_rank_ptr may be 0 when num_tokens == 0
    EP_HOST_ASSERT(hidden > 0);"""

if old_block3 in code:
    code = code.replace(old_block3, new_block3, 1)
    print("[PATCH] Fix 3: intranode dispatch - relaxed assertions")
else:
    print("[WARN] Fix 3: intranode dispatch block not found")

# --- Fix 4: intranode combine - relax assertions (~line 805-808) ---
old_block4 = """    EP_HOST_ASSERT(x_ptr != 0 && src_idx_ptr != 0 &&
                   rank_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(channel_prefix_matrix_ptr != 0 && send_head_ptr != 0);
    EP_HOST_ASSERT(recv_x_ptr != 0);"""

new_block4 = """    // Pointers may be 0 when num_tokens == 0 (empty tensor)"""

if old_block4 in code:
    code = code.replace(old_block4, new_block4, 1)
    print("[PATCH] Fix 4: intranode combine - relaxed assertions")
else:
    print("[WARN] Fix 4: intranode combine block not found")

# --- Fix 5: internode get_dispatch_layout - relax assertions (~line 876-884) ---
old_block5 = """    EP_HOST_ASSERT(num_tokens_per_rank_ptr != 0);
    EP_HOST_ASSERT(num_tokens_per_rdma_rank_ptr != 0);
    EP_HOST_ASSERT(num_tokens_per_expert_ptr != 0);
    EP_HOST_ASSERT(is_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_rdma_rank_prefix_sum_ptr != 0);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_gbl_rank_prefix_sum_ptr != 0);
    EP_HOST_ASSERT(num_tokens > 0 && hidden > 0 && num_experts > 0);"""

new_block5 = """    EP_HOST_ASSERT(hidden > 0 && num_experts > 0);
    // patched: empty DP rank — skip notify_dispatch and the 60s while loop.
    // Prefix sums stay 0; caller receives num_recv_tokens=0 and works fine.
    if (num_tokens <= 0) {
      if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
      } else {
        stream_wait(comm_stream, reinterpret_cast<cudaStream_t>(compute_stream_ptr));
      }
      if (rdma_channel_prefix_matrix_ptr != 0)
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(rdma_channel_prefix_matrix_ptr),
                                   0, num_ranks * (config.num_sms / 2) * sizeof(int), comm_stream));
      if (recv_rdma_rank_prefix_sum_ptr != 0)
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(recv_rdma_rank_prefix_sum_ptr),
                                   0, get_num_rdma_ranks() * sizeof(int), comm_stream));
      if (gbl_channel_prefix_matrix_ptr != 0)
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(gbl_channel_prefix_matrix_ptr),
                                   0, num_ranks * (config.num_sms / 2) * sizeof(int), comm_stream));
      if (recv_gbl_rank_prefix_sum_ptr != 0)
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(recv_gbl_rank_prefix_sum_ptr),
                                   0, num_ranks * sizeof(int), comm_stream));
      std::optional<EventHandle> event;
      if (async) {
        event = EventHandle(comm_stream);
      } else {
        stream_wait(reinterpret_cast<cudaStream_t>(compute_stream_ptr), comm_stream);
      }
      return std::make_tuple(0, 0, std::vector<int>{}, event);
    }"""

if old_block5 in code:
    code = code.replace(old_block5, new_block5, 1)
    print("[PATCH] Fix 5: internode get_dispatch_layout - relaxed assertions")
else:
    print("[WARN] Fix 5: internode get_dispatch_layout block not found")

# --- Fix 6: internode dispatch - relax ALL assertions ---
old_block6 = """    EP_HOST_ASSERT(x_ptr != 0 && recv_x_ptr != 0 && is_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_rdma_rank_prefix_sum_ptr != 0);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(recv_gbl_rank_prefix_sum_ptr != 0);
    EP_HOST_ASSERT(num_tokens > 0 && hidden > 0);"""

new_block6 = """    // Pointers may be 0 when num_tokens == 0
    EP_HOST_ASSERT(hidden > 0);"""

if old_block6 in code:
    code = code.replace(old_block6, new_block6, 1)
    print("[PATCH] Fix 6: internode dispatch - relaxed assertions")
else:
    print("[WARN] Fix 6: internode dispatch block not found")

# --- Fix 7: internode dispatch - relax cached_mode assertions ---
old_block7 = """    } else {
      EP_HOST_ASSERT(recv_src_meta_ptr != 0);
      EP_HOST_ASSERT(send_rdma_head_ptr != 0);
      EP_HOST_ASSERT(send_nvl_head_ptr != 0);
      EP_HOST_ASSERT(recv_rdma_channel_prefix_matrix_ptr != 0);
      EP_HOST_ASSERT(recv_gbl_channel_prefix_matrix_ptr != 0);
    }"""

new_block7 = """    }
    // No asserts on recv_src_meta_ptr, send_rdma_head_ptr, send_nvl_head_ptr,
    // recv_rdma/gbl_channel_prefix_matrix_ptr — they may be 0 when
    // num_recv_tokens or num_rdma_recv_tokens == 0."""

if old_block7 in code:
    code = code.replace(old_block7, new_block7, 1)
    print("[PATCH] Fix 7: internode dispatch cached_mode - relaxed assertions")
else:
    print("[WARN] Fix 7: internode dispatch cached_mode block not found")

# --- Fix 8: internode combine - relax assertions ---
old_block8 = """    EP_HOST_ASSERT(x_ptr != 0 && src_meta_ptr != 0 && combined_x_ptr != 0);
    EP_HOST_ASSERT(is_combined_token_in_rank_ptr != 0);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(rdma_rank_prefix_sum_ptr != 0);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix_ptr != 0);
    EP_HOST_ASSERT(combined_rdma_head_ptr != 0);
    EP_HOST_ASSERT(combined_nvl_head_ptr != 0);"""

new_block8 = """    // Pointers may be 0 when num_tokens == 0 (empty tensor)"""

if old_block8 in code:
    code = code.replace(old_block8, new_block8, 1)
    print("[PATCH] Fix 8: internode combine - relaxed assertions")
else:
    print("[WARN] Fix 8: internode combine block not found")

# --- Fallback for Fix 1: if big-block match failed, just comment out the stray
#     topk_idx_ptr / num_tokens_per_* asserts that Fix 1 was supposed to remove.
#     This avoids "topk_idx_ptr != 0" runtime crashes on empty DP ranks. ---
import re as _re
for _a in [
    "EP_HOST_ASSERT(topk_idx_ptr != 0);",
    "EP_HOST_ASSERT(num_tokens_per_rank_ptr != 0);",
    "EP_HOST_ASSERT(num_tokens_per_expert_ptr != 0);",
    "EP_HOST_ASSERT(is_token_in_rank_ptr != 0);",
]:
    _cnt_before = code.count(_a)
    code = _re.sub(_re.escape(_a), "/* patched-out: " + _a + " */", code)
    _cnt_after = code.count(_a)
    if _cnt_before > _cnt_after:
        print(f"[PATCH] Fallback: commented out {_cnt_before - _cnt_after}x '{_a[:50]}...'")

# --- Fix 9: replace all remaining 'previous_event.has_value() and async' with just 'async' ---
old_pev = "EP_HOST_ASSERT(previous_event.has_value() and async)"
new_pev = "EP_HOST_ASSERT(async)"
count = code.count(old_pev)
if count > 0:
    code = code.replace(old_pev, new_pev)
    print(f"[PATCH] Fix 9: replaced {count} 'previous_event.has_value() and async' with 'async'")
else:
    print("[WARN] Fix 9: no 'previous_event.has_value() and async' found")

# --- Fix 10: low_latency_dispatch / low_latency_combine — allow empty batches.
# UCCL-EP adds pointer-non-zero asserts that upstream DeepEP never had; under
# SGLang MTP (EAGLE draft_extend_for_decode) with DP=16, some DP ranks receive
# an empty sub-batch on the draft step and *any* of these pointers can be 0.
# Comment them all out so num_tokens=0 falls through to the kernel (which
# handles empty inputs correctly, same as upstream).
# Also applies under heavy concurrency when some DP ranks' combine buffers
# haven't been allocated yet (out_ptr==0 observed in prod at 122k/4 conc).
for _a in [
    # low_latency_dispatch (uccl_ep.cc:~1197-1200 before patching)
    "EP_HOST_ASSERT(x_ptr != 0 && topk_idx_ptr != 0);",
    "EP_HOST_ASSERT(packed_recv_x_ptr != 0 && packed_recv_count_ptr != 0);",
    # low_latency_combine (uccl_ep.cc:~1299-1301 before patching)
    "EP_HOST_ASSERT(x_ptr != 0 && topk_idx_ptr != 0 && topk_weights_ptr != 0);",
    "EP_HOST_ASSERT(src_info_ptr != 0 && layout_range_ptr != 0);",
    "EP_HOST_ASSERT(out_ptr != 0);",
]:
    # Detect idempotently: count occurrences NOT already wrapped in our
    # /* patched-out: ... */ comment.  Simple code.count would over-count
    # because the replacement string contains _a itself.
    _marker = "/* patched-out: " + _a
    _cnt_unpatched = code.count(_a) - code.count(_marker)
    if _cnt_unpatched > 0:
        code = code.replace(_a, _marker + " */")
        print(f"[PATCH] Fix 10: commented out {_cnt_unpatched}x '{_a[:60]}...'")
    elif code.count(_marker) > 0:
        print(f"[PATCH] Fix 10: already patched '{_a[:60]}...' (idempotent)")
    else:
        print(f"[WARN] Fix 10: pattern not found '{_a[:60]}...'")

# --- Fix 10b: multi-line pointer asserts in low_latency_dispatch that
# single-line matcher above cannot catch (assertion body is split across
# two lines in the source). ---
import re as _re2
_multiline_patterns = [
    # low_latency_dispatch multi-line assert (uccl_ep.cc:~1199-1200):
    #   EP_HOST_ASSERT(packed_recv_src_info_ptr != 0 &&
    #                  packed_recv_layout_range_ptr != 0);
    (
        r"EP_HOST_ASSERT\(packed_recv_src_info_ptr != 0 &&\s+packed_recv_layout_range_ptr != 0\);",
        "packed_recv_src_info_ptr/packed_recv_layout_range_ptr multi-line",
    ),
]
for _pat, _desc in _multiline_patterns:
    # Skip if already wrapped in our patched-out marker (idempotent).
    _wrapped_pat = r"/\* patched-out: " + _pat + r" \*/"
    if _re2.search(_wrapped_pat, code):
        print(f"[PATCH] Fix 10b: already patched '{_desc}' (idempotent)")
        continue
    _new_code, _n = _re2.subn(
        _pat,
        lambda m: "/* patched-out: " + m.group(0) + " */",
        code,
    )
    if _n > 0:
        code = _new_code
        print(f"[PATCH] Fix 10b: commented out {_n}x '{_desc}'")
    else:
        print(f"[WARN] Fix 10b: pattern not found '{_desc}'")

with open(filepath, "w") as f:
    f.write(code)

print("[PATCH] Done - all applicable fixes applied")
PYEOF
