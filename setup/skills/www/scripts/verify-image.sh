#!/bin/bash
# 推理镜像验收 — 在任一 GPU 节点执行
# 校验项对应 efa-validation BUILD_MATRIX.md §Verification
set -u
IMG="${1:-public.ecr.aws/n3l4x8f3/sglang-mooncake-uccl:2026.05.02-h200.dp16}"
echo "验收镜像: $IMG"

docker run --rm --gpus all "$IMG" bash -c '
set -e
python -c "import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.version.cuda)"
python -c "import sglang; print(\"sglang:\", sglang.__version__)"
python -c "from mooncake.engine import TransferEngine; print(\"mooncake.engine: OK\")"
python -c "import uccl.ep; print(\"uccl.ep: OK\")"
python -c "import deep_ep; m=deep_ep.Config.__module__; assert m.startswith(\"uccl\"), f\"deep_ep 未指向 uccl: {m}\"; print(\"deep_ep->uccl: OK\")"
python - <<EOF
import inspect
import sglang.srt.distributed.device_communicators.mooncake_transfer_engine as m
src = inspect.getsource(m)
assert "\"efa\"," in src, "efa 协议补丁缺失 — KV 会回落 TCP!"
assert "\"rdma\"," not in src, "rdma token 残留"
print("mooncake efa patch: OK")
EOF
test -x /usr/bin/ninja && echo "ninja: OK"
test -x /opt/mooncake/install/bin/transfer_engine_bench && echo "transfer_engine_bench: OK"
echo "--- BUILD_INFO ---"
cat /opt/BUILD_INFO.json
'

echo ""
echo "=== EFA 直通验证 (host network + privileged) ==="
docker run --rm --network host --privileged "$IMG" fi_info -p efa -l | head -20
echo "期望: 16 个 rdmapXXs0 设备"
