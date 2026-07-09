#!/bin/bash
# 开机后节点验证: GPU / EFA / 磁盘 / 引导日志
set -u
PASS=0; FAIL=0
ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "=== 1. GPU (期望 8x H200) ==="
GPUS=$(nvidia-smi -L 2>/dev/null | grep -c H200 || echo 0)
[ "$GPUS" -eq 8 ] && ok "8x H200" || bad "H200 数量 = $GPUS (期望 8)"

echo "=== 2. EFA 设备 (期望 16) ==="
EFAS=$(ls /sys/class/infiniband/ 2>/dev/null | wc -l | xargs)
[ "$EFAS" -eq 16 ] && ok "16 个 EFA 设备: $(ls /sys/class/infiniband/ | xargs)" \
                   || bad "EFA 设备数 = $EFAS (期望 16); 检查安全组自引用 egress"
/opt/amazon/efa/bin/fi_info -p efa -l >/dev/null 2>&1 && ok "fi_info -p efa 正常" || bad "fi_info 无输出"

echo "=== 3. 磁盘 ==="
mountpoint -q /data && ok "/data 已挂载 ($(df -h /data | awk 'NR==2{print $2}'))" || bad "/data 未挂载"
mountpoint -q /mnt/nvme && ok "/mnt/nvme 已挂载 ($(df -h /mnt/nvme | awk 'NR==2{print $2}'))" || bad "/mnt/nvme 未挂载"
grep -q 'raid0' /proc/mdstat 2>/dev/null && ok "RAID0 active" || bad "RAID0 未组建"

echo "=== 3.5 内核态组件 (DLAMI 预装) ==="
lsmod | grep -q nvidia && ok "nvidia 内核模块已加载" || bad "nvidia 内核模块未加载"
lsmod | grep -q '^efa' && ok "efa.ko 已加载 ($(modinfo efa 2>/dev/null | awk '/^version/{print $2}'))" || bad "efa.ko 未加载"

echo "=== 4. 容器运行时 ==="
systemctl is-active docker >/dev/null 2>&1 && ok "docker active" || bad "docker 未运行"
command -v nvidia-ctk >/dev/null && ok "nvidia-ctk 存在" || bad "nvidia-container-toolkit 缺失"

echo "=== 5. 引导日志尾部 ==="
sudo tail -5 /var/log/p5en-bootstrap.log 2>/dev/null || echo "  (无引导日志)"

echo ""
echo "结果: $PASS PASS / $FAIL FAIL"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
