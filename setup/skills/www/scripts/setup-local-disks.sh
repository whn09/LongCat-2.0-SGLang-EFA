#!/bin/bash
# =====================================================================
# setup-local-disks.sh  —  每次开机自愈本地 NVMe 盘
# 由 systemd 服务 setup-local-disks.service 在 docker/containerd 之前运行。
#
# 背景: 本地 NVMe (Instance Store) 物理绑定宿主机, stop/start 后换新宿主机
# = 一组全新空盘 (RAID 元数据 + 文件系统全没了)。所以必须每次开机:
#   1. 重新发现本地盘 -> 组 RAID0 -> mkfs -> 挂 /mnt/nvme
#   2. bind /var/lib/docker 和 /var/lib/containerd 到 /mnt/nvme 上
# 让 docker/containerd 每次都落回高速本地 RAID (盘上数据本身是临时的,丢失可接受)。
# 幂等: 已挂载/已 bind 则跳过, 可安全重复运行。
# =====================================================================
set -uo pipefail
log() { echo "[setup-local-disks] $*"; }

MNT=/mnt/nvme
MD=/dev/md0   # 默认名, 仅空盘新建时使用; reboot 场景会被下面的探测覆盖

# 1. 发现本地 instance-store NVMe 盘 (优先稳定的 by-id, 回退 model 字符串)
DISKS=()
if [ -d /dev/disk/by-id ]; then
  while IFS= read -r d; do [ -n "$d" ] && DISKS+=("$d"); done \
    < <(find -L /dev/disk/by-id/ -xtype l -name '*NVMe_Instance_Storage_*' 2>/dev/null | xargs -r -n1 realpath | sort -u)
fi
if [ "${#DISKS[@]}" -eq 0 ]; then
  for s in /sys/block/nvme*n1; do
    [ -e "$s" ] || continue
    m=$(cat "$s/device/model" 2>/dev/null | tr -d ' ')
    case "$m" in *InstanceStorage*) DISKS+=("/dev/$(basename "$s")") ;; esac
  done
fi
if [ "${#DISKS[@]}" -eq 0 ]; then
  log "未发现本地 instance-store NVMe 盘, 跳过 (docker/containerd 将用根盘)"
  exit 0
fi
log "本地盘: ${DISKS[*]}"

# mdadm 是硬前置: 缺失时绝不能继续走 bind 分支
# (2026-07-05 真机事故: 公有子网无公网 IP 时 dnf 拉不到 repo -> mdadm 缺失
#  -> RAID 未建 -> bind 把根盘目录挂上, 服务仍 active, 数据静默落根盘)
if ! command -v mdadm >/dev/null; then
  log "FATAL: mdadm 不存在, 拒绝继续 (否则 docker/containerd 会静默落根盘)"
  log "修复: dnf install -y mdadm && systemctl restart setup-local-disks.service"
  exit 1
fi

# 2. 组建/挂载 RAID0 到 $MNT (幂等)
# reboot 与 stop/start 是两条路径:
#   - stop/start 换宿主机 = 全新空盘 -> assemble 失败 -> create $MD
#   - reboot 不换宿主机, 盘上超级块还在, 内核在本服务运行前就已把阵列
#     自动组装成 /dev/md127 (2026-07-07 真机实锤: 硬编码 md0 时 create 报
#     Device or resource busy -> 服务 fail -> 本地盘整轮未被使用)
#   -> 所以先探测这些盘是否已属于某个 md 设备, 有就直接用现成的
EXISTING_MD=""
for d in "${DISKS[@]}"; do
  # -r 必须: 默认输出带树形字符(└─md127), 拼不出合法设备路径
  EXISTING_MD=$(lsblk -rno NAME,TYPE "$d" 2>/dev/null | awk '$2 ~ /^raid/ {print $1; exit}')
  [ -n "$EXISTING_MD" ] && { MD="/dev/$EXISTING_MD"; log "检测到已组装阵列 $MD (reboot 场景), 直接复用"; break; }
done
if mountpoint -q "$MNT"; then
  log "$MNT 已挂载, 跳过 RAID 组建"
else
  # 组 bind 前需要 docker/containerd 未占用其数据目录; 开机时服务在它们之前跑,
  # 这里 stop 主要是为了兜住"首启 userdata 里 docker 已自启"的情形。
  systemctl stop docker containerd 2>/dev/null || true
  if ! mdadm --detail "$MD" >/dev/null 2>&1; then
    # 全新空盘: assemble 会失败, 直接 create
    mdadm --assemble "$MD" "${DISKS[@]}" 2>/dev/null \
      || mdadm --create --force --run --verbose "$MD" --level=0 --raid-devices="${#DISKS[@]}" "${DISKS[@]}"
    sleep 2
  fi
  if [ -z "$(lsblk -no FSTYPE "$MD" 2>/dev/null)" ]; then
    mkfs.ext4 -F -E nodiscard,lazy_itable_init=1,lazy_journal_init=1 -m 0 "$MD"
  fi
  mkdir -p "$MNT"
  mount -o noatime,nodiratime "$MD" "$MNT"
  chmod 1777 "$MNT"
  log "已挂载 $MD -> $MNT"
fi

# fail-fast: bind 前必须确认 $MNT 真的是 RAID 设备, 防止任何失败路径把根盘目录 bind 上去
SRC_DEV=$(findmnt -no SOURCE "$MNT" 2>/dev/null)
if [ "$SRC_DEV" != "$MD" ]; then
  log "FATAL: $MNT 挂载源是 '$SRC_DEV' 而非 $MD, 拒绝 bind (数据会静默落根盘)"
  exit 1
fi

# 3. bind docker + containerd 数据目录到 RAID (幂等; 覆盖客户用 docker 或 containerd 两种)
for svc in docker containerd; do
  src=/var/lib/$svc
  dst=$MNT/$svc
  mkdir -p "$dst" "$src"
  if mountpoint -q "$src"; then
    log "$src 已 bind, 跳过"
  else
    systemctl stop "$svc" 2>/dev/null || true
    mount --bind "$dst" "$src"
    log "bind $dst -> $src"
  fi
done

log "完成: docker & containerd -> $MNT/{docker,containerd}"
