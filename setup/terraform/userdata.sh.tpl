#!/bin/bash
# p5en node bootstrap (container-only, no K8s):
#   [1] /data = EBS xfs (persistent)
#   [2] install self-heal systemd service: EVERY boot rebuilds local NVMe RAID0
#       and bind-mounts /var/lib/docker + /var/lib/containerd onto it
#       (local instance-store is wiped on stop/start, so this restores the fast
#        disk layout automatically; the data itself is ephemeral)
#   [3] EFA userspace (pinned)
#   [4] nvidia runtime for docker+containerd
set -ex
exec > >(tee /var/log/p5en-bootstrap.log) 2>&1

echo "=== [1/4] EBS data disk -> /data ==="
DATA_DISK=""
for i in $(seq 1 60); do
  for sys_path in /sys/block/nvme*n1; do
    [ -e "$sys_path" ] || continue
    model=$(cat "$sys_path/device/model" 2>/dev/null | xargs)
    case "$model" in *"Elastic Block Store"*) ;; *) continue ;; esac
    dev="/dev/$(basename "$sys_path")"
    if [ "$(lsblk -no NAME "$dev" | wc -l)" -eq 1 ] && [ -z "$(lsblk -no FSTYPE "$dev" | xargs)" ]; then
      DATA_DISK="$dev"; break 2
    fi
  done
  echo "waiting for EBS data disk... ($i/60)"; sleep 1
done
if [ -n "$DATA_DISK" ]; then
  mkfs.xfs -f "$DATA_DISK"
  mkdir -p /data
  mount -o noatime "$DATA_DISK" /data
  UUID=$(blkid -s UUID -o value "$DATA_DISK")
  grep -q "$UUID" /etc/fstab || echo "UUID=$UUID /data xfs defaults,noatime,nofail 0 2" >> /etc/fstab
  mkdir -p /data/models /data/logs /data/cache
  chmod 1777 /data
else
  echo "WARN: EBS data disk not found; /data unavailable"
fi

echo "=== [2/4] install local-disk self-heal service (rebuild RAID0 + bind docker/containerd every boot) ==="
# mdadm 是 RAID 硬前置; 开机初期网络可能未就绪(多网卡下无公网 IP 时更甚), 带重试
for i in $$(seq 1 30); do
  command -v mdadm >/dev/null && break
  dnf install -y mdadm && break
  echo "waiting for network to install mdadm... ($$i/30)"; sleep 10
done
command -v mdadm >/dev/null || echo "FATAL: mdadm unavailable after retries; setup-local-disks.service will fail closed"
echo 'IyEvYmluL2Jhc2gKIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0KIyBzZXR1cC1sb2NhbC1kaXNrcy5zaCAg4oCUICDmr4/mrKHlvIDmnLroh6rmhIjmnKzlnLAgTlZNZSDnm5gKIyDnlLEgc3lzdGVtZCDmnI3liqEgc2V0dXAtbG9jYWwtZGlza3Muc2VydmljZSDlnKggZG9ja2VyL2NvbnRhaW5lcmQg5LmL5YmN6L+Q6KGM44CCCiMKIyDog4zmma86IOacrOWcsCBOVk1lIChJbnN0YW5jZSBTdG9yZSkg54mp55CG57uR5a6a5a6/5Li75py6LCBzdG9wL3N0YXJ0IOWQjuaNouaWsOWuv+S4u+acugojID0g5LiA57uE5YWo5paw56m655uYIChSQUlEIOWFg+aVsOaNriArIOaWh+S7tuezu+e7n+WFqOayoeS6hinjgILmiYDku6Xlv4Xpobvmr4/mrKHlvIDmnLo6CiMgICAxLiDph43mlrDlj5HnjrDmnKzlnLDnm5ggLT4g57uEIFJBSUQwIC0+IG1rZnMgLT4g5oyCIC9tbnQvbnZtZQojICAgMi4gYmluZCAvdmFyL2xpYi9kb2NrZXIg5ZKMIC92YXIvbGliL2NvbnRhaW5lcmQg5YiwIC9tbnQvbnZtZSDkuIoKIyDorqkgZG9ja2VyL2NvbnRhaW5lcmQg5q+P5qyh6YO96JC95Zue6auY6YCf5pys5ZywIFJBSUQgKOebmOS4iuaVsOaNruacrOi6q+aYr+S4tOaXtueahCzkuKLlpLHlj6/mjqXlj5cp44CCCiMg5bmC562JOiDlt7LmjILovb0v5beyIGJpbmQg5YiZ6Lez6L+HLCDlj6/lronlhajph43lpI3ov5DooYzjgIIKIyA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0Kc2V0IC11byBwaXBlZmFpbApsb2coKSB7IGVjaG8gIltzZXR1cC1sb2NhbC1kaXNrc10gJCoiOyB9CgpNTlQ9L21udC9udm1lCk1EPS9kZXYvbWQwCgojIDEuIOWPkeeOsOacrOWcsCBpbnN0YW5jZS1zdG9yZSBOVk1lIOebmCAo5LyY5YWI56iz5a6a55qEIGJ5LWlkLCDlm57pgIAgbW9kZWwg5a2X56ym5LiyKQpESVNLUz0oKQppZiBbIC1kIC9kZXYvZGlzay9ieS1pZCBdOyB0aGVuCiAgd2hpbGUgSUZTPSByZWFkIC1yIGQ7IGRvIFsgLW4gIiRkIiBdICYmIERJU0tTKz0oIiRkIik7IGRvbmUgXAogICAgPCA8KGZpbmQgLUwgL2Rldi9kaXNrL2J5LWlkLyAteHR5cGUgbCAtbmFtZSAnKk5WTWVfSW5zdGFuY2VfU3RvcmFnZV8qJyAyPi9kZXYvbnVsbCB8IHhhcmdzIC1yIC1uMSByZWFscGF0aCB8IHNvcnQgLXUpCmZpCmlmIFsgIiR7I0RJU0tTW0BdfSIgLWVxIDAgXTsgdGhlbgogIGZvciBzIGluIC9zeXMvYmxvY2svbnZtZSpuMTsgZG8KICAgIFsgLWUgIiRzIiBdIHx8IGNvbnRpbnVlCiAgICBtPSQoY2F0ICIkcy9kZXZpY2UvbW9kZWwiIDI+L2Rldi9udWxsIHwgdHIgLWQgJyAnKQogICAgY2FzZSAiJG0iIGluICpJbnN0YW5jZVN0b3JhZ2UqKSBESVNLUys9KCIvZGV2LyQoYmFzZW5hbWUgIiRzIikiKSA7OyBlc2FjCiAgZG9uZQpmaQppZiBbICIkeyNESVNLU1tAXX0iIC1lcSAwIF07IHRoZW4KICBsb2cgIuacquWPkeeOsOacrOWcsCBpbnN0YW5jZS1zdG9yZSBOVk1lIOebmCwg6Lez6L+HIChkb2NrZXIvY29udGFpbmVyZCDlsIbnlKjmoLnnm5gpIgogIGV4aXQgMApmaQpsb2cgIuacrOWcsOebmDogJHtESVNLU1sqXX0iCgojIG1kYWRtIOaYr+ehrOWJjee9rjog57y65aSx5pe257ud5LiN6IO957un57ut6LWwIGJpbmQg5YiG5pSvCiMgKDIwMjYtMDctMDUg55yf5py65LqL5pWFOiDlhazmnInlrZDnvZHml6DlhaznvZEgSVAg5pe2IGRuZiDmi4nkuI3liLAgcmVwbyAtPiBtZGFkbSDnvLrlpLEKIyAgLT4gUkFJRCDmnKrlu7ogLT4gYmluZCDmiormoLnnm5jnm67lvZXmjILkuIosIOacjeWKoeS7jSBhY3RpdmUsIOaVsOaNrumdmem7mOiQveagueebmCkKaWYgISBjb21tYW5kIC12IG1kYWRtID4vZGV2L251bGw7IHRoZW4KICBsb2cgIkZBVEFMOiBtZGFkbSDkuI3lrZjlnKgsIOaLkue7nee7p+e7rSAo5ZCm5YiZIGRvY2tlci9jb250YWluZXJkIOS8mumdmem7mOiQveagueebmCkiCiAgbG9nICLkv67lpI06IGRuZiBpbnN0YWxsIC15IG1kYWRtICYmIHN5c3RlbWN0bCByZXN0YXJ0IHNldHVwLWxvY2FsLWRpc2tzLnNlcnZpY2UiCiAgZXhpdCAxCmZpCgojIDIuIOe7hOW7ui/mjILovb0gUkFJRDAg5YiwICRNTlQgKOW5guetiSkKaWYgbW91bnRwb2ludCAtcSAiJE1OVCI7IHRoZW4KICBsb2cgIiRNTlQg5bey5oyC6L29LCDot7Pov4cgUkFJRCDnu4Tlu7oiCmVsc2UKICAjIOe7hCBiaW5kIOWJjemcgOimgSBkb2NrZXIvY29udGFpbmVyZCDmnKrljaDnlKjlhbbmlbDmja7nm67lvZU7IOW8gOacuuaXtuacjeWKoeWcqOWug+S7rOS5i+WJjei3kSwKICAjIOi/memHjCBzdG9wIOS4u+imgeaYr+S4uuS6huWFnOS9jyLpppblkK8gdXNlcmRhdGEg6YeMIGRvY2tlciDlt7Loh6rlkK8i55qE5oOF5b2i44CCCiAgc3lzdGVtY3RsIHN0b3AgZG9ja2VyIGNvbnRhaW5lcmQgMj4vZGV2L251bGwgfHwgdHJ1ZQogIGlmICEgbWRhZG0gLS1kZXRhaWwgIiRNRCIgPi9kZXYvbnVsbCAyPiYxOyB0aGVuCiAgICAjIOWFqOaWsOepuuebmDogYXNzZW1ibGUg5Lya5aSx6LSlLCDnm7TmjqUgY3JlYXRlCiAgICBtZGFkbSAtLWFzc2VtYmxlICIkTUQiICIke0RJU0tTW0BdfSIgMj4vZGV2L251bGwgXAogICAgICB8fCBtZGFkbSAtLWNyZWF0ZSAtLWZvcmNlIC0tcnVuIC0tdmVyYm9zZSAiJE1EIiAtLWxldmVsPTAgLS1yYWlkLWRldmljZXM9IiR7I0RJU0tTW0BdfSIgIiR7RElTS1NbQF19IgogICAgc2xlZXAgMgogIGZpCiAgaWYgWyAteiAiJChsc2JsayAtbm8gRlNUWVBFICIkTUQiIDI+L2Rldi9udWxsKSIgXTsgdGhlbgogICAgbWtmcy5leHQ0IC1GIC1FIG5vZGlzY2FyZCxsYXp5X2l0YWJsZV9pbml0PTEsbGF6eV9qb3VybmFsX2luaXQ9MSAtbSAwICIkTUQiCiAgZmkKICBta2RpciAtcCAiJE1OVCIKICBtb3VudCAtbyBub2F0aW1lLG5vZGlyYXRpbWUgIiRNRCIgIiRNTlQiCiAgY2htb2QgMTc3NyAiJE1OVCIKICBsb2cgIuW3suaMgui9vSAkTUQgLT4gJE1OVCIKZmkKCiMgZmFpbC1mYXN0OiBiaW5kIOWJjeW/hemhu+ehruiupCAkTU5UIOecn+eahOaYryBSQUlEIOiuvuWkhywg6Ziy5q2i5Lu75L2V5aSx6LSl6Lev5b6E5oqK5qC555uY55uu5b2VIGJpbmQg5LiK5Y67ClNSQ19ERVY9JChmaW5kbW50IC1ubyBTT1VSQ0UgIiRNTlQiIDI+L2Rldi9udWxsKQppZiBbICIkU1JDX0RFViIgIT0gIiRNRCIgXTsgdGhlbgogIGxvZyAiRkFUQUw6ICRNTlQg5oyC6L295rqQ5pivICckU1JDX0RFVicg6ICM6Z2eICRNRCwg5ouS57udIGJpbmQgKOaVsOaNruS8mumdmem7mOiQveagueebmCkiCiAgZXhpdCAxCmZpCgojIDMuIGJpbmQgZG9ja2VyICsgY29udGFpbmVyZCDmlbDmja7nm67lvZXliLAgUkFJRCAo5bmC562JOyDopobnm5blrqLmiLfnlKggZG9ja2VyIOaIliBjb250YWluZXJkIOS4pOenjSkKZm9yIHN2YyBpbiBkb2NrZXIgY29udGFpbmVyZDsgZG8KICBzcmM9L3Zhci9saWIvJHN2YwogIGRzdD0kTU5ULyRzdmMKICBta2RpciAtcCAiJGRzdCIgIiRzcmMiCiAgaWYgbW91bnRwb2ludCAtcSAiJHNyYyI7IHRoZW4KICAgIGxvZyAiJHNyYyDlt7IgYmluZCwg6Lez6L+HIgogIGVsc2UKICAgIHN5c3RlbWN0bCBzdG9wICIkc3ZjIiAyPi9kZXYvbnVsbCB8fCB0cnVlCiAgICBtb3VudCAtLWJpbmQgIiRkc3QiICIkc3JjIgogICAgbG9nICJiaW5kICRkc3QgLT4gJHNyYyIKICBmaQpkb25lCgpsb2cgIuWujOaIkDogZG9ja2VyICYgY29udGFpbmVyZCAtPiAkTU5UL3tkb2NrZXIsY29udGFpbmVyZH0iCg==' | base64 -d > /usr/local/sbin/setup-local-disks.sh
chmod +x /usr/local/sbin/setup-local-disks.sh
cat > /etc/systemd/system/setup-local-disks.service <<'UNIT'
[Unit]
Description=Rebuild local NVMe RAID0 and bind docker/containerd data dirs (self-heal on every boot)
DefaultDependencies=no
Wants=systemd-udev-settle.service
After=systemd-udev-settle.service local-fs.target
Before=docker.service containerd.service
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/setup-local-disks.sh
StandardOutput=journal+console
StandardError=journal+console
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable setup-local-disks.service
systemctl start setup-local-disks.service

echo "=== [3/4] EFA userspace ${efa_installer_version} ==="
if [ ! -x /opt/amazon/efa/bin/fi_info ] || ! /opt/amazon/efa/bin/fi_info --version | grep -q "${efa_installer_version}"; then
  cd /tmp
  curl -fsSLO "https://efa-installer.amazonaws.com/aws-efa-installer-${efa_installer_version}.tar.gz"
  tar -xf "aws-efa-installer-${efa_installer_version}.tar.gz"
  cd aws-efa-installer
  ./efa_installer.sh -y --skip-kmod 2>&1 | tail -20 || echo "WARN: efa_installer failed"
fi
/opt/amazon/efa/bin/fi_info -p efa -l || echo "WARN: no EFA devices visible yet"

echo "=== [4/4] container runtime: nvidia runtime for docker+containerd (data dirs already bound to RAID) ==="
command -v nvidia-ctk >/dev/null || dnf install -y nvidia-container-toolkit || true
nvidia-ctk runtime configure --runtime=docker --set-as-default 2>/dev/null || true
nvidia-ctk runtime configure --runtime=containerd --set-as-default 2>/dev/null || true
systemctl enable docker containerd 2>/dev/null || true
systemctl restart containerd 2>/dev/null || true
systemctl restart docker 2>/dev/null || true
nvidia-smi -pm 1 || true

echo "=== p5en bootstrap complete ==="
