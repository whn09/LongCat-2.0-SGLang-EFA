#!/bin/bash
# K8s 前置配置 — AL2023 DLAMI 版（所有节点执行, master + worker）
# 参考 ec2-self-k8s-GPU-kubeflow.skill 的 k8s_prereq.sh (Ubuntu 版), 包管理改为 dnf/rpm
set -e

echo ">>> [1/5] 关闭 swap"
sudo swapoff -a
sudo sed -i '/swap/d' /etc/fstab

echo ">>> [2/5] 内核模块 + sysctl"
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF
sudo modprobe overlay
sudo modprobe br_netfilter
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF
sudo sysctl --system > /dev/null

echo ">>> [3/5] containerd: SystemdCgroup + NVIDIA runtime"
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml > /dev/null
if command -v nvidia-ctk &>/dev/null; then
  sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default
fi
# 坑: nvidia-container-toolkit 1.19 的 nvidia-ctk 会把 nvidia runtime 段的
# SystemdCgroup 重置回 false, 必须在 nvidia-ctk 之后再改并校验
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
if grep -q 'SystemdCgroup = false' /etc/containerd/config.toml; then
  echo "ERROR: SystemdCgroup=false 残留 — kubelet(systemd) 与 runc(cgroupfs) 将不一致" >&2
  echo "症状: runc create failed: expected cgroupsPath to be of format 'slice:prefix:name'" >&2
  exit 1
fi
sudo systemctl enable containerd
sudo systemctl restart containerd

echo ">>> [4/5] kubeadm / kubelet / kubectl v1.31"
cat <<EOF | sudo tee /etc/yum.repos.d/kubernetes.repo
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.31/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v1.31/rpm/repodata/repomd.xml.key
EOF
sudo dnf install -y kubelet kubeadm kubectl conntrack-tools --disableexcludes=kubernetes
sudo systemctl enable kubelet

echo ">>> [5/5] 完成: $(kubelet --version)"
echo ""
echo "下一步:"
echo "  Master: sudo kubeadm init --apiserver-advertise-address=<MASTER_IP> \\"
echo "            --pod-network-cidr=10.244.0.0/16 --node-name=node0"
echo "  Worker: 用 init 输出的 kubeadm join 命令"
