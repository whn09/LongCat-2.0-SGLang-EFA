#!/bin/bash
# K8s prerequisites for EC2 GPU instances.
# Run on EVERY node (master and worker) before kubeadm init/join.
#
# What this script does:
#   1. Disables swap (required by kubelet)
#   2. Loads kernel modules for container networking
#   3. Sets sysctl params for bridge/iptables
#   4. Configures containerd with SystemdCgroup + NVIDIA runtime
#   5. Installs kubeadm, kubelet, kubectl v1.31
#   6. Installs conntrack (required by kubeadm preflight)
set -e

echo ">>> [1/6] Disabling swap"
sudo swapoff -a
sudo sed -i '/swap/d' /etc/fstab

echo ">>> [2/6] Loading kernel modules"
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF
sudo modprobe overlay
sudo modprobe br_netfilter

echo ">>> [3/6] Setting sysctl params"
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF
sudo sysctl --system > /dev/null 2>&1

echo ">>> [4/6] Configuring containerd"
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml > /dev/null
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sudo systemctl restart containerd
sudo systemctl enable containerd

# Configure NVIDIA container runtime for containerd (if nvidia-ctk available)
if command -v nvidia-ctk &>/dev/null; then
  echo ">>> Configuring NVIDIA runtime for containerd"
  sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default
  sudo systemctl restart containerd
elif [ -f /usr/bin/nvidia-smi ]; then
  echo ">>> Installing NVIDIA container toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit > /dev/null 2>&1
  sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default
  sudo systemctl restart containerd
fi

echo ">>> [5/6] Installing kubeadm, kubelet, kubectl"
sudo apt-get update -qq
sudo apt-get install -y -qq apt-transport-https ca-certificates curl gpg > /dev/null 2>&1

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | \
  sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg 2>/dev/null
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /' | \
  sudo tee /etc/apt/sources.list.d/kubernetes.list > /dev/null

sudo apt-get update -qq
sudo apt-get install -y -qq kubelet kubeadm kubectl > /dev/null 2>&1
sudo apt-mark hold kubelet kubeadm kubectl
sudo systemctl enable kubelet

echo ">>> [6/6] Installing conntrack"
sudo apt-get install -y -qq conntrack > /dev/null 2>&1

echo ">>> Done: $(kubelet --version)"
