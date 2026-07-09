# Self-Managed Kubernetes on EC2 GPU Instances with Kubeflow PyTorchJob

Deploy a self-managed Kubernetes cluster on AWS EC2 GPU instances (tested on g7e.48xlarge) with NVIDIA GPU and AWS EFA device plugin support, then run distributed PyTorch training via Kubeflow PyTorchJob using `torchrun` with c10d elastic rendezvous.

## Architecture

```
+---------------------------+     +---------------------------+
|  Master Node (g7e.48xl)   |     |  Worker Node (g7e.48xl)   |
|  - kubeadm control plane  |     |  - kubeadm worker         |
|  - 8x NVIDIA GPU          |     |  - 8x NVIDIA GPU          |
|  - 4x EFA NIC             |     |  - 4x EFA NIC             |
|                           |     |                           |
|  PyTorchJob Master Pod    |     |  PyTorchJob Worker Pod    |
|  torchrun --nnodes=2      |<--->|  torchrun --nnodes=2      |
|  --nproc-per-node=8       | EFA |  --nproc-per-node=8       |
|  --rdzv-backend=c10d      |RDMA |  --rdzv-backend=c10d      |
+---------------------------+     +---------------------------+
```

**Components:**
- Kubernetes v1.31 (kubeadm) with Flannel CNI (bootstrap) + AWS VPC CNI v1.21.1 (production)
- NVIDIA device plugin (exposes `nvidia.com/gpu`)
- AWS EFA device plugin (exposes `vpc.amazonaws.com/efa`)
- Kubeflow Training Operator v1.7.0 (PyTorchJob CRD)
- Container image: AWS DLC with PyTorch + CUDA + EFA support

## Prerequisites

1. **2x EC2 GPU instances** (e.g., g7e.48xlarge) in the same VPC/subnet
   - Ubuntu 24.04
   - Docker + containerd installed
   - NVIDIA drivers installed
   - EFA enabled (security group allows all traffic within the group)
2. **SSH access** from a bastion host to both instances via public IP + PEM key
3. **AWS CLI** configured on both instances (for ECR image pull)
4. **IAM instance profile** with `AmazonEKS_CNI_Policy` attached (required for VPC CNI to manage ENIs and assign secondary IPs)

## Instance Layout

| Role       | Node Name | Private IP       | Public IP       |
|------------|-----------|------------------|-----------------|
| Master     | master    | `<MASTER_PRIV>`  | `<MASTER_PUB>`  |
| Worker     | worker    | `<WORKER_PRIV>`  | `<WORKER_PUB>`  |

## Deployment Steps

### Step 1: Install K8s Prerequisites (both nodes)

Run `scripts/k8s_prereq.sh` on **both** nodes:

```bash
SSH="ssh -i /path/to/key.pem -o StrictHostKeyChecking=no"
SCP="scp -i /path/to/key.pem -o StrictHostKeyChecking=no"

for IP in <MASTER_PUB> <WORKER_PUB>; do
  $SCP scripts/k8s_prereq.sh ubuntu@$IP:/tmp/k8s_prereq.sh
  $SSH ubuntu@$IP 'bash /tmp/k8s_prereq.sh'
done
```

This script:
- Disables swap
- Loads kernel modules (overlay, br_netfilter)
- Configures sysctl for K8s networking
- Configures containerd with SystemdCgroup and NVIDIA runtime
- Installs kubeadm, kubelet, kubectl v1.31

### Step 2: Initialize K8s Master

```bash
$SSH ubuntu@<MASTER_PUB> 'sudo kubeadm init \
  --apiserver-advertise-address=<MASTER_PRIV> \
  --pod-network-cidr=10.244.0.0/16 \
  --node-name=master'
```

Then configure kubectl and install Flannel CNI:

```bash
$SSH ubuntu@<MASTER_PUB> '
  mkdir -p $HOME/.kube
  sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
  sudo chown $(id -u):$(id -g) $HOME/.kube/config
  kubectl taint nodes master node-role.kubernetes.io/control-plane:NoSchedule-
  kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
'
```

### Step 3: Join Worker Node

Use the `kubeadm join` command output from Step 2:

```bash
$SSH ubuntu@<WORKER_PUB> 'sudo kubeadm join <MASTER_PRIV>:6443 \
  --token <TOKEN> \
  --discovery-token-ca-cert-hash sha256:<HASH> \
  --node-name=worker'
```

### Step 4: Install NVIDIA GPU Device Plugin

```bash
$SSH ubuntu@<MASTER_PUB> 'kubectl apply -f \
  https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml'
```

### Step 5: Install AWS EFA Device Plugin

The EFA device plugin image must be pulled from authenticated ECR, then imported into containerd:

```bash
# On both nodes: pull via Docker, import to containerd
EFA_IMG="602401143452.dkr.ecr.us-west-2.amazonaws.com/eks/aws-efa-k8s-device-plugin:v0.5.7"
for IP in <MASTER_PUB> <WORKER_PUB>; do
  $SSH ubuntu@$IP "
    aws ecr get-login-password --region us-west-2 | \
      docker login --username AWS --password-stdin 602401143452.dkr.ecr.us-west-2.amazonaws.com
    docker pull $EFA_IMG
    docker save $EFA_IMG | sudo ctr -n k8s.io images import -
  "
done
```

Then deploy the DaemonSet:

```bash
$SSH ubuntu@<MASTER_PUB> 'kubectl apply -f /path/to/scripts/efa-device-plugin.yaml'
```

The EFA plugin requires privileged access and host mounts for `/dev/infiniband` and `/sys` to discover EFA devices.

### Step 6: Replace Flannel with AWS VPC CNI

Flannel is used for initial cluster bootstrap. For production, replace it with AWS VPC CNI which assigns real VPC subnet IPs to pods (no overlay, no encapsulation).

#### 6a. Add IAM permissions

The instance role needs EC2 network management permissions:

```bash
aws iam attach-role-policy --role-name <INSTANCE_ROLE_NAME> \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
```

#### 6b. Pre-pull VPC CNI images (both nodes)

VPC CNI images are in authenticated ECR. Pull via Docker and import to containerd:

```bash
ECR_REGISTRY="602401143452.dkr.ecr.<REGION>.amazonaws.com"
for IP in <MASTER_PUB> <WORKER_PUB>; do
  $SSH ubuntu@$IP "
    aws ecr get-login-password --region <REGION> | \
      docker login --username AWS --password-stdin $ECR_REGISTRY
    docker pull $ECR_REGISTRY/amazon-k8s-cni-init:v1.21.1
    docker pull $ECR_REGISTRY/amazon-k8s-cni:v1.21.1
    docker pull $ECR_REGISTRY/amazon/aws-network-policy-agent:v1.3.1
    docker save $ECR_REGISTRY/amazon-k8s-cni-init:v1.21.1 | sudo ctr -n k8s.io images import -
    docker save $ECR_REGISTRY/amazon-k8s-cni:v1.21.1 | sudo ctr -n k8s.io images import -
    docker save $ECR_REGISTRY/amazon/aws-network-policy-agent:v1.3.1 | sudo ctr -n k8s.io images import -
  "
done
```

#### 6c. Download and customize the VPC CNI manifest

```bash
curl -sL -o aws-k8s-cni.yaml \
  https://raw.githubusercontent.com/aws/amazon-vpc-cni-k8s/v1.21.1/config/master/aws-k8s-cni.yaml
```

**Critical modifications** for self-managed clusters with EFA:

1. Change ECR region to match your deployment region:
   ```
   sed -i 's/us-west-2/<REGION>/g' aws-k8s-cni.yaml
   ```

2. Set `imagePullPolicy: Never` (since images are pre-imported):
   ```
   sed -i 's/imagePullPolicy: Always/imagePullPolicy: Never/g' aws-k8s-cni.yaml
   ```

3. **Add `MAX_ENI=1` and set `WARM_ENI_TARGET=0`** to prevent VPC CNI from assigning secondary IPs to EFA ENIs (see Known Issues #9):
   ```yaml
   # In the aws-node container env section, add:
   - name: MAX_ENI
     value: "1"
   # And change WARM_ENI_TARGET from "1" to "0":
   - name: WARM_ENI_TARGET
     value: "0"
   ```

#### 6d. Deploy VPC CNI

```bash
$SCP aws-k8s-cni.yaml ubuntu@<MASTER_PUB>:/home/ubuntu/aws-k8s-cni.yaml
$SSH ubuntu@<MASTER_PUB> 'kubectl apply -f /home/ubuntu/aws-k8s-cni.yaml'
```

Wait for aws-node pods to be Running:
```bash
$SSH ubuntu@<MASTER_PUB> 'kubectl get pods -n kube-system -l k8s-app=aws-node'
```

#### 6e. Delete Flannel and clean up

```bash
# Delete Flannel DaemonSet
$SSH ubuntu@<MASTER_PUB> 'kubectl delete -f \
  https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml'

# Remove Flannel CNI config from both nodes
for IP in <MASTER_PUB> <WORKER_PUB>; do
  $SSH ubuntu@$IP 'sudo rm -f /etc/cni/net.d/10-flannel.conflist'
done
```

#### 6f. Set iptables FORWARD policy to ACCEPT

Flannel sets the iptables FORWARD chain default policy to DROP. VPC CNI requires ACCEPT (matching EKS default behavior). This must be done on **both nodes** and persisted:

```bash
for IP in <MASTER_PUB> <WORKER_PUB>; do
  $SSH ubuntu@$IP '
    sudo iptables -P FORWARD ACCEPT

    # Clean up Flannel iptables chains
    sudo iptables -D FORWARD -j FLANNEL-FWD -m comment --comment "flanneld forward" 2>/dev/null || true
    sudo iptables -F FLANNEL-FWD 2>/dev/null || true
    sudo iptables -X FLANNEL-FWD 2>/dev/null || true

    # Remove Flannel interfaces and routes
    sudo ip link delete cni0 2>/dev/null || true
    sudo ip link delete flannel.1 2>/dev/null || true
    sudo ip route del 10.244.0.0/24 2>/dev/null || true
    sudo ip route del 10.244.1.0/24 2>/dev/null || true

    # Persist FORWARD ACCEPT across reboots
    sudo tee /etc/systemd/system/iptables-forward-accept.service > /dev/null <<EOF
[Unit]
Description=Set iptables FORWARD policy to ACCEPT for VPC CNI
After=network.target
Before=kubelet.service

[Service]
Type=oneshot
ExecStart=/usr/sbin/iptables -P FORWARD ACCEPT
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable iptables-forward-accept.service
  '
done
```

#### 6g. Restart pods to get VPC IPs

```bash
$SSH ubuntu@<MASTER_PUB> '
  kubectl rollout restart deployment coredns -n kube-system
  kubectl rollout restart daemonset nvidia-device-plugin-daemonset -n kube-system
'
```

#### 6h. Verify VPC CNI

```bash
$SSH ubuntu@<MASTER_PUB> '
  # Pods should have VPC subnet IPs (172.31.x.x), not Flannel IPs (10.244.x.x)
  kubectl get pods -A -o wide

  # Cross-node connectivity test
  ping -c 2 <WORKER_POD_IP>

  # DNS resolution
  kubectl run dns-test --image=busybox:1.36 --rm --restart=Never -i -- \
    nslookup kubernetes.default.svc.cluster.local
'
```

### Step 7: Install Kubeflow Training Operator

```bash
$SSH ubuntu@<MASTER_PUB> 'kubectl apply -k \
  "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"'
```

### Step 8: Pre-load Container Image (optional, for large images)

For large DLC images (~30-60GB), pre-import via Docker to avoid long pull times:

```bash
DLC_IMG="public.ecr.aws/deep-learning-containers/sglang:0.5.8-gpu-py312-cu129-ubuntu24.04-sagemaker-v1.11-soci"
for IP in <MASTER_PUB> <WORKER_PUB>; do
  $SSH ubuntu@$IP "docker pull $DLC_IMG && docker save $DLC_IMG | sudo ctr -n k8s.io images import -"
done
```

### Step 9: Submit PyTorchJob

```bash
$SCP templates/pytorchjob.yaml ubuntu@<MASTER_PUB>:/home/ubuntu/pytorchjob.yaml
$SSH ubuntu@<MASTER_PUB> 'kubectl apply -f /home/ubuntu/pytorchjob.yaml'
```

### Step 10: Verify

```bash
# Check job status
$SSH ubuntu@<MASTER_PUB> 'kubectl get pytorchjob'

# Check pod status
$SSH ubuntu@<MASTER_PUB> 'kubectl get pods -l app=torchrun-g7e -o wide'

# View training logs
$SSH ubuntu@<MASTER_PUB> 'kubectl logs torchrun-distributed-g7e-master-0'
```

Expected output:
```
Rank 0/16 (local_rank=0) initialized on torchrun-distributed-g7e-master-0
Epoch 1/5, Loss: 2.3456
Epoch 2/5, Loss: 2.1234
...
Training complete!
```

## Verification Checklist

After Step 7, verify all components are running:

```bash
kubectl get nodes                        # 2 nodes Ready
kubectl get pods -A | grep nvidia        # 2 nvidia-device-plugin pods Running
kubectl get pods -A | grep efa           # 2 efa-device-plugin pods Running
kubectl get pods -A | grep aws-node      # 2 aws-node pods Running (VPC CNI)
kubectl get pods -n kubeflow             # 1 training-operator pod Running
kubectl get crd pytorchjobs.kubeflow.org # CRD registered
```

Node resources should show:
```
nvidia.com/gpu: 8           # per node
vpc.amazonaws.com/efa: 4    # per node
```

## PyTorchJob Environment Variables (auto-injected)

The Kubeflow Training Operator automatically injects these env vars into every pod:

| Variable       | Value                                     | Source                              |
|---------------|-------------------------------------------|-------------------------------------|
| `MASTER_ADDR` | Master pod hostname (e.g., `...-master-0`)| Training Operator + headless Service|
| `MASTER_PORT` | `23456` (from containerPort)              | Master container's first port       |
| `WORLD_SIZE`  | Total replica count (Master + Workers)    | Computed from replicas              |
| `RANK`        | Global rank (0 for Master, 1+ for Workers)| Assigned per replica                |

These are used by `torchrun --rdzv-endpoint=$(MASTER_ADDR):$(MASTER_PORT)` for c10d elastic rendezvous.

## Known Issues

1. **`NCCL_SOCKET_IFNAME` must be `eth0` in pods**: K8s pods (both Flannel and VPC CNI) use `eth0` as the in-pod interface name. Using the host interface name (e.g., `enp*`) causes `NCCL WARN Bootstrap : no socket interface found`. Always set `NCCL_SOCKET_IFNAME=eth0` for pod-based workloads.

2. **CPU resource requests**: Do not request the full node CPU count (e.g., 192 for g7e.48xlarge). System pods (kubelet, kube-proxy, flannel, device plugins) consume some CPU. Use ~180 to leave headroom.

3. **EFA device plugin image pull**: The AWS EFA device plugin image at `602401143452.dkr.ecr.us-west-2.amazonaws.com/eks/aws-efa-k8s-device-plugin:v0.5.7` requires ECR authentication. Pull via Docker and import to containerd with `imagePullPolicy: Never`.

4. **EFA device plugin requires host mounts**: The plugin needs `/dev/infiniband` and `/sys` mounted to discover EFA RDMA devices. Without these, it reports `No valid EFA devices found` and enters CrashLoopBackOff.

5. **`conntrack` missing on Ubuntu 24.04**: kubeadm preflight check fails with `conntrack not found`. Install with `sudo apt-get install -y conntrack` before `kubeadm init`.

6. **Large container images**: DLC images can be 30-60GB. Pre-import from Docker to containerd on all nodes (`docker save | ctr -n k8s.io images import`) to avoid long `ContainerCreating` wait times.

7. **Control-plane taint**: By default, kubeadm taints the master node with `NoSchedule`. For a 2-node cluster where both nodes need to run workload pods, remove it with:
   ```
   kubectl taint nodes master node-role.kubernetes.io/control-plane:NoSchedule-
   ```

8. **`node.kubernetes.io/instance-type` label not set**: Unlike EKS, self-managed kubeadm clusters do not set instance-type labels automatically. Do not use `nodeSelector` with this label. Add it manually if needed:
   ```
   kubectl label nodes master node.kubernetes.io/instance-type=g7e.48xlarge
   kubectl label nodes worker node.kubernetes.io/instance-type=g7e.48xlarge
   ```

9. **VPC CNI assigns secondary IPs to EFA ENIs**: On instances with EFA ENIs (e.g., g7e.48xlarge with 4 EFA NICs), VPC CNI discovers all ENIs and assigns secondary IPs to them — including EFA ENIs. Pod IPs on EFA ENIs are unreachable cross-node because EFA ENIs don't handle regular IP forwarding. **Fix**: Set `MAX_ENI=1` and `WARM_ENI_TARGET=0` in the aws-node DaemonSet to restrict VPC CNI to only the primary ENI. If EFA ENIs already have secondary IPs, clean them up:
   ```bash
   # Find EFA ENIs with secondary IPs
   aws ec2 describe-network-interfaces \
     --filters "Name=attachment.instance-id,Values=<INSTANCE_ID>" \
     --query "NetworkInterfaces[?Description!=\`\`].{Id:NetworkInterfaceId,Desc:Description}"

   # Unassign secondary IPs from EFA ENI
   aws ec2 unassign-private-ip-addresses \
     --network-interface-id <EFA_ENI_ID> \
     --private-ip-addresses <IP_LIST>
   ```

10. **iptables FORWARD DROP after Flannel removal**: Flannel sets the iptables FORWARD chain default policy to DROP and adds `FLANNEL-FWD` rules for its `10.244.0.0/16` CIDR. When Flannel is removed and replaced with VPC CNI, the DROP policy remains but the FLANNEL-FWD rules no longer match VPC subnet traffic, causing all cross-node pod communication to fail. **Fix**: Set `iptables -P FORWARD ACCEPT` on all nodes and persist via systemd service (see Step 6f).

11. **VPC CNI images require ECR authentication**: The VPC CNI images (`amazon-k8s-cni`, `amazon-k8s-cni-init`, `aws-network-policy-agent`) are hosted in authenticated ECR at `602401143452.dkr.ecr.<REGION>.amazonaws.com`. Pre-pull via Docker and import to containerd with `imagePullPolicy: Never`, same approach as the EFA device plugin.

12. **EFA RDMA works with both Flannel and VPC CNI**: NCCL uses the `aws-ofi-nccl` plugin to communicate over EFA RDMA, completely bypassing the pod CNI network. NCCL logs confirm `NET/OFI Selected provider is efa, fabric is efa-direct` and `Using transport protocol RDMA` regardless of which CNI is used. The CNI choice (Flannel vs VPC CNI) only affects TCP/IP control-plane traffic (torchrun rendezvous, K8s service networking).

## Teardown

```bash
# Delete PyTorchJob
kubectl delete pytorchjob torchrun-distributed-g7e

# Delete Kubeflow
kubectl delete -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"

# Delete VPC CNI (cleans up secondary IPs from ENIs)
kubectl delete -f /home/ubuntu/aws-k8s-cni.yaml

# Reset worker (on worker node)
sudo kubeadm reset -f

# Reset master (on master node)
sudo kubeadm reset -f
sudo rm -rf /etc/kubernetes /var/lib/kubelet /var/lib/etcd $HOME/.kube
```
