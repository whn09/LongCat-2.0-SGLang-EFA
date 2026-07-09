---
name: p5en-self-k8s-sglang-pd-efa
description: >
  Deploy a SGLang Prefill-Decode disaggregated inference system on AWS p5en.48xlarge
  (8x H200, 16x EFA) purchased via Capacity Block, on a SELF-MANAGED Kubernetes cluster
  (kubeadm, NOT EKS). KV cache transfer uses Mooncake Transfer Engine (USE_EFA=ON);
  MoE all-to-all uses UCCL-EP (deep_ep drop-in, EFA-compatible DeepEP replacement).
  Target topology: Prefill 32 GPUs (pp4/tp8/ep32, 4 nodes) + Decode 64 GPUs (dp64/ep64,
  8 nodes). Validation path: 2P2D on 4 nodes first, then scale out. Includes Terraform
  AND CloudFormation launch automation (16x EFA NICs, 1TB root + 2TB gp3 data volume,
  NVMe RAID0 via userdata).
---

# p5en 自建 K8s + SGLang PD 分离 + EFA 适配部署

## 适用场景

- 客户在 AWS 上用 **Capacity Block 一次开 8 台 p5en.48xlarge**（8x H200 141GB / 16x EFA / 3200Gbps / 8x 3.5TB NVMe）
- 客户**自建 K8s（kubeadm），非 EKS**
- SGLang PD 分离推理：Mooncake KV transfer + UCCL-EP MoE all-to-all，全部跑在 EFA 上
- 目标：Prefill 32 卡（pp4 tp8 ep32）+ Decode 64 卡（dp64 ep64）；先 2P2D 验证部署，再**切换**到 4P8D（并行策略完全不同，属方案切换非扩容）

## 架构与三条通信链路

```
                      客户端
                        |
              +---------v---------+
              |   sglang_router   |  --pd-disaggregation (port 8000)
              +----+---------+----+
                   |         |
        +----------v--+   +--v-----------+
        | Prefill 集群 |   | Decode 集群  |
        | 4 节点 32 卡 |   | 8 节点 64 卡 |
        | pp4 tp8 ep32|   | dp64 ep64    |
        +------+------+   +------+-------+
               |   KV cache      |
               +-- Mooncake TE --+        <-- EFA RDMA (SRD)
                  (protocol=efa)

① MoE all-to-all: sglang --moe-a2a-backend deepep -> deep_ep(UCCL wrapper) -> uccl.ep -> EFA
② KV 传输:        sglang --disaggregation-transfer-backend mooncake -> Mooncake TE(USE_EFA=ON) -> EFA
③ TP/DP 集合通信:  torch.distributed(NCCL) -> NVLink(机内) + aws-ofi-nccl(跨机 EFA)
```

**为什么是 UCCL-EP 而不是原生 DeepEP**：EP all-to-all 依赖 IBGDA（InfiniBand GPUDirect
Async），EFA 是 SRD + libfabric，没有 ibverbs GDA 路径，原生 DeepEP / Mooncake-EP
（mlx5 IBGDA-only）在 EFA 上跑不了。UCCL-EP 用 CPU proxy + libfabric 实现同等语义，
并附带 deep_ep drop-in wrapper（`import deep_ep` 重定向到 `uccl.ep`），SGLang 无感使用。

## 版本 Pin（来源: KevinZhao/efa-validation BUILD_MATRIX.md）

| 组件 | 版本 | 关键点 |
|---|---|---|
| 基础 AMI | AL2023 Base Deep Learning AMI | 含 NVIDIA driver + EFA 内核模块 + Docker |
| 推理镜像 | `public.ecr.aws/n3l4x8f3/sglang-mooncake-uccl:2026.05.02-h200.dp16` | **大 DP 必须此标签及以后**（含 Mooncake PR #2023） |
| SGLang | 0.5.10 | 镜像内置 rdma→efa 协议 sed 补丁 |
| Mooncake TE | v0.3.11.post1（USE_EFA=ON, WITH_EP=OFF） | PR #2023 修 DP>1 endpoint 键冲突 |
| UCCL-EP | @8ac850bd + deep_ep wrapper, PER_EXPERT_BATCHING=1 | |
| EFA installer | 1.48.0 | 宿主机与容器内版本对齐 |
| aws-ofi-nccl | v1.19.0 | NCCL 跨机走 EFA |
| K8s | v1.31 kubeadm | NVIDIA plugin v0.17.1 + EFA plugin v0.5.7 |

## 机型与扩展建议

**超过 750B 的模型建议用 p6-b300（B300）**：p5en 单节点 1.128TB HBM，750B 级 FP8
权重加载后 KV cache 余量紧张（长 context/大 batch 受限）；p6-b300 单节点 8x 288GB=2.3TB
（约 2 倍），长文本场景余量宽裕。B300 软件栈差异（BUILD_MATRIX B300 stack）：
SGLang ≥0.5.12.post1 + Dockerfile.customer-b300；`--attention-backend flashinfer`（禁 fa3）；
UCCL 需含 PR #950（16 EFA 双 NIC/GPU 选卡，缺则 proxy SIGABRT）；网卡布局差异：
B300 的 card0 纯 ENA 不承载 EFA（card1-16 各一个 EFA），p5en 的 card0 是
ENA(dev0)+EFA(dev1) 双接口；Mooncake 版本不变，构建加 `CUDA_ARCHITECTURES="100;103"`。

**负载均衡/数据并行扩容 = 整组复制，机器 8→16，其他全不变**：单组（4P+8D）吞吐到顶后
再开一组完全相同的集群。启动模板/镜像/SGLang 参数/并行策略零改动，只需：CB 与
instance_count 翻倍；第二组独立 dist-init-addr（组间无通信）；router 层分发
（sglang_router 多组 --prefill/--decode，或 NLB/ALB 挂两个 router 前面）。
整组复制优于加大单组并行（dp64→dp128）：无新通信半径、故障域隔离、容量线性。

## 三个全局硬性约束

1. **单 AZ**：CB 预留绑定 AZ；PD 分离跨 AZ 会挂。所有节点、子网同一 AZ。
2. **安全组自引用全通**：EFA 要求 SG 对自身开放全部 ingress **和 egress**，缺 egress 自引用 EFA 无法通信。
3. **模型权重放本地 NVMe**：同 Region S3 → `/mnt/nvme`（RAID0），`--model-path` 只指本地。
   禁 FSx（跨 AZ + 并发 mmap 慢 8x）、禁 HF 直拉（限流）、禁跨 AZ 共享。

---

# 第一阶段：EC2 开机（Terraform 或 CloudFormation）

## p5en.48xlarge 启动模板要点

| 配置 | 值 | 原因 |
|---|---|---|
| CB 双要素 | LT 内嵌 `market_type=capacity-block` + `capacity_reservation_id` | 缺一开机报错 |
| 网卡 | **17 个接口** = 1x `interface`(纯 ENA 主网卡) + **16x** `efa-only` | AWS 官方推荐；主网卡不能是 efa-only；efa-only 不占 IP |
| NetworkCardIndex/DeviceIndex | 主网卡 (card0, dev0)；EFA 为 **(card0, dev1)** + (card1..15, dev0) | **card0 有两个接口**——dev1 的 EFA 容易漏，漏了只有 15 个 EFA 设备 |
| 根盘 | 1024 GiB gp3 加密 | 镜像层 + 系统 |
| 数据盘 | 2048 GiB gp3 **16000 IOPS / 1000 MBps** 加密 | 默认 3000/125 对权重分发太慢 |
| Placement Group | cluster 策略 | EFA 低延迟 |
| IMDS | `http_tokens=required`, `hop_limit=2` | hop=2 容器内可访问 IMDS |
| 开机方式 | 固定数量 `aws_instance`（**不用 ASG**） | CB 到期实例被回收，ASG 补机只会报错 |
| IAM | SSM + ECR ReadOnly + EKS_CNI_Policy | CNI policy 备 VPC CNI 切换用 |

## 操作步骤

### 1. 购买 CB 并记录 cr-id / AZ

```bash
aws ec2 describe-capacity-block-offerings --region <R> \
  --instance-type p5en.48xlarge --instance-count 8 --capacity-duration-hours 168 --output table
aws ec2 purchase-capacity-block --region <R> \
  --capacity-block-offering-id cbo-xxxx --instance-platform Linux/UNIX
aws ec2 describe-capacity-reservations --region <R> \
  --filters Name=instance-type,Values=p5en.48xlarge \
  --query 'CapacityReservations[].{id:CapacityReservationId,az:AvailabilityZone,state:State}'
```

### 2. 查 AL2023 DLAMI

```bash
aws ssm get-parameter --region <R> \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id \
  --query Parameter.Value --output text
```

### 3. 部署前预检（强烈建议）

开机前先跑预检,提前拦截最常见的 `InsufficientInstanceCapacity` / 网卡报错
(CR 未 active、AZ/机型/数量不符、子网 AZ 不匹配、IP 不足):

```bash
REGION=<R> CR_ID=cr-xxxx SUBNET_ID=subnet-xxxx COUNT=8 \
  bash scripts/preflight-cr.sh
# 全 PASS(退出码 0)再执行下面的 terraform apply / create-stack
```

### 3a. Terraform 开机（terraform/ 目录）

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # 填 region/vpc/subnet/cr-id/ami
# 2P2D 测试: instance_count=4; 全量: 8
terraform init && terraform apply
terraform output private_ips
```

### 3b. CloudFormation 开机（cloudformation/p5en-cb-cluster.yaml）

```bash
aws cloudformation create-stack --region <R> --stack-name p5en-inference \
  --template-body file://cloudformation/p5en-cb-cluster.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=VpcId,ParameterValue=vpc-xxx \
    ParameterKey=SubnetId,ParameterValue=subnet-xxx \
    ParameterKey=CapacityReservationId,ParameterValue=cr-xxx \
    ParameterKey=AmiId,ParameterValue=ami-xxx \
    ParameterKey=InstanceCount,ParameterValue=4
# InstanceCount 只允许 2/4/6/8（CFN 无 count，用 Condition 分档）
```

userdata（两套 IaC 相同）自动完成：
- EBS 2TB 数据盘（按 NVMe model `Elastic Block Store` 识别）→ xfs → `/data`（**持久**：stop/start 后靠 fstab UUID 自动重挂）
- **本地盘自愈服务** `setup-local-disks.service`（systemd oneshot，`Before=docker/containerd`，**每次开机都跑**）：
  by-id 发现 8x 3.5TB 本地 NVMe → mdadm RAID0 → ext4 → 挂 `/mnt/nvme` → **bind `/var/lib/docker` 和 `/var/lib/containerd` 到 `/mnt/nvme`**（客户用 Docker 或 containerd 都覆盖，镜像/层落 28T 高速盘、不占根盘）
  ⚠️ **为什么要 systemd 服务而非一次性 userdata**：本地 instance-store 在 stop/start 后是全新空盘（RAID+fs 全没），userdata 默认只首启跑一次；做成每次开机的服务才能**自愈**（数据本身仍是临时的，丢失可接受，但 RAID+bind 自动恢复）
- EFA userspace 固定版本 1.48.0（`--skip-kmod`，DLAMI 已带内核模块）
- nvidia-container-toolkit 对 docker+containerd 注册为默认运行时 + `nvidia-smi -pm 1`

**真机验证结论（p5en.48xlarge, us-west-2, DLAMI, CFN 部署）**：
- 首启：服务 `active(exited)`，RAID0(8盘)+`/mnt/nvme`+docker/containerd bind 全部就绪；16 EFA；`docker run hello-world` OK；`ctr` OK。
- **stop/start 自愈已验证**：重启后 userdata 不重跑,但服务本次开机自动重建 RAID0 + 重新 bind（注意设备名会变 nvme2↔nvme3,by-id 发现健壮）；`/var/lib/docker`、`/var/lib/containerd` 均落回 `/dev/md0`；容器服务正常；本地数据丢失（预期），`/data`(EBS) 持久。
- ⚠️ **reboot ≠ stop/start（2026-07-07 真机实锤 + 已修复）**：软重启不换宿主机，本地盘超级块还在，内核在服务运行前就把阵列自动组装成 `/dev/md127`；旧脚本硬编码 `md0` + 假设空盘 → `mdadm create` 报 `Device or resource busy` → 服务 fail（fail-fast 正确拦截，未落根盘，但本地盘整轮未被使用）。**修复**：脚本先用 `lsblk -rno NAME,TYPE` 探测本地盘是否已属于某个 md 设备，有则直接复用（如 md127），仅全新空盘才 create md0；数据在 reboot 后**保留**（同一组盘）。两条路径均已真机验证。

### 4. 开机验证（每节点）

```bash
systemctl is-active setup-local-disks.service  # active（每次开机自愈本地盘）
nvidia-smi -L                            # 8x H200
/opt/amazon/efa/bin/fi_info -p efa -l    # 16 个设备
ls /sys/class/infiniband/                # rdmapXXs0 x16
findmnt /var/lib/docker /var/lib/containerd  # 均 /dev/md0[/...]（bind 到本地 RAID）
df -h /data /mnt/nvme                    # ~2T / ~28T
cat /proc/mdstat                         # md0 active raid0 8 disks
sudo tail -50 /var/log/p5en-bootstrap.log
```

常见开机失败：
- `InsufficientInstanceCapacity`：CB 未到开始时间 / cr-id、AZ、数量与预留不符
- 网卡报错：子网 AZ 不对，或索引组合不合法（合法：card0/dev0=interface、card0/dev1=efa-only、card1-15/dev0=efa-only）
- EFA 设备只有 15 个：LT 漏了 card0/dev1 的 efa-only 接口
- `fi_info` 无输出：SG 缺自引用 egress，或 EFA 安装失败
- `associatePublicIPAddress cannot be specified`（API 400 → ROLLBACK）：多网卡机型禁止该参数，
  见上文网络方案（NAT）

### 4.1 私有子网访问与调试（SSM，真机验证流程）

**前提已全部内置**：节点 role 含 `AmazonSSMManagedInstanceCore`（模板自带）、AL2023 DLAMI
预装 SSM Agent、私有子网经 NAT 出网 → 开机即自动注册 SSM，零配置。
（完全封闭子网才需要 ssm/ssmmessages/ec2messages 三个 Interface Endpoint。）

**SSH key**：CFN `KeyName` 参数在 create-stack 时传入即可（EC2 启动时自动注入
`ec2-user`，无需 send-command 手工注入）；`AllowedSshCidr` **留空**——SSH 走 SSM 隧道，
不需要任何 22 端口入站规则。

```bash
# 交互 shell（最简，连 key 都不需要）
aws ssm start-session --target i-xxxx --region <R>

# 批量执行
aws ssm send-command --instance-ids i-a i-b --document-name AWS-RunShellScript \
  --parameters 'commands=["nvidia-smi -L"]'

# SSH over SSM（scp/VSCode Remote 可用；~/.ssh/config）:
Host gpu-*
  User ec2-user
  ProxyCommand aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters portNumber=22
# 使用: ssh gpu-i-xxxx（HostName 直接写 instance-id）

# 端口转发调试推理服务（本地 curl localhost:8000 直达节点上的 router）:
aws ssm start-session --target i-xxxx --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8000"],"localPortNumber":["8000"]}'
```

**外部流量接入**：内网调用直接打 router 私有 IP:8000；VPC 外/公网客户端加
internet-facing **NLB**（TCP 透传，target=router 节点私有 IP）——NLB 与 router 是两层：
router 管 PD 分发/LB（业务层），NLB 管接入/高可用（网络层），单 router 内网场景可不建 NLB。

### 5. 实例拓扑检查（可选）

```bash
aws ec2 describe-instance-topology --region <R> \
  --filters Name=instance-type,Values=p5en.48xlarge \
  --query 'Instances[].{id:InstanceId,nodes:NetworkNodes}'
# NetworkNodes Layer1/Layer2 相同的实例互访延迟最低。
# 拓扑稍远的节点安排为 Prefill(计算密集); Decode 节点组(MoE 小消息, 延迟敏感)放最近的一组。
```

## 固化自定义 AMI（CB 换块快速开机 — 必做）

**CB 到期实例被自动回收**。环境配好后（EFA + K8s 组件 + 推理镜像已导入 containerd、
kubeadm init/join 之前）打 golden AMI，下一个 CB 块用新 AMI 开机，省 ~30 分钟/节点：

```bash
# 已 join 的节点先清集群身份:
sudo kubeadm reset -f && sudo rm -rf /etc/kubernetes /var/lib/kubelet /var/lib/etcd
# 打镜像:
aws ec2 create-image --region <R> --instance-id i-xxxx \
  --name "p5en-inference-golden-$(date +%Y%m%d)"
aws ec2 wait image-available --region <R> --image-ids ami-yyyy
```

要点：
- AMI 只含 EBS 卷；**/mnt/nvme（实例存储）不进 AMI**，换块后模型权重需重新 S3 分发
- kubeadm init/join 不能固化（集群身份），换块后重新组集群
- 数据盘不需固化时 `--block-device-mappings '[{"DeviceName":"/dev/xvdb","NoDevice":""}]'` 排除，省快照时间/费用
- 推理镜像升级后重打 AMI；**CB 到期前必须打好**（回收后无机会）
- 换块流程：新 cr-id → IaC 改 `ami_id` + `capacity_reservation_id` → apply → verify-node.sh → 权重分发 → kubeadm init/join

## 磁盘方案设计边界与竞态知识（仅知识沉淀，不改脚本）

本方案 docker/containerd 数据目录**不落根盘**的实现 = `setup-local-disks.service`
（systemd `Before=docker/containerd`）把空目录 bind 到本地 NVMe RAID0。
它与参考实现 eks-enterprise-quickstart 的 **EBS+LVM 迁移方案**是两条路线，边界如下：

**为什么 v11 无 containerd 迁移竞态**（对照 eks-quickstart system-nodegroup userdata 记录的真实事故：
containerd 先在根盘启动、未等 boltdb 落盘就 rsync 活 content store、错误被 `|| true` 吞掉
→ metadata.db 引用的 blob 未落盘 → 少数节点 `blob not found` / ImagePullBackOff）。
竞态需三个条件同时成立，v11 一个都不满足：

| 竞态条件（eks-quickstart） | v11 |
|---|---|
| containerd 已在根盘启动、有未落盘数据 | 重启路径 systemd `Before=` 排序保证 daemon 不可能先启动；首启 `systemctl stop` 同步等待后才 bind |
| 对活 store 做 rsync 迁移 | **不迁移**——bind 空目录，根盘旧数据只被遮蔽，无 torn-copy |
| 空 store 致命（EKS AMI pin `sandbox_image=localhost/kubernetes/pause` 不可重拉，见 awslabs/amazon-eks-ami#2000/#2122） | DLAMI+kubeadm，pause 来自 registry.k8s.io 可重拉；业务镜像本就走 `ctr import` |

因此 eks-quickstart 的 `mask containerd + EXIT trap + rsync fail-fast` 重手段在 v11 **不需要也不要引入**。

**适用边界——方案依赖 instance-store NVMe，客户只用 p5en 故成立**：

| 机型 | 本方案覆盖情况 |
|---|---|
| p5en.48xlarge（本项目全部节点） | 覆盖：8x 3.5T 本地盘，真机已验证（含 stop/start 自愈） |
| CPU `d` 机型（m6id/c6id 等，未来若加） | 天然覆盖，脚本零修改（单盘也组 RAID0+bind） |
| CPU EBS-only 机型（m6i/c6i 等，未来若加） | **不覆盖**：脚本 `exit 0`，docker/containerd 落根盘 → 镜像层/日志写满根盘 → kubelet DiskPressure 驱逐；专职 master 上还与 etcd 抢根盘 IOPS。需换 `d` 机型（推荐）或移植 eks-quickstart 的 EBS 方案（届时 mask/trap/fail-fast 缺一不可） |

**2026-07-05 真机事故与修复（上段"静默退化"预言被踩中）**：
事故链——多网卡 LT 会**忽略子网 `MapPublicIpOnLaunch`** → 公有子网里主 ENI 无公网 IP、开机
初期无出网 → `dnf install mdadm` 拉 repo 超时（DLAMI 不预装 mdadm）→ `|| true` 吞掉 →
RAID 未建、bind 把根盘目录挂上、服务仍 active。已落地三层修复：
1. **网络（2026-07-06 修正）**：多网卡启动**不允许** `AssociatePublicIpAddress`（EC2 API 400，
   真机 ROLLBACK 实锤）且忽略子网 auto-assign → **p5en 开机不可能有公网 IP**。
   标准方案 = **私有子网 + NAT Gateway**（次选 = 开机后手工挂 EIP）；
   `preflight-cr.sh` 已加出网路由检查（默认路由 igw-* 直接 FAIL）
2. **userdata**：mdadm 安装改为 30 次×10s 重试（等网络就绪）
3. **setup-local-disks.sh fail-fast**：mdadm 缺失即 `exit 1`；bind 前校验
   `findmnt -no SOURCE /mnt/nvme == /dev/md0`，不符即 `exit 1`——服务进 `failed` 状态可见可告警，
   **杜绝静默落根盘**。人工检测不变：`grep md0 /proc/mounts` 应 3 行。
golden AMI 仍建议预装 mdadm（换块开机彻底免疫，且不依赖网络）。

---

# 第二阶段：容器镜像

## 直接拉取（推荐）

```bash
# public ECR 无需认证；containerd 环境需二次导入
docker pull public.ecr.aws/n3l4x8f3/sglang-mooncake-uccl:2026.05.02-h200.dp16
docker save public.ecr.aws/n3l4x8f3/sglang-mooncake-uccl:2026.05.02-h200.dp16 \
  | sudo ctr -n k8s.io images import -
```

**标签红线**：Decode dp64 强依赖 Mooncake PR #2023（DP>1 时所有 DP worker 共用同一
EfaEndPoint 槽位 → fi_av_remove/insert 抖动 → "Re-establish" 刷屏、decode #running 掉 1）。
`2026.05.02-h200.dp16` 起修复。

## 自建（需定制时）

```bash
git clone https://github.com/KevinZhao/efa-validation.git && cd efa-validation
docker build -f common/Dockerfile.customer-h200 \
  --build-arg WITH_UCCL=true --build-arg VARIANT=uccl \
  -t <registry>/sglang-mooncake-uccl:<date>-h200 .
```

自建注意（历次 hotfix 教训，删了会复现线上事故）：
- **勿删 rdma→efa sed 补丁段**：sglang 硬编码 protocol "rdma"，Mooncake USE_EFA=ON 构建
  收到非 "efa" 会静默回落 TCP，KV 传输分钟级延迟（h200.3 事故）
- **runtime 层必须有** `libpython3.10`（Mooncake binding dlopen，h200.1 事故）、
  `ninja-build`（SGLang TP worker JIT 编译，h200.2 事故）

## 镜像验收

```bash
bash scripts/verify-image.sh   # 见 scripts/，逐项校验 torch/sglang/mooncake/uccl/efa patch
```

---

# 第三阶段：自建 K8s（kubeadm + EFA 适配）

完整命令见 `scripts/k8s-prereq-al2023.sh`（AL2023 版，dnf/rpm）。流程：

1. **所有节点**：关 swap → overlay/br_netfilter 模块 + sysctl → containerd
   SystemdCgroup=true + `nvidia-ctk runtime configure` → kubeadm/kubelet/kubectl v1.31 + conntrack-tools
2. **Master**：`kubeadm init --pod-network-cidr=10.244.0.0/16` → 去 control-plane 污点
   （master 也跑推理）→ Flannel CNI
3. **其余节点**：`kubeadm join`
4. **NVIDIA device plugin** v0.17.1（官方静态 yaml 直接 apply）
5. **EFA device plugin** v0.5.7：镜像在需认证 ECR `602401143452.dkr.ecr.<region>.amazonaws.com`，
   自建 K8s 无凭据链 → 每节点 docker pull + `ctr -n k8s.io images import` + `imagePullPolicy: Never`；
   DaemonSet 必须挂 `/dev/infiniband` 和 `/sys`（否则 `No valid EFA devices found` CrashLoop）。
   manifest 见 `manifests/efa-device-plugin.yaml`

验收：每节点 Allocatable 必须是 `nvidia.com/gpu: 8` + `vpc.amazonaws.com/efa: 16`。

**SystemdCgroup 坑**：nvidia-container-toolkit 1.19 的 `nvidia-ctk runtime configure`
会把 nvidia runtime 段的 SystemdCgroup 重置回 false（eks-quickstart userdata 实录）。
必须在 nvidia-ctk **之后**再 sed 一次并 grep 校验无 `SystemdCgroup = false` 残留。
症状：Pod 报 `runc create failed: expected cgroupsPath to be of format "slice:prefix:name"`。

## GPU 可观测性组件（生产必装, 移植自 eks-gpu-stack standard 模式）

| 组件 | 作用 | 装法 |
|---|---|---|
| GFD | 自动打 `nvidia.com/gpu.product` 等节点标签 | nvidia-device-plugin helm chart `gfd.enabled=true` |
| dcgm-exporter | GPU 指标 :9400/metrics（利用率/显存/XID） | helm repo `nvidia.github.io/dcgm-exporter/helm-charts` |
| node-problem-detector | XID/内核 hang 上报节点 Condition | helm repo `charts.deliveryhero.io` |
| gpu-health-check | 开机 nvidia-smi 探测, 失败打 `gpu-unhealthy:NoSchedule` taint | `manifests/gpu-health-check.yaml`（已随本 skill 提供） |

```bash
helm upgrade -i nvidia-device-plugin nvdp/nvidia-device-plugin -n kube-system \
  --version 0.17.1 --set gfd.enabled=true --set mofedEnabled=false
# mofedEnabled=false 必须: 让 AWS EFA plugin 独占 /dev/infiniband/uverbs*
helm upgrade -i dcgm-exporter dcgm/dcgm-exporter -n kube-system
helm upgrade -i node-problem-detector deliveryhero/node-problem-detector -n kube-system
```

移植注意：eks-gpu-stack 的 values 用 `nodeSelector: workload-type=gpu` — 自建集群无此标签，
去掉 nodeSelector 或先 `kubectl label nodes --all workload-type=gpu`。

## K8s 网络设计决策

- **RDMA 与 CNI 无关**：NCCL/Mooncake/UCCL 直接走 EFA 设备，CNI 只承载控制面 TCP
- **所有推理 Pod 用 hostNetwork: true**：PD 分离要求 Pod 以节点 IP 互访
- Flannel 够用。若切 AWS VPC CNI **必须** `MAX_ENI=1` + `WARM_ENI_TARGET=0`
  （否则 CNI 往 EFA ENI 分 secondary IP，Pod 跨节点失联）；移除 Flannel 后
  `iptables -P FORWARD ACCEPT` 并 systemd 持久化

## 自建 K8s 特有坑

- 不自动打 `node.kubernetes.io/instance-type` 标签，nodeSelector 别用它
- CPU request 留余量（p5en 192 vCPU 用 ~180）
- 11GB+ 镜像提前预热到每节点，避免 ContainerCreating 卡死
- Pod 内 NCCL_SOCKET_IFNAME：hostNetwork 用 `enp`，CNI 网络用 `eth0`

---

# 第四阶段：模型分发

```bash
sudo mkdir -p /mnt/nvme/models && sudo chmod 1777 /mnt/nvme/models
s5cmd cp "s3://<bucket>/models/<MODEL>/*" /mnt/nvme/models/<MODEL>/
# 或 aws s3 cp --recursive
du -sh /mnt/nvme/models/<MODEL>   # 各节点一致
```

节点 stop/start 后 `/mnt/nvme` 清空需重拉（本地盘换新空盘；`setup-local-disks.service` 会自动重建 RAID+bind，但**数据需重新分发**）。JIT 缓存（DeepGEMM/Triton/torchinductor）
也放 NVMe 或 /data：冷启 JIT 10-20 min，热缓存秒级。

---

# 第五阶段：2P2D 部署验证（4 节点）

Pod 模板见 `manifests/pod-template.yaml`；环境变量与启动命令见
`scripts/launch-2p2d.sh`。要点：

## Pod 规格（每节点 1 个 Pod 整机独占）

```yaml
hostNetwork: true / hostIPC: true / privileged: true + IPC_LOCK
resources: nvidia.com/gpu: 8, vpc.amazonaws.com/efa: 16, cpu: 180, memory: 1500Gi
volumes: /mnt/nvme/models -> /models (hostPath), /dev/shm emptyDir Memory 512Gi
command: ["sleep","infinity"]   # kubectl exec 启服务便于调试
```

## 通用环境变量（所有角色）

```bash
# EFA/libfabric 必备
FI_PROVIDER=efa  FI_EFA_USE_DEVICE_RDMA=1  FI_EFA_FORK_SAFE=1
FI_EFA_USE_HUGE_PAGE=0  FI_MR_CACHE_MONITOR=disabled
# NCCL
NCCL_SOCKET_IFNAME=enp  NCCL_PROTO=Simple  NCCL_CROSS_NIC=1  NCCL_NVLS_ENABLE=1
# UCCL-EP
UCCL_SOCKET_IFNAME=enp  UCCL_IB_MAX_INFLIGHT_LOW_LATENCY=128
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
# PD 超时放宽
SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=10
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=300
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=300
```

## EFA rail 切分（customer_glm_5.1 实践）

16 EFA 分两半：低 8 rail 给 Mooncake KV（`--disaggregation-ib-device`），
高 8 rail 给 UCCL-EP（`UCCL_IB_HCA` + `FI_EFA_IFACE`）。
设备名 `rdmapXXs0` **因机而异**，先 `ls /sys/class/infiniband/` 枚举再填。

## 启动参数（2P2D）

| | Prefill (node0/1) | Decode (node2/3) |
|---|---|---|
| disaggregation-mode | prefill | decode |
| transfer-backend | mooncake | mooncake |
| 并行 | `--tp-size 8 --pp-size 2 --dp-size 1` | `--tp-size 16 --dp-size 16 --enable-dp-attention` |
| deepep-mode | normal（大 batch 吞吐） | low_latency（逐 token 延迟） |
| nnodes / rank | 2 / 0..1 | 2 / 0..1 |
| dist-init-addr | `<P_master>:5757` | `<D_master>:5757`（两组独立） |
| port | 30000 | 30001 |
| mem-fraction-static | 0.85 | 0.74 |
| 其他 | `--chunked-prefill-size 16384 --watchdog-timeout 3600` | `--cuda-graph-max-bs 16 --max-running-requests 256 --prefill-round-robin-balance` |
| **两侧必须一致** | `--page-size 64` | `--page-size 64` |

共同：`--moe-a2a-backend deepep --ep-dispatch-algorithm dynamic --eplb-algorithm deepseek --trust-remote-code`

**page-size 不一致 → 首个请求触发 KV 传输即 `AssertionError: Page size mismatch` 连环 crash。**

## Router

```bash
# 等两侧日志 "The server is fired up and ready to roll!" 后:
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://<P_master>:30000 --decode http://<D_master>:30001 \
  --host 0.0.0.0 --port 8000
```

## 2P2D 验收清单

| 检查 | 方法 | 通过标准 |
|---|---|---|
| KV 走 EFA | prefill 日志 | **无** `Installing TCP transport` |
| NCCL 走 EFA | NCCL_DEBUG=INFO | `Selected provider is efa` + `RDMA` |
| UCCL-EP | `python -c "import deep_ep; print(deep_ep.Config.__module__)"` | `uccl` 开头 |
| 端到端 | curl router /v1/chat/completions | 正常返回 |
| 压测 | `sglang.bench_serving` rate 1→32 | 成功率 100% |

冷启动 30-35 min 属正常（模型加载 + UCCL proxy init + JIT + CUDA graph）；热缓存 ~5 min。

---

# 第六阶段：切换到 4P8D 目标部署方案

**这是部署方案切换，不是扩容**：2P2D 与 4P8D 并行策略完全不同——权重切分方式变
（pp2→pp4 / tp16dp16→tp64dp64）、KV pool 布局变、JIT 缓存对新形状全部失效（重编 10-20min）、
全部 P/D 服务停服按新参数重启重组（dist-init 组 nnodes 2→4/8）。
**可复用**：基础设施（LT/SG/PG）、镜像、AMI、K8s 集群本体。
操作：开机数量改 8 → 新节点 join + 预热 + 权重分发 → 按新参数全量重新拉起：

| | 2P2D（验证方案） | 4P8D（目标方案） |
|---|---|---|
| Prefill | tp8/pp2, nnodes 2 | **tp8/pp4, nnodes 4**（ep32） |
| Decode | tp16/dp16, nnodes 2 | **tp64/dp64, nnodes 8**（ep64），`--max-running-requests 1024` |

## 大 DP/EP 三个关口

1. **Mooncake PR #2023**：dp64 必须 `2026.05.02-h200.dp16` 及以后镜像
2. **UCCL dispatch tokens**：rank≥32 遇 dispatch 超时，把 `deep_ep/buffer.py` 对应
   rank 档 `num_max_dispatch_tokens` 128→512 并同步
   `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512`（64-rank 需实测调参）
3. **DP idle-batch 死锁**：PD+DP 下空转 rank 卡 all_gather（上游 bug），修法：
   `scheduler_dp_attn_mixin.py` 的 `prepare_mlp_sync_batch_raw()` 条件加
   `or local_batch.forward_mode.is_idle()`

## dp64 容量提示

DP attention 下单请求只落一个 DP rank，最大输入长度受**单 rank KV pool** 限制
（dp16 实测单 rank max_total_num_tokens≈44800）。dp64 单 rank 更小，长文场景
看启动日志实测值设业务上限，或降 dp 换单 rank 容量。

---

# 故障速查表

| 症状 | 根因 | 处置 |
|---|---|---|
| KV 分钟级延迟 + `Installing TCP transport` | protocol "rdma" 回落 TCP | 用 h200.3+ 镜像；自建勿删 sed 补丁 |
| NCCL `no socket interface found` | IFNAME 不符 | hostNetwork=`enp`，CNI 网络=`eth0` |
| `ImportError: DeepEP is not installed` | deep_ep 残缺/未指向 uccl | 校验 `deep_ep.Config.__module__` 以 uccl 开头 |
| "Re-establish" 刷屏、decode 塌陷 | Mooncake DP>1 键冲突 | 换 dp16 镜像 |
| `Page size mismatch` crash | P/D page-size 不一致 | 两侧 `--page-size 64` |
| `DeepEP error: timeout (dispatch CPU)` | RDMA 压力/tokens 不足 | 勿用 `--enable-two-batch-overlap`（UCCL 不兼容）；调 dispatch tokens |
| UCCL proxy segfault @ CUDA graph 捕获 | dispatch/combine 进 graph | 兜底 `--disable-cuda-graph` |
| Pod ContainerCreating 卡死 | 现场拉 11GB 镜像 | 预热 docker pull + ctr import |
| 单侧重启后 KV 报错 | RDMA channel 残留 | P/D 成对重启 |
| VPC CNI 下 Pod 跨节点不通 | EFA ENI 被分 secondary IP / FORWARD DROP | MAX_ENI=1 + WARM_ENI_TARGET=0；FORWARD ACCEPT 持久化 |
| 启动 >30min | JIT 冷缓存 + graph 捕获 | 正常；缓存持久化后 ~5min |

# 压测与诊断工具（镜像自带）

```bash
python3 -m sglang.bench_serving --backend sglang --host <ROUTER> --port 8000 \
  --dataset-name random --num-prompts 100 \
  --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.5 \
  --max-concurrency <rate> --request-rate <rate>
/opt/mooncake/install/bin/transfer_engine_bench   # Mooncake 裸带宽
/opt/mooncake/install/bin/efa_transport_test      # 10 项 gtest
```

# 文件清单

| 路径 | 用途 |
|---|---|
| `terraform/` | CB 开机方案 A（LT/SG/PG/IAM/userdata 全套） |
| `cloudformation/p5en-cb-cluster.yaml` | CB 开机方案 B（已过 validate-template；已实测 CREATE_COMPLETE、每台 17 NIC/16 EFA） |
| `scripts/preflight-cr.sh` | **部署前预检**：CR active/机型匹配/可用数≥台数/子网AZ==CR AZ/IP 足够（防 InsufficientInstanceCapacity） |
| `scripts/setup-local-disks.sh` | **本地盘自愈**（每次开机 systemd 跑）：重建 RAID0 + bind docker/containerd 到 /mnt/nvme；已随 userdata base64 内嵌,此为可读源 |
| `scripts/k8s-prereq-al2023.sh` | 全节点 K8s 前置（AL2023 dnf 版） |
| `scripts/verify-node.sh` | 开机后节点验证（GPU/EFA/磁盘） |
| `scripts/verify-image.sh` | 镜像验收 |
| `scripts/launch-2p2d.sh` | 2P2D 启动（prefill/decode/router 三角色） |
| `manifests/efa-device-plugin.yaml` | EFA device plugin DaemonSet（自建 K8s 版） |
| `manifests/gpu-health-check.yaml` | GPU 健康探测 DaemonSet（失败打 taint 阻止调度） |
| `manifests/pod-template.yaml` | 推理 Pod 模板（hostNetwork 整机独占） |

# 性能预期（RoCE 迁移客户参考）

AWS Summit 2026《750B MoE 分离推理：从 RoCE 到 EFA 的全栈验证》实测
（GLM-5.1 750B/256E/Top-8, SGLang, 2P2D 4x p5en, 120K 长文本）：

| 指标 | RoCE→EFA | 解读 |
|---|---|---|
| TTFT | +9% | Prefill 计算密集, 网络非瓶颈 |
| TPOT | +31% | Decode MoE 每步 ~150 次小消息, 单次差 ~28μs |
| Max ITL | **−73%**（434→117ms） | SRD 多路径喷射, 尾延迟好 3.7x |

平均延迟略高属预期；尾延迟大幅改善，P99 SLA 场景整体更稳，且无 PFC/ECN 交换机运维负担。

# 参考来源

- KevinZhao/efa-validation — Dockerfile.customer-h200 / BUILD_MATRIX.md / customer_glm_5.1 / setup-nvme.sh
- aws-samples/sample-eks-enterprise-quickstart — eks-gpu-nodegroup 启动模板（p5en efa_only_count=15）/ userdata / eks-gpu-stack
- yuhuiaws/ML-study — sglang-2p2d-ucclep-nixl.skill / eks-h200-deepseek-v4-pd.skill
- 本地 ec2-self-k8s-GPU-kubeflow.skill — 自建 K8s + EFA device plugin 12 条踩坑
