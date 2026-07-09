# Longcat-H200

p5en.48xlarge (H200) 多机多卡推理部署指导：自建 K8s (kubeadm) + SGLang PD 分离 + Mooncake KV Transfer + UCCL-EP + 16x EFA，基于 Capacity Block 批量开机。

## 阅读入口

| 文档 | 内容 |
|---|---|
| [p5en-inference-deployment.html](p5en-inference-deployment.html) | **主文档**：EC2 开机（Terraform/CFN）→ 容器镜像 → 自建 K8s → 模型分发 → 2P2D/4P8D 部署 → 压测排障 |
| [userdata-explained.html](userdata-explained.html) | **UserData 说明**（客户友好版）：开机自动做的 4 件事/磁盘布局/验证清单/排障速查 |
| [aws-cli-setup-and-handover.html](aws-cli-setup-and-handover.html) | **SRE→研发交接指南**：CLI/SSM 插件安装、研发 IAM 用户授权（仅 PowerUserAccess）、日常操作速查、交接清单 |
| [p5en-vs-p6b300.html](p5en-vs-p6b300.html) | **机型对比**：p5en(H200) vs p6-b300(B300)，GPU 显存/网络带宽为主，EC2 API 实查数据+官方链接 |
| [comm-stack-relationship.html](comm-stack-relationship.html) | 通信栈关系：NCCL / Mooncake TE / UCCL-EP 三条链路与 EFA 的关系 |
| [mooncake-efa-analysis.html](mooncake-efa-analysis.html) | Mooncake EFA 改版与上游原版的代码差异分析 |
| [skills/www/SKILL.md](skills/www/SKILL.md) | 主文档的精简可执行版（含文件清单与踩坑记录） |
| [skills/ec2-self-k8s-GPU-kubeflow/SKILL.md](skills/ec2-self-k8s-GPU-kubeflow/SKILL.md) | 自建 K8s + GPU + Kubeflow 通用参考 |

## 目录结构

```
terraform/            CB 开机方案 A（LT/SG/PG/IAM/userdata 全套）
cloudformation/       CB 开机方案 B（p5en-cb-cluster.yaml，已实测 CREATE_COMPLETE）
skills/www/           SKILL.md + scripts/（预检、本地盘自愈、验证）+ manifests/ + IaC 副本
skills/ec2-self-k8s-GPU-kubeflow/   通用自建 K8s 参考
```

## 关键版本

| 组件 | 版本 |
|---|---|
| SGLang | 0.5.10（镜像内置 rdma→efa 补丁） |
| Mooncake TE | v0.3.11.post1（USE_EFA=ON, 含 PR #2023） |
| EFA installer | 1.48.0（宿主机与容器对齐，--skip-kmod） |
| K8s | v1.31 kubeadm + NVIDIA plugin v0.17.1 + EFA plugin v0.5.7 |

## 使用流程（速览）

1. `scripts/preflight-cr.sh` 预检 CR/子网/容量 → 全 PASS 再开机
2. Terraform 或 CloudFormation 开机（userdata 自动完成 /data、本地盘 RAID0 自愈服务、EFA、nvidia runtime）
3. `scripts/verify-node.sh` 逐节点验证（8x H200 / 16 EFA / RAID0 / findmnt bind）
4. 按主文档 §5-§10 完成镜像、K8s、模型分发与部署验证
