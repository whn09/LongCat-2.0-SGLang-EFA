# =====================================================================
# p5en.48xlarge Capacity Block 推理集群 — 变量定义
# 参考: aws-samples/sample-eks-enterprise-quickstart terraform/modules/eks-gpu-nodegroup
# =====================================================================

variable "region" {
  type        = string
  description = "部署 Region（必须与 Capacity Block 预留的 Region 一致）"
}

variable "cluster_name" {
  type        = string
  description = "集群名称前缀，用于命名 LT/ASG/SG/IAM 等资源"
  default     = "p5en-inference"
}

variable "vpc_id" {
  type        = string
  description = "VPC ID"
}

variable "subnet_id" {
  type        = string
  description = "单个私有子网 ID。必须与 Capacity Block 预留的 AZ 一致；EFA 集群必须单 AZ"
}

variable "capacity_reservation_id" {
  type        = string
  description = "Capacity Block 预留 ID（cr-xxxxxxxx）。CB 购买后在 EC2 控制台 Capacity Reservations 查看"
}

variable "instance_count" {
  type        = number
  description = "开机台数。目标 8 台（CB 一次开 8 台）；2P2D 测试阶段可先设 4"
  default     = 8
}

variable "instance_type" {
  type    = string
  default = "p5en.48xlarge"
}

variable "ami_id" {
  type        = string
  description = <<-EOT
    基础 AMI。推荐 AL2023 Deep Learning AMI（含 NVIDIA driver + EFA 内核模块 + Docker）。
    查询命令:
    aws ssm get-parameter --region <region> \
      --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id \
      --query Parameter.Value --output text
  EOT
}

variable "key_name" {
  type        = string
  description = "EC2 SSH key pair 名称"
  default     = ""
}

variable "root_volume_size" {
  type        = number
  description = "根盘大小 GiB（gp3）。需求: 1TB"
  default     = 1024
}

variable "data_volume_size" {
  type        = number
  description = "数据持久化盘大小 GiB（gp3）。需求: 2TB"
  default     = 2048
}

# gp3 基线 3000 IOPS / 125 MBps 对模型加载太慢。
# 2TB 盘按容量自动适配到较高档位；gp3 上限 16000 IOPS / 1000 MBps。
variable "data_volume_iops" {
  type        = number
  description = "数据盘 IOPS（gp3 范围 3000-16000）"
  default     = 16000
}

variable "data_volume_throughput" {
  type        = number
  description = "数据盘吞吐 MBps（gp3 范围 125-1000）"
  default     = 1000
}

variable "efa_installer_version" {
  type        = string
  description = "aws-efa-installer 版本。固定版本保证节点可重复构建"
  default     = "1.48.0"
}

variable "allowed_ssh_cidr" {
  type        = string
  description = "允许 SSH 进入节点的 CIDR（如堡垒机网段）。留空则不开 SSH 入口"
  default     = ""
}

variable "extra_tags" {
  type        = map(string)
  description = "附加到所有资源的标签"
  default     = {}
}
