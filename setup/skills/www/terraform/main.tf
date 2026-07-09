# =====================================================================
# p5en.48xlarge Capacity Block 推理集群
#
# 关键设计:
#   1. p5en.48xlarge 网卡布局（AWS 官方 EFA 多网卡推荐配置）:
#      card 0 / device 0  = interface (纯 ENA 主网卡; 主网卡不能是 efa-only)
#      card 0 / device 1  = efa-only  ← 容易漏! card 0 也承载一个 EFA 设备
#      card 1-15 / device 0 = efa-only x15
#      合计 17 个接口 = 1 ENA + 16 EFA (fi_info 应显示 16 个 EFA 设备)
#      参考: docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-acc-inst-types.html
#   2. Capacity Block 要求 InstanceType + InstanceMarketOptions(capacity-block)
#      + CapacityReservationTarget 全部嵌入 Launch Template
#   3. EFA 必须: 单 AZ + cluster placement group + 自引用全通安全组
#   4. 磁盘: 根盘 1TB gp3 + 数据盘 2TB gp3 16000 IOPS / 1000 MBps
# =====================================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

locals {
  # p5en.48xlarge: 16 network cards; card0 = ENA主网卡(dev0) + EFA(dev1), card1-15 = EFA-only
  efa_only_count = 15
  common_tags = merge({
    Project = var.cluster_name
  }, var.extra_tags)
}

# ---------------------------------------------------------------------
# IAM: 节点角色（SSM 运维 + ECR 拉镜像）
# ---------------------------------------------------------------------
resource "aws_iam_role" "node" {
  name = "${var.cluster_name}-node-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "ecr" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# 自建 K8s 若后续切换 AWS VPC CNI，节点需要 ENI 管理权限
resource "aws_iam_role_policy_attachment" "cni" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_instance_profile" "node" {
  name = "${var.cluster_name}-node-profile"
  role = aws_iam_role.node.name
}

# ---------------------------------------------------------------------
# 安全组: EFA 要求组内全通（自引用 ingress/egress all）
# ---------------------------------------------------------------------
resource "aws_security_group" "node" {
  name        = "${var.cluster_name}-node-sg"
  description = "p5en nodes with EFA - self-referencing all traffic"
  vpc_id      = var.vpc_id
  tags        = merge(local.common_tags, { Name = "${var.cluster_name}-node-sg" })
}

resource "aws_vpc_security_group_ingress_rule" "self_all" {
  security_group_id            = aws_security_group.node.id
  ip_protocol                  = "-1"
  referenced_security_group_id = aws_security_group.node.id
  description                  = "EFA/NCCL/K8s self-allow all"
}

resource "aws_vpc_security_group_egress_rule" "self_all" {
  security_group_id            = aws_security_group.node.id
  ip_protocol                  = "-1"
  referenced_security_group_id = aws_security_group.node.id
  description                  = "EFA self-egress (required by EFA docs)"
}

resource "aws_vpc_security_group_egress_rule" "internet" {
  security_group_id = aws_security_group.node.id
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
  description       = "outbound internet (image pull, efa installer)"
}

resource "aws_vpc_security_group_ingress_rule" "ssh" {
  count             = var.allowed_ssh_cidr != "" ? 1 : 0
  security_group_id = aws_security_group.node.id
  ip_protocol       = "tcp"
  from_port         = 22
  to_port           = 22
  cidr_ipv4         = var.allowed_ssh_cidr
  description       = "SSH from bastion"
}

# ---------------------------------------------------------------------
# Cluster placement group: EFA 低延迟必须
# 注意: CB 预留本身已绑定容量位置，placement group 仍建议保留以显式声明拓扑
# ---------------------------------------------------------------------
resource "aws_placement_group" "cluster" {
  name     = "${var.cluster_name}-pg"
  strategy = "cluster"
  tags     = local.common_tags
}

# ---------------------------------------------------------------------
# Launch Template
# ---------------------------------------------------------------------
resource "aws_launch_template" "p5en" {
  name          = "${var.cluster_name}-p5en-lt"
  description   = "p5en.48xlarge CB, 16x EFA, 1TB root + 2TB data gp3"
  image_id      = var.ami_id
  instance_type = var.instance_type # CB 要求 InstanceType 写入 LT
  key_name      = var.key_name != "" ? var.key_name : null

  user_data = base64encode(templatefile("${path.module}/userdata.sh.tpl", {
    efa_installer_version = var.efa_installer_version
  }))

  iam_instance_profile {
    name = aws_iam_instance_profile.node.name
  }

  # Capacity Block 双要素: market options + reservation target
  instance_market_options {
    market_type = "capacity-block"
  }

  capacity_reservation_specification {
    capacity_reservation_target {
      capacity_reservation_id = var.capacity_reservation_id
    }
  }

  placement {
    group_name = aws_placement_group.cluster.name
    tenancy    = "default"
  }

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2 # 容器内访问 IMDS 需要 hop 2
    http_endpoint               = "enabled"
  }

  # 根盘 1TB gp3
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = var.root_volume_size
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  # 数据盘 2TB gp3, IOPS/吞吐拉满（模型分发/持久化）
  block_device_mappings {
    device_name = "/dev/xvdb"
    ebs {
      volume_size           = var.data_volume_size
      volume_type           = "gp3"
      iops                  = var.data_volume_iops
      throughput            = var.data_volume_throughput
      encrypted             = true
      delete_on_termination = true
    }
  }

  # 主网卡: card 0 / device 0, 纯 ENA（官方要求: 主网卡不能是 efa-only）
  network_interfaces {
    network_card_index    = 0
    device_index          = 0
    interface_type        = "interface"
    subnet_id             = var.subnet_id
    delete_on_termination = true
    security_groups       = [aws_security_group.node.id]
    # ⚠️ EC2 硬限制: 多网卡启动不允许 associate_public_ip_address(API 400),
    # 且忽略子网 MapPublicIpOnLaunch → 节点开机时无公网 IP。
    # 出网必须二选一: ①子网路由走 NAT Gateway(推荐) ②开机后给主 ENI 挂 EIP。
  }

  # card 0 / device 1: EFA-only —— 容易漏掉的一个! card 0 同样承载 EFA 设备
  network_interfaces {
    network_card_index    = 0
    device_index          = 1
    interface_type        = "efa-only"
    subnet_id             = var.subnet_id
    delete_on_termination = true
    security_groups       = [aws_security_group.node.id]
  }

  # card 1..15 / device 0: EFA-only x15（efa-only 不占 IP、不走 TCP/IP，仅 RDMA）
  # 合计 16 个 EFA 设备
  dynamic "network_interfaces" {
    for_each = range(1, local.efa_only_count + 1)
    content {
      network_card_index    = network_interfaces.value
      device_index          = 0
      interface_type        = "efa-only"
      subnet_id             = var.subnet_id
      delete_on_termination = true
      security_groups       = [aws_security_group.node.id]
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(local.common_tags, {
      Name = "${var.cluster_name}-p5en-node"
    })
  }

  tag_specifications {
    resource_type = "volume"
    tags = merge(local.common_tags, {
      Name = "${var.cluster_name}-p5en-volume"
    })
  }

  tags = local.common_tags
}

# ---------------------------------------------------------------------
# 开机: 直接用 aws_instance 批量创建（数量固定、不需要 ASG 自愈；
# CB 到期后实例会被回收，ASG 自动补机反而会报错）
# ---------------------------------------------------------------------
resource "aws_instance" "p5en" {
  count = var.instance_count

  launch_template {
    id      = aws_launch_template.p5en.id
    version = aws_launch_template.p5en.latest_version
  }

  tags = merge(local.common_tags, {
    Name = "${var.cluster_name}-p5en-${count.index}"
  })

  lifecycle {
    ignore_changes = [launch_template] # LT 迭代不触发重建，滚动替换手动控制
  }
}
