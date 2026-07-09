#!/bin/bash
# =====================================================================
# preflight-cr.sh — 部署前 Capacity Block / CR 预检
# 在 terraform apply 或 cloudformation create-stack 之前运行,
# 提前拦截最常见的开机失败 (InsufficientInstanceCapacity / 网卡报错):
#   - CR 是否存在且 state=active
#   - CR 机型是否 == 期望机型 (默认 p5en.48xlarge)
#   - CR 可用数量是否 >= 计划开机台数
#   - 传入子网的 AZ 是否 == CR 的 AZ (EFA 必须单 AZ 同区)
#   - 子网可用 IP 是否够 (每实例仅 1 个私有 IP; efa-only 不占 IP)
#   - Capacity Block: 是否已到 StartDate、离 EndDate 还有多久
#
# 用法:
#   REGION=us-west-2 CR_ID=cr-xxxx SUBNET_ID=subnet-xxxx COUNT=8 \
#     bash preflight-cr.sh
#   可选: INSTANCE_TYPE=p5en.48xlarge
# 退出码: 0=全部通过; 1=有阻断项
# =====================================================================
set -uo pipefail
export AWS_PAGER=""

REGION="${REGION:?Set REGION, e.g. us-west-2}"
CR_ID="${CR_ID:?Set CR_ID, e.g. cr-xxxxxxxx}"
SUBNET_ID="${SUBNET_ID:?Set SUBNET_ID (must be in the same AZ as the CR)}"
COUNT="${COUNT:-8}"
INSTANCE_TYPE="${INSTANCE_TYPE:-p5en.48xlarge}"

command -v aws >/dev/null || { echo "ERROR: aws CLI not found"; exit 1; }
command -v jq  >/dev/null || { echo "ERROR: jq not found"; exit 1; }

PASS=0; FAIL=0
ok()  { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

echo "=== Capacity Reservation 预检: ${CR_ID} (region=${REGION}) ==="
CR=$(aws ec2 describe-capacity-reservations \
      --capacity-reservation-ids "${CR_ID}" --region "${REGION}" \
      --query 'CapacityReservations[0]' --output json 2>/dev/null)
if [ -z "${CR}" ] || [ "${CR}" = "null" ]; then
  echo "  [FAIL] CR ${CR_ID} 在 ${REGION} 未找到 (区域是否写错?)"; exit 1
fi

CR_TYPE=$(echo "${CR}"  | jq -r '.InstanceType')
CR_AZ=$(echo "${CR}"    | jq -r '.AvailabilityZone')
CR_STATE=$(echo "${CR}" | jq -r '.State')
CR_AVAIL=$(echo "${CR}" | jq -r '.AvailableInstanceCount')
CR_TOTAL=$(echo "${CR}" | jq -r '.TotalInstanceCount')
CR_RTYPE=$(echo "${CR}" | jq -r '.ReservationType // "on-demand"')
CR_START=$(echo "${CR}" | jq -r '.StartDate // "n/a"')
CR_END=$(echo "${CR}"   | jq -r '.EndDate // "n/a"')

echo "  机型=${CR_TYPE}  AZ=${CR_AZ}  state=${CR_STATE}  可用=${CR_AVAIL}/${CR_TOTAL}  类型=${CR_RTYPE}"
echo "  有效期: ${CR_START}  ->  ${CR_END}"

[ "${CR_STATE}" = "active" ] && ok "CR 处于 active" \
  || bad "CR 非 active (state=${CR_STATE}); Capacity Block 未到 StartDate 时会一直 InsufficientInstanceCapacity"

[ "${CR_TYPE}" = "${INSTANCE_TYPE}" ] && ok "机型匹配 (${CR_TYPE})" \
  || bad "机型不匹配: CR=${CR_TYPE} 但期望=${INSTANCE_TYPE}"

if [ "${CR_AVAIL}" != "null" ] && [ "${CR_AVAIL}" -ge "${COUNT}" ]; then
  ok "可用数量足够 (${CR_AVAIL} >= ${COUNT})"
else
  bad "可用数量不足: 可用 ${CR_AVAIL} < 计划 ${COUNT} (调小 instance_count 或核对 CR)"
fi

echo "=== 出网路由校验 (p5en 多网卡开机拿不到公网 IP, IGW 直连路由无效) ==="
RT_JSON=$(aws ec2 describe-route-tables --region "${REGION}" \
  --filters "Name=association.subnet-id,Values=${SUBNET_ID}" \
  --query 'RouteTables[0]' --output json 2>/dev/null)
if [ -z "${RT_JSON}" ] || [ "${RT_JSON}" = "null" ]; then
  # 无显式关联 = 用 VPC 主路由表
  VPC_OF_SUBNET=$(aws ec2 describe-subnets --region "${REGION}" --subnet-ids "${SUBNET_ID}" \
    --query 'Subnets[0].VpcId' --output text)
  RT_JSON=$(aws ec2 describe-route-tables --region "${REGION}" \
    --filters "Name=vpc-id,Values=${VPC_OF_SUBNET}" "Name=association.main,Values=true" \
    --query 'RouteTables[0]' --output json)
fi
DEFAULT_TGT=$(echo "${RT_JSON}" | jq -r '.Routes[] | select(.DestinationCidrBlock=="0.0.0.0/0") | (.NatGatewayId // .GatewayId // .TransitGatewayId // "none")' | head -1)
case "${DEFAULT_TGT}" in
  nat-*) ok "默认路由走 NAT Gateway (${DEFAULT_TGT}), 开机即有出网" ;;
  igw-*) bad "默认路由走 IGW (${DEFAULT_TGT}): 多网卡启动禁用 associatePublicIPAddress 且忽略子网 MapPublicIpOnLaunch, 开机无公网 IP → IGW 路由无效, dnf/镜像拉取全失败。改 NAT Gateway 或开机后挂 EIP" ;;
  tgw-*) echo "  [WARN] 默认路由走 TGW (${DEFAULT_TGT}), 请自行确认可达公网" ;;
  none|"") bad "子网无默认路由 (0.0.0.0/0): 节点无出网, userdata 必然失败" ;;
  *) echo "  [WARN] 默认路由目标 ${DEFAULT_TGT}, 请自行确认出网可达" ;;
esac

echo "=== 子网校验: ${SUBNET_ID} ==="
SUBNET=$(aws ec2 describe-subnets --subnet-ids "${SUBNET_ID}" --region "${REGION}" \
          --query 'Subnets[0]' --output json 2>/dev/null)
if [ -z "${SUBNET}" ] || [ "${SUBNET}" = "null" ]; then
  bad "子网 ${SUBNET_ID} 未找到"
else
  SUBNET_AZ=$(echo "${SUBNET}"  | jq -r '.AvailabilityZone')
  SUBNET_FREE=$(echo "${SUBNET}"| jq -r '.AvailableIpAddressCount')
  echo "  子网 AZ=${SUBNET_AZ}  可用 IP=${SUBNET_FREE}"
  [ "${SUBNET_AZ}" = "${CR_AZ}" ] && ok "子网 AZ 与 CR 一致 (${CR_AZ})" \
    || bad "子网 AZ=${SUBNET_AZ} != CR AZ=${CR_AZ} (EFA 集群必须单 AZ、且与 CB 同 AZ)"
  # 每实例只占 1 个私有 IP(主 ENA 网卡);efa-only 网卡不占 IP。
  if [ "${SUBNET_FREE}" -ge "${COUNT}" ]; then
    ok "子网 IP 足够 (${SUBNET_FREE} >= ${COUNT}, 每实例 1 IP)"
  else
    bad "子网 IP 不足: ${SUBNET_FREE} < ${COUNT}"
  fi
fi

echo ""
echo "结果: ${PASS} PASS / ${FAIL} FAIL"
if [ "${FAIL}" -eq 0 ]; then
  echo "预检通过 ✅  可以执行 terraform apply / cloudformation create-stack"
  exit 0
else
  echo "预检未通过 ❌  修复以上 FAIL 项后再部署"
  exit 1
fi
