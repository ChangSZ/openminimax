#!/usr/bin/env bash
# Fast global g6e.12xlarge capacity probe — WITHOUT building a CFN stack per attempt.
#
# AWS exposes no "is there capacity" API; only a real launch reveals it. So this does
# a minimal RunInstances into each region's DEFAULT VPC and, on success, TERMINATES it
# within seconds. A miss (InsufficientInstanceCapacity) returns in ~1s. This turns the
# slow "build VPC -> fail -> rollback -> delete" loop (2-3 min/miss) into a ~1s/miss
# scan, so you find a live AZ fast, then deploy the real roaming stack there with
# deploy_gpu_global.sh (or gpu-region.yaml) pinned to that region+AZ.
#
#   ./probe_capacity.sh              # scan, report first region+AZ with capacity
#   ./probe_capacity.sh --all        # don't stop at first; map capacity everywhere
#
# A probe instance exists for only the moment between launch and terminate (seconds),
# so cost is negligible (per-second billing) — but it IS a real launch, hence the
# terminate is best-effort-hardened below. Uses the PyTorch DLAMI per region.
set -euo pipefail

ALL=false; [[ "${1:-}" == "--all" ]] && ALL=true
REGIONS="${REGIONS:-us-east-1 us-west-2 us-east-2 eu-central-1 ap-northeast-1 ap-south-1}"
ITYPE="${ITYPE:-g6e.12xlarge}"
AMI_PARAM="/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.12-ubuntu-24.04/latest/ami-id"

found=""
for region in $REGIONS; do
  offered=$(aws ec2 describe-instance-type-offerings --region "$region" \
    --location-type availability-zone --filters Name=instance-type,Values="$ITYPE" \
    --query 'InstanceTypeOfferings[].Location' --output text 2>/dev/null | tr '\t' ' ')
  [[ -z "$offered" ]] && { echo "== $region: $ITYPE not offered, skip"; continue; }

  ami=$(aws ssm get-parameter --region "$region" --name "$AMI_PARAM" --query 'Parameter.Value' --output text 2>/dev/null || echo "")
  [[ -z "$ami" || "$ami" == "None" ]] && { echo "== $region: no DLAMI, skip"; continue; }
  vpc=$(aws ec2 describe-vpcs --region "$region" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text 2>/dev/null)

  echo "== $region ($ITYPE in: $offered)"
  for az in $offered; do
    subnet=$(aws ec2 describe-subnets --region "$region" \
      --filters Name=vpc-id,Values="$vpc" Name=availability-zone,Values="$az" Name=default-for-az,Values=true \
      --query 'Subnets[0].SubnetId' --output text 2>/dev/null)
    [[ -z "$subnet" || "$subnet" == "None" ]] && { echo "   $az: no default subnet, skip"; continue; }

    # Real launch attempt. Success -> capacity exists; terminate immediately.
    iid=$(aws ec2 run-instances --region "$region" --instance-type "$ITYPE" \
      --image-id "$ami" --subnet-id "$subnet" --count 1 \
      --instance-initiated-shutdown-behavior terminate \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=openminimax-probe}]' \
      --query 'Instances[0].InstanceId' --output text 2>/tmp/mmh3-probe.err) || iid=""

    if [[ -n "$iid" && "$iid" != "None" ]]; then
      echo "   $az: ✅ CAPACITY (probe $iid) — terminating probe"
      aws ec2 terminate-instances --region "$region" --instance-ids "$iid" >/dev/null 2>&1 || \
        echo "   !! WARN: could not terminate $iid in $region — TERMINATE IT MANUALLY" >&2
      found="$region $az"
      $ALL || { echo; echo "FIRST CAPACITY: region=$region az=$az"; echo \
        "Deploy there:  REGIONS=\"$region\" AZ_ONLY=$az ./deploy_gpu_global.sh APPROVE"; exit 0; }
    else
      reason=$(grep -oiE 'InsufficientInstanceCapacity|Unsupported[A-Za-z]*|VcpuLimitExceeded|[A-Za-z]+Limit[A-Za-z]*' /tmp/mmh3-probe.err | head -1)
      echo "   $az: ${reason:-no-capacity}"
    fi
  done
done

if [[ -n "$found" ]]; then echo; echo "Capacity seen at: $found"; else
  echo; echo "!!! no $ITYPE capacity in any probed region: $REGIONS" >&2; exit 3
fi
