#!/usr/bin/env bash
# Deploy the openminimax infrastructure stack into YOUR account.
# Region defaults to us-west-2; override with AWS_REGION=... (see STACK/AWS_REGION below).
#
# Two deliberately-separate spend tiers:
#   ./deploy.sh base            # free/near-free: VPC, SGs, IAM, empty bucket. NO GPU.
#   ./deploy.sh gateway         # + t3.small controller (~$15/mo) + NAT (~$32/mo)
#   ./deploy.sh gpu   APPROVE   # + g6e.12xlarge (~$10/hr) — REQUIRES the APPROVE arg
#
# The GPU tier refuses to run without the literal word APPROVE as $2, so nobody
# starts a ~$10/hr instance by muscle memory. See docs/PLAN.md §3 (Phase 0) and §7.
set -euo pipefail

STACK="${STACK:-openminimax}"
REGION="${AWS_REGION:-us-west-2}"
TEMPLATE="$(dirname "$0")/template.yaml"
TIER="${1:-base}"

launch_gpu=false
launch_gateway=false
gpu_ami_arg=()

case "$TIER" in
  base) ;;
  gateway) launch_gateway=true ;;
  gpu)
    if [[ "${2:-}" != "APPROVE" ]]; then
      echo "REFUSING: the 'gpu' tier launches a g6e.12xlarge (~\$10/hr)." >&2
      echo "Re-run with explicit approval:  ./deploy.sh gpu APPROVE" >&2
      exit 2
    fi
    launch_gpu=true
    launch_gateway=true
    # Resolve the current Deep Learning AMI from SSM, only now. Default: the PyTorch
    # DLAMI (torch + CUDA preinstalled -> less to set up for SGLang). Override
    # DLAMI_PARAM to use the driver-only base-oss AMI instead.
    DLAMI_PARAM="${DLAMI_PARAM:-/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.12-ubuntu-24.04/latest/ami-id}"
    echo "Resolving DLAMI from SSM: $DLAMI_PARAM"
    GPU_AMI=$(aws ssm get-parameter --region "$REGION" --name "$DLAMI_PARAM" \
              --query 'Parameter.Value' --output text)
    echo "GPU AMI: $GPU_AMI"
    gpu_ami_arg=(GpuAmiId="$GPU_AMI")
    ;;
  *)
    echo "usage: $0 [base|gateway|gpu APPROVE]" >&2; exit 1 ;;
esac

deploy_once() {  # $1 = ComputeAz override ('' to leave as-is). Streams+captures output.
  local az_arg=()
  [[ -n "${1:-}" ]] && az_arg=(ComputeAz="$1")
  aws cloudformation deploy \
    --region "$REGION" \
    --stack-name "$STACK" \
    --template-file "$TEMPLATE" \
    --capabilities CAPABILITY_IAM \
    --no-fail-on-empty-changeset \
    --parameter-overrides \
      LaunchGpuInstance="$launch_gpu" \
      LaunchGatewayInstance="$launch_gateway" \
      "${az_arg[@]}" \
      "${gpu_ami_arg[@]}" 2>&1 | tee /tmp/mmh3-deploy.out
  return "${PIPESTATUS[0]}"
}

# Did THIS attempt fail specifically on g6e capacity? Check the just-captured output
# AND the stack's own failed events for this run — never stale history.
this_attempt_hit_capacity() {
  grep -qiE 'sufficient .*capacity|InsufficientInstanceCapacity' /tmp/mmh3-deploy.out && return 0
  aws cloudformation describe-stack-events --region "$REGION" --stack-name "$STACK" \
    --query 'StackEvents[?ResourceStatus==`CREATE_FAILED` && LogicalResourceId==`GpuInstance`].ResourceStatusReason' \
    --output text 2>/dev/null | grep -qiE 'sufficient .*capacity'
}

echo "Deploying stack '$STACK' in $REGION (tier=$TIER, gpu=$launch_gpu, gateway=$launch_gateway)"

if [[ "$launch_gpu" == "true" ]]; then
  # g6e capacity is per-AZ and flaps. The network (subnets in every AZ) is built once;
  # if the GPU can't get capacity in one AZ we retry with the next — a fast
  # instance-only replacement, no subnet/NAT churn. Order is overridable.
  AZ_SWEEP="${AZ_SWEEP:-${REGION}a ${REGION}b ${REGION}c ${REGION}d}"
  ok=false
  for az in $AZ_SWEEP; do
    echo ">>> trying ComputeAz=$az"
    if deploy_once "$az"; then ok=true; echo ">>> launched in $az"; break; fi
    if this_attempt_hit_capacity; then
      echo ">>> no g6e capacity in $az — rolling forward to the next AZ"; continue
    fi
    echo "!!! deploy failed for a NON-capacity reason (see output above) — stopping." >&2
    exit 1
  done
  if [[ "$ok" != "true" ]]; then
    echo "!!! no g6e capacity in any of: $AZ_SWEEP" >&2
    echo "    Options: wait and retry, request a Capacity Reservation/Capacity Block," >&2
    echo "    or try another region (all IaC is region-parameterized). See docs/PLAN.md." >&2
    exit 3
  fi
else
  deploy_once ""
fi

echo "--- outputs ---"
aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query 'Stacks[0].Outputs[].{Key:OutputKey,Value:OutputValue}' --output table
