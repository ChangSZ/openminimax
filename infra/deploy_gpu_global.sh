#!/usr/bin/env bash
# Global g6e capacity hunt: deploy the roaming GPU stack (gpu-region.yaml) into the
# first region+AZ that has g6e.12xlarge capacity. The control plane (DynamoDB tables,
# result bucket, KMS) stays FIXED in the home region; only the GPU box roams.
#
# Why this works: the GPU box is dial-out only (worker reads the DDB queue, writes S3
# cross-region over the AWS backbone). Nothing connects INTO it, so it can live in any
# region and its SG has zero inbound — "no public port" holds everywhere.
#
#   ./deploy_gpu_global.sh APPROVE
#
# Requires the literal APPROVE (each landed instance is ~$10/hr). Stops at the FIRST
# success. Skips regions where g6e.12xlarge isn't offered. Non-capacity errors in a
# region hard-stop that region and move on.
set -euo pipefail

[[ "${1:-}" == "APPROVE" ]] || { echo "REFUSING: launches a g6e.12xlarge (~\$10/hr). Re-run: $0 APPROVE" >&2; exit 2; }

HOME_REGION="${HOME_REGION:-us-west-2}"
STACK="${GPU_STACK:-openminimax-gpu}"
TEMPLATE="$(dirname "$0")/gpu-region.yaml"
# Ordered by likely capacity + proximity. Override with REGIONS="...".
REGIONS="${REGIONS:-us-west-2 us-east-1 us-east-2 us-east-1 ap-northeast-1}"

# Control-plane ARNs (home region). Auto-discovered if not supplied via env.
BASE_STACK="${BASE_STACK:-openminimax}"
KEYS_TABLE="${KEYS_TABLE:-openminimax-serverless-keys}"
TASKS_TABLE="${TASKS_TABLE:-openminimax-serverless-tasks}"
# The result bucket name is account-specific (the base stack names it with your
# account id), so it is NEVER defaulted to a literal — discover it from the base
# stack's output, or pass RESULT_BUCKET=... explicitly.
if [[ -z "${RESULT_BUCKET:-}" ]]; then
  RESULT_BUCKET="$(aws cloudformation describe-stacks --region "$HOME_REGION" --stack-name "$BASE_STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='ResultBucketName'].OutputValue" --output text 2>/dev/null || true)"
fi
if [[ -z "$RESULT_BUCKET" || "$RESULT_BUCKET" == "None" ]]; then
  echo "ERROR: could not resolve the result bucket. Set RESULT_BUCKET=... or ensure base" >&2
  echo "       stack '$BASE_STACK' in $HOME_REGION exports ResultBucketName." >&2
  exit 1
fi
echo "Discovering control-plane ARNs in $HOME_REGION ..."
KEYS_ARN="${KEYS_ARN:-$(aws dynamodb describe-table --region "$HOME_REGION" --table-name "$KEYS_TABLE" --query 'Table.TableArn' --output text)}"
TASKS_ARN="${TASKS_ARN:-$(aws dynamodb describe-table --region "$HOME_REGION" --table-name "$TASKS_TABLE" --query 'Table.TableArn' --output text)}"
RESULT_ARN="arn:aws:s3:::${RESULT_BUCKET}"
# KMS key that encrypts the result bucket (from the base stack).
DATAKEY_ARN="${DATAKEY_ARN:-$(aws cloudformation describe-stack-resources --region "$HOME_REGION" --stack-name "$BASE_STACK" \
  --query "StackResources[?ResourceType=='AWS::KMS::Key'].PhysicalResourceId" --output text \
  | xargs -I{} aws kms describe-key --region "$HOME_REGION" --key-id {} --query 'KeyMetadata.Arn' --output text)}"
echo "  keys=$KEYS_ARN"
echo "  tasks=$TASKS_ARN"
echo "  bucket=$RESULT_ARN"
echo "  kms=$DATAKEY_ARN"

offered_azs() {  # $1 region -> space-separated AZ names offering g6e.12xlarge
  aws ec2 describe-instance-type-offerings --region "$1" --location-type availability-zone \
    --filters Name=instance-type,Values=g6e.12xlarge \
    --query 'InstanceTypeOfferings[].Location' --output text 2>/dev/null | tr '\t' ' '
}

try_region_az() {  # $1 region  $2 az  -> 0 on launch success
  local region="$1" az="$2"
  echo ">>> $region / $az"
  aws cloudformation deploy \
    --region "$region" --stack-name "$STACK" \
    --template-file "$TEMPLATE" --capabilities CAPABILITY_IAM \
    --no-fail-on-empty-changeset \
    --parameter-overrides \
      ComputeAz="$az" HomeRegion="$HOME_REGION" \
      KeysTableArn="$KEYS_ARN" TasksTableArn="$TASKS_ARN" \
      KeysTableName="$KEYS_TABLE" TasksTableName="$TASKS_TABLE" \
      ResultBucketArn="$RESULT_ARN" ResultBucketName="$RESULT_BUCKET" \
      DataKeyArn="$DATAKEY_ARN" 2>&1 | tee /tmp/mmh3-gpu-global.out
  local rc="${PIPESTATUS[0]}"
  [[ "$rc" == 0 ]] && return 0
  # capacity? -> caller tries next AZ/region. else hard-stop.
  if grep -qiE 'sufficient .*capacity|InsufficientInstanceCapacity' /tmp/mmh3-gpu-global.out || \
     aws cloudformation describe-stack-events --region "$region" --stack-name "$STACK" \
       --query 'StackEvents[?ResourceStatus==`CREATE_FAILED` && LogicalResourceId==`GpuInstance`].ResourceStatusReason' \
       --output text 2>/dev/null | grep -qiE 'sufficient .*capacity'; then
    echo ">>> no capacity in $region/$az"
    # roll the failed stack back to nothing so the next region starts clean
    aws cloudformation delete-stack --region "$region" --stack-name "$STACK" 2>/dev/null || true
    aws cloudformation wait stack-delete-complete --region "$region" --stack-name "$STACK" 2>/dev/null || true
    return 1
  fi
  echo "!!! $region/$az failed for a NON-capacity reason (see output). Stopping." >&2
  exit 1
}

for region in $REGIONS; do
  azs="$(offered_azs "$region")"
  if [[ -z "$azs" ]]; then echo "== $region: g6e.12xlarge not offered, skipping"; continue; fi
  echo "== $region offers g6e.12xlarge in: $azs"
  for az in $azs; do
    if try_region_az "$region" "$az"; then
      echo
      echo "############################################################"
      echo "# GPU LANDED: region=$region az=$az stack=$STACK"
      aws cloudformation describe-stacks --region "$region" --stack-name "$STACK" \
        --query 'Stacks[0].Outputs[].{K:OutputKey,V:OutputValue}' --output table
      echo "# Next: SSM in; start serving/run_h3_server.sh (diffusers Turbo shim) + python -m app.worker_main (SGLANG_STEPS=9)"
      echo "############################################################"
      exit 0
    fi
  done
done

echo "!!! no g6e.12xlarge capacity in any region tried: $REGIONS" >&2
echo "    Try again later, add regions (REGIONS=...), or reserve capacity (ODCR/Capacity Block)." >&2
exit 3
