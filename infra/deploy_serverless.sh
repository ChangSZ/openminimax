#!/usr/bin/env bash
# Package the Lambda code and deploy the serverless API layer (infra/serverless.yaml)
# on top of the base stack (infra/template.yaml).
#
# The Lambda bundle is just the gateway's `app/` + `lambdas/` packages. The handlers
# use only the stdlib + boto3 (boto3 is in the Lambda runtime), so NO dependencies are
# vendored — FastAPI/uvicorn are for the EC2 gateway path only, not the Lambda path.
#
# Usage:
#   ./deploy_serverless.sh <artifact-bucket> [gpu-instance-id]
#
#   <artifact-bucket>   an S3 bucket you own for the code zip (any private bucket)
#   [gpu-instance-id]   base-stack output GpuInstanceId; enables the autostop schedule
#                       (omit before the GPU tier is launched)
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
BASE_STACK="${BASE_STACK:-openminimax}"
STACK="${STACK:-openminimax-serverless}"
ARTIFACT_BUCKET="${1:?usage: $0 <artifact-bucket> [gpu-instance-id]}"
GPU_INSTANCE_ID="${2:-}"

HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_DIR="$HERE/../gateway"
KEY="openminimax/lambda-$(date +%Y%m%d%H%M%S).zip"
ZIP="$(mktemp -d)/lambda.zip"

echo "Packaging Lambda bundle from $GATEWAY_DIR (app/ + lambdas/, no deps)"
( cd "$GATEWAY_DIR" && zip -q -r "$ZIP" app lambdas \
    -x '*/__pycache__/*' '*.pyc' 'app/main.py' )   # main.py = FastAPI, not used in Lambda
echo "  -> $(du -h "$ZIP" | cut -f1)  ($ZIP)"

echo "Uploading to s3://$ARTIFACT_BUCKET/$KEY"
aws s3 cp --region "$REGION" "$ZIP" "s3://$ARTIFACT_BUCKET/$KEY"

# Derive the results bucket + KMS key from the base stack so the poll Lambda can
# sign a fresh presigned GET at read time (publish.presign_s3_ref). Best-effort:
# if the lookups fail, deploy without them (poll returns the s3:// ref unsigned).
_out() { aws cloudformation describe-stacks --region "$REGION" --stack-name "$BASE_STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text 2>/dev/null; }
RESULT_BUCKET="$(_out ResultBucketName)"; [ "$RESULT_BUCKET" = "None" ] && RESULT_BUCKET=""
# KMS key isn't a stack output; resolve the well-known alias to its ARN.
RESULT_KMS_ARN="$(aws kms describe-key --region "$REGION" \
  --key-id "alias/${BASE_STACK}-openminimax" --query 'KeyMetadata.Arn' --output text 2>/dev/null || true)"
[ "$RESULT_KMS_ARN" = "None" ] && RESULT_KMS_ARN=""
echo "  result bucket: '${RESULT_BUCKET:-none}'  kms: '${RESULT_KMS_ARN:-none}'"

echo "Deploying stack '$STACK' (base='$BASE_STACK', gpu='${GPU_INSTANCE_ID:-none}')"
aws cloudformation deploy \
  --region "$REGION" \
  --stack-name "$STACK" \
  --template-file "$HERE/serverless.yaml" \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides \
    BaseStackName="$BASE_STACK" \
    LambdaCodeS3Bucket="$ARTIFACT_BUCKET" \
    LambdaCodeS3Key="$KEY" \
    GpuInstanceId="$GPU_INSTANCE_ID" \
    ResultBucket="$RESULT_BUCKET" \
    ResultKmsKeyArn="$RESULT_KMS_ARN"

echo "--- outputs ---"
aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query 'Stacks[0].Outputs[].{Key:OutputKey,Value:OutputValue}' --output table

echo
echo "Next:"
echo "  1. issue a key:   KEYS_TABLE=${STACK}-keys python -m app.admin_keys issue --label team-1"
echo "     (run from gateway/, with operator AWS creds)"
echo "  2. point your client: set MINIMAX_BASE_URL to the ApiEndpoint above"
