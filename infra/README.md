# infra — network, security, and (approval-gated) compute

CloudFormation for self-hosting MiniMax-H3 in your own AWS account (region defaults to
`us-west-2`, override with `AWS_REGION`),
built so that **a base deploy costs ~nothing and launching the GPU is an explicit,
separate, approved step**.

Two templates:

| template | what | public surface |
|---|---|---|
| [`template.yaml`](template.yaml) | **base**: private VPC, locked-down SGs, private result bucket, least-priv IAM, KMS, (optional) GPU + gateway EC2 | **none** |
| [`serverless.yaml`](serverless.yaml) | **API layer**: HTTP API + **Lambda authorizer (validates the self-signed key)** + submit/poll Lambda + DynamoDB (keys+tasks) + DynamoDB VPC endpoint + EventBridge autostop Lambda | **only the HTTPS API Gateway**, and every route requires a valid key |

The serverless layer is the recommended shape (see [`../docs/PLAN.md`](../docs/PLAN.md) §2.1):
the **only thing exposed to the internet is API Gateway over TLS**, and it lets nothing
through without the Lambda authorizer approving the caller's `mmh3_` key. Admin key
issuance is an IAM-gated CLI (`app.admin_keys`), not a public route.

## Security model (the hard requirement: no public ports)

Enforced structurally in [`template.yaml`](template.yaml), not by convention — see
[`../docs/PLAN.md`](../docs/PLAN.md) §7:

- **SGLang `:30010` is never public.** Its security-group ingress references the
  **gateway's security group** (`SourceSecurityGroupId`), not a CIDR. There is no
  `0.0.0.0/0` rule anywhere in the template.
- **The GPU box has no public IP** (private subnet, `MapPublicIpOnLaunch: false`)
  and **no SSH** — no key pair, no port 22. Operate it via **SSM Session Manager**
  (`AmazonSSMManagedInstanceCore` on the instance role).
- **The gateway SG defines no ingress by default.** The client-facing entrypoint
  (PrivateLink / internal ALB + private DNS / 443+WAF+Bearer) is added deliberately
  as a decided step (§7.2), never a blanket open port.
- **Result bucket is private:** Block Public Access (all four), KMS-encrypted,
  versioned, `Retain`, TLS-only bucket policy, 7-day expiry. Clients only ever get
  short-lived **presigned** URLs.
- **Least-privilege IAM:** `ec2:Start/StopInstances` is scoped by
  `aws:ResourceTag/Project = openminimax`, so the controller can touch only this
  project's box. S3/KMS actions are scoped to this bucket/key.
- **IMDSv2 required**, EBS KMS-encrypted, egress via a **NAT gateway** (created only
  when compute exists) + an **S3 gateway endpoint** (keeps result traffic off NAT).

## Cost tiers (deploy.sh)

| command | creates | ~cost |
|---|---|---|
| `./deploy.sh base` | VPC, subnets, SGs, IAM, KMS, empty private bucket. **No NAT, no compute.** | ~$0 |
| `./deploy.sh gateway` | + t3.small controller + NAT gateway | ~$15/mo + ~$32/mo NAT |
| `./deploy.sh gpu APPROVE` | + **g6e.12xlarge** (4×L40S) | **~$10/hr** |

The `gpu` tier **refuses to run without the literal `APPROVE` argument**, so nobody
starts a ~$10/hr instance by accident. It resolves the current Deep Learning OSS
Nvidia AMI from SSM only at that point.

```bash
export AWS_REGION=us-west-2
./deploy.sh base                 # safe, free scaffolding — validated, reversible
# ...when you're ready to measure Phase 0 (docs/PLAN.md §3):
./deploy.sh gpu APPROVE          # launches the GPU box (~$10/hr) — approved spend
```

## Serverless API layer (deploy_serverless.sh)

Deploy on top of the base stack. Packages `gateway/app` + `gateway/lambdas` (stdlib +
boto3 only — no vendored deps), uploads to an artifact bucket, and creates the HTTP
API + authorizer + DynamoDB + autostop:

```bash
export AWS_REGION=us-west-2
./deploy.sh base                                   # base network/IAM/bucket first
./deploy_serverless.sh <your-artifact-bucket>      # API layer, no GPU yet
# later, once the GPU instance exists (deploy.sh gpu APPROVE), enable autostop:
./deploy_serverless.sh <your-artifact-bucket> <GpuInstanceId>
```

Output `ApiEndpoint` is what a client's `MINIMAX_BASE_URL` points at. The
authorizer caches results ~300s (tunable via `AuthorizerCacheTtlSeconds`) so a
client's frequent polling doesn't verify against DynamoDB each time.

## Scale-to-zero (the real cost lever, docs/PLAN.md §2)

Two implementations of the same rule (start on queued work; stop after idle with
nothing queued **and** nothing running); pick the one matching your path:

- **serverless** — [`lambdas/autostop.py`](../gateway/lambdas/autostop.py), invoked by
  an EventBridge rule every minute (in `serverless.yaml`). No always-on box.
- **single-box** — [`autostop_controller.py`](autostop_controller.py) as a systemd
  unit on a t3.small ([`openminimax-autostop.service`](openminimax-autostop.service)).

Both measure idle on **queue state, not CPU**, so they can never stop the box
mid-generation — a running task counts as work. The video path is async
and tolerates the GPU cold start. Verified against the real
task schema.

## Validation

```bash
cfn-lint template.yaml serverless.yaml       # both clean (0 findings)
aws cloudformation validate-template --region us-west-2 --template-body file://template.yaml
aws cloudformation validate-template --region us-west-2 --template-body file://serverless.yaml
```

All pass. `cfn-guard` (compliance) can be added when available; the secure defaults
above already cover the S3/IMDSv2/encryption/DynamoDB-SSE checks it would assert.

## After launch — wiring a client (Phase 3, docs/PLAN.md §3)

**Serverless path (recommended):**
1. `./deploy.sh base` then `./deploy_serverless.sh <artifact-bucket>` → note `ApiEndpoint`.
2. `./deploy.sh gpu APPROVE` → note `GpuInstanceId`; re-run `deploy_serverless.sh` with
   it to enable autostop. Ship this repo to the box (scp/rsync/tar over SSM — **no git
   needed on the box**) and run the one-shot deploy:
   `sudo RESULT_BUCKET=<result-bucket> bash serving/bootstrap.sh`. It installs deps +
   weights, deploys `gateway/`+`serving/` to `/opt/openminimax`, writes the account values
   to `/etc/openminimax.env`, and **installs + `enable`s both systemd units so they
   auto-start on every boot** — the **diffusers Turbo shim**
   ([`openminimax-serve.service`](openminimax-serve.service) → `serving/h3_turbo_server.py`,
   binds the private `:30010`; NOT the deprecated `serve_h3.sh`, which serves noise —
   see [`../serving/README.md`](../serving/README.md)) and the worker
   ([`openminimax-worker.service`](openminimax-worker.service), `SGLANG_URL`/**`SGLANG_STEPS=9`**
   inline, `RESULT_BUCKET`/tables from `/etc/openminimax.env`). After a stop/start the box
   comes back request-ready with zero manual steps. First `start`:
   `sudo systemctl start openminimax-serve openminimax-worker`.
3. Issue keys: `KEYS_TABLE=openminimax-serverless-keys python -m app.admin_keys issue --label team-1`
   (run from `gateway/`, operator AWS creds).
4. Point your client at it by setting `MINIMAX_BASE_URL=<ApiEndpoint>`. See [`../docs/API.md`](../docs/API.md).

**Single-box fallback:** run the FastAPI gateway on the t3.small
([`openminimax-gateway.service`](openminimax-gateway.service)) with `SGLANG_URL`,
`RESULT_BUCKET`, `ADMIN_TOKEN`; front it with your own private ingress.
