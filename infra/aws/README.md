# Deploying SKF Skill Studio on AWS

One EC2 app host runs the whole application with Docker Compose behind Caddy (automatic TLS). State lives in
managed services: Postgres on RDS (recommended) and artifacts in S3. The worker on the app host drives the GPU
training box over SSH and Kaggle over its API, and runs CPU evaluation stages itself.

```
                    Internet ── 443/80 ──►  EC2 app host (public subnet, Elastic IP, IAM instance role)
                                            ┌──────────────────────────────────────────────────────────┐
  staff browsers ── https://studio.… ──►    │ caddy ─► web (Next.js) ─► api (FastAPI) ◄─ worker (arq)  │
  Kaggle / AWS jobs ── /ingest/* ──────►    │   └────────────────────────► api        redis            │
                                            └──────┬────────────┬───────────────┬──────────┬───────────┘
                                                   │ 5432       │ S3 (role)     │ SSM, ECR │ SSH 22, EC2 API
                                            RDS Postgres 17   artifacts bucket  config    g1-train* GPU box
                                            (private subnets) (versioned)                 (start/stop only)
```

Only `/` (web) and `/ingest/*` (job phone-home, per-stage tokens) are reachable from the internet; `/api/v1`,
`/docs` and `/openapi.json` are blocked at Caddy and the API port is never published.

| File | Purpose |
|---|---|
| `infra/compose.prod.yaml` | services on the host: caddy, web, web-migrate, api, migrate, worker, redis, optional postgres |
| `infra/Caddyfile` | TLS, routing, security headers |
| `infra/aws/user-data.sh` | first-boot setup of the host: Docker + compose plugin, AWS CLI, jq, ECR credential helper, backup cron |
| `infra/aws/deploy.sh` | run from a workstation: sync files, roll to an image tag, roll back |
| `infra/aws/host/*.sh` | run on the host: `release.sh` (pull, migrate, start, wait healthy), `render-env.sh` (SSM → env files), `backup-db.sh` |
| `infra/aws/iam/app-host-policy.json` | least-privilege instance role policy |
| `infra/aws/s3/*.json` | bucket lifecycle and TLS-only bucket policy |

Region: the examples use `eu-north-1` (Stockholm, closest to Gothenburg). The existing GPU box lives in
`us-east-1`; cross-region works, see step 2.

## 0. Prerequisites and variables

An AWS account with admin access for the setup (AWS CLI v2), a DNS name you control, Docker with buildx on the
machine that builds images, and `envsubst` (gettext). Set these once per shell:

```bash
export AWS_REGION=eu-north-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ARTIFACT_BUCKET=skf-studio-artifacts-$AWS_ACCOUNT_ID
export SKF_DOMAIN=studio.example.com         # the public hostname
export ADMIN_CIDR=203.0.113.10/32            # your office/VPN IP, for SSH to the app host
export REGISTRY=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

## 1. Network (VPC and subnets)

A dedicated VPC keeps the app isolated; the default VPC also works for a pilot.

```bash
VPC_ID=$(aws ec2 create-vpc --cidr-block 10.40.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=skf-studio},{Key=Project,Value=skf-studio}]' \
  --query Vpc.VpcId --output text)
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames
IGW_ID=$(aws ec2 create-internet-gateway --query InternetGateway.InternetGatewayId --output text)
aws ec2 attach-internet-gateway --vpc-id $VPC_ID --internet-gateway-id $IGW_ID

# public subnet for the app host
PUBLIC_SUBNET=$(aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.40.0.0/24 --availability-zone ${AWS_REGION}a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=skf-studio-public-a}]' --query Subnet.SubnetId --output text)
RTB_ID=$(aws ec2 create-route-table --vpc-id $VPC_ID --query RouteTable.RouteTableId --output text)
aws ec2 create-route --route-table-id $RTB_ID --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID
aws ec2 associate-route-table --route-table-id $RTB_ID --subnet-id $PUBLIC_SUBNET

# two private subnets (two AZs) for RDS; no route to the internet
PRIVATE_A=$(aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.40.10.0/24 --availability-zone ${AWS_REGION}a --query Subnet.SubnetId --output text)
PRIVATE_B=$(aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.40.11.0/24 --availability-zone ${AWS_REGION}b --query Subnet.SubnetId --output text)
```

## 2. Security groups

```bash
APP_SG=$(aws ec2 create-security-group --vpc-id $VPC_ID --group-name skf-studio-app \
  --description "SKF Skill Studio app host" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $APP_SG --ip-permissions \
  'IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0}],Ipv6Ranges=[{CidrIpv6=::/0}]' \
  'IpProtocol=udp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0}],Ipv6Ranges=[{CidrIpv6=::/0}]' \
  'IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0}],Ipv6Ranges=[{CidrIpv6=::/0}]' \
  "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$ADMIN_CIDR,Description=admin}]"

DB_SG=$(aws ec2 create-security-group --vpc-id $VPC_ID --group-name skf-studio-db \
  --description "SKF Skill Studio Postgres" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $DB_SG --protocol tcp --port 5432 --source-group $APP_SG
```

Port 80 is needed for the ACME HTTP challenge and the HTTP→HTTPS redirect; UDP 443 carries HTTP/3.
If you use SSM Session Manager (step 5) you can drop the SSH rule entirely.

**Training box.** The worker SSHes into the GPU box, so its security group must allow TCP 22 from the app host:
- same region and VPC: `aws ec2 authorize-security-group-ingress --group-id <training-sg> --protocol tcp --port 22 --source-group $APP_SG`
- another region or VPC (like today's `g1-train` in `us-east-1`): allow the app host's Elastic IP (step 8):
  `aws ec2 authorize-security-group-ingress --region us-east-1 --group-id sg-01c410eb3570de544 --protocol tcp --port 22 --cidr <EIP>/32`

## 3. S3 artifact bucket

```bash
aws s3api create-bucket --bucket $ARTIFACT_BUCKET --create-bucket-configuration LocationConstraint=$AWS_REGION
aws s3api put-public-access-block --bucket $ARTIFACT_BUCKET --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-ownership-controls --bucket $ARTIFACT_BUCKET \
  --ownership-controls 'Rules=[{ObjectOwnership=BucketOwnerEnforced}]'
aws s3api put-bucket-versioning --bucket $ARTIFACT_BUCKET --versioning-configuration Status=Enabled
aws s3api put-bucket-encryption --bucket $ARTIFACT_BUCKET --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'
aws s3api put-bucket-lifecycle-configuration --bucket $ARTIFACT_BUCKET --lifecycle-configuration file://infra/aws/s3/lifecycle.json
envsubst < infra/aws/s3/bucket-policy.json > /tmp/skf-bucket-policy.json
aws s3api put-bucket-policy --bucket $ARTIFACT_BUCKET --policy file:///tmp/skf-bucket-policy.json
```

Lifecycle (`s3/lifecycle.json`): incomplete multipart uploads are aborted after 7 days; overwritten or deleted
object versions are kept 30 days (3 newest always kept); raw job logs (`log` artifacts, which the API uploads with the object
tag `skf-kind=log`) move to Standard-IA after 30 days and expire after 180; database dumps under `backups/postgres/` expire after 35 days.
Browsers fetch artifacts with presigned URLs (15 minutes); the bucket itself stays private.

## 4. Container registry (ECR)

```bash
for repo in api web web-tools worker; do
  aws ecr create-repository --repository-name skf-studio/$repo \
    --image-scanning-configuration scanOnPush=true --image-tag-mutability IMMUTABLE >/dev/null
  aws ecr put-lifecycle-policy --repository-name skf-studio/$repo --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"description":"keep the last 30 images","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":30},"action":{"type":"expire"}}]}' >/dev/null
done
```

Tags are immutable: every release is a git SHA, and rollback is redeploying an older SHA.

## 5. IAM role and instance profile

```bash
aws iam create-role --role-name skf-studio-app-host \
  --assume-role-policy-document file://infra/aws/iam/app-host-trust.json
envsubst < infra/aws/iam/app-host-policy.json > /tmp/skf-app-host-policy.json
aws iam put-role-policy --role-name skf-studio-app-host --policy-name skf-studio-app-host \
  --policy-document file:///tmp/skf-app-host-policy.json
# optional but recommended: shell access through SSM Session Manager instead of SSH
aws iam attach-role-policy --role-name skf-studio-app-host \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam create-instance-profile --instance-profile-name skf-studio-app-host
aws iam add-role-to-instance-profile --instance-profile-name skf-studio-app-host --role-name skf-studio-app-host
```

What the policy allows, and nothing else:
- S3: list the artifact bucket; read, write, delete and tag objects in it.
- EC2: `Describe*` everywhere; `StartInstances`/`StopInstances` only on instances tagged `Project=skf-studio`
  or with a `Name` starting `g1-train`. No `RunInstances`, so `create_if_missing` must stay `false`.
- SSM: read parameters under `/skf-studio/`. SecureStrings encrypted with the AWS-managed `aws/ssm` key need no
  extra KMS permission; with a customer-managed key, add `kms:Decrypt` on that key.
- ECR: pull `skf-studio/*` images.

If the training box's EBS volume is encrypted with a customer-managed KMS key, starting it also needs
`kms:CreateGrant` on that key.

## 6. Database

**Recommended: RDS for PostgreSQL 17** (automated backups, point-in-time recovery, minor-version patching).

```bash
aws rds create-db-subnet-group --db-subnet-group-name skf-studio \
  --db-subnet-group-description "SKF Skill Studio" --subnet-ids $PRIVATE_A $PRIVATE_B
aws rds create-db-instance --db-instance-identifier skf-studio \
  --engine postgres --engine-version 17 --db-instance-class db.t4g.small \
  --allocated-storage 20 --max-allocated-storage 200 --storage-type gp3 --storage-encrypted \
  --master-username skf --manage-master-user-password \
  --db-name skf --db-subnet-group-name skf-studio --vpc-security-group-ids $DB_SG --no-publicly-accessible \
  --backup-retention-period 14 --copy-tags-to-snapshot --deletion-protection \
  --auto-minor-version-upgrade --tags Key=Project,Value=skf-studio
```

RDS keeps the master password in Secrets Manager; read it once to build the connection URLs for step 7
(`aws secretsmanager get-secret-value --secret-id <arn from describe-db-instances MasterUserSecret>`).
URL-encode the password. After the app host is up (step 8), create the two schemas once:

```bash
ssh ubuntu@$SKF_DOMAIN 'docker run --rm -i postgres:17-alpine psql "postgresql://skf:<password>@<rds-endpoint>:5432/skf?sslmode=require"' \
  <<'SQL'
CREATE SCHEMA IF NOT EXISTS auth AUTHORIZATION skf;
CREATE SCHEMA IF NOT EXISTS app AUTHORIZATION skf;
SQL
```

**Alternative: Postgres in a container on the app host** (cheaper, single point of failure). Set
`/skf-studio/host/COMPOSE_PROFILES=localdb` and `/skf-studio/postgres/POSTGRES_PASSWORD`; the container runs
`infra/postgres/init.sql` on first start, its data lives in the `pgdata` volume, and `bin/backup-db.sh` dumps it
to S3 every night at 02:15 UTC. Use `postgres:5432` as the host in the URLs below.

## 7. Configuration and secrets (SSM Parameter Store)

The host renders env files from these parameters on every deploy (`host/render-env.sh`); the last path segment
is the variable name. Use `SecureString` for secrets. Values must not contain `'`, `\` or newlines.

| Parameter | Type | Value |
|---|---|---|
| `/skf-studio/host/SKF_DOMAIN` | String | `studio.example.com` |
| `/skf-studio/host/S3_ASSET_ORIGIN` | String | `https://<bucket>.s3.<region>.amazonaws.com` (CSP for videos/images) |
| `/skf-studio/host/SKF_IMAGE_REGISTRY` | String | `<account>.dkr.ecr.<region>.amazonaws.com` |
| `/skf-studio/host/BACKUP_BUCKET` | String | the artifact bucket (only used with `localdb`) |
| `/skf-studio/host/COMPOSE_PROFILES` | String | `localdb` only when not using RDS |
| `/skf-studio/host/SKF_WORKER_CPUS` | String | optional CPU cap for the worker container (default `3`) |
| `/skf-studio/api/DATABASE_URL` | SecureString | `postgresql+asyncpg://skf:<pw>@<rds-endpoint>:5432/skf?ssl=require` |
| `/skf-studio/api/SECRET_KEY` | SecureString | `openssl rand -base64 32` |
| `/skf-studio/api/S3_BUCKET` | String | the artifact bucket |
| `/skf-studio/api/S3_REGION` | String | `eu-north-1` |
| `/skf-studio/api/PUBLIC_INGEST_URL` | String | `https://studio.example.com/ingest/v1` |
| `/skf-studio/web/DATABASE_URL` | SecureString | `postgresql://skf:<pw>@<rds-endpoint>:5432/skf?sslmode=require&uselibpqcompat=true` |
| `/skf-studio/web/BETTER_AUTH_SECRET` | SecureString | `openssl rand -base64 32` |
| `/skf-studio/web/BETTER_AUTH_URL` | String | `https://studio.example.com` |
| `/skf-studio/web/ADMIN_EMAIL` | String | first admin (seed only) |
| `/skf-studio/web/ADMIN_PASSWORD` | SecureString | first admin's initial password; delete after step 10 |
| `/skf-studio/postgres/POSTGRES_PASSWORD` | SecureString | only with `localdb` |

`S3_ENDPOINT_URL`, `S3_PUBLIC_ENDPOINT_URL` and the S3 keys stay unset: the SDK uses real S3 and the instance
role. Everything in-network (Redis, JWKS URL, internal ingest URL, paths) is fixed in `compose.prod.yaml`.

```bash
put() { aws ssm put-parameter --name "$1" --type "${3:-String}" --value "$2" --overwrite >/dev/null; }
put /skf-studio/host/SKF_DOMAIN "$SKF_DOMAIN"
put /skf-studio/api/SECRET_KEY "$(openssl rand -base64 32)" SecureString
put /skf-studio/web/BETTER_AUTH_SECRET "$(openssl rand -base64 32)" SecureString
# ... and the rest of the table
```

## 8. Launch the app host

`m7i.xlarge` (4 vCPU, 16 GiB) is the minimum: CPU evaluation stages run on this host (the worker is capped at
3 CPUs by default, `/skf-studio/host/SKF_WORKER_CPUS`). Use `c7i.2xlarge` if evaluations queue up.

```bash
AMI=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text)
# check AWS_REGION at the top of user-data.sh first
INSTANCE_ID=$(aws ec2 run-instances --image-id $AMI --instance-type m7i.xlarge \
  --subnet-id $PUBLIC_SUBNET --security-group-ids $APP_SG --key-name <your-key-pair> \
  --iam-instance-profile Name=skf-studio-app-host \
  --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=80,VolumeType=gp3,Encrypted=true,DeleteOnTermination=false}' \
  --user-data file://infra/aws/user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=skf-studio-app},{Key=Project,Value=skf-studio}]' \
    'ResourceType=volume,Tags=[{Key=Name,Value=skf-studio-app},{Key=Project,Value=skf-studio}]' \
  --query 'Instances[0].InstanceId' --output text)
EIP_ALLOC=$(aws ec2 allocate-address --query AllocationId --output text)
aws ec2 associate-address --instance-id $INSTANCE_ID --allocation-id $EIP_ALLOC
aws ec2 describe-addresses --allocation-ids $EIP_ALLOC --query 'Addresses[0].PublicIp' --output text
```

`HttpPutResponseHopLimit=2` matters: containers reach the instance role through IMDSv2 one network hop further
than the host, and with the default of 1 the SDK in the api/worker containers finds no credentials.
The `Project=skf-studio` tag on the app host also lets its own role stop/start it, which is harmless.
Wait for `cloud-init status --wait` on the host to finish (about 3 minutes).

## 9. DNS

Create an `A` record `studio.example.com → <Elastic IP>` (Route 53 or SKF's DNS). Caddy requests the
certificate on first start, so the record must resolve before the first deploy.

## 10. Build, push and deploy

```bash
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY
make push-images REGISTRY=$REGISTRY                       # TAG defaults to the current git SHA (12 chars)
SKF_DEPLOY_HOST=ubuntu@$SKF_DOMAIN infra/aws/deploy.sh "$(git rev-parse --short=12 HEAD)"
```

`deploy.sh` copies `compose.prod.yaml`, `Caddyfile`, `postgres/init.sql` and `host/*.sh` to `/opt/skf-studio`,
then runs `bin/release.sh <tag>` on the host: render env files from SSM, pull images (ECR credential helper, no
stored password), run both migrations (`migrate`, `web-migrate`), start everything and wait until healthy.

First deploy only: create the default compute targets and the first admin, then remove its password from SSM.

```bash
ssh ubuntu@$SKF_DOMAIN 'cd /opt/skf-studio && c="docker compose -f compose.prod.yaml --env-file deploy.env" &&
  $c exec api skf-api seed-targets && $c run --rm web-migrate pnpm seed:admin'
aws ssm delete-parameter --name /skf-studio/web/ADMIN_PASSWORD
```

Skills are loaded from the image at API startup. The seeded Kaggle and AWS targets start disabled: add their
credentials and enable them in **Compute**, see
[RUNBOOK §3](../../docs/webapp/RUNBOOK.md#3-compute-targets-and-their-secrets).

Smoke checks:

```bash
curl -fsS https://$SKF_DOMAIN/api/auth/ok                            # {"ok":true}
curl -s -o /dev/null -w '%{http_code}\n' https://$SKF_DOMAIN/api/v1/me   # 404: internal API not exposed
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  https://$SKF_DOMAIN/ingest/v1/stages/00000000-0000-0000-0000-000000000000/heartbeat   # 401: ingest reachable
```

## 11. Updates and rollback

- **Release:** `make push-images REGISTRY=$REGISTRY && infra/aws/deploy.sh <new-sha>`. Idempotent; re-running
  with the live tag only re-renders config and recreates what changed (use it after changing SSM parameters).
- **Rollback:** `infra/aws/deploy.sh --rollback` switches to the previously live tag (running it twice switches
  back). `infra/aws/deploy.sh --status` shows the live and previous tags and container health.
- **Migrations are forward-only.** An image rollback does not downgrade the schema, so migrations must stay
  backward-compatible with the previous release (add columns first, drop them a release later). If a release's
  migration must be undone, restore the pre-deploy snapshot (next section) and redeploy the old tag.
- **Long runs survive deploys:** remote stages keep running on Kaggle/AWS and are reconciled by the new worker;
  local CPU stages in progress fail with "worker restarted" and can be retried. Deploy when the dashboard shows no
  active local stages.
- **Host OS:** `unattended-upgrades` applies security updates; reboot in a quiet window
  (`sudo reboot`; restart policies bring the stack back).

## 12. Backups and restore

| Data | Protection | Restore |
|---|---|---|
| RDS | automated backups, 14 days PITR; take a manual snapshot before releases with migrations (`aws rds create-db-snapshot`) | `aws rds restore-db-instance-to-point-in-time`, then point `DATABASE_URL` at the new endpoint and redeploy |
| Containerised Postgres | nightly `pg_dump` to `s3://<bucket>/backups/postgres/` (35 days) | `bin/backup-db.sh --restore s3://<bucket>/backups/postgres/<file>.dump` on the host |
| Artifacts | S3 versioning; old versions kept 30 days | `aws s3api list-object-versions`, copy the version back |
| Configuration | SSM parameter history | `aws ssm get-parameter-history` |
| TLS certificates | `caddy_data` volume; re-issued automatically if lost | nothing to do |

Rehearse a restore into a scratch RDS instance once per quarter.

## 13. Operations

- Logs: `ssh ubuntu@$SKF_DOMAIN 'cd /opt/skf-studio && docker compose -f compose.prod.yaml --env-file deploy.env logs -f --tail 200 api worker'`
  (JSON logs are rotated at 5 × 50 MB per container).
- Health: `infra/aws/deploy.sh --status`; the API's `/readyz` checks Postgres and Redis.
- Costs (eu-north-1, on demand): m7i.xlarge ≈ $0.20/h, db.t4g.small ≈ $0.03/h, S3 and data transfer by
  volume. The GPU box is billed only while running; the worker stops it when idle.

## 14. Teardown

Stop and terminate the app host, release the Elastic IP, delete the RDS instance (final snapshot), then the ECR
repositories, the bucket (after emptying all versions), the SSM parameters under `/skf-studio/`, the IAM role and
instance profile, the security groups and the VPC.
