#!/usr/bin/env bash
# EC2 user data for the SKF Skill Studio app host (Ubuntu 24.04 LTS, x86_64 or arm64). Runs once, as root,
# on first boot. Installs Docker Engine + the compose plugin, the AWS CLI, jq and the ECR credential helper,
# and prepares /opt/skf-studio for infra/aws/deploy.sh. Caddy is not installed on the host: it runs as the
# `caddy` container in compose.prod.yaml, so its config and version ship with each release.
#
# Check the values below before launching (see infra/aws/README.md, step 8).
set -euo pipefail

AWS_REGION=eu-north-1          # region of the SSM parameters, S3 bucket and ECR registry
DEPLOY_USER=ubuntu
APP_DIR=/opt/skf-studio

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q ca-certificates curl gnupg jq rsync unzip unattended-upgrades amazon-ecr-credential-helper

# Container logs are also capped per service in compose.prod.yaml; this covers anything run by hand.
install -m 0755 -d /etc/docker
cat >/etc/docker/daemon.json <<'JSON'
{
  "log-driver": "json-file",
  "log-opts": {"max-size": "50m", "max-file": "5"},
  "live-restore": true
}
JSON

# Docker Engine and the compose plugin from Docker's apt repository (Ubuntu's docker.io lags behind).
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
# shellcheck source=/dev/null
. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  >/etc/apt/sources.list.d/docker.list
apt-get update -q
apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
usermod -aG docker "$DEPLOY_USER"

# AWS CLI v2 (the snap is published by AWS).
snap install aws-cli --classic

cat >/etc/profile.d/skf-studio.sh <<EOF
export AWS_REGION=$AWS_REGION
export APP_DIR=$APP_DIR
EOF

install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" -m 0750 "$APP_DIR" "$APP_DIR/bin" "$APP_DIR/env"

# Nightly database backup; a no-op when Postgres is RDS (see bin/backup-db.sh).
cat >/etc/cron.d/skf-studio-backup <<EOF
PATH=/snap/bin:/usr/local/bin:/usr/bin:/bin
AWS_REGION=$AWS_REGION
15 2 * * * $DEPLOY_USER [ -x $APP_DIR/bin/backup-db.sh ] && $APP_DIR/bin/backup-db.sh 2>&1 | logger -t skf-backup
EOF
chmod 0644 /etc/cron.d/skf-studio-backup

echo "skf-studio: host ready; deploy with infra/aws/deploy.sh" | logger -t skf-user-data
