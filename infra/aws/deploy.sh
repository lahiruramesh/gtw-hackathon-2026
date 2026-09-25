#!/usr/bin/env bash
# Deploy SKF Skill Studio to the EC2 app host (run from an admin workstation or CI). Idempotent.
#
#   infra/aws/deploy.sh <image-tag>      sync compose + Caddyfile + host scripts, then roll to <image-tag>
#   infra/aws/deploy.sh --rollback       roll back to the tag that was live before the current one
#   infra/aws/deploy.sh --status         show live/previous tags and container health
#
# Environment:
#   SKF_DEPLOY_HOST   ssh destination of the app host, e.g. ubuntu@studio.example.com (required)
#   SKF_SSH_KEY       private key for that host (optional; default ssh config / agent)
#
# Images must already be in the registry (make push-images REGISTRY=... TAG=...). Configuration and secrets
# are read on the host from SSM Parameter Store; nothing secret passes through this script.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HOST=${SKF_DEPLOY_HOST:?set SKF_DEPLOY_HOST, e.g. ubuntu@studio.example.com}
APP_DIR=/opt/skf-studio
ssh_opts=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
[[ -n ${SKF_SSH_KEY:-} ]] && ssh_opts+=(-i "$SKF_SSH_KEY")

case "${1:-}" in
  --status | --rollback) action=$1 ;;
  "" | -*) echo "usage: $0 <image-tag> | --rollback | --status" >&2; exit 2 ;;
  *) action=$1 ;;
esac

if [[ $action != --status ]]; then
  rsync -az -e "ssh ${ssh_opts[*]}" \
    --include compose.prod.yaml --include Caddyfile --include 'postgres/' --include 'postgres/init.sql' \
    --exclude '*' "$ROOT/infra/" "$HOST:$APP_DIR/"
  # host scripts are 0755 in git and -a keeps the mode (macOS openrsync has no --chmod)
  rsync -az --delete-after -e "ssh ${ssh_opts[*]}" "$ROOT/infra/aws/host/" "$HOST:$APP_DIR/bin/"
fi

# Login shell so /etc/profile.d/skf-studio.sh (AWS_REGION) is loaded. ssh joins its arguments into one
# string for the remote shell, so the command is quoted twice: once for bash -c, once for the remote shell.
release_cmd=$(printf '%q ' "$APP_DIR/bin/release.sh" "$action")
# shellcheck disable=SC2029  # expanding on the client is the point: the quoted command is built here
ssh "${ssh_opts[@]}" "$HOST" "bash -lc $(printf '%q' "$release_cmd")"
