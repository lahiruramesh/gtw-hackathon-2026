#!/usr/bin/env bash
# Nightly logical backup of the containerised Postgres to S3 (cron, installed by user-data.sh).
#
# Only does work with COMPOSE_PROFILES=localdb; with RDS, automated backups and snapshots cover this and
# the script exits quietly. Dumps land in s3://$BACKUP_BUCKET/backups/postgres/ (35-day lifecycle rule).
#
#   bin/backup-db.sh                    dump now
#   bin/backup-db.sh --restore <s3-uri> restore a dump into the running postgres (stops api/worker/web first)
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/skf-studio}
cd "$APP_DIR"
compose() { docker compose -f compose.prod.yaml --env-file deploy.env "$@"; }
setting() { sed -n "s/^$1='\(.*\)'$/\1/p" deploy.env; }

if [[ $(setting COMPOSE_PROFILES) != *localdb* ]]; then
  echo "backup-db: Postgres is external (RDS); nothing to do"
  exit 0
fi
bucket=$(setting BACKUP_BUCKET)
[[ -n $bucket ]] || { echo "backup-db: BACKUP_BUCKET is not set in /skf-studio/host" >&2; exit 1; }

if [[ ${1:-} == --restore ]]; then
  src=${2:?usage: $0 --restore s3://bucket/backups/postgres/<file>.dump}
  compose stop web api worker
  aws s3 cp "$src" - | compose exec -T postgres pg_restore -U skf -d skf --clean --if-exists --no-owner
  compose up -d --wait
  echo "backup-db: restored $src"
  exit 0
fi

key="backups/postgres/skf-$(date -u +%Y%m%dT%H%M%SZ).dump"
compose exec -T postgres pg_dump -U skf -d skf --format=custom \
  | aws s3 cp - "s3://$bucket/$key" --sse AES256 --only-show-errors
echo "backup-db: wrote s3://$bucket/$key"
