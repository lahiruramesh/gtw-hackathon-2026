#!/usr/bin/env bash
# Render the app host's configuration from SSM Parameter Store (runs on the host, as the deploy user).
#
#   /skf-studio/host/*      -> deploy.env        compose interpolation: domain, registry, profiles, backup bucket
#   /skf-studio/api/*       -> env/api.env       api, migrate, worker
#   /skf-studio/web/*       -> env/web.env       web, web-migrate
#   /skf-studio/postgres/*  -> env/postgres.env  only with COMPOSE_PROFILES=localdb
#
# Each parameter's last path segment is the variable name. Files are written 0600 and replaced atomically,
# so a failed read never leaves a half-written env file behind.
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/skf-studio}
SSM_PREFIX=${SKF_SSM_PREFIX:-/skf-studio}
: "${AWS_REGION:?AWS_REGION must be set (see /etc/profile.d/skf-studio.sh)}"

env_lines() {  # env_lines <ssm-path>: NAME='value' per parameter
  aws ssm get-parameters-by-path --region "$AWS_REGION" --path "$1" --with-decryption --recursive --output json \
    | jq -r '.Parameters[] | [(.Name | split("/") | last), .Value] | @tsv' \
    | while IFS=$'\t' read -r name value; do
        # compose's env-file parser has no escapes inside single quotes, and @tsv escapes tabs, newlines and
        # backslashes with a backslash, so refuse both rather than write a value that parses differently.
        case $value in
          *\'* | *\\*) echo "render-env: $1/$name contains a quote, backslash or newline" >&2; exit 1 ;;
        esac
        printf "%s='%s'\n" "$name" "$value"
      done
}

render() {  # render <ssm-path> <output-file>
  local content tmp
  content=$(env_lines "$1")
  tmp=$(mktemp "$2.XXXXXX")
  chmod 600 "$tmp"
  printf '%s\n' "$content" >"$tmp"
  mv "$tmp" "$2"
}

mkdir -p "$APP_DIR/env"
render "$SSM_PREFIX/host" "$APP_DIR/deploy.env"
render "$SSM_PREFIX/api" "$APP_DIR/env/api.env"
render "$SSM_PREFIX/web" "$APP_DIR/env/web.env"
render "$SSM_PREFIX/postgres" "$APP_DIR/env/postgres.env"

for required in SKF_DOMAIN S3_ASSET_ORIGIN SKF_IMAGE_REGISTRY; do
  grep -q "^$required=" "$APP_DIR/deploy.env" || { echo "render-env: missing $SSM_PREFIX/host/$required" >&2; exit 1; }
done
echo "render-env: wrote deploy.env and env/{api,web,postgres}.env from $SSM_PREFIX"
