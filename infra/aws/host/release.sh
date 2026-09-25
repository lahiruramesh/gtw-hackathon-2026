#!/usr/bin/env bash
# Roll the app host to an image tag (runs on the host; infra/aws/deploy.sh calls it over SSH).
#
#   bin/release.sh <image-tag>     render config from SSM, pull, migrate, start, wait until healthy
#   bin/release.sh --rollback      the same with the tag that was live before the current one
#   bin/release.sh --status        print the live and previous tags and container health
#
# Idempotent: re-running with the live tag re-renders config and recreates only what changed.
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/skf-studio}
STATE_DIR=$APP_DIR/.deploy
cd "$APP_DIR"
mkdir -p "$STATE_DIR"

compose() { docker compose -f compose.prod.yaml --env-file deploy.env "$@"; }
current_tag() { cat "$STATE_DIR/current_tag" 2>/dev/null || true; }

case "${1:-}" in
  --status)
    echo "live:     $(current_tag)"
    echo "previous: $(cat "$STATE_DIR/previous_tag" 2>/dev/null || true)"
    compose ps
    exit 0 ;;
  --rollback)
    tag=$(cat "$STATE_DIR/previous_tag" 2>/dev/null) || { echo "release: no previous tag to roll back to" >&2; exit 1; } ;;
  "" | -*)
    echo "usage: $0 <image-tag> | --rollback | --status" >&2; exit 2 ;;
  *)
    tag=$1 ;;
esac
[[ $tag =~ ^[A-Za-z0-9._-]+$ ]] || { echo "release: invalid image tag '$tag'" >&2; exit 2; }

"$APP_DIR/bin/render-env.sh"
printf "SKF_IMAGE_TAG='%s'\n" "$tag" >>deploy.env

# Pull from ECR through the instance role (amazon-ecr-credential-helper); no registry password is stored.
registry=$(sed -n "s/^SKF_IMAGE_REGISTRY='\(.*\)'$/\1/p" deploy.env)
docker_config=~/.docker/config.json
mkdir -p ~/.docker
[[ -f $docker_config ]] || echo '{}' >"$docker_config"
jq --arg r "$registry" '.credHelpers[$r] = "ecr-login"' "$docker_config" >"$docker_config.tmp"
mv "$docker_config.tmp" "$docker_config"

compose pull --quiet
compose up -d --remove-orphans --wait --wait-timeout 600

live=$(current_tag)
if [[ $live != "$tag" ]]; then
  [[ -n $live ]] && echo "$live" >"$STATE_DIR/previous_tag"
  echo "$tag" >"$STATE_DIR/current_tag"
fi
docker image prune -f --filter "until=168h" >/dev/null
echo "release: $tag is live"
