#!/usr/bin/env bash
set -euo pipefail

host="${DEPLOY_HOST:-root@31.220.54.145}"
domain="${DEPLOY_DOMAIN:-justinstone.online}"
release="$(date -u +%Y%m%dT%H%M%SZ)"
remote_root="/srv/www/${domain}"
remote_release="${remote_root}/releases/${release}"

bundle exec jekyll build

ssh "$host" "mkdir -p '${remote_root}/releases' '${remote_root}/shared'"
rsync -az --delete _site/ "${host}:${remote_release}/"
ssh "$host" "ln -sfn '${remote_release}' '${remote_root}/current' && nginx -t && systemctl reload nginx"

echo "Deployed ${domain} to ${remote_release}"
