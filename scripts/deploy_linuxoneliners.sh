#!/usr/bin/env bash
set -euo pipefail

host="${DEPLOY_HOST:-root@31.220.54.145}"
domain="${DEPLOY_DOMAIN:-linuxoneliners.com}"
release="$(date -u +%Y%m%dT%H%M%SZ)"
site_dir="sites/linuxoneliners.com"
remote_root="/srv/www/${domain}"
remote_release="${remote_root}/releases/${release}"
remote_app="/srv/apps/${domain}/analytics"

ruby "${site_dir}/build.rb"

ssh "$host" "mkdir -p '${remote_root}/releases' '${remote_root}/shared'"
rsync -az --delete "${site_dir}/dist/" "${host}:${remote_release}/"
ssh "$host" "mkdir -p '${remote_app}'"
rsync -az "${site_dir}/api/event_collector.py" "${host}:${remote_app}/event_collector.py"
ssh "$host" "ln -sfn '${remote_release}' '${remote_root}/current' && systemctl restart linuxoneliners-analytics.service && nginx -t && systemctl reload nginx"

echo "Deployed ${domain} to ${remote_release}"
