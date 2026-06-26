#!/usr/bin/env bash
set -euo pipefail

host="${DEPLOY_HOST:-root@31.220.54.145}"
domain="${DEPLOY_DOMAIN:-justinstone.online}"
release="$(date -u +%Y%m%dT%H%M%SZ)"
remote_root="/srv/www/${domain}"
remote_release="${remote_root}/releases/${release}"

if command -v nix >/dev/null 2>&1 && [ -f flake.nix ]; then
  nix --extra-experimental-features 'nix-command flakes' develop -c bundle exec jekyll build
else
  bundle exec jekyll build
fi

ssh "$host" "mkdir -p '${remote_root}/releases' '${remote_root}/shared'"
rsync -az --delete _site/ "${host}:${remote_release}/"
if [ -f api/comment_service.py ]; then
  ssh "$host" "mkdir -p /srv/apps/justinstone.online/comments"
  rsync -az api/comment_service.py "${host}:/srv/apps/justinstone.online/comments/comment_service.py"
fi
ssh "$host" "ln -sfn '${remote_release}' '${remote_root}/current' && if systemctl list-unit-files justinstone-comments.service >/dev/null 2>&1; then systemctl restart justinstone-comments.service; fi && nginx -t && systemctl reload nginx"

echo "Deployed ${domain} to ${remote_release}"
