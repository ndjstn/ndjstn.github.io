#!/usr/bin/env bash
set -euo pipefail

host="${DEPLOY_HOST:-root@31.220.54.145}"
remote_app="/srv/apps/justinstone.online/comments"
remote_data="/srv/data/justinstone.online"
env_file="/etc/default/justinstone-comments"

ssh "$host" "mkdir -p '${remote_app}' '${remote_data}' && chown -R www-data:www-data '${remote_data}'"
rsync -az api/comment_service.py "${host}:${remote_app}/comment_service.py"
rsync -az deploy/vps/systemd/justinstone-comments.service "${host}:/etc/systemd/system/justinstone-comments.service"
rsync -az deploy/vps/justinstone.online.conf "${host}:/etc/nginx/sites-available/justinstone.online"

ssh "$host" "if [ ! -f '${env_file}' ]; then umask 077; cat > '${env_file}' <<EOF
JS_COMMENTS_HOST=127.0.0.1
JS_COMMENTS_PORT=8092
JS_COMMENTS_DB=/srv/data/justinstone.online/comments.sqlite3
JS_COMMENTS_PUBLIC_BASE_URL=https://justinstone.online
JS_COMMENTS_COOKIE_SECRET=$(openssl rand -base64 48)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
EOF
fi"
ssh "$host" "systemctl daemon-reload && systemctl enable justinstone-comments.service && nginx -t"
ssh "$host" "systemctl restart justinstone-comments.service && systemctl reload nginx"

echo "Installed justinstone comments service. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in ${env_file}."
