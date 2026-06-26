#!/usr/bin/env bash
set -euo pipefail

host="${DEPLOY_HOST:-root@31.220.54.145}"
db="/srv/data/justinstone.online/comments.sqlite3"
action="${1:-pending}"
id="${2:-}"

case "$action" in
  pending)
    ssh "$host" "sqlite3 -header -column '${db}' \"select comments.id, comments.page_url, users.email, users.display_name, substr(comments.body, 1, 120) as body, datetime(comments.created_at_ms / 1000, 'unixepoch') as created_utc from comments join users on users.id = comments.user_id where comments.status = 'pending' order by comments.created_at_ms asc limit 50;\""
    ;;
  approve|reject)
    if ! [[ "$id" =~ ^[0-9]+$ ]]; then
      echo "Usage: $0 ${action} COMMENT_ID" >&2
      exit 2
    fi
    status="approved"
    if [ "$action" = "reject" ]; then
      status="rejected"
    fi
    ssh "$host" "sqlite3 '${db}' \"update comments set status = '${status}' where id = ${id}; select id, status from comments where id = ${id};\""
    ;;
  *)
    echo "Usage: $0 [pending|approve COMMENT_ID|reject COMMENT_ID]" >&2
    exit 2
    ;;
esac
