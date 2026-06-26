#!/usr/bin/env bash
set -euo pipefail

host="${DEPLOY_HOST:-root@31.220.54.145}"
remote_snapshot_dir="/srv/backups/sqlite-snapshots/$(date -u +%Y%m%dT%H%M%SZ)"
local_dest="${1:-}"

ssh "$host" "mkdir -p '${remote_snapshot_dir}'"

ssh "$host" "if [ -f /srv/data/justinstone.online/comments.sqlite3 ]; then sqlite3 /srv/data/justinstone.online/comments.sqlite3 \".backup '${remote_snapshot_dir}/justinstone-comments.sqlite3'\"; fi"
ssh "$host" "if [ -f /srv/data/linuxoneliners.com/analytics.sqlite3 ]; then sqlite3 /srv/data/linuxoneliners.com/analytics.sqlite3 \".backup '${remote_snapshot_dir}/linuxoneliners-analytics.sqlite3'\"; fi"
ssh "$host" "find '${remote_snapshot_dir}' -type f -name '*.sqlite3' -exec sha256sum {} + > '${remote_snapshot_dir}/SHA256SUMS'"

if [ -n "$local_dest" ]; then
  mkdir -p "$local_dest"
  rsync -az "${host}:${remote_snapshot_dir}/" "${local_dest}/"
fi

echo "Created SQLite snapshots at ${host}:${remote_snapshot_dir}"
