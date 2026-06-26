#!/usr/bin/env bash
set -euo pipefail

host="${DEPLOY_HOST:-root@31.220.54.145}"
base="${DEPLOY_BASE_URL:-https://linuxoneliners.com}"
db="${LOL_ANALYTICS_DB:-/srv/data/linuxoneliners.com/analytics.sqlite3}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="${1:-artifacts/linuxoneliners-analytics-proof/${stamp}}"

mkdir -p "$out_dir"

curl -fsSL "${base}/api/analytics/proof" > "${out_dir}/public-proof.json"

ssh "$host" "sqlite3 '$db' '.schema events'" > "${out_dir}/events-schema.sql"

ssh "$host" "sqlite3 '$db'" > "${out_dir}/database-summary.tsv" <<'SQL'
.headers on
.mode tabs
select count(*) as total_events from events;
select count(distinct nullif(session_id, '')) as unique_sessions from events;
select count(distinct nullif(visitor_id, '')) as unique_visitors from events;
select datetime(min(received_at_ms) / 1000, 'unixepoch') as first_event_utc from events;
select datetime(max(received_at_ms) / 1000, 'unixepoch') as latest_event_utc from events;
select event_name, count(*) as events from events group by event_name order by events desc, event_name;
select path, count(*) as events from events group by path order by events desc, path limit 50;
select id as latest_event_id, event_hash as latest_event_hash from events where event_hash is not null order by id desc limit 1;
pragma integrity_check;
SQL

ssh "$host" "sha256sum '$db' /var/log/nginx/linuxoneliners.com.access.log* 2>/dev/null || true" > "${out_dir}/remote-source-sha256.txt"

cat > "${out_dir}/README.md" <<'MD'
# LinuxOneLiners Analytics Proof Package

This package is intended for private buyer, broker, or escrow diligence.

Files:

- `public-proof.json`: public aggregate proof from `/api/analytics/proof`.
- `events-schema.sql`: SQLite schema for the first-party analytics events table.
- `database-summary.tsv`: private aggregate query output from the live SQLite database.
- `remote-source-sha256.txt`: SHA256 checksums for the live analytics database and available Nginx access logs.
- `manifest.sha256`: local checksums for this package.

Verification approach:

1. Re-fetch `https://linuxoneliners.com/api/analytics/proof`.
2. Compare `event_count`, `latest_event_id`, and `latest_event_hash` against this package.
3. Compare the SQLite database checksum and Nginx log checksums if fresh server access is available.
4. Treat this as first-party operational evidence, not a third-party audit or revenue certification.
MD

(
  cd "$out_dir"
  find . -type f ! -name manifest.sha256 -print0 | sort -z | xargs -0 sha256sum
) > "${out_dir}/manifest.sha256"

printf 'wrote %s\n' "$out_dir"
