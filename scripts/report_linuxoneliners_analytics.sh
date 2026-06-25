#!/usr/bin/env bash
set -euo pipefail

host="${DEPLOY_HOST:-root@31.220.54.145}"
db="${LOL_ANALYTICS_DB:-/srv/data/linuxoneliners.com/analytics.sqlite3}"

ssh "$host" "sqlite3 '$db'" <<'SQL'
.headers on
.mode column
select count(*) as total_events from events;
select event_name, count(*) as events from events group by event_name order by events desc, event_name;
select path, count(*) as events from events group by path order by events desc, path limit 20;
select datetime(received_at_ms / 1000, 'unixepoch') as utc_time, event_name, path, label, json_extract(performance_json, '$.loadMs') as load_ms
from events
order by id desc
limit 20;
SQL
