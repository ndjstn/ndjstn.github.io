#!/usr/bin/env bash
set -euo pipefail

host="${DEPLOY_HOST:-root@31.220.54.145}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"

remote_summary_script="/usr/local/sbin/write-vps-traffic-summary"
remote_logrotate="/etc/logrotate.d/nginx"

scp scripts/write_vps_traffic_summary.sh "${host}:${remote_summary_script}" >/dev/null

ssh "$host" "chmod 0755 '${remote_summary_script}' && cp '${remote_logrotate}' '${remote_logrotate}.bak.${stamp}'"

ssh "$host" "cat > '${remote_logrotate}'" <<'REMOTE'
/var/log/nginx/*.log {
	daily
	missingok
	rotate 90
	maxage 120
	compress
	delaycompress
	notifempty
	create 0640 www-data adm
	sharedscripts
	prerotate
		/usr/local/sbin/write-vps-traffic-summary >/dev/null 2>&1 || true
		if [ -d /etc/logrotate.d/httpd-prerotate ]; then \
			run-parts /etc/logrotate.d/httpd-prerotate; \
		fi \
	endscript
	postrotate
		invoke-rc.d nginx rotate >/dev/null 2>&1
	endscript
}
REMOTE

ssh "$host" "logrotate -d '${remote_logrotate}' >/tmp/linuxoneliners-logrotate-debug.txt && '${remote_summary_script}'"
ssh "$host" "tail -40 /tmp/linuxoneliners-logrotate-debug.txt && find /srv/data/site-traffic-summaries -maxdepth 2 -type f | sort | tail -20"
