# Web Hosting, DNS, SSL, VPS Candidates

## Nginx Triage

- nginx-test-config: `sudo nginx -t`
- nginx-reload-safely: `sudo systemctl reload nginx`
- nginx-show-enabled-sites: `ls -l /etc/nginx/sites-enabled/`
- nginx-find-server-name: `grep -R "server_name" /etc/nginx/sites-enabled/`
- nginx-tail-access-log: `sudo tail -f /var/log/nginx/access.log`
- nginx-tail-error-log: `sudo tail -f /var/log/nginx/error.log`
- nginx-count-status-codes: `awk '{print $9}' /var/log/nginx/access.log | sort | uniq -c | sort -nr`
- nginx-find-404s: `awk '$9==404 {print $7}' /var/log/nginx/access.log | sort | uniq -c | sort -nr | head`
- nginx-find-502s: `grep ' 502 ' /var/log/nginx/access.log | tail -20`
- nginx-active-connections: `curl -s http://127.0.0.1/nginx_status`

## DNS Debugging

- dig-a-record: `dig +short example.com A`
- dig-authoritative-ns: `dig +short NS example.com`
- dig-trace-domain: `dig +trace example.com`
- compare-dns-resolvers: `for r in 1.1.1.1 8.8.8.8; do dig @$r +short example.com; done`
- check-cname-chain: `dig +short www.example.com CNAME`
- check-mx-records: `dig +short example.com MX`
- check-txt-records: `dig +short example.com TXT`
- dns-cache-bust-query: `dig +search +short example.com`
- reverse-dns-lookup: `dig +short -x 31.220.54.145`
- detect-split-dns: `getent hosts example.com && dig +short example.com`

## SSL / HTTPS

- openssl-cert-dates
- openssl-cert-subject
- curl-show-tls
- certbot-list-certs
- certbot-dry-run-renew
- check-http-redirect
- check-hsts-header
- find-expiring-certs
- ssl-name-mismatch
- force-curl-sni-ip

## VPS / Server Health

- check-uptime-load
- memory-pressure
- disk-free-human
- inode-usage
- biggest-journals
- vacuum-journal-size
- failed-services
- service-restart-count
- top-cpu-processes
- check-reboot-required

## Web Hosting Operations

- check-open-http-ports
- test-localhost-site
- inspect-response-headers
- find-web-root-owner
- find-recent-web-files
- backup-site-tarball
- dry-run-delete-old-releases
- check-nginx-symlink-target
- verify-live-release-marker
- watch-access-by-referrer
