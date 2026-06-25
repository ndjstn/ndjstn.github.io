#!/usr/bin/env bash
set -euo pipefail

domain="${1:-${DEPLOY_DOMAIN:-justinstone.online}}"
ip="${2:-${DEPLOY_IP:-31.220.54.145}}"

if [[ -z "$domain" || -z "$ip" ]]; then
  echo "Usage: $0 <domain> <ip>" >&2
  exit 2
fi

www_domain="www.${domain}"

echo "== Authoritative DNS =="
for ns in ns1.dns-parking.com ns2.dns-parking.com; do
  host "$domain" "$ns"
  host "$www_domain" "$ns"
done

echo
echo "== Forced HTTP/HTTPS =="
curl -sI "http://${ip}/" | sed -n '1,12p'
curl -sI --resolve "${domain}:80:${ip}" "http://${domain}/" | sed -n '1,12p'
curl -skI --resolve "${domain}:443:${ip}" "https://${domain}/" | sed -n '1,16p'
curl -skI --resolve "${www_domain}:443:${ip}" "https://${www_domain}/" | sed -n '1,16p'

echo
echo "== Public resolver check =="
host "$domain" || true
host "$www_domain" || true
