#!/usr/bin/env bash
set -euo pipefail

base="${LOL_BASE_URL:-https://linuxoneliners.com}"

check_contains() {
  local url="$1"
  local pattern="$2"
  curl -sk "$url" | rg -Fq "$pattern"
  printf 'ok %s contains %s\n' "$url" "$pattern"
}

check_header() {
  local url="$1"
  local pattern="$2"
  curl -skI "$url" | rg -qi "$pattern"
  printf 'ok %s header %s\n' "$url" "$pattern"
}

check_contains "$base/" "Fix the Linux problem"
check_contains "$base/" "Browse by what broke"
check_contains "$base/" "demos passed"
check_contains "$base/demos/summary.json" '"failed": []'
check_contains "$base/exports/analytics-events.json" "command_copy"
check_header "$base/" "x-content-type-options: nosniff"
check_header "$base/" "permissions-policy:"
curl -sk "$base/api/health" | rg -q '"ok":true'
printf 'ok %s/api/health\n' "$base"
