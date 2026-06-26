#!/usr/bin/env bash
set -euo pipefail

summary_root="${TRAFFIC_SUMMARY_ROOT:-/srv/data/site-traffic-summaries}"
stamp="$(date -u +%F)"
generated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
out_dir="${summary_root}/${stamp}"
tmp_dir="$(mktemp -d)"

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

mkdir -p "$out_dir"

cat > "${tmp_dir}/README.md" <<EOF
# Site Traffic Summary

Generated: ${generated_at}

This summary is derived from Nginx access logs on the VPS. It is intended for operational review and due diligence. It preserves aggregate counts and source-file checksums without needing to keep unlimited raw logs.
EOF

printf 'generated_at\t%s\nhostname\t%s\nsummary_date\t%s\n' "$generated_at" "$(hostname)" "$stamp" > "${tmp_dir}/summary-metadata.tsv"
printf 'sha256\tbytes\tlines\tfile\n' > "${tmp_dir}/log-inventory.tsv"
printf 'file\trequests\tunique_ips\tbot_or_tool_ua\tbrowser_like_ua\tscanner_like_paths\tstatus_summary\n' > "${tmp_dir}/traffic-summary.tsv"
printf 'file\trequests\tpath\n' > "${tmp_dir}/top-paths.tsv"
printf 'file\trequests\tuser_agent\n' > "${tmp_dir}/top-user-agents.tsv"
printf 'file\trequests\treferrer\n' > "${tmp_dir}/top-referrers.tsv"
printf 'file\trequests\tpath\n' > "${tmp_dir}/scanner-paths.tsv"
mkdir -p "${tmp_dir}/sites"

for log_path in /var/log/nginx/*access.log /var/log/nginx/*access.log.[0-9] /var/log/nginx/*access.log.*.gz; do
  [ -e "$log_path" ] || continue

  base_name="$(basename "$log_path")"
  if [[ "$base_name" == access.log* ]]; then
    site_name="_default"
  else
    site_name="${base_name%%.access.log*}"
  fi
  site_dir="${tmp_dir}/sites/${site_name}"
  mkdir -p "$site_dir"
  for site_file in log-inventory traffic-summary top-paths top-user-agents top-referrers scanner-paths; do
    case "$site_file" in
      log-inventory) header='sha256\tbytes\tlines\tfile' ;;
      traffic-summary) header='file\trequests\tunique_ips\tbot_or_tool_ua\tbrowser_like_ua\tscanner_like_paths\tstatus_summary' ;;
      top-paths) header='file\trequests\tpath' ;;
      top-user-agents) header='file\trequests\tuser_agent' ;;
      top-referrers) header='file\trequests\treferrer' ;;
      scanner-paths) header='file\trequests\tpath' ;;
    esac
    if [ ! -f "${site_dir}/${site_file}.tsv" ]; then
      printf '%b\n' "$header" > "${site_dir}/${site_file}.tsv"
    fi
  done

  stream_cmd=(cat "$log_path")
  if [[ "$log_path" == *.gz ]]; then
    stream_cmd=(zcat "$log_path")
  fi

  line_count="$("${stream_cmd[@]}" 2>/dev/null | wc -l)"
  byte_count="$(wc -c < "$log_path" | tr -d ' ')"
  sha="$(sha256sum "$log_path" | awk '{print $1}')"
  inventory_row="$(printf '%s\t%s\t%s\t%s\n' "$sha" "$byte_count" "$line_count" "$log_path")"
  printf '%s\n' "$inventory_row" >> "${tmp_dir}/log-inventory.tsv"
  printf '%s\n' "$inventory_row" >> "${site_dir}/log-inventory.tsv"

  summary_row="$("${stream_cmd[@]}" 2>/dev/null | awk -v file="$base_name" -F'"' '
    {
      split($1, left, " "); ip = left[1];
      split($2, req, " "); path = req[2];
      split($3, tail, " "); status = tail[1];
      ua = $6;
      total++;
      ips[ip] = 1;
      statuses[status]++;
      bot = (ua ~ /(bot|Bot|crawler|Crawler|spider|Spider|curl|Headless|python|Go-http|MJ12|Claude|GPT|OAI|Semrush|Ahrefs|Bytespider)/);
      scan = (path ~ /(wp-|wp\/|xmlrpc|phpmyadmin|admin|login|\.env|vendor|boaform|cgi-bin|HNAP1|actuator|shell|\.php|\.git|server-status)/);
      if (bot) bots++; else browser++;
      if (scan) scans++;
    }
    END {
      for (ip in ips) unique++;
      status_summary = "";
      for (status in statuses) status_summary = status_summary status "=" statuses[status] " ";
      gsub(/[[:space:]]+$/, "", status_summary);
      printf "%s\t%d\t%d\t%d\t%d\t%d\t%s\n", file, total + 0, unique + 0, bots + 0, browser + 0, scans + 0, status_summary;
    }
  ')"
  printf '%s\n' "$summary_row" >> "${tmp_dir}/traffic-summary.tsv"
  printf '%s\n' "$summary_row" >> "${site_dir}/traffic-summary.tsv"

  "${stream_cmd[@]}" 2>/dev/null | awk -v file="$base_name" -F'"' '
    { split($2, req, " "); if (req[2] != "") paths[req[2]]++; }
    END { for (path in paths) print paths[path] "\t" path; }
  ' | sort -k1,1nr | head -50 | awk -v file="$base_name" '{count=$1; $1=""; sub(/^\t? */, ""); printf "%s\t%s\t%s\n", file, count, $0}' | tee -a "${tmp_dir}/top-paths.tsv" >> "${site_dir}/top-paths.tsv"

  "${stream_cmd[@]}" 2>/dev/null | awk -v file="$base_name" -F'"' '
    { ua[$6]++; }
    END { for (agent in ua) print ua[agent] "\t" agent; }
  ' | sort -k1,1nr | head -50 | awk -v file="$base_name" '{count=$1; $1=""; sub(/^\t? */, ""); printf "%s\t%s\t%s\n", file, count, $0}' | tee -a "${tmp_dir}/top-user-agents.tsv" >> "${site_dir}/top-user-agents.tsv"

  "${stream_cmd[@]}" 2>/dev/null | awk -v file="$base_name" -F'"' '
    { ref = $4; if (ref == "" || ref == "-") ref = "(direct)"; refs[ref]++; }
    END { for (ref in refs) print refs[ref] "\t" ref; }
  ' | sort -k1,1nr | head -50 | awk -v file="$base_name" '{count=$1; $1=""; sub(/^\t? */, ""); printf "%s\t%s\t%s\n", file, count, $0}' | tee -a "${tmp_dir}/top-referrers.tsv" >> "${site_dir}/top-referrers.tsv"

  "${stream_cmd[@]}" 2>/dev/null | awk -v file="$base_name" -F'"' '
    {
      split($2, req, " "); path = req[2];
      if (path ~ /(wp-|wp\/|xmlrpc|phpmyadmin|admin|login|\.env|vendor|boaform|cgi-bin|HNAP1|actuator|shell|\.php|\.git|server-status)/) scans[path]++;
    }
    END { for (path in scans) print scans[path] "\t" path; }
  ' | sort -k1,1nr | head -50 | awk -v file="$base_name" '{count=$1; $1=""; sub(/^\t? */, ""); printf "%s\t%s\t%s\n", file, count, $0}' | tee -a "${tmp_dir}/scanner-paths.tsv" >> "${site_dir}/scanner-paths.tsv"
done

(
  cd "$tmp_dir"
  find . -type f ! -name manifest.sha256 -print0 | sort -z | xargs -0 sha256sum
) > "${tmp_dir}/manifest.sha256"

rsync -a --delete "${tmp_dir}/" "${out_dir}/"
find "$summary_root" -mindepth 1 -maxdepth 1 -type d -mtime +730 -exec rm -rf {} +

printf 'wrote %s\n' "$out_dir"
