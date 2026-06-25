# Candidate Batch Inventory

Current live slugs are checked before promotion:

```text
check-listening-ports
curl-resolve-site-check
disk-usage-by-folder
file-owner-and-mode
find-large-files
grep-errors-in-logs
grep-line-numbers
inspect-permissions
list-processes-by-memory
recent-log-errors
rsync-dry-run
rsync-show-deletes
show-top-files-human
tail-live-logs
```

Subagent candidate pass, June 25, 2026:

```text
web-hosting-dns-ssl-vps: 50 candidates
cybersecurity-defensive-triage: 40 candidates
apple-dev-cicd-git-data-logs: 60 candidates
total: 150 candidates
```

Promotion rule:

```text
candidate -> duplicate check -> fixture -> curation check -> vessel run -> live page -> campaign link
```

Do not add candidates directly to `content/lessons.json` without a fixture and a passing vessel transcript.

## Next Promotion Waves

Wave 1: hosting operations

- nginx-test-config
- nginx-show-enabled-sites
- dig-a-record
- check-http-redirect
- inspect-response-headers
- check-open-http-ports
- test-localhost-site
- check-nginx-symlink-target
- verify-live-release-marker
- disk-free-human

Wave 2: cybersecurity triage

- failed-ssh-logins
- top-failed-ssh-ips
- accepted-ssh-logins
- users-with-login-shells
- world-writable-files
- world-writable-dirs
- recently-modified-etc
- check-key-permissions
- processes-by-cpu
- environment-secret-names

Wave 3: Apple/developer workflow

- macos-find-large-downloads
- macos-list-listening-ports
- macos-dns-cache-flush-note
- find-todos-in-project
- check-json-validity
- find-hardcoded-localhost
- git-show-staged-summary
- git-safe-clean-preview
- csv-show-columns
- json-pretty-print

## Candidate Sources

Full raw candidate output came from three subagent passes:

- Hosting, DNS, SSL, VPS/server management
- Cybersecurity and defensive triage
- Apple/macOS, developer workflow, CI/CD, Git, data workflows, logs

Store promoted candidates in structured lesson JSON only after they pass the normal content gate.
