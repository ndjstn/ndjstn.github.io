# Linux Issue Seed Backlog

Goal: build toward 250 high-quality issue/fix pages first, then expand only where search data and engagement justify it.

Each seed must become:

```text
problem query -> one-liner -> fake fixture -> vessel output -> page -> short -> LinkedIn hook -> tracking link
```

## Batch 1: Survival And Triage

- disk full linux find large files
- linux find which folder is using disk
- linux show biggest files human readable
- linux check memory usage by process
- linux check cpu usage by process
- linux check open ports
- linux check if nginx is listening
- linux tail live logs
- linux grep errors in logs
- linux grep line numbers in logs
- linux show recent errors
- linux inspect file permissions
- linux check owner and mode
- linux rsync dry run
- linux rsync show deletes

## Batch 2: Web Server Rescue

- nginx test config before reload
- nginx reload without downtime
- nginx find enabled site
- nginx 404 wrong root
- nginx 502 upstream refused
- nginx check access log status codes
- nginx show top client errors
- curl show response headers
- curl follow redirects
- curl time request phases
- openssl check certificate expiry
- certbot renew dry run
- check DNS A record
- check DNS CNAME
- compare local DNS resolver to public resolver

## Batch 3: Files And Permissions

- find files modified today
- find files larger than 100MB
- find empty files
- find writable files
- find world writable directories
- inspect parent directory permissions
- show numeric chmod mode
- compare file checksums
- count files in directory
- find broken symlinks
- safely preview rm targets
- tar create backup
- tar list archive contents
- tar extract to safe directory
- zip directory from terminal

## Batch 4: Logs And Text

- grep case insensitive
- grep multiple patterns
- grep before and after context
- grep only filenames
- grep count matches
- awk print column
- awk sum column
- sed print line range
- sort by number
- sort unique counts
- cut delimiter field
- tail from multiple logs
- journalctl since time
- journalctl service errors
- journalctl follow service

## Batch 5: Processes And Services

- systemctl status service
- systemctl list failed
- systemctl restart service
- systemctl show logs
- ps sort by cpu
- ps sort by memory
- kill process safely
- pgrep by name
- pkill dry-run alternative
- uptime load average
- free memory human readable
- vmstat quick check
- iostat disk pressure
- lsof open files
- lsof listening ports

## Batch 6: Networking

- ip addr show
- ip route show
- ping gateway
- traceroute target
- curl local service
- check port with nc
- ss listening tcp
- ss established connections
- show process by port
- DNS lookup with dig
- reverse DNS lookup
- test IPv6 disabled
- show firewall status
- list ufw rules
- check public IP

## Batch 7: Cybersecurity Triage

- list listening services safely
- find world writable directories
- find suid files
- check failed ssh logins
- check recent successful logins
- list users with login shells
- show sudo group members
- find files changed in last hour
- check suspicious cron jobs
- list systemd timers
- check open firewall ports
- show nginx requests by status code
- show top client IPs in access log
- detect repeated 404 probes
- check file hashes before and after deploy
- verify package ownership of a file
- list processes with deleted binaries
- find private keys with loose permissions
- check SSH authorized keys
- check TLS certificate issuer and expiry

## Batch 8: Web Hosting And Server Management

- nginx config test
- nginx reload safely
- show active nginx server names
- find nginx site root
- check certbot certificates
- certbot renew dry run
- inspect systemd service file
- show failed systemd units
- show service logs since reboot
- check disk inode usage
- check backup directory size
- rsync backup dry run
- verify backup archive contents
- check domain A record
- check domain www redirect
- curl force host header
- curl resolve to server IP
- measure response time with curl
- show largest Nginx access log paths
- show top referrers from access logs

## Batch 9: Apple And macOS Terminal

- show macOS version
- list Homebrew packages
- update Homebrew safely
- find large files in Downloads
- show listening ports on macOS
- flush DNS cache macOS
- show Wi-Fi interface info
- list launch agents
- inspect launch daemon
- find quarantine attribute
- remove quarantine attribute safely
- show file extended attributes
- convert HEIC to JPG if tool exists
- list iPhone backups
- find Xcode derived data size
- clean Xcode derived data preview
- show shell path on macOS
- check Rosetta architecture
- list installed fonts
- show battery cycle count

## Batch 10: Developer Workflow

- git show changed files
- git restore preview
- git clean dry run
- git find large tracked files
- npm list outdated
- python virtualenv path check
- find duplicate env files
- check port before starting dev server
- curl local API health
- jq extract JSON field
- watch command output
- run command every second
- compare two directories
- generate checksum manifest
- find TODO comments

## Expansion Rule

Do not bulk publish junk.

Scale in waves:

```text
14 -> 25 -> 50 -> 100 -> 250
```

At each wave, review:

- search impressions
- page visits
- copy command events
- scroll depth
- LinkedIn clicks
- YouTube clicks
- comments or saves
- vessel pass rate
- moderation risk
- revenue or affiliate intent

If a category gets no signal, pause that category and test another.
