# Cybersecurity Defensive Triage Candidates

## SSH And Login Triage

- failed-ssh-logins
- top-failed-ssh-ips
- accepted-ssh-logins
- recent-login-history
- recent-sudo-commands
- users-with-login-shells
- empty-password-accounts
- locked-user-accounts
- expired-passwords
- lastlog-never-logged-in

## Network And Web Server Triage

- established-connections
- count-connections-by-ip
- web-4xx-by-ip
- web-5xx-by-url
- top-web-user-agents
- suspicious-web-methods
- largest-web-responses
- recent-nginx-errors
- tls-cert-expiry
- dns-answer-trace

## File Integrity And Permission Triage

- world-writable-files
- world-writable-dirs
- suid-files
- sgid-files
- recently-modified-etc
- new-files-last-hour
- hash-important-configs
- find-private-keys
- check-key-permissions
- find-hidden-files

## Process, Service, And Runtime Triage

- processes-by-cpu
- deleted-open-files
- service-failed-units
- service-restart-history
- cron-jobs-all-users
- user-crontabs
- docker-running-containers
- docker-recent-container-errors
- environment-secret-names
- path-hijack-check
