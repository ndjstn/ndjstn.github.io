# Placeholder releases

Use placeholders when an old public site should be parked while the replacement
is rebuilt.

The placeholder must be:

- static
- visually acceptable on mobile and desktop
- clear that the site is being rebuilt
- free of dead API calls
- deployed through the same release/current symlink pattern as finished sites

## Linux One Liners

Live placeholder source:

```text
deploy/vps/placeholders/linuxoneliners.com/index.html
```

Live release layout:

```text
/srv/www/linuxoneliners.com/
  releases/<timestamp>-placeholder/
  current -> releases/<timestamp>-placeholder
```

Bootstrap command:

```bash
release="/srv/www/linuxoneliners.com/releases/$(date -u +%Y%m%dT%H%M%SZ)-placeholder"
ssh root@31.220.54.145 "mkdir -p '$release'"
scp deploy/vps/placeholders/linuxoneliners.com/index.html root@31.220.54.145:"$release/index.html"
ssh root@31.220.54.145 "ln -sfn '$release' /srv/www/linuxoneliners.com/current && nginx -t && systemctl reload nginx"
```

The current canonical domain is:

```text
linuxoneliners.com
```

Aliases should redirect to the canonical domain after their DNS records point to
the VPS:

```text
www.linuxoneliners.com
linuxoneliner.com
www.linuxoneliner.com
```

Do not issue a multi-name certificate until all alias DNS records resolve to the
VPS. Until then, forced-resolution checks can verify Nginx behavior, but public
TLS validation may fail.
