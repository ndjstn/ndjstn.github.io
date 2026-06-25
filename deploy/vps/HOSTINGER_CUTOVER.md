# Hostinger VPS cutover runbook

This is the repeatable path used to move `justinstone.online` from Hostinger
Website Builder/CDN to the VPS at `31.220.54.145`.

Use this as the base process for future domains, including a rebuilt
`linuxoneliners.com`.

## 1. Access

Use delegated Hostinger access where possible. For this account, the workable
path was:

1. Send Hostinger shared access to the collaborator email.
2. Accept the invite from that inbox.
3. Log in with Google SSO as the collaborator.
4. Confirm hPanel shows impersonation mode for the owner account.

Do not put the owner email password in the browser session.

## 2. SSH bootstrap

Add the local public SSH key through hPanel:

```text
VPS -> Settings -> SSH keys
```

Then verify:

```bash
ssh -i ~/.ssh/id_ed25519 root@31.220.54.145 'hostname && whoami'
```

The Hostinger web terminal is useful for visible work, but it is a serial
console. It asks for an operating-system username and password, not the hPanel
or Google login. If a temporary console user is needed, create it only for the
session, give it narrowly timed sudo access, and remove it before finishing.

## 3. Static deploy

Build and deploy the static Jekyll release:

```bash
./scripts/check_public_build.sh
DEPLOY_HOST=root@31.220.54.145 DEPLOY_DOMAIN=justinstone.online ./scripts/deploy_static_vps.sh
```

The deploy layout is:

```text
/srv/www/<domain>/
  releases/<timestamp>/
  current -> releases/<timestamp>
  shared/
```

## 4. Nginx

Install a server block before DNS cutover:

```bash
scp deploy/vps/justinstone.online.conf root@31.220.54.145:/etc/nginx/sites-available/justinstone.online
ssh root@31.220.54.145 'ln -sfn /etc/nginx/sites-available/justinstone.online /etc/nginx/sites-enabled/justinstone.online && nginx -t && systemctl reload nginx'
```

Verify the site before public DNS moves:

```bash
curl -sI --resolve justinstone.online:80:31.220.54.145 http://justinstone.online/
curl -s --resolve justinstone.online:80:31.220.54.145 http://justinstone.online/ | sed -n '1,20p'
```

Expected result before SSL: `HTTP/1.1 200 OK` from `nginx` and the Jekyll
portfolio HTML.

## 5. DNS cutover

For a Hostinger-managed domain using `ns1.dns-parking.com` and
`ns2.dns-parking.com`, remove builder/CDN records such as:

```text
ALIAS @ -> <domain>.cdn.hstgr.net
CNAME www -> www.<domain>.cdn.hstgr.net
```

Add:

```text
A @   -> 31.220.54.145
A www -> 31.220.54.145
```

If the hPanel form is brittle, the browser session can call Hostinger's same DNS
API after validating the payload:

```js
await fetch('/api/dns/v1/direct/zone/resource-records/validate', {
  method: 'POST',
  credentials: 'include',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    domain: 'justinstone.online',
    resource_records: [
      { name: '@', type: 'A', ttl: 14400, records: [{ content: '31.220.54.145', disabled: false }] },
      { name: 'www', type: 'A', ttl: 14400, records: [{ content: '31.220.54.145', disabled: false }] }
    ]
  })
})
```

Then apply the same `resource_records` with `PATCH` to
`/api/dns/v1/direct/zone/resource-records`.

Read the zone back afterward:

```js
await fetch('/api/dns/v1/direct/zone/resource-records?domain=justinstone.online', {
  credentials: 'include'
}).then(r => r.json())
```

## 6. SSL

Wait until the authoritative nameservers return the VPS IP:

```bash
host justinstone.online ns1.dns-parking.com
host www.justinstone.online ns1.dns-parking.com
host justinstone.online ns2.dns-parking.com
host www.justinstone.online ns2.dns-parking.com
```

Then issue TLS:

```bash
ssh root@31.220.54.145 \
  'certbot --nginx --non-interactive --agree-tos --redirect \
    -m jstone1@cougars.ccis.edu \
    -d justinstone.online -d www.justinstone.online'
```

Verify:

```bash
./scripts/check_vps_site.sh justinstone.online 31.220.54.145
ssh root@31.220.54.145 'nginx -t && certbot certificates --cert-name justinstone.online'
```

Certbot rewrites the Nginx file to add the HTTPS listener, certificate paths,
and HTTP-to-HTTPS redirect. Keep the checked-in Nginx file as the bootstrap
config unless we later template Certbot-managed production configs.

After Certbot, make the HTTP default server canonical instead of leaving the
generated `return 404` behavior. That prevents `http://<server-ip>/` from
looking like a dead site:

```nginx
listen 80 default_server;
listen [::]:80 default_server;
server_name justinstone.online www.justinstone.online;
return 301 https://justinstone.online$request_uri;
```

If local browsers still show "server not found" after authoritative DNS is
correct, check the recursive resolver before debugging Nginx:

```bash
host -v justinstone.online
host -v www.justinstone.online
```

During the first cutover, the local router at `192.168.10.1` cached the old
empty/NXDOMAIN state while Cloudflare and Hostinger authoritative DNS already
returned `31.220.54.145`. In that case, wait for the negative-cache TTL, flush
the local/router DNS cache, change the client DNS resolver, or use forced
resolution for verification.

## 7. Cleanup

Remove any temporary web-terminal user created for visible console work:

```bash
ssh root@31.220.54.145 'rm -f /etc/sudoers.d/90-opsconsole; userdel -r opsconsole 2>/dev/null || true'
```

Check logs after cutover:

```bash
ssh root@31.220.54.145 'tail -n 80 /var/log/nginx/justinstone.online.error.log; journalctl -u nginx -n 80 --no-pager'
```

## Rebuild rule

For both `justinstone.online` and `linuxoneliners.com`, rebuild from the same
baseline:

- Jekyll static site first
- site manifest under `ops/sites/`
- checked build and leak test
- release-folder deploy
- Nginx server block
- Hostinger DNS to VPS
- Certbot TLS
- health check script
- dashboard hooks later
