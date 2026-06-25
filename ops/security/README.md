# Site Security Baseline

Assume failure. Design every site so a bad deploy, compromised dependency, stale credential, or defacement attempt has a small blast radius.

## Default Rules

- Static first when possible.
- No database until the product actually needs one.
- No user accounts until there is a real user workflow.
- No secrets in repos, static builds, screenshots, videos, or lesson fixtures.
- One deploy user per server once automation is formalized.
- One site root per domain.
- One database per app when a database becomes necessary.
- One backup and restore path per site.
- Logs retained long enough to debug, not forever by default.

## Static Site Baseline

For static sites:

- serve from `/srv/www/<domain>/current`
- deploy to timestamped release directories
- update `current` by symlink
- keep at least the last known-good release
- run `nginx -t` before reload
- force HTTPS
- set security headers
- disable directory listing
- avoid server-side includes
- avoid writable files under the web root

Recommended headers:

```nginx
add_header X-Content-Type-Options "nosniff" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header X-Frame-Options "SAMEORIGIN" always;
add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
```

Add a Content Security Policy after each site has stable asset and analytics requirements.

## Dynamic App Baseline

For apps:

- run app containers as non-root
- bind app containers to localhost or a private Docker/Podman network
- put Nginx in front
- store secrets in `/srv/secrets/<domain>.env`
- use one database and one database user per site
- deny cross-site database access
- keep uploads outside the app image
- virus-scan or type-check uploads if uploads exist
- rate-limit login, forms, and API write endpoints
- back up database and uploads separately

## Content Production Security

For generated terminal demos:

- use disposable containers or VMs
- never mount real home directories
- never mount real SSH keys
- never record real tokens, cookies, emails, private URLs, or internal IP lists
- use fake domains like `example.com` in public demos unless the real domain is intentionally part of the lesson
- review frames before publishing Shorts
- keep fixtures reproducible

## Pre-Publish Checklist

Before a site or major feature is public:

- `nginx -t` passes
- HTTPS works
- HTTP redirects to HTTPS
- security headers present
- no directory listing
- no secrets in generated files
- no private email addresses in generated files
- no accidental GitHub/API tokens in generated files
- backup exists before deploy
- rollback path is known
- analytics are privacy-appropriate
- privacy policy exists if collecting personal data

## Incident Checklist

If defacement or compromise is suspected:

1. Put the site in maintenance or roll back to last known-good static release.
2. Preserve logs before restarting everything.
3. Rotate exposed credentials.
4. Check deploy artifacts for injected files.
5. Check Nginx configs and enabled sites.
6. Check cron/systemd timers.
7. Check recent SSH logins.
8. Check app dependencies and lockfiles.
9. Restore from known-good source.
10. Write the incident note before moving on.
