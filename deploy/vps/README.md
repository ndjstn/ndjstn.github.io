# VPS deployment notes

This site is intended to run as the canonical portfolio at `justinstone.online`.
The current low-maintenance production shape is:

- build static Jekyll output locally or in CI
- upload each release to `/srv/www/justinstone.online/releases/<timestamp>`
- point `/srv/www/justinstone.online/current` at the active release
- let Nginx serve static files directly

That keeps the portfolio independent from GitHub Pages while avoiding a full app
server for static content.

## Suggested host layout

```text
/srv/www/
  justinstone.online/
    current -> releases/20260625T033000Z
    releases/
      20260625T033000Z/
    shared/
  linuxoneliners.com/
    current -> releases/...
    releases/
  future-site.example/
    current -> releases/...
    releases/
```

Use the same layout for future sub-business or sellable sites. Static sites can
use this exact deploy flow. Dynamic apps should sit behind Nginx as local
services or Docker Compose stacks, with one Nginx server block per domain.

## Operating model

Hostinger VPS can be rebuilt as the owned production host because the existing
`linuxoneliners.com` site has no meaningful user base yet. The lowest-risk
production path is still phased:

1. make the current Ubuntu VPS reproducible with Nix-managed tooling, Docker
   Compose, Nginx config, and checked-in deployment docs
2. move all live domains behind that model
3. only then consider a full NixOS rebuild or second staging VPS

Nix should be used all around for repeatability: dev shells, static builds,
asset generation, deploy tooling, config templates, and eventually service
definitions. Mutable runtime state should stay outside Nix.

Recommended split:

- Static portfolio sites: build locally/CI, upload immutable release folders.
- Dynamic experiments: one Docker Compose project per domain or subdomain.
- Databases: one logical database per site, with separate users and passwords.
- Secrets: per-site `.env` files outside git, ideally under `/srv/secrets`.
- Backups: host snapshot plus per-database logical dumps.
- Logs/events: per-site tables or schemas; never mix visitor/user data between projects.
- Dashboard/control plane: one private admin app that reads site manifests,
  health checks, analytics summaries, contact queues, policy state, and deploy
  history.

This keeps a failed experiment from taking down the portfolio and makes a site
sale or handoff cleaner: code, deployment recipe, domain, database export, and
asset bundle can be separated from personal infrastructure.

## Data boundaries

For any site that collects user data:

- collect the smallest useful dataset
- publish a simple privacy policy before collecting emails, accounts, analytics identifiers, or form submissions
- keep raw user/contact data in that site's database only
- use aggregate analytics for portfolio/business decisions
- do not sell or transfer personal data unless the privacy policy and user consent model explicitly allow it
- prefer exporting anonymized or aggregate business metrics when selling a site

The default sale package should be:

- source code
- deployment docs
- domain transfer instructions
- static assets
- anonymized analytics summary
- optional database export only when the data policy allows transfer

## Portfolio comments and SSO

`justinstone.online` uses first-party comments instead of a hosted discussion
widget. The static site calls a local Python service through Nginx:

```text
Browser -> https://justinstone.online/api/comments -> Nginx -> 127.0.0.1:8092
```

Runtime files stay outside the static release:

- service code: `/srv/apps/justinstone.online/comments/comment_service.py`
- database: `/srv/data/justinstone.online/comments.sqlite3`
- secrets: `/etc/default/justinstone-comments`
- systemd unit: `justinstone-comments.service`

Comments require Google SSO. New submissions are stored as `pending`; only
`approved` comments render publicly. Moderation is intentionally operator-side:

```bash
scripts/moderate_justinstone_comments.sh pending
scripts/moderate_justinstone_comments.sh approve 123
scripts/moderate_justinstone_comments.sh reject 123
```

The OAuth client should use:

- authorized JavaScript origin: `https://justinstone.online`
- authorized redirect URI:
  `https://justinstone.online/auth/google/callback`

Never commit OAuth secrets. Put them in `/etc/default/justinstone-comments`.

For NAS/offsite backups, snapshot SQLite first and rsync the snapshot, not the
hot database file:

```bash
scripts/snapshot_vps_databases.sh /path/to/nas/site-snapshots/latest
```

## Future multi-site pattern

Each new idea should start with a small manifest:

```text
domain:
site_type: static | app | marketplace | content
repo:
runtime:
database:
data_collected:
privacy_policy:
backup_policy:
handoff_package:
```

This gives us a repeatable way to build, test, monetize, and potentially sell
sites without commingling infrastructure or user data.

## Experiment lane

The platform should make small launches cheap:

- build a static landing page or tiny app
- point a domain or subdomain at it
- add analytics and one clear conversion target
- publish supporting LinkedIn, GitHub, Kaggle, YouTube, or Medium content
- review traffic and leads after a fixed period
- kill, keep, or promote the site

Only promoted sites should get heavier infrastructure such as a dedicated
database, authenticated user accounts, background jobs, or separate VPS hosting.
This keeps experimentation fast while preventing one weak idea from becoming
permanent operational clutter.

## Static-site template rule

Use Jekyll as the default for new public sites unless the idea clearly needs a
server-side app. A Jekyll site is enough for portfolios, project writeups,
landing pages, small content sites, docs, SEO experiments, and saleable static
properties.

Each new Jekyll-based site should start with:

- site manifest under `ops/sites/`
- clear success metric
- one public call to action
- social preview image
- analytics plan
- privacy/contact policy if any user data is collected
- local build check before deploy

## Sellable site package

A site becomes saleable when it is more than a folder of code. The handoff
package should let a normal owner run it:

- domain transfer notes
- hosting/deploy instructions
- edit guide for pages, images, products, or services
- analytics summary
- contact/lead export instructions
- privacy/terms notes
- backup and restore instructions
- clear list of what the buyer owns and what remains excluded

For shops and catalogs, start with simple owner-editable data such as Markdown,
JSON, or CSV. Add a lightweight admin tool only when it reduces support work.

## Dashboards

Use two dashboards:

- Factory dashboard: private dashboard for us to create sites, manage domains,
  deploy, monitor health, review analytics, decide keep/kill/promote/sell, and
  prepare handoff packages.
- Site owner dashboard: one small dashboard per site for editing content,
  publishing, checking messages, viewing simple analytics, exporting allowed
  data, and reading renewal/support instructions.

Do not give site owners access to the factory dashboard.

## Control dashboard

The home end should eventually expose a private dashboard for:

- site inventory and DNS status
- deploy history and rollback pointer
- uptime checks and error logs
- resource usage by site or container
- form submissions, lead queues, and outbound follow-up status
- privacy policy, cookie policy, terms, and data-retention status
- backup freshness and restore checks
- buyer/handoff readiness for any site that may be sold

That dashboard should not be the public portfolio. It should run behind a
separate admin hostname, VPN, or identity gate. A public portfolio outage should
not expose or depend on private operations tooling.

## Resource management

Use the VPS like a small platform, not like one big app:

- Nginx or Traefik at the edge for domain routing and TLS.
- Static sites served directly from release folders.
- Dynamic sites isolated as Docker Compose projects.
- One Postgres instance is acceptable early, but each site gets its own
  database, role, password, backups, and retention policy.
- Add Redis, queues, or background workers only when a site actually needs
  them.
- Use CPU and memory limits on containers once multiple dynamic sites are live.
- Use monitoring before scaling decisions: uptime checks, logs, disk, memory,
  CPU, and database size.

Load balancing on one VPS is mostly process supervision and reverse proxying.
It can route traffic between multiple containers, but it does not protect
against the VPS itself failing. True load balancing becomes useful when a site
has multiple app containers, multiple VPS instances, or a separate database
host. Until then, preventive resource management is the right goal.

## First cutover checklist

1. Confirm SSH access to the VPS.
2. Copy `deploy/vps/justinstone.online.conf` to `/etc/nginx/sites-available/`.
3. Symlink it into `/etc/nginx/sites-enabled/`.
4. Run `sudo nginx -t`.
5. Deploy the static build with `scripts/deploy_static_vps.sh`.
6. Point Hostinger DNS for `justinstone.online` and `www.justinstone.online` to the VPS.
7. Issue TLS with Certbot after DNS resolves to the VPS.

The detailed browser, DNS, Nginx, SSL, and cleanup process is captured in
`deploy/vps/HOSTINGER_CUTOVER.md`.

After cutover, run:

```bash
./scripts/check_vps_site.sh justinstone.online 31.220.54.145
```

Use the same checklist for a future rebuild of `linuxoneliners.com`; only the
domain, site root, Nginx file, and manifest should change.

For parked or partially rebuilt sites, use a placeholder release from
`deploy/vps/placeholders/` instead of leaving stale apps or dead API routes
public.

## CI/CD

The repository has two GitHub Actions lanes:

- `Site CI`: runs on pushes and pull requests. It validates ops manifests,
  generates the private site registry, builds Jekyll, checks that private
  `ops/` and `deploy/` files do not leak into `_site`, and runs internal-link
  checks.
- `Deploy justinstone.online to VPS`: manual workflow for production deploys.
  It runs the same public build checks, installs an SSH key from GitHub
  secrets, and runs `scripts/deploy_static_vps.sh`.

Required GitHub secrets for VPS deploy:

```text
VPS_DEPLOY_HOST=root@31.220.54.145
VPS_SSH_PRIVATE_KEY=<private deploy key>
VPS_KNOWN_HOSTS=<known_hosts line for 31.220.54.145>
```

Use a deploy-only SSH key once the VPS is ready. Avoid using a personal key for
long-term automation.

## DNS target

The VPS currently serving `linuxoneliners.com` resolves to:

```text
31.220.54.145
```

Cutover should not happen until Nginx is configured for `justinstone.online`.
