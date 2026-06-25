# Owned hosting architecture

This is the target shape for replacing GitHub Pages with `justinstone.online`
and using the Hostinger VPS as a small, repeatable hosting platform.

## Goals

- Make `justinstone.online` the canonical portfolio and hiring surface.
- Keep `linuxoneliners.com` and future sites isolated enough to change, sell,
  or remove without disturbing the portfolio.
- Use Nix for repeatable local and server workflows.
- Keep databases, uploads, logs, analytics, and contact records separated by
  site.
- Build an admin dashboard for operational control instead of managing sites by
  memory.
- Launch many small experiments quickly and promote only the ones that show
  traffic, leads, revenue, or strategic value.

## Non-goals

- Do not build a Kubernetes-style platform before there is traffic.
- Do not combine every idea into one database.
- Do not depend on GitHub Pages for the paid domain.
- Do not transfer or sell raw user data unless the site policy and consent model
  explicitly allow that.
- Do not over-engineer every idea before the market gives a reason.

## Layers

```text
Internet
  |
DNS
  |
Nginx or Traefik edge
  |
  +-- static release: justinstone.online
  +-- static release: linuxoneliners.com
  +-- compose app: future-site-a
  +-- compose app: future-site-b
  |
Postgres
  +-- db_justinstone_admin
  +-- db_future_site_a
  +-- db_future_site_b
```

## Nix role

Nix should own reproducibility:

- `nix develop` for local tooling
- static site builds
- image and video generation tools
- deployment scripts
- generated Nginx, Compose, and backup templates
- future NixOS module experiments on a staging host

Nix should not own mutable production state:

- database files
- uploaded files
- secrets
- logs
- TLS private keys
- analytics events
- customer/contact records

## Runtime role

Use boring runtime pieces first:

- Nginx for static files, TLS, compression, and reverse proxying.
- Docker Compose for dynamic app isolation.
- Postgres for relational data.
- SQLite only for small single-purpose tools where export/handoff is trivial.
- Systemd timers or a small backup container for scheduled dumps.

This gives clean boundaries without creating a heavy platform too early.

## Admin dashboard

The admin dashboard should be a private site that reads a manifest for every
managed site.

Minimum objects:

```text
site
domain
repo
runtime
deployment
database
backup
policy
analytics_summary
lead_or_contact
handoff_package
```

Minimum dashboard views:

- sites
- deployments
- health checks
- resource usage
- leads and messages
- policies
- backups
- sale readiness

The dashboard should start read-only. Write actions like deploy, rollback,
policy publish, or user export can come after the read model is trustworthy.

Each site should have an explicit state so one person can operate the portfolio
like a small bot-managed fleet:

```text
idea -> draft -> live -> promote -> maintain -> sell
                  |        |          |
                  v        v          v
                paused   retire   maintenance
```

Useful states:

- `draft`: exists in repo but has no public DNS route.
- `live`: public and monitored.
- `paused`: public route can remain, but promotion and active work stop.
- `maintenance`: public route can show a static maintenance page while fixes
  happen.
- `promote`: worth pushing across LinkedIn, GitHub, YouTube, Kaggle, Medium,
  and search.
- `sell`: prepare handoff package and anonymized metrics.
- `retire`: remove from promotion, archive code, and preserve only allowed data.

The first automation should be conservative: read status, check health, build,
deploy, backup, and produce a to-do list. Destructive actions like delete,
domain transfer, raw data export, or policy changes should stay manual.

## Experiment lane

Most new ideas should start in the cheapest possible lane:

```text
idea -> landing page/static site -> tracking -> first links -> keep/kill/promote
```

Default launch package:

- one domain or subdomain
- static page or tiny app
- clear call to action
- basic analytics
- contact capture only if there is a privacy policy
- one README describing the audience, offer, and success metric
- one manifest entry for the dashboard

Promotion triggers:

- consistent inbound traffic from search, LinkedIn, YouTube, Kaggle, or GitHub
- contact forms, emails, waitlist signups, purchases, or recruiter messages
- a project becomes useful enough to demo publicly
- a site becomes saleable as a niche asset
- the data collected becomes valuable enough to protect separately

Kill triggers:

- no traffic after distribution attempts
- no clear audience
- no follow-up content angle
- high maintenance burden
- legal, privacy, or platform-policy risk outweighs the upside

Promotion path:

1. Static experiment on shared VPS.
2. Static site with analytics and contact capture.
3. Dynamic app in its own Compose project.
4. Dedicated database, backups, and admin dashboard visibility.
5. Dedicated VPS or external service if traffic, revenue, or sale value justifies it.

This lets us throw many ideas at the wall while keeping the serious ones clean
enough to scale or sell.

## Data model

Every public site gets a data manifest:

```text
domain:
owner:
purpose:
repo:
runtime:
database_name:
database_user:
data_collected:
personal_data: yes | no
analytics:
privacy_policy_url:
retention_period:
backup_schedule:
export_allowed:
sale_handoff:
success_metric:
kill_date:
promotion_status:
state:
automation:
```

Default rule: aggregate analytics can inform business decisions. Raw personal
data stays with the site and is not sold or transferred unless the policy and
consent model allow it.

## Scaling path

1. One VPS, Nginx, static releases.
2. Add Docker Compose for dynamic sites.
3. Add Postgres with one database and role per site.
4. Add private dashboard and monitoring.
5. Add container limits and backup verification.
6. Move high-value or high-traffic sites to their own VPS.
7. Add real load balancing only when multiple app instances or servers exist.

This keeps the system understandable while still leaving room to grow.
