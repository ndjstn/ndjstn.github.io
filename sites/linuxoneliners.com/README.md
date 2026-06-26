# Linux One Liners Scaffold

Linux One Liners should be a production pipeline, not just a page collection.

One structured lesson should generate:

- a search-friendly web page
- a YouTube Shorts script
- a LinkedIn hook and question
- a terminal demo plan
- a transcript/caption seed
- future course or paid-pack metadata
- experiment notes for A/B testing

## Current Bet

Build a static command rescue library first:

```text
commands + failures + safer fixes + short-form video metadata
```

Do not start with accounts, a database, self-hosted video, comments, or a SaaS dashboard. Those add operating burden before the site has traffic.

## Review Of The Plan

What is good:

- Static pages are cheap, fast, portable, and easy to sell later.
- YouTube handles video hosting and discovery.
- LinkedIn gets short practical hooks instead of generic article links.
- Structured lesson data lets us regenerate pages, Shorts, scripts, captions, and course units.
- A/B experiments are tracked at the lesson level instead of guessed from memory.

What needs discipline:

- Every lesson needs original terminal output or a reproducible fixture.
- Dangerous commands need recovery notes and clear warnings.
- Ads should stay off until traffic is real.
- User data should not be collected until there is a concrete reason.
- Runtime containers should be used for demos and generators, not for a simple static site.
- Security should be boring by default: static releases, no writable web root, no secrets in demos, and no user data until it pays for its complexity.

## Traffic And Monetization Gates

Ads are not first. Useful traffic is first.

Review ads after roughly:

```text
10,000 monthly page views
```

Review paid products after roughly:

```text
250 email subscribers
```

Review sponsorship or affiliate outreach after roughly:

```text
50,000 monthly video views
```

Early monetization should favor:

- affiliate links for VPS, monitoring, books, and tools
- paid runbook packs
- downloadable cheat sheets
- consulting leads
- sponsored command series once traffic is proven

## A/B Testing

Each lesson can define experiments in `content/lessons.json`.

Start with simple tests:

- LinkedIn hook A vs. B
- YouTube Short title A vs. B
- thumbnail/caption first line
- call-to-action wording
- page layout: command first vs. problem first

Track:

- YouTube 3-second hold
- YouTube 15-second retention
- YouTube click-through rate
- LinkedIn comment rate
- LinkedIn save/share rate
- site click-through to related lessons
- email or contact conversion

## Buyer Evidence

Track these monthly under `ops/valuation/`:

- traffic by source
- top pages and search queries
- YouTube Shorts views and retention
- LinkedIn click/comment/save rates
- revenue by source
- expenses
- net profit
- content added
- experiments run
- operational incidents
- transfer risks

This keeps a running valuation defensible instead of vibes-based.

## Security Baseline

Use `ops/security/README.md` as the default gate.

For this site, the safest early shape is:

```text
Nginx -> static release directory
```

Generated demos should run in disposable containers or VMs with fake data. The public site should not mount real keys, use real secrets, or collect user data until there is a product reason.

## First-Party Analytics

The site now has an owned analytics collector:

```text
Nginx -> /api/events -> linuxoneliners-analytics.service -> SQLite
```

Current database:

```text
/srv/data/linuxoneliners.com/analytics.sqlite3
```

Quick report:

```bash
./scripts/report_linuxoneliners_analytics.sh
```

Public aggregate proof endpoint:

```text
https://linuxoneliners.com/api/analytics/proof
```

The proof endpoint exposes aggregate counts, the observed event time range, and the latest event hash-chain tip. It does not expose raw visitor IDs, IP addresses, user agents, or event rows.

Private diligence export:

```bash
./scripts/export_linuxoneliners_analytics_proof.sh
```

The export creates a local package with the public proof, event schema, aggregate database summaries, remote SQLite/access-log checksums, and a manifest. This is intended for buyer, broker, or escrow review. It is first-party operational evidence, not a third-party revenue audit.

Traffic-log retention:

```bash
./scripts/install_vps_log_retention.sh
```

The VPS keeps Nginx access logs bounded with daily compressed rotation and 90 retained rotations. Before each rotation it writes compact summaries under:

```text
/srv/data/site-traffic-summaries/YYYY-MM-DD/
```

Those summaries include source-log checksums, status counts, top paths, user-agent counts, and scanner-like path counts. They are much smaller than raw logs and are kept for longer due-diligence history.

Events currently include:

- page views
- command copy clicks
- export link clicks
- scroll depth
- referrer
- browser language
- timezone
- viewport
- connection hints where the browser exposes them
- navigation timing
- user agent and request IP from the server side

This is intentionally site-owned data for load planning, content testing, monetization tests, and future sale diligence.

## Disposable Demo Vessels

Run command demos in isolated containers:

```bash
./scripts/run_linuxoneliners_demos.sh
```

Artifacts:

```text
sites/linuxoneliners.com/artifacts/demos/
```

Published paths after build:

```text
/demos/summary.json
/demos/<lesson-slug>/terminal.txt
/demos/<lesson-slug>/demo.json
```

Current isolation controls:

- network disabled
- capabilities dropped
- no-new-privileges
- memory limit
- CPU limit
- PID limit
- fake fixtures
- secret-pattern scan before artifact write

These artifacts are the source material for CI checks, screenshots, terminal animations, Shorts, and voiceover scripts.

## Build

```bash
ruby sites/linuxoneliners.com/build.rb
```

## Content Check

```bash
./scripts/check_linuxoneliners_content.py
```

The curation gate blocks lessons that are missing safety notes, demo steps, Shorts metadata, LinkedIn prompts, or experiment fields.

## Lesson Packs

Seed lessons live in:

```text
content/lessons.json
```

Scaled batches live in:

```text
content/packs/*.json
```

The builder, content checker, and demo vessel runner load both. This keeps the site workable when the library grows from dozens to hundreds or thousands of one-liners.

## Template Model

The site extends shared ERB templates:

```text
templates/layout.erb
templates/index.erb
templates/lesson.erb
```

All one-liner pages share `lesson.erb`, so comments, ads, demo output, SSO prompts, related lessons, and conversion widgets can be added once and rolled across every generated page.

Output:

```text
sites/linuxoneliners.com/dist/
```

## Next Granular Build Slices

1. Add ten Linux Survival Basics lessons.
2. Add a terminal fixture runner that executes commands in disposable containers.
3. Add terminal capture with asciinema or a screenshot renderer.
4. Add Shorts metadata export as CSV/JSON.
5. Add a deployment script for `linuxoneliners.com`.
6. Add privacy-safe analytics events for copy buttons and outbound video clicks.
7. Add a local quality gate that blocks lessons missing danger, undo, expected output, or experiment notes.
