# Traffic, Conversion, And Safety Tracking

Traffic needs to be reliable enough for decisions:

- content iteration
- load planning
- ad testing
- LinkedIn/YouTube campaign comparison
- valuation and sale diligence

## Required Tracking

Every public link we control should use campaign params:

```text
?src=linkedin&campaign=linux_lab_001&variant=a
?src=youtube_shorts&campaign=linux_lab_001&variant=a
?src=github&campaign=linux_lab_001&variant=repo
```

The backend should track:

- page view
- source/campaign/variant
- referrer
- visitor id
- session id
- viewport
- connection hints
- page timing
- scroll depth
- command copy
- demo export click
- YouTube outbound click
- LinkedIn inbound click
- comment/signup/purchase/conversion

## Reliability Checks

Run after deploy:

```bash
./scripts/check_linuxoneliners_live.sh
./scripts/report_linuxoneliners_analytics.sh
```

Weekly:

- export analytics DB
- check Nginx error logs
- check top 404s
- check slow pages
- check event volume by path
- check source/campaign conversions
- review moderation queue
- verify backups restore

## Safety Checks

Each command page must pass:

- curation validator
- disposable vessel run
- secret-pattern scan
- generated artifact review
- no real personal values
- no real credentials
- no destructive command as primary command without dry-run and recovery note

## Scale Target

First serious target:

```text
250 curated one-liners
```

Long-run ceiling:

```text
2,000-20,000 pages only if generated from real issue clusters and quality gates
```

The limiting factor is not page generation. It is quality, search intent, safety, and conversion signal.

## Content Verticals

Track each vertical separately:

- Linux survival basics
- web hosting operations
- server management
- cybersecurity triage
- dangerous commands
- Apple/macOS terminal
- developer workflow
- data/CSV/log workflows
- cloud/VPS operations
- CI/CD troubleshooting

Each vertical should have its own:

- campaign naming
- source links
- conversion goals
- moderation risk rating
- affiliate/product map
- sale value notes
