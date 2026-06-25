# Retained Operator Platform

This repository can host public site code, but the business should treat the operator platform as a separate retained asset.

The operator platform is the private system used to create, measure, moderate, and package sites.

## Keep

- cross-site dashboard
- content generation tooling
- container demo runner
- video rendering pipeline
- voiceover pipeline
- moderation prompts/models
- deployment playbooks
- valuation dashboard
- site-factory templates
- private experiments across all sites

## Sell With A Site

- domain
- public source for that site
- site database export
- site analytics export
- user/comment data if privacy terms allow transfer
- revenue proof
- expense proof
- campaign history for that site
- moderation records for that site

## Dashboard Scope

The private dashboard should track:

- live traffic and load
- LinkedIn campaign visits
- YouTube campaign visits
- command copy events
- comments and moderation queue
- ad impressions and clicks
- affiliate clicks
- product/email conversions
- revenue and expenses
- estimated valuation
- export readiness

## Carve-Out Rule

Every table, file, and service should be tagged as one of:

```text
sellable_site_asset
retained_operator_asset
shared_open_template
```

If it is not tagged, assume it is retained until reviewed.
