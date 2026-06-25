# LinuxOneLiners Backend Plan

The public site can stay static while the business backend grows behind it.

The backend has two asset classes:

## Sellable Site Assets

These are part of `linuxoneliners.com` if the site is sold, subject to the privacy policy and sale terms:

- public content
- command lesson catalog
- generated demo artifacts
- site-specific analytics
- site-specific users
- comments
- saved commands
- ad placements and performance history
- affiliate/product conversion data
- site-specific database schema and exports
- domain-specific brand assets

## Retained Operator Platform

These tools are ours by default and should not transfer with a normal site sale:

- content generation pipelines
- video/Shorts generation engine
- voiceover automation
- moderation models and prompts
- cross-site analytics dashboard
- site factory templates
- deployment automation
- valuation dashboard across all owned sites
- private playbooks and experiments

The boundary matters. A buyer can receive the site, its database, and its documented operating process without receiving the machinery that lets us create the next ten sites.

## Backend Modules

Start with one site-specific backend, then extract shared operator tooling later.

```text
/api/events          first-party analytics ingestion
/api/campaigns       LinkedIn/YouTube/source tracking links
/api/conversions     newsletter, affiliate, product, signup, comment events
/api/accounts        users, sessions, preferences
/api/commands        command submissions, saves, ratings
/api/comments        comments and discussion threads
/api/moderation      spam, abuse, unsafe command, PII review queue
/api/ads             placements, experiments, impressions, clicks
/admin               private dashboard
```

## Dashboard Jobs

The dashboard should answer:

- Which LinkedIn posts drove visits?
- Which YouTube Shorts drove visits?
- Which commands got copied?
- Which lessons got scroll depth?
- Which visitors returned?
- Which pages produced affiliate clicks?
- Which campaigns produced comments, signups, or purchases?
- Which commands produce support risk or moderation burden?
- Which pages are worth turning into longer videos?
- What is the current estimated site value?

## Campaign Tracking

Use explicit links instead of guessing from platform analytics:

```text
https://linuxoneliners.com/lessons/find-large-files/?src=linkedin&campaign=linux_lab_001&variant=a
https://linuxoneliners.com/lessons/find-large-files/?src=youtube_shorts&campaign=linux_lab_001&variant=b
```

Store:

- source
- campaign
- variant
- landing page
- session id
- visitor id
- event path
- conversion event
- revenue if any

## Moderation

Commands and comments need moderation because unsafe Linux instructions can be harmful.

Moderation queue reasons:

- spam
- harassment
- unsafe destructive command
- credential or secret leakage
- personal information
- malware-like behavior
- low-quality duplicate content
- affiliate/ad disclosure issue

Default behavior:

- comments are published only after passing automated checks or manual approval
- submitted commands are drafts until reviewed
- destructive commands require warnings and recovery notes
- user-submitted content can be removed without deleting internal moderation records

## Databases

Keep databases isolated:

```text
db_linuxoneliners_public      sellable site data
db_operator_platform          retained cross-site tools
```

For now, SQLite is acceptable for first-party analytics. Before accounts/comments/ads, move site data to Postgres with one database and one user per site.

## Sale Readiness

A clean sale package should include:

- site source
- site database export
- analytics export
- revenue proof
- expense record
- campaign performance
- moderation policy
- privacy policy
- transfer checklist
- excluded-asset list

Excluded by default:

- cross-site dashboard
- generation engine
- moderation model prompts
- private analytics across other sites
- site factory automation
