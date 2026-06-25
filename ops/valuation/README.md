# Site Valuation Tracking

Track each site as if a skeptical buyer will inspect it later.

Flippa guidance consistently points to a few practical themes:

- profitable sites are commonly valued from monthly net profit or SDE
- buyers want verifiable traffic and revenue evidence
- clean financial records improve credibility
- traffic source diversity matters
- transferable assets and low operational burden reduce buyer risk

Useful references:

- https://flippa.com/blog/how-much-is-my-website-worth/
- https://flippa.com/blog/how-much-can-you-sell-your-website-for/
- https://flippa.com/blog/how-much-is-your-blog-worth-the-complete-valuation-guide/
- https://flippa.com/blog/due-diligence-checklist/
- https://support.flippa.com/hc/en-us/articles/360001508695-Maximizing-views-on-your-Flippa-listing

## Monthly Inputs

Minimum monthly metrics:

```text
page_views
unique_visitors
organic_search_visits
direct_visits
social_visits
youtube_views
email_subscribers
gross_revenue
affiliate_revenue
product_revenue
ad_revenue
hosting_cost
tooling_cost
contractor_cost
net_profit
owner_hours
```

Quality notes:

```text
top_pages
top_queries
top_referrers
best_video
best_linkedin_post
experiments_run
content_added
operational_issues
transfer_risks
```

## Valuation Model

Use a rough range, not a fake precise number:

```text
low = trailing_average_monthly_net_profit * 30
mid = trailing_average_monthly_net_profit * 35
high = trailing_average_monthly_net_profit * 45
```

For sites without profit, track an internal build value separately:

```text
traffic_asset_score
content_inventory
email_list_quality
domain_quality
transferability
```

Do not pretend traffic alone is exit value. Traffic is a signal. Profit is what usually prices the deal.

## A/B Testing Log

Each test needs:

```text
hypothesis
variant_a
variant_b
start_date
end_date
primary_metric
result
decision
```

Default metrics:

- YouTube Shorts: 3-second hold, 15-second retention, average view duration, subscribers gained
- LinkedIn: comment rate, save rate, profile clicks, link clicks
- Site: page CTR, related lesson clicks, copy-button events, email conversion
- Revenue: affiliate clicks, RPM, product conversion, net profit
