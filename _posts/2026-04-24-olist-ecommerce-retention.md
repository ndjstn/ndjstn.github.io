---
title: "93,000 Customers and No Repeats: Olist Brazilian E-Commerce Analytics"
date: 2026-04-24 00:00:08 -0500
description: "Olist's 2016-2018 marketplace data shows 97 percent single-purchase customers. Revenue growth is all acquisition; retention is near-zero by construction."
image:
  path: /assets/img/posts/olist-ecommerce-retention/hero.png
  alt: "Cohort retention heatmap for Olist: rows are first-purchase month, columns are months since, near-zero retention beyond month 0."
tags:
  - "E-Commerce"
  - "RFM Segmentation"
  - "Cohort Analysis"
  - "Customer Retention"
  - "Business Intelligence"
categories:
  - "Data Science"
---

Olist is a Brazilian marketplace platform. The public dataset carries 100,000 orders across 93,000 customers from 2016 to 2018, spread across nine CSV tables that join into a clean star schema. No machine learning in this project. The interesting work is joining the tables, building an RFM segmentation that a marketing team could actually use, and looking at cohort retention to answer the question every e-commerce operator wants answered: how many of these customers come back?

Almost none. More than 97 percent of Olist customers appear in exactly one delivered order during the observation window. Platform revenue growth through 2017-2018 comes from acquiring new customers, not repeat purchases. That shapes everything else.

<!-- YouTube embed will go here once the walkthrough video is published -->

## The cohort reality

![Cohort retention heatmap: rows are the first-purchase month, columns are months since first purchase, colour is retention rate.](/assets/img/posts/olist-ecommerce-retention/hero.png)

*Row is the month a customer first purchased. Column is months since. Cell is the share of the cohort that ordered again in that month. Month 0 is 100 percent by construction. Months 1 through 11 are nearly dark — almost every cell sits below 3 percent retention.*

This isn't a retention problem, it's a business model. Olist sells mostly one-time-purchase items — appliances, home goods, furniture, electronics. Customers come in when they need a specific thing, buy it, and don't return until they need another specific thing, which may be years later. The effective customer "lifetime" in the data is about 45 days; past that, almost nobody comes back.

For an operator looking at this data, the practical takeaway is that customer acquisition cost matters more than customer lifetime value, because LTV collapses to first-order value. Acquisition-channel efficiency and order-level economics are where optimisation pays off.

## Monthly revenue

![Monthly Olist revenue 2016-2018 with a November 2017 spike for Brazilian Black Friday.](/assets/img/posts/olist-ecommerce-retention/monthly-revenue.png)

*Revenue ramps from zero in late 2016 through Q1 2017 growth, a clear Black Friday spike in November 2017, and steady 2018 levels around R$ 1M/month. Total across the window: R$ 15.4M.*

## RFM segmentation

Six archetypes, quintile-scored on Recency / Frequency / Monetary:

| Segment | Customers | Revenue (BRL) | Revenue/customer |
| --- | ---: | ---: | ---: |
| Big spenders | 10,337 | 3,021,467 | 292 |
| Champions | 14,871 | 2,631,536 | 177 |
| At risk | 14,919 | 2,529,831 | 170 |
| New / Recent | 14,984 | 2,448,694 | 163 |
| Lost | 14,986 | 2,441,760 | 163 |
| Others | 23,261 | 2,346,486 | 101 |

![Two-panel bar chart: customer count per RFM segment on the left, total revenue per segment on the right.](/assets/img/posts/olist-ecommerce-retention/rfm-segments.png)

*Big spenders is the smallest segment but pulls the highest revenue total at R$ 292 per customer. At-risk customers are where a retention program would point first — 14,919 of them, historically frequent, currently dormant.*

![Recency-vs-monetary scatter with log y-axis, coloured by frequency. Frequency stays at 1 for most customers.](/assets/img/posts/olist-ecommerce-retention/rfm-scatter.png)

*Most customers sit in a dense low-monetary band across the recency range. The high-monetary tail is sparse and mostly single-purchase customers who bought one expensive item. Frequency barely exceeds 1 anywhere on the chart.*

## Delivery time drives review scores

![Bar chart of mean review score against delivery-time bucket.](/assets/img/posts/olist-ecommerce-retention/review-vs-delivery.png)

*0-3 day delivery averages 4.5-star reviews. 15-21 days averages 3.3. Past 30 days, below 3.0 — where customers stop recommending a retailer. Brazilian geography makes 40-day delivery not rare in this data, and the cost shows up directly in the review distribution.*

## Revenue by state

![Animation stepping through monthly revenue across the top 8 Brazilian states, with São Paulo (SP) highlighted.](/assets/img/posts/olist-ecommerce-retention/state-revenue-animation.gif)

*São Paulo dominates every month by 2-3x over the next state. Rio, Minas Gerais, and Rio Grande do Sul form the next cluster. The ordering is stable — no state overtakes SP at any point in the observation window.*

## What this isn't

Not a revenue forecast. The 2018 tail of the dataset is incomplete; the last-month numbers are partially a cutoff artifact.

Not a full LTV model. With 97 percent single-purchase customers, the usual LTV math collapses to first-order value plus a tiny tail. A full churn-model LTV is overkill for this data.

Not a vendor-economics analysis. Seller IDs are present but seller costs aren't, so the margin side of the marketplace isn't visible.

## Reproducibility note

Source, notebook, outputs at [github.com/ndjstn/olist-ecommerce-analytics](https://github.com/ndjstn/olist-ecommerce-analytics). Dataset: Olist Brazilian e-commerce dataset on Kaggle ([Olist, n.d.](#ref-olist)).

## References

<div id="ref-olist" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Olist. (n.d.). <em>Brazilian e-commerce public dataset by Olist</em> [Data set]. Kaggle. <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce">https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce</a>
</div>

<div id="ref-fader" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Fader, P. S., Hardie, B. G. S., &amp; Lee, K. L. (2005). RFM and CLV: Using iso-value curves for customer base analysis. <em>Journal of Marketing Research, 42</em>(4), 415-430.
</div>

<div id="ref-blattberg" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Blattberg, R. C., Getz, G., &amp; Thomas, J. S. (2001). <em>Customer Equity: Building and Managing Relationships as Valuable Assets</em>. Harvard Business School Press.
</div>
