---
title: "From 541,909 Invoice Lines to Four Customer Segments"
date: 2026-05-31 00:00:00 -0500
description: "RFM-based K-means segmentation on the UCI Online Retail dataset turns invoice-level transactions into four testable marketing groups: high-value repeat buyers, regular buyers, occasional high-spend customers, and inactive customers."
image:
  path: /assets/img/posts/online-retail-rfm-segmentation/hero.png
  alt: "Bubble chart of customer segments by average invoices and average spend, with bubble size showing segment count."
tags:
  - "Customer Segmentation"
  - "RFM"
  - "KMeans"
  - "Retail Analytics"
  - "Marketing Analytics"
categories:
  - "Data Science"
---

The raw Online Retail file is not a customer dataset at first. It is an invoice-line dataset: 541,909 rows of products, quantities, invoice dates, prices, and customer IDs from a United Kingdom online gift retailer. That format is useful for accounting. It is not directly useful for deciding whether a customer should receive loyalty recognition, a product recommendation, a seasonal reminder, or a low-cost win-back message.

This project turns the invoice lines into customer-level RFM records: recency, frequency, and monetary value. The final modeling table has 4,338 customers. From there, a four-cluster K-means segmentation produces a practical set of marketing hypotheses: high-value repeat buyers, regular buyers, occasional high-spend customers, and inactive or at-risk customers.

The main point is restrained. These clusters do not prove business impact. They organize customer behavior into groups that a retailer could test with holdouts, repeat-purchase tracking, conversion measurement, and average-order-value comparisons.

## The unit of analysis changed

![Bar chart showing the data preparation path from raw transaction rows to final customer RFM records.](/assets/img/posts/online-retail-rfm-segmentation/cleaning_counts.png)

The largest row loss happens immediately. Of the 541,909 raw rows, 135,080 do not have `CustomerID`, so they cannot become customer-level records. After removing canceled invoices and invalid quantity or price values, 397,884 transaction rows remain. Aggregating those rows by customer produces the 4,338-record RFM table used for segmentation.

![Horizontal bar chart showing rows removed by missing CustomerID, canceled invoice, and invalid quantity or price.](/assets/img/posts/online-retail-rfm-segmentation/rows_removed.png)

The important decision is not just cleaning; it is choosing the right row. Clustering invoice lines would answer the wrong question because one customer can have many product lines in a single purchase. The business question is about customer behavior, so each customer needs one profile.

## RFM features

The baseline model uses only three fields:

| Feature | Definition | Business meaning |
| --- | --- | --- |
| Recency | Days from December 10, 2011 to the customer's most recent invoice date | How recently the customer purchased |
| Frequency | Count of unique invoices by `CustomerID` | How regularly the customer buys |
| Monetary | Sum of `Quantity * UnitPrice` by `CustomerID` | How much value the customer generated |

Frequency is counted by unique invoice, not product line. That matters because a customer with one large basket should not look like a frequent buyer just because the invoice contains many line items.

## Why four clusters

![Line chart of K-means cluster-count diagnostics, showing inertia falling as k increases and silhouette peaking at k equals 4.](/assets/img/posts/online-retail-rfm-segmentation/cluster_diagnostics.png)

K-means was fit after scaling the RFM fields with `StandardScaler`. The report tested `k = 2` through `k = 7` and selected `k = 4` because it gave the best balance of silhouette score and business interpretability. Inertia keeps falling as more clusters are added, which is expected. The useful check is whether additional clusters make the segmentation clearer. Past four, the added complexity did not improve the business read.

| k | Inertia / 1,000 | Silhouette | Decision |
| ---: | ---: | ---: | --- |
| 2 | 25.6 | 0.41 | Reject |
| 3 | 17.8 | 0.46 | Maybe |
| 4 | 12.6 | 0.52 | Select |
| 5 | 10.9 | 0.49 | Reject |
| 6 | 9.7 | 0.43 | Reject |
| 7 | 8.9 | 0.39 | Reject |

## The four segments

![Bar chart showing final segment counts: high-value repeat, regular buyers, occasional high-spend, and inactive at-risk.](/assets/img/posts/online-retail-rfm-segmentation/segment_counts.png)

| Segment | Customers | Recency, days | Invoices | Avg. spend, GBP | Business read |
| --- | ---: | ---: | ---: | ---: | --- |
| High-value repeat | 304 | 16.4 | 18.2 | 5,920 | Protect and reward |
| Regular buyers | 2,082 | 43.7 | 4.6 | 1,120 | Grow average value |
| Occasional high-spend | 911 | 84.1 | 2.0 | 1,850 | Encourage return timing |
| Inactive / at-risk | 1,041 | 219.6 | 1.3 | 321 | Re-engage carefully |

The largest segment is regular buyers. That matters operationally because small changes in repeat purchase rate or average order value could affect many customers. The smallest segment, high-value repeat buyers, is the strongest by frequency and average spend. That group should be protected carefully rather than casually discounted.

![Grouped bar chart showing normalized recency score, frequency score, and monetary score by segment.](/assets/img/posts/online-retail-rfm-segmentation/segment_rfm_profile.png)

For the profile chart, recency is inverted into a `RecencyScore` so higher values mean more recent purchase activity. That keeps the visual direction consistent: higher is better for recent activity, invoice frequency, and spend.

## What to test

High-value repeat customers should not automatically receive broad discounts. They are already active and valuable, so a better first test is loyalty recognition, early access, or personalized recommendations.

Regular buyers are the cleanest growth segment. They are active but not at the top of the value distribution. A useful experiment would compare bundle offers, recommendation emails, and a holdout group.

Occasional high-spend customers need a different interpretation. Lower frequency does not mean low value. Seasonal reminders, replenishment timing, or bulk-order messaging fit this group better than generic reactivation.

Inactive or at-risk customers should receive low-cost reactivation first. A win-back message or small incentive can be tested before spending heavily on a segment that may not respond.

## What this is not

Not a supervised model. There is no accuracy score because there is no labeled target. The validation evidence is internal clustering diagnostics plus whether the segment profiles are coherent enough to support business tests.

Not proof of marketing lift. The output is a segmentation hypothesis. The next step is an experiment with a holdout or comparable baseline.

Not a full customer-lifetime-value model. The baseline intentionally uses only RFM fields. Product category, seasonality, discount exposure, acquisition source, and return behavior are all outside this first pass.

Not an identity model. `CustomerID` is a grouping key, not a model feature. The segments describe historical buying behavior, not customer personalities.

## Reproducibility note

Source code, notebook, generated figures, output tables, and presentation transcript are at [github.com/ndjstn/online-retail-rfm-segmentation](https://github.com/ndjstn/online-retail-rfm-segmentation). The notebook downloads the UCI Online Retail file, rebuilds the customer-level RFM table, and writes the portfolio figures.

Dataset: Online Retail dataset from the UCI Machine Learning Repository ([Chen, 2015](#ref-chen)).

## References

<div id="ref-chen" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Chen, D. (2015). <em>Online Retail</em> [Data set]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5BW33">https://doi.org/10.24432/C5BW33</a>
</div>

<div id="ref-chen-sain-guo" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Chen, D., Sain, S. L., &amp; Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. <em>Journal of Database Marketing &amp; Customer Strategy Management, 19</em>(3), 197-208. <a href="https://doi.org/10.1057/dbm.2012.17">https://doi.org/10.1057/dbm.2012.17</a>
</div>

<div id="ref-fader" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Fader, P. S., Hardie, B. G. S., &amp; Lee, K. L. (2005). RFM and CLV: Using iso-value curves for customer base analysis. <em>Journal of Marketing Research, 42</em>(4), 415-430.
</div>

<div id="ref-macqueen" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. <em>Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1</em>, 281-297.
</div>

<div id="ref-rousseeuw" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. <em>Journal of Computational and Applied Mathematics, 20</em>, 53-65. <a href="https://doi.org/10.1016/0377-0427(87)90125-7">https://doi.org/10.1016/0377-0427(87)90125-7</a>
</div>

<div id="ref-pedregosa" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &amp; Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. <em>Journal of Machine Learning Research, 12</em>, 2825-2830. <a href="https://jmlr.org/papers/v12/pedregosa11a.html">https://jmlr.org/papers/v12/pedregosa11a.html</a>
</div>
