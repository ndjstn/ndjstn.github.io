---
title: "Five Archetypes at the Top of YouTube: Creator Segmentation on the Global Top-995"
date: 2026-04-24 00:00:09 -0500
description: "KMeans on the Global YouTube Statistics 2023 dataset produces five distinct creator archetypes: mega-scale, mainstream, low-engagement, music-video, and upload machines."
image:
  path: /assets/img/posts/youtube-top995-five-archetypes/hero.png
  alt: "Scatter plot of YouTube creators on subscribers × views-per-subscriber, coloured by KMeans cluster assignment."
tags:
  - "YouTube"
  - "KMeans"
  - "Creator Economy"
  - "Clustering"
  - "Data Analysis"
categories:
  - "Data Science"
---

The Global YouTube Statistics 2023 dataset has 995 top creators with subscriber counts, lifetime video views, upload counts, category, country, and a handful of derived stats. It's a snapshot from mid-2023, so forecasting from it is a mistake — the numbers have shifted since. What's worth doing is finding the structure. KMeans at k=5 on standardised log-features produces five clusters that map onto five recognisable archetypes of top creator, which is a more interesting output than another ranking of MrBeast.

<!-- YouTube embed will go here once the walkthrough video is published -->

## Five archetypes

![Scatter plot of creators on subscribers × views-per-subscriber axes, coloured by cluster assignment.](/assets/img/posts/youtube-top995-five-archetypes/hero.png)

| Cluster | Median subs | Median views | Median uploads | Views/sub | n | Label |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 36.5M | 22.6B | 1,331 | 592 | 135 | **Mega-scale creators** |
| 1 | 16.0M | 7.5B | 719 | 436 | 367 | **Mainstream large channels** |
| 2 | 14.9M | 2.2B | 461 | 133 | 158 | **Low-engagement large channels** |
| 3 | 20.7M | 9.4B | 12 | 445 | 114 | **Music-video channels** |
| 4 | 16.9M | 9.5B | 10,022 | 548 | 175 | **Upload machines** |

**Cluster 0** is the top-tier scale phenomenon. 36M median subs, 22B median views, 1,331 uploads. Mr. Beast, PewDiePie, the top 10 or so individual creators plus a hundred other channels at similar scale. Views per subscriber of 592 means each subscriber has, on average, watched nearly 600 videos on the channel over its lifetime — absurd viewer depth.

**Cluster 1** is the mainstream large-channel group — 367 channels at 16M median subs and 7.5B median views. Big, not mega. The largest bucket by count.

**Cluster 2** is interesting for what's missing. 14.9M subs but only 2.2B views. Views per subscriber is 133 — a quarter of every other cluster. Large subscriber counts, low engagement per subscriber. The explanation is usually one of two things: subscribers accumulated during a past hit run and most don't watch new content (common for kids-content channels that gamed the 2015-2020 recommendation algorithm), or the channel's audience is fragmented across short-form content where each upload reaches only a slice.

**Cluster 3** is the music-video cluster. Median uploads: 12. An order of magnitude below every other cluster. 20M subs, 9B views from a handful of extremely viral videos. Major-label music channels and one-hit-wonders. Views per upload is 668 million — highest in the dataset, because each upload is a full music release that accumulates hundreds of millions of plays.

**Cluster 4** is the upload-machine cluster. 10,022 median uploads against 16.9M subs and 9.5B views. News networks, clip-compilation channels, daily-vlog channels, and automated content farms. Views per upload is just 767k — two orders of magnitude below the music-video cluster — but high posting volume compensates.

![KMeans elbow plot: SSE vs. k from 2 to 10, bend at k=5.](/assets/img/posts/youtube-top995-five-archetypes/kmeans-elbow.png)

*The elbow sits between k=4 and k=5. At k=5 the resulting clusters separate interpretably; adding a sixth cluster splits one of the existing groups along a dimension that doesn't correspond to an obvious archetype.*

## Revealing clusters one at a time

![Animation revealing each cluster from smallest to largest median-subscriber, highlighting over a faded scatter of the full population.](/assets/img/posts/youtube-top995-five-archetypes/cluster-reveal-animation.gif)

*Steps through clusters in ascending order of median subscriber count.*

## Where the creators sit geographically

![Bar chart of top-15 countries by channel count in the top-995.](/assets/img/posts/youtube-top995-five-archetypes/top-countries.png)

*US leads, India and Brazil close behind. Long tail extends through most of Latin America, Europe, and the Gulf.*

![Bar chart of top-15 categories.](/assets/img/posts/youtube-top995-five-archetypes/top-categories.png)

*Entertainment leads at 241. Music second at 201. News & Politics at 26 — notably sparse for a dataset drawn from top-subscriber channels rather than top-influence channels.*

Per-capita, the ranking flips entirely.

![Top-15 countries by channels per 10M population, with 5+ top-995 channels.](/assets/img/posts/youtube-top995-five-archetypes/creator-density.png)

*Small English-speaking countries with concentrated creator economies punch above their weight — New Zealand, Ireland, Canada, Netherlands. The US sits mid-pack once calculated per capita; India ranks low after the 1.4 billion is divided through.*

## Estimated earnings by category

![Horizontal bar chart of median yearly earnings by category.](/assets/img/posts/youtube-top995-five-archetypes/category-earnings.png)

*Midpoints of Social Blade's reported earnings bands. Treat absolute numbers with significant skepticism — they're CPM-based estimates, not actual revenue. Category ordering is more trustworthy than the levels.*

## Subscribers track views, approximately

![Log-log scatter of subscribers vs. lifetime video views.](/assets/img/posts/youtube-top995-five-archetypes/subs-vs-views.png)

*A roughly linear relationship on log scale — an order of magnitude more views per order of magnitude more subs. The scatter around the line is what the clustering picks apart.*

## What this isn't

Not a longitudinal analysis. The dataset is a single snapshot. Trends can't be inferred from it.

Not representative of YouTube creators. It's the top 995 by subscriber count. Conclusions generalise only to the top of the distribution.

The earnings numbers are Social Blade CPM-based estimates. Creators routinely make more or less than those bands depending on sponsorships, merch, and content type. The midpoint I used is illustrative, not validated.

## Reproducibility note

Source, notebook, outputs at [github.com/ndjstn/youtube-global-stats](https://github.com/ndjstn/youtube-global-stats). Dataset: Global YouTube Statistics 2023 by Nelgiriye Withana on Kaggle ([Nelgiriye Withana, 2023](#ref-nelgiriye)).

## References

<div id="ref-nelgiriye" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Nelgiriye Withana, N. (2023). <em>Global YouTube Statistics 2023</em> [Data set]. Kaggle. <a href="https://www.kaggle.com/datasets/nelgiriyewithana/global-youtube-statistics-2023">https://www.kaggle.com/datasets/nelgiriyewithana/global-youtube-statistics-2023</a>
</div>

<div id="ref-socialblade" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Social Blade. (2023). <em>YouTube rankings and statistics</em>. <a href="https://socialblade.com">https://socialblade.com</a>
</div>

<div id="ref-macqueen" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. <em>Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1</em>, 281-297.
</div>
