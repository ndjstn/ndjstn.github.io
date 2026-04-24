---
title: "A California Housing Model and a 35-Year Zillow Reality Check"
date: 2026-04-23 00:10:00 -0500
description: "A four-phase regression on the 1990 California Housing benchmark reaches RMSE 0.4834 with a clustered gradient boosting model, then checks itself against the 2025 Zillow Home Value Index across 36 California counties."
image:
  path: /assets/img/posts/california-housing-1990-model-2025-zillow/hero.png
  alt: "Side-by-side California maps showing 1990 block-group median values on the left and 2025 Zillow county ZHVI on the right, both colored on a shared yellow-to-red ramp."
tags:
  - "California Housing"
  - "Regression"
  - "Gradient Boosting"
  - "KMeans"
  - "Zillow"
  - "Scikit-learn"
categories:
  - "Data Science"
---

The California Housing benchmark is one of the oldest teaching datasets in applied machine learning. I have already written [a separate post about the dataset itself]({% post_url 2026-04-21-california-housing-prices-dataset %}) — what the rows mean, why the target is capped, why location is the dominant signal. This post is about what happens when you actually run a modeling project on it, then check the 1990 result against 2025.

The modeling half is a four-phase regression sequence. Raw features, engineered features, engineered-plus-coastline, and finally a geographic partition of California followed by a separate gradient boosting regressor fit inside each partition. The comparison half joins the 1990 block-group medians to the current Zillow Home Value Index for California counties and asks what the last thirty-five years did to the prices the model was trained on.

Both halves agree on one thing. Geography carried most of the signal in 1990 and has carried most of the dispersion in appreciation since.

{% include embed/youtube.html id='oVCvzCFw9fA' %}

## Four phases and why they exist

I ran a 4×4 grid. Four model families — ordinary least squares, ridge with cross-validated alpha, random forest, and gradient boosting — across four data configurations. The grid exists because the interesting question is not "which model wins" in the abstract. It is "which improvement step actually does the work."

![Five-stage pipeline diagram running from raw census features through engineering, coastline, clustering, and model fitting.](/assets/img/posts/california-housing-1990-model-2025-zillow/pipeline-diagram.png)

*The pipeline as a diagram. Each stage is a feature-engineering step, and each column in the grid is a model family fit under those features.*

Phase 1 is the baseline on the raw eight census features: median income, house age, average rooms and bedrooms per household, population, average occupancy, latitude, longitude. Phase 2 adds five engineered features: rooms per household, bedrooms per room, population density, income per capita, and an approximate distance-to-coast computed from a simplified California coastline polyline. Phase 3 isolates the coastline feature on top of the engineered configuration to see whether that spatial signal is additive. Phase 4 replaces the coastline feature with a geographic partition: KMeans with k=3 on latitude and longitude, then a separate gradient boosting regressor fit inside each partition.

The quick profile I worked against:

| Item | Value |
| --- | ---: |
| Rows | 20,640 |
| Raw features | 8 |
| Engineered features added | 5 |
| Target ceiling (hit by 4.68% of rows) | 5.0001 (in $100k units) |
| Median income correlation with target | 0.688 |
| Split | 80/20 with fixed seed |

The dataset ships through `fetch_california_housing` in scikit-learn ([scikit-learn, 2026](#ref-sklearn)), which wraps a California subset of the 1990 United States Census ([Pace &amp; Barry, 1997](#ref-pace-barry)).

## The signal is already almost entirely in latitude, longitude, and income

Before any of the phase work, the linear correlation story is simple. Median income correlates with median house value at about 0.688. Latitude and longitude look like weaker predictors as individual linear correlations, but together they carry most of the spatial structure that the dataset's geography proxies for — distance to coast, distance to major metros, climate, local policy.

![Dataset overview showing the row count, the feature table, and the target distribution with the capping bar at the right edge.](/assets/img/posts/california-housing-1990-model-2025-zillow/data-overview.png)

*The overview panel shows the shape of the problem. The target distribution has a distinctive spike at the right edge that is not a natural market pattern.*

That spike is the ceiling. The scikit-learn version of the dataset caps the target at 5.0001 (in hundred-thousand-dollar units), and 4.68 percent of rows sit at exactly that value. A regression model cannot distinguish a block group with a true median of $520,000 from one with a true median of $900,000 once both get flattened into the same endpoint. This shows up as a thin vertical stripe in any predicted-versus-actual plot parallel to the forty-five-degree line at the top end.

![Residuals from the best gradient boosting configuration showing a heavy central mass around zero and a right tail driven by the censored top-end values.](/assets/img/posts/california-housing-1990-model-2025-zillow/residuals.png)

*Residuals from the winning model. The right tail is not a modeling failure; it is the ceiling the data was preprocessed with, showing up exactly where it should.*

## Cluster first, then regress inside each cluster

Phase 4 is the interesting one. Instead of adding a spatial feature to the existing feature set, I partitioned California into three regions with KMeans on latitude and longitude, then fit a separate gradient boosting regressor inside each partition.

![KMeans k=3 clustering overlaid on block-group coordinates, separating the Bay Area corridor, the Los Angeles basin, and the inland-northern interior.](/assets/img/posts/california-housing-1990-model-2025-zillow/cluster-map.png)

*Three clusters. The Bay Area and Central Coast on one side, the Los Angeles basin on another, and inland and northern California on the third.*

The choice of k=3 matches California geography. Higher k values produced marginal changes in aggregate RMSE at the cost of interpretability and smaller per-cluster training sets that made the within-cluster fits noisier. Lower k (k=2) collapsed the distinct coastal regions into one bucket and lost the within-coast variation that was the whole point of segmenting.

The clustered configuration reached **RMSE 0.4834, MAE 0.3179, and R² 0.8217** on the held-out test split, in hundred-thousand-dollar units. That is a 5.6 percent RMSE reduction over the Phase 1 Random Forest baseline at 0.5118. Over the strongest single-model Phase 2 configuration (engineered Random Forest at RMSE 0.4885), the clustered gain is about 1.1 percent.

The phase-improvement chart makes those margins visible on one axis.

![Bar chart of best-model RMSE per phase showing a stepwise improvement from baseline through engineered through coast and into clustered.](/assets/img/posts/california-housing-1990-model-2025-zillow/phase-improvement.png)

*RMSE drops with each feature-engineering step. The largest gain is engineering; the next-largest is clustering. The coast-only feature in Phase 3 does not outperform engineered-alone.*

The full model comparison across all phases and all model families is the chart below.

![Full comparison matrix showing RMSE for each of four model families across four phases on a shared axis.](/assets/img/posts/california-housing-1990-model-2025-zillow/model-comparison.png)

*Every cell on one scale. The clustered gradient boosting cell wins on RMSE; the gap over engineered-only Random Forest is real but narrow.*

Coast-alone not outperforming engineered-alone is a useful negative result. It suggests that the spatial signal the coastline variable is carrying is already available to tree-based models through latitude and longitude once the engineered features are in place. The feature doesn't add; it duplicates.

Feature importance on the clustered model places median income and the two coordinate features in the top three, with the engineered ratios contributing secondary signal rather than driving the ranking.

![Horizontal bar chart of feature importance scores for the winning clustered gradient boosting model, with median income highest followed by latitude and longitude.](/assets/img/posts/california-housing-1990-model-2025-zillow/feature-importance.png)

*Median income dominates; latitude and longitude together carry the spatial structure; the engineered ratios contribute at a smaller scale.*

## Then I asked what Zillow said had happened since

The 1990 California Housing dataset is thirty-five years old. Any modern reader raises the same question: what does this model actually say about California housing today? The honest answer is not much, and the second half of the project is the exercise of saying so with numbers instead of assertion.

I joined the 1990 block-group medians to the current Zillow Home Value Index for California counties ([Zillow Research, 2025](#ref-zillow)). The two datasets do not align natively — the 1990 data is indexed at block-group granularity, Zillow at the county level — so each block group was assigned to one of thirty-six California counties using a bounding-box approximation based on county-extent rectangles. Block-group medians were aggregated to county medians before joining. The thirty-six county pairs became the comparison surface.

The appreciation distribution is heavy and very uneven.

![Histogram of per-county appreciation multipliers from 1990 to 2025, with a long right tail carrying the coastal and metro counties.](/assets/img/posts/california-housing-1990-model-2025-zillow/appreciation-distribution.png)

*Every matched county rose by a large multiple. The coastal and peninsula counties carry most of the absolute gains; some inland counties, starting from lower 1990 baselines, posted larger percent multipliers.*

The relative ranking of counties shifted. The coastal cluster led by the Bay Area pulled away far enough that some inland counties that were similar to coastal counties in 1990 are now in different tiers entirely. The pattern is not random. It is geographic. And it is the same geographic structure the modeling half surfaced — just amplified by thirty-five years of uneven supply, zoning politics, and commute-distance constraints.

## What this means for the 1990 model

The 0.4834 RMSE on the 1990 distribution is honest. It does not transfer to 2025 decisions. A model trained on 1990 features, even a good one, cannot recover the dispersion that appeared in the next three decades. The coastal counties whose 2025 values pulled away are counties that were already flagged as expensive in 1990, but the scale of their 2025 appreciation is not contained in the 1990 training signal.

The interpretive move, which I want to state plainly rather than bury in a limitations section, is this. The modeling half of the project is a controlled study in feature engineering and spatial segmentation. The comparison half shows how much the world has changed since 1990 relative to the training data. Taken together, they make a case for approaching any time-gap dataset with both a model and a descriptive reality check.

What I would not do with this model is use it for any consequential present-day decision. Not lending. Not tenant screening. Not any valuation task. The feature set omits variables that drive modern prices — school quality, commute infrastructure, property condition, local regulation — and the 1990 training distribution is no longer the distribution the predictions would be applied against. Both of those limits would surface in any serious fairness audit before the model left a notebook.

## Reproducibility note

Every number in this post comes from a pipeline that runs end-to-end from the scikit-learn loader through preprocessing, four-phase model fitting, threshold-free evaluation on a fixed train-test split, and the figure outputs used in the walkthrough video. The code, the notebook for the 1990-to-2025 comparison, and all of the figures live in the public repository at [github.com/ndjstn/california-housing](https://github.com/ndjstn/california-housing). The Zillow Home Value Index file is a static snapshot of the most recent month available at the time of analysis, checked in under the repository's comparison directory.

## References

<div id="ref-pace-barry" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Pace, R. K., &amp; Barry, R. (1997). Sparse spatial autoregressions. <em>Statistics &amp; Probability Letters, 33</em>(3), 291-297. <a href="https://spatial-statistics.com/pace_manuscripts/spletters_ms_dir/statistics_prob_lets/html/ms_sp_lets1.html">https://spatial-statistics.com/pace_manuscripts/spletters_ms_dir/statistics_prob_lets/html/ms_sp_lets1.html</a>
</div>

<div id="ref-sklearn" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
scikit-learn developers. (2026). <em>fetch_california_housing</em>. scikit-learn 1.8.0 documentation. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html">https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html</a>
</div>

<div id="ref-zillow" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Zillow Research. (2025). <em>Zillow Home Value Index (ZHVI) — data methodology and county-level downloads</em>. Zillow. <a href="https://www.zillow.com/research/data/">https://www.zillow.com/research/data/</a>
</div>

<div id="ref-geron2019" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Géron, A. (2019). <em>Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow</em> (2nd ed.). O'Reilly Media.
</div>

<div id="ref-pedregosa2011" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &amp; Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. <em>Journal of Machine Learning Research, 12</em>, 2825-2830. <a href="https://www.jmlr.org/papers/v12/pedregosa11a/pedregosa11a.pdf">https://www.jmlr.org/papers/v12/pedregosa11a/pedregosa11a.pdf</a>
</div>
