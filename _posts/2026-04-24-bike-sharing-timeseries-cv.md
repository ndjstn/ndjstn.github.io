---
title: "When a Naive Baseline Beats LightGBM: Bike-Sharing Demand with Proper Time-Series CV"
date: 2026-04-24 00:00:07 -0500
description: "Random 5-fold CV makes LightGBM look like the winner on UCI hourly bike-sharing data. Time-series CV says a naive seasonal-mean baseline beats it."
image:
  path: /assets/img/posts/bike-sharing-timeseries-cv/hero.png
  alt: "Grouped bar chart comparing random 5-fold and time-series 5-fold RMSLE for naive seasonal, Ridge, and LightGBM models."
tags:
  - "Time Series"
  - "Cross-Validation"
  - "Bike Sharing"
  - "LightGBM"
  - "Forecasting"
categories:
  - "Data Science"
---

Run the UCI hourly bike-sharing dataset through a random 5-fold cross-validation and LightGBM comes out at 0.405 RMSLE and looks like the obvious winner. Run the same data through a `TimeSeriesSplit` — train on everything before a certain date, validate on what comes next — and LightGBM lands at 0.595 RMSLE. It gets beaten by a naive "seasonal mean of (weekday, hour)" baseline that scores 0.509.

That's the whole story. Validation strategy matters more than model choice when your data has time structure. Random k-fold shuffles the time dimension away, so training folds always contain information from the future relative to the validation fold. The model learns from data it won't have at inference time, and the CV score flatters it accordingly.

<!-- YouTube embed will go here once the walkthrough video is published -->

## The dataset

17,379 hourly observations from UCI's Bike Sharing Dataset (Capital Bikeshare, Washington DC, 2011-2012). The target is `cnt`, the total rentals in that hour. 14 weather and calendar features per row. Data runs continuously from January 2011 through December 2012.

![Heatmap of mean hourly bike rentals by weekday and hour of day.](/assets/img/posts/bike-sharing-timeseries-cv/hour-weekday-heatmap.png)

*Weekday mornings hit a sharp 8am peak and a bigger 5-6pm peak — classic commuter pattern. Weekends are flatter with one broad afternoon bulge. That bimodal-weekday vs. unimodal-weekend pattern is why a naive seasonal baseline indexed on (weekday, hour) does well.*

![Monthly rental totals across 2011 and 2012 showing strong seasonality and year-over-year growth.](/assets/img/posts/bike-sharing-timeseries-cv/monthly-seasonality.png)

*Summer peaks, winter troughs. 65 percent growth between 2011 and 2012 as the program matured. That growth is another reason time-series CV matters: random k-fold lets a model see 2012 growth during training on "2011 test" folds, which doesn't happen in deployment.*

## The headline comparison

![Grouped bar chart: random k-fold vs. time-series 5-fold for naive seasonal, Ridge, and LightGBM. Naive seasonal wins under time-series CV.](/assets/img/posts/bike-sharing-timeseries-cv/hero.png)

Three models, two cross-validation strategies, six bars.

| Model | Random k-fold | Time-series 5-fold |
| --- | ---: | ---: |
| Naive seasonal (weekday × hour) | 0.644 | **0.509** |
| Ridge with cyclical features | 1.151 | 1.230 |
| LightGBM with seasonal features | **0.405** | 0.595 |

Under random k-fold, LightGBM wins by a wide margin. Under time-series CV, the naive seasonal baseline wins. LightGBM's time-series score is 47 percent worse than its k-fold score. The naive baseline's time-series score is actually 21 percent better than its k-fold score, which is a separate quirk — time-series CV happens to evaluate on the later, more-data-rich second year where the seasonal mean is more stable.

Ridge is terrible under both strategies. Log-RMSLE above 1.0 means multiplicative errors of 2-3x on typical predictions. Linear models with `sin(hour) + cos(hour)` don't capture demand shape well enough; a tree model gets much more of the non-linearity for free.

## Why random k-fold is misleading here

![Animation showing time-series 5-fold being revealed fold by fold: training sets grow, validation windows slide forward through the two-year range.](/assets/img/posts/bike-sharing-timeseries-cv/timeseries-cv-animation.gif)

*Time-series split: for fold k, train on everything before timestamp T(k), validate on what comes after. Training grows each fold; validation slides forward. Mirrors deployment — train on the past, predict the future.*

Random 5-fold ignores the time dimension. A training fold pulled uniformly at random from the 17,379 rows contains observations from December 2012 and April 2011 mixed together. Validating on the held-out 20 percent, which also spans all 24 months, the model sees future-like information during training that it would never see at inference time.

The leakage shows up even in the naive baseline — not a trained model, just `cnt.groupby(["weekday", "hr"]).mean()`. Under random k-fold, the seasonal mean is computed on training data spanning all 24 months; the validation fold also spans 24 months; both see the same year-over-year growth level; residuals are small. Under time-series CV, the seasonal mean is computed on pre-validation data that doesn't include the validation period's growth; residuals are larger. Net: random k-fold overstates generalisation.

## What actually wins

Nothing beats LightGBM's 0.595 among the three models I tested, in isolation. But the naive baseline's 0.509 sits inside what LightGBM achieves — the base rate of "what's the average demand for this weekday and hour" explains more variance than LightGBM's weather + cyclical + interaction features recover on truly unseen future data.

The practical implication is that deploying LightGBM as a residual predictor on top of the naive baseline (model the difference between actual demand and the seasonal mean, then add back) usually works better than deploying LightGBM directly. That approach isn't in this project, but it's the natural next step for anyone continuing the work.

## What this isn't

Not a Kaggle-competitive submission. Kaggle's Bike Sharing Demand competition uses a held-out test set from the same time window as training, so random sampling gives a reasonable approximation of what the public leaderboard measures. The point of this project is the opposite — the difference between Kaggle-style random evaluation and what you'd actually hit in deployment.

Not a test of modern time-series models either. No Prophet, no LSTM, no Temporal Fusion Transformer. The argument isn't that neural nets can't beat LightGBM here; it's that any model bound for production needs time-respecting validation so you know what score it'll actually hit.

## Reproducibility note

Source code, notebook, outputs at [github.com/ndjstn/bike-sharing-demand](https://github.com/ndjstn/bike-sharing-demand). Dataset is the `hour.csv` from the UCI Bike Sharing Dataset mirror on Kaggle ([Fanaee-T & Gama, 2014](#ref-fanaee)).

## References

<div id="ref-fanaee" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Fanaee-T, H., &amp; Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. <em>Progress in Artificial Intelligence, 2</em>(2-3), 113-127. <a href="https://doi.org/10.1007/s13748-013-0040-3">https://doi.org/10.1007/s13748-013-0040-3</a>
</div>

<div id="ref-bergmeir" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Bergmeir, C., &amp; Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. <em>Information Sciences, 191</em>, 192-213. <a href="https://doi.org/10.1016/j.ins.2011.12.028">https://doi.org/10.1016/j.ins.2011.12.028</a>
</div>

<div id="ref-hyndman" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Hyndman, R. J., &amp; Athanasopoulos, G. (2021). <em>Forecasting: Principles and Practice</em> (3rd ed.). OTexts. <a href="https://otexts.com/fpp3/">https://otexts.com/fpp3/</a>
</div>
