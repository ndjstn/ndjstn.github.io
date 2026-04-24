---
title: "Geography Beats the Odometer: NYC Taxi Trip Duration Prediction"
date: 2026-04-24 00:00:06 -0500
description: "Distance alone predicts 64 percent of trip duration. Adding geography and time of day gets you to 80 percent. The heatmap shows where the missing signal lives."
image:
  path: /assets/img/posts/nyc-taxi-geography-beats-odometer/hero.png
  alt: "Heatmap of median NYC taxi trip duration by hour and weekday, with midday Tuesday-Thursday as the slowest block."
tags:
  - "NYC Taxi"
  - "Trip Duration"
  - "XGBoost"
  - "Geospatial Features"
  - "Time Series"
categories:
  - "Data Science"
---

1.45 million NYC taxi trips from the 2016 public release. The obvious baseline is to predict trip duration from distance alone. It works — about 64 percent R² from distance by itself — and it also leaves a third of the usable signal on the table. An XGBoost model that adds geography and time features hits 80 percent R² and cuts MAE from 258 seconds to 177. Twenty-six percent less RMSLE from features the distance baseline ignores.

<!-- YouTube embed will go here once the walkthrough video is published -->

## The slow hours cluster

![Heatmap of median NYC trip duration by hour and weekday, with midday Tuesday through Thursday as the slowest block.](/assets/img/posts/nyc-taxi-geography-beats-odometer/hero.png)

*Tuesdays through Thursdays, midday, is where the slowest trips cluster. Saturday and Sunday stay flatter throughout the day. Sunday 3am-6am is the fastest block in the week. That is the single figure that tells you why distance alone isn't enough — a 3-mile trip at 8am Wednesday and the same 3-mile trip at 8am Sunday have materially different median durations.*

## The dataset

The Kaggle NYC Taxi Trip Duration mirror carries 1,458,644 trips with pickup/dropoff coordinates, pickup datetime, passenger count, vendor ID, and `trip_duration` in seconds. I filtered to trips between 60 seconds and 2 hours, coordinates inside the NYC bounding box, and Haversine distances between 0.1 and 80 km. That drops about 1.5 percent of the rows — mostly dropoffs mis-coded in the middle of the Atlantic or trips with durations under a minute.

I worked on a 150,000-row sample to keep training time under a minute per run. The full 1.4M behaves the same, just slower.

## Where the pickups happen

![Hexbin density of pickup coordinates across the NYC metropolitan area.](/assets/img/posts/nyc-taxi-geography-beats-odometer/pickup-density.png)

*Midtown and Lower Manhattan dominate. Brooklyn and Queens bands are thinner but visible. LGA and JFK show up as small bright cells in the east. Staten Island is nearly dark — cabs don't run there much.*

## Duration by hour

![Median trip duration in minutes across the 24 hours of the day, with a midday peak of 12-13 minutes and a 3am-6am valley of 8-9 minutes.](/assets/img/posts/nyc-taxi-geography-beats-odometer/duration-by-hour.png)

*The median starts around 8-9 minutes at 4am, rises through morning traffic, and plateaus at 12-13 minutes in the 10am-to-5pm window before dropping back down after 7pm. Rush hour shows up as a high plateau rather than a sharp spike because the distribution averages across many different trip geometries.*

## Two models on the same split

I fit two XGBoost models on an 80/20 split of log(duration+1). Distance-only baseline uses a single feature. The full model adds pickup/dropoff latitude and longitude, Manhattan-style rectilinear distance, hour of day, weekday, month, and passenger count.

| Model | Features | RMSLE | MAE (seconds) | R² |
| --- | --- | ---: | ---: | ---: |
| Distance only | 1 | 0.434 | 258.4 | 0.641 |
| Full | 10 | 0.323 | 176.6 | 0.802 |

Distance-only already explains 64 percent of variance. That's what makes this a useful baseline — the naive model works better than you might expect because distance really is the dominant predictor. But the full model cuts RMSLE by 26 percent and MAE by 32 percent, which is too big a gap to ignore.

![Feature importance bar chart showing dropoff lat/lon, pickup lat/lon, Manhattan distance, and Haversine distance at the top.](/assets/img/posts/nyc-taxi-geography-beats-odometer/feature-importance.png)

*The four coordinate features together outweigh the two distance features. Manhattan distance edges Haversine — NYC streets are a grid, and the sum of latitude and longitude differences captures actual driving distance better than as-the-crow-flies. Hour of day sits fifth. Passenger count is below 1 percent and mostly noise.*

## An hour-by-hour pickup tour

![Animation stepping through 24 hours of the day showing pickup locations and the median trip duration per hour.](/assets/img/posts/nyc-taxi-geography-beats-odometer/hourly-pickup-animation.gif)

*Twenty-four frames. The Manhattan core stays dominant across every hour. What changes is the outer-borough mix — LGA inbound pickups cluster at 7-9am, JFK outbound at 5-7pm — and the median trip duration, which moves from 8 minutes at 4am to 12 minutes at 2pm.*

## What this isn't

Not a live routing model. The dataset is seven years old; Manhattan traffic patterns have shifted with congestion pricing and the post-pandemic rebound. Anyone building a live duration estimator would want current data and weather covariates.

Not a competition-winning submission either. Top leaderboard solutions use OSRM routing distances, weather data, and stacked ensembles. 0.32 RMSLE lands mid-pack; winning entries reach 0.28.

## Reproducibility note

Source code, notebook, and outputs in [github.com/ndjstn/nyc-taxi-trip-duration](https://github.com/ndjstn/nyc-taxi-trip-duration). Dataset is the `NYC.csv` from yasserh's mirror of the Kaggle competition data ([Kaggle, 2017](#ref-kaggle)).

## References

<div id="ref-kaggle" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Kaggle. (2017). <em>NYC Taxi Trip Duration</em> [Data set]. <a href="https://www.kaggle.com/competitions/nyc-taxi-trip-duration">https://www.kaggle.com/competitions/nyc-taxi-trip-duration</a>
</div>

<div id="ref-tlc" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
New York City Taxi &amp; Limousine Commission. (2019). <em>TLC Trip Record Data</em>. <a href="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page">https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page</a>
</div>

<div id="ref-chen" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Chen, T., &amp; Guestrin, C. (2016). XGBoost: A scalable tree boosting system. <em>Proceedings of the 22nd ACM SIGKDD International Conference</em>, 785-794. <a href="https://doi.org/10.1145/2939672.2939785">https://doi.org/10.1145/2939672.2939785</a>
</div>
