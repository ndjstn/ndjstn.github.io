---
title: "The Default Threshold Is the Bug: Credit Card Fraud on the ULB Benchmark"
date: 2026-04-24 00:00:00 -0500
description: "A fraud classifier reaches 0.98 ROC-AUC on the famous ULB dataset and still ships a policy most fraud teams would reject. The 0.5 decision threshold is the bug."
image:
  path: /assets/img/posts/credit-card-fraud-default-threshold/hero.png
  alt: "Title card for a credit card fraud classifier project showing a cost curve bottoming out near threshold zero point zero one."
tags:
  - "Credit Card Fraud"
  - "Class Imbalance"
  - "XGBoost"
  - "Cost-Sensitive Learning"
  - "Machine Learning"
  - "Anomaly Detection"
categories:
  - "Data Science"
---

The Kaggle ULB credit card fraud dataset has 284,807 transactions and 492 of them are fraudulent. That is a 0.1727 percent positive rate, roughly one fraud case for every 580 legitimate charges. A classifier that predicts "legitimate" for every row is correct 99.83 percent of the time. If a coworker presents that classifier in a meeting with accuracy as the headline, the meeting is over.

The usual next move is to train a proper model and report 0.98 ROC-AUC. That number is real, and it is also beside the point. The operating question is whether the deployed threshold catches fraud at a cost the business can actually pay. On this dataset the 0.5 default threshold is a policy decision almost no one would sign off on if they saw the confusion matrix underneath it.

<!-- YouTube embed will go here once the walkthrough video is published -->

## Start with the base rate and nothing else

Every evaluation choice downstream has to be made in light of the class balance, so the class balance is the first thing to look at.

![Class balance with the fraud bar shown on a log scale alongside legitimate transactions.](/assets/img/posts/credit-card-fraud-default-threshold/class-balance.png)

*The fraud bar is small enough that a trivial classifier clears 99.83 percent accuracy without learning anything. The log scale is the only reason the positive bar is visible.*

At that prevalence, accuracy is a useless metric. The majority-class baseline is already above any reasonable threshold anyone might set for a model. ROC-AUC is less useless, but it treats every misclassification equally and rewards score rank-ordering that may or may not translate into a usable operating point. The diagnostic that matters is precision-recall, and the metric that ultimately matters is the expected cost of the decisions the classifier makes at a specific threshold.

Here is the quick profile I worked against:

| Item | Value |
| --- | ---: |
| Rows | 284,807 |
| Positive cases | 492 |
| Positive rate | 0.1727% |
| Test set positives | 123 |
| Majority-class accuracy | 99.83% |
| Stratified 75/25 split | yes |

The stratified split is not optional at this prevalence. A purely random 25 percent holdout could hand me a test set with fewer than 80 positives if the draw went badly, and recall estimates off that many positives are noisy enough that a single misclassified row shifts the number by a full percentage point.

## Three models and the one that does not belong

I fit three classifiers against a stratified 75/25 split. Logistic regression with balanced class weights. XGBoost with `scale_pos_weight` set to the empirical negative-to-positive ratio. And an Isolation Forest trained only on the legitimate transactions, which treats fraud as an anomaly-detection problem and does not see the labels during training.

The Isolation Forest is the one that does not belong, and I included it on purpose. The pitch for unsupervised fraud detection is that you do not need labels, which matters for novel fraud patterns that have never been labelled in the historical data. The counter-pitch is that labels carry information supervised models can use that geometry alone cannot recover. Running both on the same split makes the tradeoff concrete.

![Precision-recall curves for the three classifiers with average precision in the legend.](/assets/img/posts/credit-card-fraud-default-threshold/precision-recall.png)

*Precision-recall is the right diagnostic at this prevalence. Logistic regression reaches AP 0.7062, XGBoost reaches AP 0.863, and the Isolation Forest reaches AP 0.1133. The gap between supervised and unsupervised is two orders of magnitude wide here.*

XGBoost has the highest average precision, which makes sense: the PCA-transformed features separate the classes geometrically, and a boosted ensemble with an AUC-PR objective is a good fit for that kind of signal. Logistic regression is not far behind and has the property that its scores are relatively well calibrated out of the box, which matters for the threshold conversation that follows. The Isolation Forest is not a serious contender at this prevalence.

## The threshold is not a hyperparameter

The usual reflex is to report accuracy or F1 at the default 0.5 threshold and move on. That reflex is wrong on this dataset. The right question is what threshold minimises the total business cost, and to answer that I had to pick a cost model.

The one I used treats a missed fraud as a 100-EUR loss and a false positive as a 5-EUR manual review. Those numbers are illustrative rather than calibrated to any specific institution. They capture the order-of-magnitude asymmetry most fraud teams work under: a missed fraud is an order of magnitude more expensive than a manual review of a flagged transaction. Swapping either number for a real institution is a trivial edit; the framework does not change.

With that cost model in hand I swept the threshold from 0.01 to 0.99 for each classifier and computed expected cost at every step.

![Cost curve for XGBoost showing total cost bottoming out at threshold 0.01.](/assets/img/posts/credit-card-fraud-default-threshold/cost-curve-xgboost.png)

*Missed-fraud cost in red, review cost in blue, total in black. The total-cost curve bottoms out at a decision threshold of 0.01, which is not where most tutorials point the reader.*

That 0.01 number is startling the first time you see it. A classifier that flags a transaction as fraud as soon as it assigns a one-in-a-hundred probability is not a standard recommendation. But it is exactly right for this problem. At a 0.17 percent base rate, a posterior probability of 0.01 is a fifty-seven-fold lift over prior. The review-cost line stays shallow enough that the optimum sits further left than intuition suggests, and the missed-fraud cost is steep enough to dominate any choice of threshold to the right of the minimum.

At the XGBoost minimum-cost threshold of 0.01, the classifier catches 106 of 123 fraud cases, misses 17, and flags 79 legitimate transactions for review. Total cost: 2,095 EUR against a no-model baseline of 12,300 EUR.

![Confusion matrix for XGBoost at the min-cost threshold.](/assets/img/posts/credit-card-fraud-default-threshold/confusion-xgboost.png)

*The true-positive block dominates. The false-positive block is non-trivial, but 79 flagged legitimate transactions out of 71,079 in the test set is a review rate the fraud team can actually handle.*

Logistic regression lands in a similar place with a very different threshold number. Its cost curve bottoms out at 0.97 and catches 105 fraud cases for 2,310 EUR total cost. The two threshold numbers — 0.01 and 0.97 — are not comparable across models; XGBoost and logistic regression produce scores with different calibrations, so only the resulting cost and recall are comparable.

![Cost curve for logistic regression showing its optimum at threshold 0.97.](/assets/img/posts/credit-card-fraud-default-threshold/cost-curve-logistic_regression.png)

*Logistic regression reaches almost the same operating point from a different direction. The threshold is 0.97; the resulting cost is 2,310 EUR; the score scale is simply different.*

The Isolation Forest at its own minimum-cost threshold catches 72 of 123 fraud but flags 710 legitimate transactions for review. Total cost: 8,650 EUR. At the prices I assumed, it does worse than either supervised baseline by a factor of four. Unsupervised anomaly detection is not viable on this dataset when manual review has a non-trivial cost.

## What changes when the costs change

The ranking of the three models is not robust to the cost ratio. With a 500-EUR cost of a missed fraud, XGBoost's optimum moves further left and it misses fewer cases; the logistic regression optimum moves from 0.97 toward 0.80 and recall climbs. With a 50-EUR missed-fraud cost and a 10-EUR review cost, the optimum thresholds move right and recall drops. The ordering of XGBoost and logistic regression stays consistent across regimes, but the gap between them and the Isolation Forest widens further every time the missed-fraud cost goes up.

> The threshold is not a modelling decision. It is a policy decision.

A fraud analytics team deploying this classifier has to know its own cost structure before it can reason about where to place the threshold. That structure is almost never 1:1, which is what the 0.5 default implicitly assumes. Reporting a single operating point as "the model's performance" hides the decision that actually determines whether the classifier is useful in production.

## What this is not

This is an educational analysis on a public benchmark, not a deployable fraud system. A few honest limits have to be named before anyone takes this further.

The PCA anonymisation is the single largest constraint. Every feature except `Time` and `Amount` is a principal component, which prevents any meaningful feature-importance analysis or domain-driven feature engineering. I cannot tell you what V14 or V17 carry; I can only tell you that the supervised models extract usable signal from the combination.

The two-day observation window is short enough that seasonal effects are invisible. The geography is constrained to European cardholders from a specific 2013 sample, which makes generalisation to the 2026 fraud landscape an act of extrapolation. Real fraud detection moves by modelling the attacker, and the attack surface in 2026 is not the 2013 attack surface.

The cost model is illustrative. I chose 100 EUR and 5 EUR because they capture the order-of-magnitude asymmetry fraud teams work under, but any institution taking this approach to production should replace those numbers with its own before the cost sweep runs.

The test set has 123 positives, which means the recall estimates have confidence intervals of several percentage points. A ten-fold cross-validated version of the cost sweep would produce more stable estimates of the optimal threshold, and that is the natural next step for anyone taking this further.

## Reproducibility note

Everything here comes from a single pipeline that runs end-to-end from the Kaggle CSV through preprocessing, stratified splitting, all three model fits, the cost sweep, and the figure generation. Source code, the analysis notebook, and the figure outputs are in the public repository at [github.com/ndjstn/credit-card-fraud](https://github.com/ndjstn/credit-card-fraud). The dataset is the `creditcard.csv` file from the Machine Learning Group ULB dataset on Kaggle ([Dal Pozzolo et al., 2015](#ref-dalpozzolo2015)).

## References

<div id="ref-dalpozzolo2015" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Dal Pozzolo, A., Caelen, O., Johnson, R. A., &amp; Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. <em>2015 IEEE Symposium Series on Computational Intelligence</em>, 159-166. <a href="https://doi.org/10.1109/SSCI.2015.33">https://doi.org/10.1109/SSCI.2015.33</a>
</div>

<div id="ref-chen2016" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Chen, T., &amp; Guestrin, C. (2016). XGBoost: A scalable tree boosting system. <em>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</em>, 785-794. <a href="https://doi.org/10.1145/2939672.2939785">https://doi.org/10.1145/2939672.2939785</a>
</div>

<div id="ref-liu2008" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Liu, F. T., Ting, K. M., &amp; Zhou, Z.-H. (2008). Isolation forest. <em>2008 Eighth IEEE International Conference on Data Mining</em>, 413-422. <a href="https://doi.org/10.1109/ICDM.2008.17">https://doi.org/10.1109/ICDM.2008.17</a>
</div>

<div id="ref-elkan2001" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Elkan, C. (2001). The foundations of cost-sensitive learning. <em>Proceedings of the Seventeenth International Joint Conference on Artificial Intelligence</em>, 973-978.
</div>

<div id="ref-kaggle" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Machine Learning Group ULB. (2013). <em>Credit card fraud detection</em> [Data set]. Kaggle. <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud</a>
</div>
