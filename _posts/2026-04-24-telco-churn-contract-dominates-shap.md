---
title: "Contract Type Eats the Rest: Telco Customer Churn with SHAP"
date: 2026-04-24 00:00:01 -0500
description: "The IBM Telco churn dataset has 19 features and one of them does most of the work. SHAP ranks contract type first with a mean absolute impact of 0.88, and the raw churn rate across contract types agrees."
image:
  path: /assets/img/posts/telco-churn-contract-dominates-shap/hero.png
  alt: "Title card for a telco churn analysis with a SHAP bar chart showing contract type dominating the feature importance ranking."
tags:
  - "Customer Churn"
  - "SHAP"
  - "XGBoost"
  - "Explainable AI"
  - "Machine Learning"
  - "Telco"
categories:
  - "Data Science"
---

The IBM-style Telco customer churn dataset has 7,043 rows, 19 features, and a 26.54 percent churn rate. You can fit an XGBoost classifier on it in under a minute and get 0.83 ROC-AUC without tuning anything. That part is not interesting. The interesting part is what happens when you point SHAP at the fitted model and ask which features the classifier is actually using.

The answer is almost embarrassing. One feature does most of the work. Contract type has a mean absolute SHAP of 0.880. The next feature is tenure at 0.538, and by the fifth-ranked feature the contribution is a factor of three below the top. For anyone who has sat through a tutorial that treats churn modelling as a balanced feature-importance exercise, the SHAP bar chart is a reality check.

<!-- YouTube embed will go here once the walkthrough video is published -->

## The setup

Before explaining the model I want to be clear about what this dataset is and what it is not. It is a sample dataset originally produced by IBM for their product demos and redistributed through Kaggle by user `blastchar`. There are no date stamps, no geographic information, no cohort identifiers. The rows are residential customers of a hypothetical telecommunications operator, and the 26.54 percent churn rate is high enough that it may have been oversampled from an underlying dataset with a lower base rate. Those are real limitations, and the conclusions in the rest of this post should be read with them in mind.

With that caveat stated, here is the quick profile:

| Item | Value |
| --- | ---: |
| Rows | 7,043 |
| Features | 19 |
| Positive (churn) | 1,869 |
| Positive rate | 26.54% |
| Feature groups | demographics, services, account |

Two preprocessing notes. `TotalCharges` arrives as a string column because of 11 rows with blank values; I coerced to numeric and median-imputed. `customerID` carries no signal and I dropped it. Everything else passes through XGBoost with `enable_categorical=True`, which handles the sixteen categorical columns natively.

## The model

Five hundred trees, depth 4, learning rate 0.05, `scale_pos_weight` set to the empirical negative-to-positive ratio, 75/25 stratified split.

The test set numbers:

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.8328 |
| Average precision | 0.6387 |
| F1-optimal threshold | 0.54 |
| Precision at F1-best | 0.548 |
| Recall at F1-best | 0.715 |
| F1 | 0.621 |

Nothing remarkable there. A retention team operating this model at the F1-optimal threshold would catch roughly seven of every ten customers who will churn, and between five and six of every ten flags would be correct. That is the envelope a retention program would design around.

![Confusion matrix at the F1-optimal threshold showing the split of true positives, false positives, true negatives, and false negatives.](/assets/img/posts/telco-churn-contract-dominates-shap/confusion-matrix.png)

*The model misses 133 actual churners out of 467 in the test set and flags 275 customers who will stay. That is the tradeoff the F1 optimiser found; a retention team might choose to recalibrate in either direction.*

## Contract type eats the rest

This is where the project departs from a standard churn tutorial. Most churn tutorials stop at the feature importance bar from `model.feature_importances_`, which is a split-count proxy and does not tell you the sign or magnitude of the effect on an individual prediction. SHAP does both.

![SHAP bar chart showing Contract dominating at 0.88 mean absolute SHAP, with tenure and MonthlyCharges behind it.](/assets/img/posts/telco-churn-contract-dominates-shap/shap-bar.png)

*Mean absolute SHAP over the 1,761-row test set. Contract type leads by a factor of 1.6 over tenure and by a factor of three over the fifth-ranked feature.*

The top three are contract, tenure, and monthly charges. Everything else is a rounding error by comparison.

The raw data confirms the ranking. The 3,875 month-to-month customers churn at a 42.71 percent rate. The 1,473 one-year customers churn at 11.27 percent. The 1,695 two-year customers churn at 2.83 percent. Fifteen to one between the extremes.

![Churn rate by contract type with month-to-month at 42.7 percent and two-year at 2.8 percent.](/assets/img/posts/telco-churn-contract-dominates-shap/churn-by-contract.png)

*A two-line group-by in pandas already tells you most of what the model is going to learn. The SHAP ranking just confirms that the model has the right priority.*

The practical reading is that contract type is closer to a business-model variable than to a customer attribute. A customer who signs a two-year contract is, by the act of signing, a customer the operator has already retained for two years. The churn rate difference between contract tiers is not entirely a fact about the customers; it is partly a fact about the product structure the operator offers.

## Tenure is the second story

![SHAP dependence plot for tenure coloured by monthly charges, showing positive SHAP below twelve months and negative SHAP above thirty-six.](/assets/img/posts/telco-churn-contract-dominates-shap/shap-dependence-tenure.png)

*The cleanest single figure in the project. Below a tenure of about twelve months, the SHAP contribution to the churn score is strongly positive. Between twelve and thirty-six months, it drops through zero. Above thirty-six months, it stabilises at a meaningfully negative contribution.*

Tenure is the second-strongest feature at mean absolute SHAP 0.538, and the dependence plot shows exactly how the model is using it. A customer in their first year gets pushed toward the churn side by around 0.7 to 1.0 log-odds; a customer past three years gets pulled toward the stay side by 0.5 to 0.8. The transition is gradual, which is the expected shape of a hazard function that decays with tenure.

The colouring by MonthlyCharges adds a secondary layer: at short tenure, higher monthly charges amplify the churn risk further; at long tenure, the monthly-charge effect attenuates. New customers on premium plans are the highest-risk cohort in the dataset.

The raw tenure statistics agree. Churned customers have a median tenure of 10 months against a median of 38 months for customers who stayed. The interquartile ranges barely overlap.

## The part of SHAP that actually matters

SHAP's reputation as an explainability tool is partly earned on the global charts above, but the real operational payoff is the per-customer waterfall.

![SHAP waterfall showing the single customer in the test set with the highest predicted churn probability.](/assets/img/posts/telco-churn-contract-dominates-shap/shap-waterfall-highest.png)

*One customer, all the features ranked by how much each one pushed the log-odds score up or down. The final probability is 0.97.*

For this specific customer, the model assigns a churn probability of 0.97. The waterfall shows exactly why: the month-to-month contract contributes +1.02 log-odds, the short tenure contributes +0.71, the fiber-optic internet service contributes +0.46, and the absence of online security adds another +0.25.

A score alone tells the retention team whom to call. The waterfall tells the retention agent what to offer. If the dominant contributors are contract and tenure, the appropriate pitch is a contract upgrade with a retention discount attached. If the dominant contributor is a missing service add-on, the appropriate pitch is that add-on at a sweetener price. The per-customer breakdown is the part of this project that an operator would actually use.

## What this is not

The model does not prove that contract type causes lower churn. It shows that contract type is the strongest predictor of churn in the observed data. Customers who sign two-year contracts are a different population of customers than those who sign month-to-month ones, and the lower observed churn among the former partly reflects the selection effect rather than a causal benefit of the contract itself. Pinning the causal fraction down would require an experiment.

SHAP values are a local attribution, not a causal decomposition. A feature with a large SHAP contribution to a prediction tells you what the model used, not what would happen if you intervened on the feature. "Upgrade this customer to a one-year contract" is a defensible retention hypothesis from the SHAP output, but it needs a randomised holdout to validate.

And the dataset is what it is. Thin provenance, a high base rate that may not reflect the operator's real churn, no dates, no geography. The methodology in this post generalises; the specific numbers should not be quoted out of context.

## Reproducibility note

Everything here runs end-to-end from the Kaggle CSV through preprocessing, the XGBoost fit, the SHAP pass, and the figure generation. Source code and outputs are in the public repository at [github.com/ndjstn/telco-churn-shap](https://github.com/ndjstn/telco-churn-shap). The dataset is the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file from the `blastchar` Kaggle dataset ([blastchar, n.d.](#ref-blastchar)).

## References

<div id="ref-lundberg2017" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Lundberg, S. M., &amp; Lee, S.-I. (2017). A unified approach to interpreting model predictions. <em>Advances in Neural Information Processing Systems, 30</em>. <a href="https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions">https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions</a>
</div>

<div id="ref-lundberg2020" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., &amp; Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. <em>Nature Machine Intelligence, 2</em>(1), 56-67. <a href="https://doi.org/10.1038/s42256-019-0138-9">https://doi.org/10.1038/s42256-019-0138-9</a>
</div>

<div id="ref-chen2016" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
Chen, T., &amp; Guestrin, C. (2016). XGBoost: A scalable tree boosting system. <em>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</em>, 785-794. <a href="https://doi.org/10.1145/2939672.2939785">https://doi.org/10.1145/2939672.2939785</a>
</div>

<div id="ref-blastchar" style="padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 0.85rem;">
blastchar. (n.d.). <em>Telco customer churn</em> [Data set]. Kaggle. <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn">https://www.kaggle.com/datasets/blastchar/telco-customer-churn</a>
</div>
