---
title: 'Classical supervised learning'
description: 'Linear and logistic models, kernel methods and SVMs, k-nearest neighbours, decision trees, random forests and gradient boosting: what to use on tabular data and why.'
aliases:
  [
    supervised learning,
    linear regression,
    logistic regression,
    regularized regression,
    kernel methods,
    kernel ridge regression,
    SVM,
    support vector machine,
    kNN,
    k-nearest neighbors,
    decision tree,
    CART,
    random forest,
    gradient boosting,
    XGBoost,
    LightGBM,
    CatBoost,
    bagging,
    boosting,
    tabular data,
    feature engineering,
    class imbalance,
    calibration
  ]
tags: [machine learning, supervised, tabular data]
order: 1
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Tabular data, prediction quality matters:* *gradient-boosted trees* (LightGBM, XGBoost, CatBoost). Usually beat neural nets on tabular data of moderate size.
- *Baseline and interpretability:* regularised linear / logistic regression. Always fit it first.
- *Robust, little tuning:* random forests (good out-of-the-box, out-of-bag error for free).
- *Small $`n`$, smooth relationships, need uncertainty:* kernel methods / Gaussian processes.
- *kNN:* simple, useful as a baseline or for local structure; poor in high dimensions.
- Use proper *cross-validation*, keep preprocessing inside folds, and check *calibration* of predicted probabilities.

## Decision table

| Method | Training cost | Strengths | Weaknesses | Use when |
| --- | --- | --- | --- | --- |
| Linear / logistic (+L1/L2) | $`O(nd^2)`$ or iterative | interpretable, fast, calibrated-ish | only linear (unless features engineered) | baseline, many features, small $`n`$ |
| Kernel ridge / GP | $`O(n^3)`$ | smooth nonlinearity, uncertainty (GP) | scales poorly in $`n`$, kernel choice | $`n \lesssim 10^4`$, smooth targets |
| SVM (RBF kernel) | $`O(n^2)`$–$`O(n^3)`$ | strong for medium-size, high-dimensional data | probabilities need calibration, tuning $`C, \gamma`$ | text/omics classification, medium $`n`$ |
| kNN | none (lazy), $`O(nd)`$ per query | trivial, nonparametric | curse of dimensionality, needs scaling | baselines, low-dimensional data |
| Decision tree (CART) | $`O(nd\log n)`$ | interpretable rules, handles mixed types | high variance, unstable | explanation, quick insight |
| Random forest | trees × bootstrap | robust, little tuning, OOB error, feature importance | less accurate than boosting, large models | strong default with little tuning |
| Gradient boosting (GBM) | sequential trees | *best accuracy on tabular data*, handles missing values | tuning, can overfit | tabular prediction tasks |
| Naive Bayes | $`O(nd)`$ | very fast, works with tiny data | independence assumption | text baselines |

## Variants

### Linear and logistic models

- Ridge (L2) for stability with correlated features; LASSO (L1) for sparse models; elastic net in between (see [sparse recovery](/ref/approx/sparse-recovery)).
- Standardise features; tune the penalty by CV; logistic regression outputs reasonably calibrated probabilities.
- Nonlinearity through features: polynomials, splines, interactions, domain-derived features.

### Kernel methods

- Replace inner products with a kernel $`k(x, x')`$; prediction $`\hat f(x) = \sum_i\alpha_ik(x_i, x)`$.
- *Kernel ridge regression* = GP posterior mean; *SVM* uses the hinge loss, giving sparse solutions (support vectors).
- Scale with Nyström / random features for large $`n`$ (see [Gaussian processes](/ref/approx/gaussian-processes)).

### Trees and ensembles

- *CART:* recursive binary splits minimising impurity (Gini, entropy) or squared error; prune by cost-complexity.
- *Random forest:* average many deep trees trained on bootstrap samples with random feature subsets at each split (decorrelates trees). Key parameter: features per split (√d classification, d/3 regression).
- *Gradient boosting:* add shallow trees sequentially, each fitting the negative gradient of the loss (residuals for squared error).
  - Key hyperparameters: learning rate (0.01–0.1), number of trees (early stopping on validation), tree depth / leaves, subsampling, regularisation.
  - Histogram-based splits (LightGBM) make it fast on large data; CatBoost handles categorical features well.

### Practical issues

- *Class imbalance:* use proper metrics (PR-AUC, balanced accuracy), class weights, threshold tuning; resampling only inside training folds.
- *Categorical features:* one-hot for linear models; target encoding (with CV) or native handling for boosting.
- *Missing values:* native handling in boosting; imputation inside CV otherwise.
- *Calibration:* check reliability diagrams; recalibrate with Platt scaling or isotonic regression.

## Other lenses

- *Statistics.* Linear and logistic regression are GLMs; regularisation = priors; random forests and boosting are nonparametric regression estimators (see [regression & GLMs](/ref/freq/regression-glms)).
- *Optimisation.* Boosting is functional gradient descent; SVMs are convex QPs (see [constrained optimisation](/ref/optimise/constrained)).
- *Explainability.* Tree ensembles admit exact, fast SHAP values (TreeSHAP); linear models are directly interpretable (see [feature attribution](/ref/explain/feature-attribution)).

## Pitfalls & diagnostics

- *Leakage:* features derived from the target or future, preprocessing on all data, duplicated samples across splits.
- *Distribution shift* between training and deployment data; check covariate shift and use grouped/temporal splits.
- *Overinterpreting feature importances* (impurity-based importances are biased towards high-cardinality features); use permutation importance or SHAP.
- *Metrics:* accuracy is misleading with imbalance; report several metrics and confidence intervals (bootstrap).
- *Tuning on the test set:* keep a final untouched test set or use nested CV (see [resampling](/ref/freq/resampling)).

## See also

- [Regression & GLMs](/ref/freq/regression-glms)
- [Regularisation & generalisation](/ref/learn/regularisation-generalisation)
- [Resampling](/ref/freq/resampling)
- [Feature attribution](/ref/explain/feature-attribution)
- [Conformal prediction](/ref/freq/conformal-prediction)

*References:*

- Hastie, Tibshirani & Friedman, _The Elements of Statistical Learning_ (2nd ed.).
- Grinsztajn, Oyallon & Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?", NeurIPS (2022).
- Chen & Guestrin, "XGBoost: a scalable tree boosting system", KDD (2016).
