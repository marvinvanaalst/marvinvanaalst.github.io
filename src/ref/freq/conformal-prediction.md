---
title: 'Conformal prediction'
description: 'Distribution-free prediction intervals with finite-sample coverage: split and full conformal, jackknife+ and CV+, conformalised quantile regression, classification sets and their assumptions.'
aliases:
  [
    conformal prediction,
    split conformal,
    inductive conformal,
    full conformal,
    jackknife+,
    CV+,
    conformalized quantile regression,
    CQR,
    prediction intervals,
    prediction sets,
    coverage guarantee,
    exchangeability,
    nonconformity score,
    adaptive conformal,
    conditional coverage,
    Mondrian conformal,
    distribution shift,
    weighted conformal
  ]
tags: [statistics, uncertainty quantification, machine learning]
order: 10
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- Wrap **any** fitted model (regression, classifier, neural net, mechanistic model) to get prediction intervals/sets with guaranteed coverage $P(y_{\text{new}} \in C(x_{\text{new}})) \ge 1 - \alpha$ in finite samples.
- Only assumption: calibration and test points are **exchangeable** (e.g. i.i.d.). No model correctness needed.
- **Split conformal:** fit on one part, compute residual scores on a held-out calibration set, take the $\lceil(n + 1)(1 - \alpha)\rceil$-th smallest score. Cheap, the default.
- **Adaptive widths:** use conformalised quantile regression (CQR) or normalised scores, so intervals widen where the model is uncertain.
- Coverage is **marginal** (on average over $x$), not conditional on each $x$.

## Decision table

| Method                                  | Model fits            | Interval adaptivity         | Use when                                             |
| --------------------------------------- | --------------------- | --------------------------- | ---------------------------------------------------- |
| Split conformal, absolute residual      | 1                     | constant width              | **default**, plenty of data                          |
| Split conformal, normalised residual    | 1 (+ scale model)     | scales with $\hat\sigma(x)$ | heteroscedastic noise                                |
| Conformalised quantile regression (CQR) | 1 (quantile model)    | fully adaptive              | heteroscedastic regression                           |
| Full (transductive) conformal           | one per candidate $y$ | any                         | tiny data, cheap models (closed forms)               |
| Jackknife+                              | $n$                   | as score                    | small data, no calibration split affordable          |
| CV+                                     | $K$                   | as score                    | small–moderate data, fewer fits than jackknife+      |
| Classification (APS, RAPS)              | 1                     | set size adapts             | classification with calibrated set sizes             |
| Mondrian / group-conditional            | 1                     | per group                   | coverage required within groups                      |
| Weighted conformal                      | 1                     | as score                    | known covariate shift (likelihood ratio)             |
| Adaptive conformal inference            | online                | as score                    | time series / distribution drift (long-run coverage) |

## Variants

### Split conformal (regression)

1. Split data into a training set and a calibration set of size $n$.
2. Fit $\hat\mu$ on the training set.
3. Scores on calibration points: $s_i = \lvert y_i - \hat\mu(x_i)\rvert$.
4. $\hat q$ = the $\lceil(n + 1)(1 - \alpha)\rceil$-th smallest score.
5. Prediction interval: $[\hat\mu(x) - \hat q,\ \hat\mu(x) + \hat q]$.

- Coverage between $1 - \alpha$ and $1 - \alpha + 1/(n + 1)$. A few hundred calibration points give stable intervals.

### Conformalised quantile regression

1. Fit lower and upper quantile models $\hat q_{\alpha/2}(x)$, $\hat q_{1-\alpha/2}(x)$ (quantile loss).
2. Scores: $s_i = \max\big(\hat q_{\alpha/2}(x_i) - y_i,\ y_i - \hat q_{1-\alpha/2}(x_i)\big)$.
3. Interval: $[\hat q_{\alpha/2}(x) - \hat Q,\ \hat q_{1-\alpha/2}(x) + \hat Q]$ with $\hat Q$ the conformal quantile of the scores.

### Jackknife+ and CV+

- Fit leave-one-out (or leave-fold-out) models; build the interval from LOO residuals combined with LOO predictions at the new point.
- Coverage $\ge 1 - 2\alpha$ in the worst case (≈ $1 - \alpha$ in practice) without a separate calibration set.

### Classification

- Score $1 - \hat p_y(x)$ gives sets $\{y : \hat p_y(x) \ge 1 - \hat q\}$.
- APS/RAPS scores accumulate sorted probabilities, giving sets whose size reflects difficulty; empty or large sets signal ambiguous inputs.

## Other lenses

- **Hypothesis testing.** Full conformal inverts a permutation test of "is $y$ a plausible label for $x$ given the data"; exchangeability makes the rank of the test score uniform.
- **Bayesian.** Bayesian predictive intervals are conditional but only valid if the model is right; conformal intervals are marginal but valid regardless. Conformalising a Bayesian model's predictive distribution combines both.
- **Calibration.** Conformal prediction is a post-hoc recalibration of uncertainty, like temperature scaling but with guarantees (see [resampling](/ref/freq/resampling)).

## Pitfalls & diagnostics

- **Exchangeability violations** (time series, distribution shift, grouped data split randomly) void the guarantee; use weighted/adaptive variants or split by group.
- **Marginal ≠ conditional coverage:** intervals can undercover in hard regions and overcover in easy ones; check coverage across subgroups or bins of $x$.
- **Calibration data reuse:** don't use the calibration set for training or model selection.
- **Bad models give valid but wide intervals;** conformal prediction can't fix poor point predictions.
- **Check empirically:** coverage and average width on a held-out test set.

## See also

- [Resampling](/ref/freq/resampling)
- [Confidence intervals](/ref/freq/confidence-intervals)
- [Classical supervised learning](/ref/learn/classical-supervised)
- [Regularisation & generalisation](/ref/learn/regularisation-generalisation)

**References:**

- Angelopoulos & Bates, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification" (2021).
- Romano, Patterson & Candès, "Conformalized quantile regression", NeurIPS (2019).
- Barber, Candès, Ramdas & Tibshirani, "Predictive inference with the jackknife+", Ann. Stat. (2021).
