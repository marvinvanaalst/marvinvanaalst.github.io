---
title: 'Resampling: bootstrap, jackknife & cross-validation'
description: 'Nonparametric, residual, parametric and block bootstraps, the jackknife, and k-fold, grouped and time-series cross-validation, with their costs and failure modes.'
aliases:
  [
    bootstrap,
    nonparametric bootstrap,
    parametric bootstrap,
    residual bootstrap,
    wild bootstrap,
    block bootstrap,
    moving block bootstrap,
    cluster bootstrap,
    jackknife,
    cross-validation,
    k-fold,
    leave-one-out,
    LOOCV,
    grouped cross-validation,
    nested cross-validation,
    time series cross-validation,
    out-of-bag,
    bias estimation,
    standard error estimation
  ]
tags: [statistics, resampling, validation]
order: 7
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Bootstrap:* resample the data with replacement, recompute the statistic, use the spread for standard errors and intervals. Works for almost any statistic; \~200 resamples for SEs, ≳ 1000–2000 for intervals.
- *Resample the right units:* i.i.d. pairs → nonparametric; fixed design with homoscedastic errors → residual bootstrap; heteroscedastic → wild bootstrap; groups → cluster bootstrap; time series → block bootstrap; a trusted model → parametric bootstrap.
- *Jackknife:* leave-one-out recomputation; cheap bias and SE estimates for smooth statistics, used inside BCa.
- *Cross-validation* estimates predictive performance: *5- or 10-fold* is the default. Use grouped CV for clustered data and forward-chaining for time series. Tune hyperparameters with *nested CV* to avoid optimistic estimates.

## Decision table

| Method | Resamples | Use when | Fails when |
| --- | --- | --- | --- |
| Nonparametric bootstrap | rows with replacement | i.i.d. observations, smooth statistics | extremes (max), very small $`n`$, dependence |
| Residual bootstrap | residuals added to fitted values | regression with fixed design, homoscedastic | heteroscedasticity, model misspecification |
| Wild bootstrap | residuals × random signs/weights | heteroscedastic regression | dependence |
| Parametric bootstrap | simulate from the fitted model | small data, trusted model, ODE models | wrong model |
| Cluster bootstrap | whole groups | repeated measures, hierarchical data | few clusters (≲ 20) |
| Block bootstrap | contiguous blocks | time series, spatial data | block length badly chosen |
| Jackknife | $`n`$ leave-one-out samples | bias/SE of smooth statistics, BCa acceleration | non-smooth statistics (median) |
| k-fold CV ($`k`$ = 5–10) | $`k`$ train/test splits | predictive performance, model selection | dependence between folds |
| Leave-one-out CV | $`n`$ fits (or closed form) | small data; linear smoothers (closed-form LOO) | high variance for unstable learners |
| Grouped / leave-group-out CV | splits by group | patients, batches, cell lines | — |
| Time-series CV (forward chaining) | train on past, test on future | forecasting | — |
| Nested CV | inner CV for tuning, outer for evaluation | honest performance with tuning | cost ($`k_{\text{out}}\times k_{\text{in}}`$ fits) |

## Variants

### Nonparametric bootstrap

1. Draw $`n`$ observations with replacement: $`y^{*}_b`$.
2. Compute $`\hat\theta^{*}_b = T(y^{*}_b)`$.
3. Repeat for $`b = 1..B`$.
4. *SE:* standard deviation of $`\hat\theta^*`$. *Bias:* $`\overline{\hat\theta^*} - \hat\theta`$. *Intervals:* percentile, BCa (see [confidence intervals](/ref/freq/confidence-intervals)).

### Regression variants

- *Pairs bootstrap:* resample $`(x_i, y_i)`$ rows; robust to heteroscedasticity and misspecification, random design.
- *Residual bootstrap:* $`y^* = \hat y + e^*`$ with resampled (leverage-adjusted) residuals; assumes the model and constant variance.
- *Wild bootstrap:* $`y^* = \hat y + e_iv_i`$ with $`v_i`$ random signs (Rademacher); handles heteroscedasticity.
- *Parametric (ODE models):* simulate new datasets from the fitted model + noise model, refit each. Gives the sampling distribution under the model.

### Dependent data

- *Block bootstrap:* resample blocks of length $`\ell`$ (moving or stationary bootstrap with random lengths), $`\ell \sim n^{1/3}`$ as a starting point.
- *Cluster bootstrap:* resample clusters, keep observations within clusters together.

### Jackknife

- $`\hat\theta_{(i)}`$ = estimate without observation $`i`$; $`\overline{\theta}_{(\cdot)}`$ their mean.
- Bias $`\approx (n - 1)(\overline\theta_{(\cdot)} - \hat\theta)`$, SE $`= \sqrt{\frac{n-1}n\sum_i(\hat\theta_{(i)} - \overline\theta_{(\cdot)})^2}`$.

### Cross-validation

1. Split data into $`k`$ folds (stratified for classification, grouped for clusters).
2. For each fold: fit on the other $`k - 1`$, evaluate the loss on the held-out fold.
3. Average the losses; report the spread across folds (or repeat with different splits).

- *Closed-form LOO* for linear smoothers: $`\text{CV} = \frac1n\sum_i\big(\frac{y_i - \hat y_i}{1 - h_{ii}}\big)^2`$; GCV replaces $`h_{ii}`$ by its average.
- *Nested CV:* hyperparameter tuning inside each outer training set; the outer folds give the performance estimate.
- *Preprocessing* (scaling, feature selection, imputation) must be fitted inside each training fold.

## Other lenses

- *Monte Carlo.* The bootstrap is Monte Carlo integration over the empirical distribution; variance reduction (balanced resampling, antithetic) applies (see [variance reduction](/ref/random/variance-reduction)).
- *Bayesian.* The Bayesian bootstrap uses Dirichlet weights; PSIS-LOO approximates LOO-CV from a single posterior fit (see [Bayesian model comparison](/ref/bayes/model-comparison)).
- *Ensembles.* Bagging averages models fitted to bootstrap samples; random forests add feature subsampling and use out-of-bag samples as built-in validation (see [classical supervised learning](/ref/learn/classical-supervised)).

## Pitfalls & diagnostics

- *Data leakage* in CV (preprocessing on all data, duplicates across folds, temporal leakage) gives overly optimistic results.
- *Ignoring structure:* random CV splits of grouped or time-series data overestimate performance on new groups/future data.
- *Bootstrap of extremes or boundary parameters* is inconsistent; use parametric or subsampling methods.
- *Refit failures* in bootstrap replicates (non-converged ODE fits) bias results if dropped silently.
- *Reporting CV error of the final tuned model* from the same CV used for tuning is optimistic; use nested CV or a held-out test set.

## See also

- [Confidence intervals](/ref/freq/confidence-intervals)
- [Model selection](/ref/freq/model-selection)
- [Conformal prediction](/ref/freq/conformal-prediction)
- [Regularisation & generalisation](/ref/learn/regularisation-generalisation)

*References:*

- Efron & Tibshirani, _An Introduction to the Bootstrap_.
- Davison & Hinkley, _Bootstrap Methods and their Application_.
- Arlot & Celisse, "A survey of cross-validation procedures for model selection", Stat. Surveys (2010).
