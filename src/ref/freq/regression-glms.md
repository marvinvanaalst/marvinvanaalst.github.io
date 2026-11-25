---
title: 'Regression & GLMs'
description: 'OLS assumptions and diagnostics, generalised linear models fitted by IRLS, mixed-effects models for grouped data, and robust regression.'
aliases:
  [
    linear regression,
    OLS,
    ordinary least squares,
    Gauss-Markov,
    residual diagnostics,
    heteroscedasticity,
    multicollinearity,
    VIF,
    GLM,
    generalized linear model,
    logistic regression,
    Poisson regression,
    negative binomial,
    link function,
    IRLS,
    deviance,
    overdispersion,
    mixed models,
    random effects,
    GLMM,
    REML,
    robust regression,
    Huber,
    quantile regression,
    splines,
    GAM
  ]
tags: [statistics, regression, modelling]
order: 6
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Continuous outcome, additive Gaussian noise:* linear regression (OLS via QR). Check residuals, not $`R^2`$.
- *Non-Gaussian outcome:* GLM with an appropriate family and link: binary → logistic; counts → Poisson (or negative binomial if overdispersed); positive skewed → Gamma with log link. Fitted by IRLS.
- *Grouped / repeated measurements* (patients, plates, cell lines, time series per subject): *mixed-effects models* with random intercepts/slopes. Ignoring grouping gives overconfident inference.
- *Outliers / heavy tails:* robust regression (Huber, bisquare) or quantile regression.
- *Nonlinear smooth effects:* splines in the linear predictor (GAMs) before reaching for black-box ML.

## Decision table

| Outcome type | Model | Link (canonical) | Notes |
| --- | --- | --- | --- |
| Continuous, Gaussian noise | linear model (OLS) | identity | check homoscedasticity, normality of residuals |
| Continuous, multiplicative noise | linear model on $`\log y`$ or Gamma GLM | log | back-transformation bias for log models |
| Binary | logistic (binomial GLM) | logit | separation → penalised (Firth) or Bayesian |
| Proportions with denominators | binomial GLM | logit | overdispersion → beta-binomial / quasi-binomial |
| Counts | Poisson GLM | log | offsets for exposure, e.g. $`\log(\text{time})`$ |
| Overdispersed counts | negative binomial | log | RNA-seq-style data |
| Positive continuous, skewed | Gamma GLM | log (inverse canonical) |  |
| Ordinal | proportional odds | cumulative logit |  |
| Grouped data | linear / generalised linear mixed model | as above | random effects per group |
| Outliers | robust M-estimation, quantile regression | — | median regression is robust |
| Smooth nonlinear effects | GAM (penalised splines) | as above | smoothing parameter by REML / GCV |

## Variants

### Linear regression

- $`y = X\beta + \varepsilon`$; $`\hat\beta = \arg\min\lVert y - X\beta\rVert^2`$, computed by QR (see [least squares](/ref/solve/least-squares)).
- *Assumptions (in order of importance):* correct mean structure (linearity, right covariates), independence, constant variance, normality (only matters for small-sample inference).
- *Inference:* $`\widehat{\operatorname{Cov}}(\hat\beta) = \hat\sigma^2(X^\top X)^{-1}`$; heteroscedasticity-robust (HC3) sandwich errors if variance isn't constant.
- *Multicollinearity:* inflates variances, not bias; VIF $`= 1/(1 - R_j^2)`$ above \~5–10 is a warning.

### GLMs and IRLS

- Model: $`g(\mathbb E[y]) = X\beta`$, $`y`$ from an exponential family with variance function $`V(\mu)`$.
- *IRLS:* each iteration solves a weighted least-squares problem:
  1. Working response $`z = \eta + (y - \mu)\,g'(\mu)`$.
  2. Weights $`w = 1/(V(\mu)\,g'(\mu)^2)`$.
  3. $`\beta \leftarrow (X^\top WX)^{-1}X^\top Wz`$ (via QR of $`W^{1/2}X`$).
  4. Repeat until convergence (= Newton/Fisher scoring on the log-likelihood).
- *Deviance* replaces residual sum of squares; differences in deviance between nested models are likelihood-ratio tests.
- *Overdispersion:* Pearson $`\chi^2/\text{df} \gg 1`$ → quasi-likelihood, negative binomial or observation-level random effects.

### Mixed-effects models

- $`y = X\beta + Zb + \varepsilon`$, $`b \sim\mathcal N(0, G)`$: fixed effects $`\beta`$, random effects $`b`$ per group.
- Fit by (RE)ML: REML for unbiased variance components; ML for comparing fixed-effect structures.
- *Random intercepts* for baseline differences between groups, *random slopes* if effects vary by group; keep the structure justified by the design ("keep it maximal" vs convergence problems).
- GLMMs need approximations (Laplace, adaptive quadrature) or Bayesian fitting.

### Robust and quantile regression

- *M-estimation:* minimise $`\sum\rho(r_i/s)`$ with Huber or Tukey bisquare $`\rho`$, via IRLS with weights $`\psi(r)/r`$.
- *Quantile regression:* minimise the check loss; models conditional medians or other quantiles; LP-based.

## Other lenses

- *Linear algebra.* OLS is a projection; leverage = diagonal of the hat matrix $`H = X(X^\top X)^{-1}X^\top`$; influential points combine leverage and residual (Cook's distance).
- *Bayesian.* Priors on coefficients regularise (ridge, horseshoe); mixed models are hierarchical models (see [Bayesian foundations](/ref/bayes/foundations-priors)).
- *Machine learning.* Logistic regression is a one-layer network with sigmoid output and cross-entropy loss; GLMs are the interpretable baseline every ML model should beat (see [classical supervised learning](/ref/learn/classical-supervised)).

## Pitfalls & diagnostics

- *Residual plots:* residuals vs fitted (nonlinearity, heteroscedasticity), Q-Q plot (tails), residuals vs each covariate and vs time/order (dependence).
- *Pseudo-replication:* treating repeated measures as independent; use mixed models or aggregate.
- *Separation in logistic regression* gives infinite estimates; use penalisation.
- *Causal language:* regression coefficients are associations unless the design supports causal claims (see [causal effect estimation](/ref/explain/causal-effect-estimation)).
- *Extrapolation* beyond the covariate range, and *stepwise selection* (biased estimates, invalid p-values).

## See also

- [Least squares](/ref/solve/least-squares)
- [Estimation & MLE](/ref/freq/estimation-mle)
- [Hypothesis tests](/ref/freq/hypothesis-tests)
- [Model selection](/ref/freq/model-selection)
- [Classical supervised learning](/ref/learn/classical-supervised)

*References:*

- Gelman, Hill & Vehtari, _Regression and Other Stories_.
- McCullagh & Nelder, _Generalized Linear Models_ (2nd ed.).
- Bates et al., "Fitting linear mixed-effects models using lme4", J. Stat. Softw. (2015).
