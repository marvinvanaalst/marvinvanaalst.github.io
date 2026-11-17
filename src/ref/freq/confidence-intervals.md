---
title: 'Confidence intervals'
description: 'Wald, likelihood-ratio / profile likelihood and bootstrap intervals (percentile, BCa), coverage, and when the cheap asymptotic interval lies.'
aliases:
  [
    confidence interval,
    Wald interval,
    standard error,
    profile likelihood,
    likelihood ratio interval,
    bootstrap confidence interval,
    percentile bootstrap,
    BCa,
    studentized bootstrap,
    coverage,
    simultaneous confidence bands,
    prediction interval,
    Clopper-Pearson,
    Wilson interval,
    Fieller
  ]
tags: [statistics, uncertainty, intervals]
order: 2
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Wald:** $\hat\theta \pm z_{1-\alpha/2}\,\mathrm{SE}$. Cheap; accurate only when the log-likelihood is close to quadratic (large data, mild nonlinearity, far from bounds).
- **Profile likelihood:** $\{\theta_j : 2[\ell(\hat\theta) - \ell_p(\theta_j)] \le \chi^2_{1, 1-\alpha}\}$. The right default for nonlinear models (ODEs, kinetics): asymmetric, invariant to reparametrisation, reveals non-identifiability.
- **Bootstrap:** resample data (or residuals) and refit. Percentile is simple; **BCa** corrects bias and skewness. Use when the sampling distribution is unclear; costs hundreds to thousands of refits.
- **Proportions:** Wilson or Clopper–Pearson, never Wald for small counts.
- Confidence interval ≠ prediction interval: the latter adds observation noise.

## Decision table

| Method                          | Cost                           | Accuracy / properties                           | Use when                                                                   |
| ------------------------------- | ------------------------------ | ----------------------------------------------- | -------------------------------------------------------------------------- |
| Wald (SE-based)                 | free after fit (Hessian)       | symmetric, not invariant, poor for nonlinearity | large samples, near-linear models                                          |
| Wald on transformed scale       | free                           | better for positive / bounded parameters        | log-rates, logit-probabilities                                             |
| Profile likelihood              | 1D optimisations per parameter | asymmetric, invariant, good coverage            | **nonlinear models**, identifiability checks                               |
| Likelihood-ratio region (joint) | grid / sampling                | joint confidence regions                        | pairs of parameters, contour plots                                         |
| Percentile bootstrap            | $B$ refits ($B \sim 1000$)     | simple; biased for skewed estimators            | general, smooth statistics                                                 |
| BCa bootstrap                   | $B$ refits + jackknife         | second-order accurate                           | skewed or biased estimators                                                |
| Studentised bootstrap           | nested resampling              | second-order accurate                           | when an SE estimate per resample is available                              |
| Parametric bootstrap            | $B$ simulations + refits       | relies on the model                             | small data, complex models, profile-free                                   |
| Wilson / Clopper–Pearson        | closed form                    | good / conservative coverage                    | binomial proportions                                                       |
| Bayesian credible interval      | posterior sampling             | different interpretation                        | when priors are acceptable (see [Bayesian](/ref/bayes/foundations-priors)) |

## Variants

### Wald intervals

- $\hat\theta_j \pm z_{1-\alpha/2}\sqrt{[I(\hat\theta)^{-1}]_{jj}}$.
- For positive parameters, build the interval for $\log\theta$ and exponentiate: asymmetric and positive.
- Fails silently: no warning when the quadratic approximation is bad.

### Profile likelihood

1. For a grid of values $\theta_j = c$, maximise $\ell$ over all other parameters: $\ell_p(c) = \max_{\theta_{-j}}\ell(c, \theta_{-j})$.
2. Interval: all $c$ with $2[\ell(\hat\theta) - \ell_p(c)] \le \chi^2_{1, 0.95} = 3.84$.
3. Trace the profile from $\hat\theta_j$ outwards with warm starts (continuation); adapt the step to the curvature.

- **Interpretation:** a profile that stays below the threshold on one side ⇒ the parameter is practically non-identifiable there (open interval); a flat profile ⇒ structurally non-identifiable.
- **Prediction profiles** give intervals for model predictions in the same way.

### Bootstrap intervals

1. Resample: data pairs (nonparametric), residuals added to fitted values (residual bootstrap), or simulate from the fitted model (parametric).
2. Refit to get $\hat\theta^{*}_b$, $b = 1..B$.
3. **Percentile:** quantiles $\alpha/2$, $1 - \alpha/2$ of $\hat\theta^*$.
4. **BCa:** adjust quantile levels using a bias correction $z_0$ (fraction of $\hat\theta^* \lt \hat\theta$) and acceleration $a$ (from jackknife skewness).

- Respect the data structure: block bootstrap for time series, cluster bootstrap for grouped data (see [resampling](/ref/freq/resampling)).

### Proportions and counts

- **Wilson score interval:** inverts the score test; good coverage even for small $n$.
- **Clopper–Pearson:** exact (inverts binomial tests), conservative.
- Poisson rates: exact intervals from chi-square quantiles, or likelihood-based.

## Other lenses

- **Hypothesis testing.** A 95% CI is the set of values not rejected by a 5% test; Wald, score and likelihood-ratio tests give the three interval families (see [hypothesis tests](/ref/freq/hypothesis-tests)).
- **Geometry.** Wald intervals assume the likelihood surface is an ellipse; profile likelihood follows the actual "banana" shapes typical of sloppy models.
- **Bayesian.** With weak priors and lots of data, credible and confidence intervals nearly coincide; they diverge for small data, bounds, and non-identifiability, where profiles stay open while posteriors fall back on priors.

## Pitfalls & diagnostics

- **Coverage:** check by simulation (refit many synthetic datasets from the fitted model, count how often the interval covers the truth).
- **Wald intervals including impossible values** (negative rates) are a sign to use transformations or profiles.
- **Bootstrap with too few resamples** ($B \lt 1000$ for percentile intervals) gives noisy endpoints; BCa needs more.
- **Refit failures** in bootstrap or profiles (non-convergence) must be counted and handled, not silently dropped.
- **Multiplicity:** many intervals at 95% each don't give 95% joint coverage; use simultaneous bands if needed.

## See also

- [Estimation & MLE](/ref/freq/estimation-mle)
- [Identifiability](/ref/freq/identifiability)
- [Resampling](/ref/freq/resampling)
- [Hypothesis tests](/ref/freq/hypothesis-tests)

**References:**

- Efron & Tibshirani, _An Introduction to the Bootstrap_.
- Raue et al., "Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood", Bioinformatics (2009).
- Brown, Cai & DasGupta, "Interval estimation for a binomial proportion", Statistical Science (2001).
