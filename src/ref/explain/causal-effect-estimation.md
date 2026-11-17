---
title: 'Causal effect estimation'
description: 'Estimating average and heterogeneous treatment effects: regression adjustment, propensity scores, IPW, doubly robust and double ML, matching, difference-in-differences and regression discontinuity.'
aliases:
  [
    treatment effect,
    average treatment effect,
    ATE,
    ATT,
    CATE,
    heterogeneous treatment effects,
    potential outcomes,
    Rubin causal model,
    regression adjustment,
    propensity score,
    inverse probability weighting,
    IPW,
    doubly robust,
    AIPW,
    double machine learning,
    DML,
    targeted maximum likelihood,
    TMLE,
    matching,
    difference-in-differences,
    DiD,
    regression discontinuity,
    synthetic control,
    positivity,
    unconfoundedness,
    causal forests,
    meta-learners
  ]
tags: [causality, statistics, estimation]
order: 3
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- **Potential outcomes:** $Y(1), Y(0)$ per unit; ATE $= \mathbb E[Y(1) - Y(0)]$. We only ever observe one of them.
- **Assumptions** for observational data: unconfoundedness (no unmeasured confounders given $X$), **positivity/overlap** (every unit could get either treatment), consistency, no interference.
- **Estimators under unconfoundedness:** outcome regression, propensity-score weighting (IPW), matching, and **doubly robust** methods (AIPW, TMLE, double ML), which are consistent if either the outcome or the propensity model is right and allow flexible ML models.
- **Design-based strategies** when unconfoundedness is implausible: difference-in-differences, regression discontinuity, instrumental variables, synthetic control.
- **Heterogeneous effects (CATE):** causal forests, meta-learners (T/S/X/R-learner), DR-learner.

## Decision table

| Method                                              | Models needed                  | Robustness                                              | Use when                                            |
| --------------------------------------------------- | ------------------------------ | ------------------------------------------------------- | --------------------------------------------------- |
| Randomised comparison                               | none                           | design-based                                            | RCTs                                                |
| Outcome regression (g-computation)                  | $\mathbb E[Y\mid T, X]$        | needs correct outcome model                             | good outcome models, few confounders                |
| Propensity score IPW                                | $e(X) = P(T = 1\mid X)$        | needs correct propensity, unstable with extreme weights | treatment mechanism well understood                 |
| Matching (propensity, Mahalanobis, coarsened exact) | distance / propensity          | transparent; bias from imperfect matches                | intuitive comparisons, ATT                          |
| AIPW / doubly robust                                | both                           | consistent if either is right                           | **default** observational ATE                       |
| Double ML (cross-fitting)                           | both, with ML learners         | valid inference with flexible models                    | many confounders, nonlinear relationships           |
| TMLE                                                | both                           | doubly robust, respects bounds                          | epidemiology, bounded outcomes                      |
| Difference-in-differences                           | before/after × treated/control | parallel trends assumption                              | policy changes, staggered adoption (new estimators) |
| Regression discontinuity                            | outcome near a threshold       | local effect at the cutoff                              | assignment by a score threshold                     |
| Instrumental variables                              | instrument                     | exclusion restriction                                   | unmeasured confounding with a valid instrument      |
| Synthetic control                                   | weighted donor units           | pre-treatment fit                                       | single treated unit (region, country)               |
| Causal forests / meta-learners                      | ML models                      | depends on learners                                     | heterogeneous effects (CATE)                        |

## Variants

### Regression adjustment (g-computation)

1. Fit $\hat\mu_t(x) = \mathbb E[Y\mid T = t, X = x]$.
2. Predict both outcomes for everyone: $\widehat{\text{ATE}} = \frac1n\sum_i[\hat\mu_1(x_i) - \hat\mu_0(x_i)]$.
3. Bootstrap for confidence intervals.

### Propensity scores and IPW

- Propensity $e(x) = P(T = 1\mid X = x)$ (logistic regression or ML).
- IPW: $\widehat{\text{ATE}} = \frac1n\sum_i\Big[\frac{T_iY_i}{\hat e(x_i)} - \frac{(1 - T_i)Y_i}{1 - \hat e(x_i)}\Big]$; normalised (Hájek) weights are more stable.
- **Overlap diagnostics:** plot propensity distributions by group; extreme scores (near 0 or 1) mean poor overlap → trim, use overlap weights, or change the estimand.
- **Balance check:** standardised mean differences of covariates after weighting/matching (≲ 0.1).

### Doubly robust: AIPW and double ML

- AIPW combines both models: $\hat\mu_1 - \hat\mu_0 + \frac{T(Y - \hat\mu_1)}{\hat e} - \frac{(1 - T)(Y - \hat\mu_0)}{1 - \hat e}$, averaged.
- **Cross-fitting:** fit nuisance models on other folds than those used for the effect estimate; allows flexible ML (boosting, forests) with valid $\sqrt n$ inference.

### Difference-in-differences

- Effect = (treated after − treated before) − (control after − control before), typically via two-way fixed-effects regression.
- Key assumption: parallel trends without treatment; check pre-treatment trends (event-study plots).
- Staggered treatment timing with heterogeneous effects breaks two-way fixed effects; use modern estimators (Callaway–Sant'Anna, Sun–Abraham).

### Regression discontinuity

- Treatment assigned when a running variable crosses a cutoff; compare outcomes just above and below with local linear regression and an optimal bandwidth.
- Check for manipulation of the running variable (density test) and covariate continuity at the cutoff.

### Heterogeneous effects

- **T-learner:** separate outcome models per arm; **S-learner:** one model with treatment as a feature; **X/R/DR-learners:** combine propensity and outcome models for better CATE estimates.
- **Causal forests:** honest splitting to estimate CATE with confidence intervals.

## Other lenses

- **Graphs.** Unconfoundedness given $X$ = $X$ satisfies the backdoor criterion in the causal DAG (see [causal graphs](/ref/explain/causal-graphs-scms)).
- **Survey sampling.** IPW is Horvitz–Thompson estimation; missing potential outcomes are a missing-data problem.
- **Experimental design.** Randomisation makes treatment independent of everything else; blocking/stratification reduces variance (see [experimental design](/ref/freq/experimental-design)).

## Pitfalls & diagnostics

- **Unmeasured confounding** remains the main threat; report sensitivity analyses (E-values, Rosenbaum bounds).
- **Positivity violations** make IPW explode and outcome models extrapolate.
- **Bad controls:** adjusting for post-treatment variables or colliders biases estimates.
- **Defining the estimand** (ATE vs ATT vs effect at the cutoff vs local effect for compliers) before choosing the method.
- **Overfitting nuisance models** without cross-fitting gives over-confident intervals.

## See also

- [Causal graphs & SCMs](/ref/explain/causal-graphs-scms)
- [Counterfactuals](/ref/explain/counterfactuals)
- [Regression & GLMs](/ref/freq/regression-glms)
- [Resampling](/ref/freq/resampling)

**References:**

- Hernán & Robins, _Causal Inference: What If_ (free online).
- Chernozhukov et al., "Double/debiased machine learning for treatment and structural parameters", Econom. J. (2018).
- Imbens & Rubin, _Causal Inference for Statistics, Social, and Biomedical Sciences_.
