---
title: 'Model selection'
description: 'AIC, AICc, BIC, likelihood-ratio tests for nested models and cross-validation: what each criterion optimises and how to choose between candidate models.'
aliases:
  [
    model selection,
    AIC,
    AICc,
    BIC,
    information criteria,
    likelihood ratio test,
    nested models,
    cross-validation,
    Akaike weights,
    overfitting,
    parsimony,
    Occam's razor,
    model averaging,
    stepwise selection,
    minimum description length,
    MDL,
    Mallows Cp
  ]
tags: [statistics, model selection, information criteria]
order: 8
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *Goal "best prediction":* AIC (AICc for small samples) or cross-validation. *Goal "find the true model" (if one is in the candidate set):* BIC.
- *Nested models:* likelihood-ratio test gives a p-value; AIC/BIC compare nested and non-nested models alike.
- Rules of thumb: $`\Delta\text{AIC} \lt 2`$ indistinguishable, 4–7 considerably less support, $`\gt 10`$ essentially none.
- *Akaike weights* quantify relative support and allow model averaging.
- Information criteria need the *same data* and properly normalised likelihoods (including constants and noise parameters) for all models.

## Decision table

| Criterion | Formula | Optimises | Use when |
| --- | --- | --- | --- |
| AIC | $`2k - 2\ell(\hat\theta)`$ | expected predictive KL | prediction, $`n \gg k`$ |
| AICc | $`\text{AIC} + \frac{2k(k + 1)}{n - k - 1}`$ | as AIC, small-sample corrected | $`n/k \lesssim 40`$ (most ODE fits) |
| BIC | $`k\ln n - 2\ell(\hat\theta)`$ | approx. marginal likelihood | identifying the data-generating model, large $`n`$ |
| Likelihood-ratio test | $`2\Delta\ell \sim\chi^2_{\Delta k}`$ | test nested hypothesis | nested models, inference on extra terms |
| Cross-validation | held-out loss | prediction error directly | any model class, incl. ML; enough data |
| Mallows $`C_p`$ | RSS-based, ≈ AIC for linear models | prediction | linear regression |
| WAIC / PSIS-LOO | Bayesian predictive | prediction | Bayesian models (see [Bayesian model comparison](/ref/bayes/model-comparison)) |
| MDL | code length | compression | information-theoretic framing |

## Variants

### AIC and AICc

- Estimates the expected out-of-sample log-likelihood (relative KL divergence to the truth), penalising each parameter by 1 log-likelihood unit.
- Asymptotically equivalent to leave-one-out CV for regular models.
- AICc corrects the small-sample bias; always use it when $`n/k`$ is small, since it converges to AIC anyway.

### BIC

- $`-\tfrac12\text{BIC}`$ approximates $`\log p(y\mid M)`$ (Laplace approximation with unit-information prior).
- Consistent: selects the true model with probability → 1 if it is among the candidates; penalises complexity more than AIC for $`n \ge 8`$.
- In practice no candidate is "true"; BIC tends to underfit for prediction.

### Akaike weights and averaging

- $`\Delta_i = \text{AIC}_i - \text{AIC}_{\min}`$, weights $`w_i = e^{-\Delta_i/2}/\sum_je^{-\Delta_j/2}`$.
- Model-averaged predictions $`\sum_iw_i\hat y_i`$; unconditional uncertainty includes between-model spread.

### Likelihood-ratio tests

- For nested models $`M_0 \subset M_1`$: $`2[\ell_1 - \ell_0] \sim\chi^2_{k_1 - k_0}`$ under $`M_0`$.
- Boundary nulls (variance components, rates = 0 on the edge of the parameter space) need mixture distributions or parametric bootstrap.

### Cross-validation for selection

- Directly measures the quantity of interest; needs refits; accounts for the whole fitting procedure (including tuning) if done properly (see [resampling](/ref/freq/resampling)).
- *One-standard-error rule:* pick the simplest model within one SE of the best CV score.

## Other lenses

- *Information theory.* AIC is an unbiased estimate of expected KL divergence; MDL and BIC are coding arguments; both trade fit against complexity (see [entropy & KL](/ref/info/entropy-kl-mi)).
- *Bayesian.* BIC ≈ evidence; AIC ≈ LOO predictive. The evidence-vs-prediction distinction explains why AIC and BIC disagree (see [Bayesian model comparison](/ref/bayes/model-comparison)).
- *Regularisation.* Penalised estimation (LASSO, ridge) is continuous model selection; the penalty strength plays the role of the complexity penalty (see [regularisation & generalisation](/ref/learn/regularisation-generalisation)).

## Pitfalls & diagnostics

- *Comparing likelihoods on different data* (different transformations of $`y`$, subsets, missing values) is invalid. Log-transformed models need the Jacobian to compare with untransformed ones.
- *Counting parameters:* include noise parameters and estimated initial conditions; non-identifiable parameters make $`k`$ ambiguous (effective number of parameters).
- *Local optima:* an under-converged complex model looks worse than it is; fit every candidate with multistart.
- *Stepwise selection* and selection followed by inference on the same data inflate significance.
- *Select, then validate:* check the chosen model's predictions on new conditions; criteria only rank the candidates you gave them.

## See also

- [Bayesian model comparison](/ref/bayes/model-comparison)
- [Resampling](/ref/freq/resampling)
- [Hypothesis tests](/ref/freq/hypothesis-tests)
- [Fitting ODE models](/ref/optimise/fitting-odes)

*References:*

- Burnham & Anderson, _Model Selection and Multimodel Inference_ (2nd ed.).
- Akaike, "A new look at the statistical model identification", IEEE TAC (1974).
- Schwarz, "Estimating the dimension of a model", Ann. Stat. (1978).
