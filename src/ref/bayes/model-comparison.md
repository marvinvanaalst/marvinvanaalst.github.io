---
title: 'Bayesian model comparison'
description: 'Marginal likelihood and Bayes factors, how to compute evidence (nested sampling, thermodynamic integration, bridge sampling), and predictive alternatives (WAIC, PSIS-LOO), plus model averaging and stacking.'
aliases:
  [
    model comparison,
    model selection,
    marginal likelihood,
    evidence,
    Bayes factor,
    nested sampling,
    thermodynamic integration,
    path sampling,
    bridge sampling,
    harmonic mean estimator,
    WAIC,
    LOO,
    PSIS-LOO,
    Pareto k,
    cross-validation,
    Bayesian model averaging,
    stacking,
    posterior predictive,
    elpd
  ]
tags: [Bayesian, model selection, evidence]
order: 7
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- Two questions, two tools:
  - *"Which model is more probable given the data?"* → marginal likelihood $`p(y\mid M)`$ and Bayes factors. Sensitive to priors (vague priors penalise a model heavily).
  - *"Which model predicts new data better?"* → expected log predictive density, estimated by *PSIS-LOO* (default) or WAIC. Robust to vague priors, cheap from existing posterior samples.
- *Computing evidence:* nested sampling (gives evidence + posterior), thermodynamic integration or SMC (from tempering), bridge sampling (from MCMC draws). *Never* the harmonic mean estimator.
- *Combining models:* stacking of predictive distributions usually beats Bayesian model averaging when no model is true.

## Decision table

| Method | Estimates | Cost | Use when | Pitfalls |
| --- | --- | --- | --- | --- |
| Laplace / BIC | evidence (approx.) | MAP + Hessian | large data, regular models | poor for small data, boundaries |
| Nested sampling | evidence + posterior | $`\sim`$ live points × information (bits) | low–moderate $`d`$ (≲ 50), multimodal | needs constrained-prior sampling |
| Thermodynamic integration | evidence | MCMC at many temperatures | when you already run tempering | discretisation of the temperature path |
| SMC samplers | evidence (by-product) | SMC run | multimodal, parallel | variance with few particles |
| Bridge sampling | evidence | posterior draws + proposal | after MCMC, moderate $`d`$ | needs a good proposal (e.g. Gaussian fit) |
| Harmonic mean | evidence | free | *never* | infinite variance |
| PSIS-LOO | elpd (predictive) | free from posterior draws | *default predictive comparison* | Pareto-$`\hat k \gt 0.7`$ → refit those points |
| WAIC | elpd (predictive) | free from posterior draws | similar to LOO | less robust diagnostics than PSIS-LOO |
| K-fold CV | elpd | $`K`$ refits | when LOO is unreliable, grouped data | cost |
| Stacking | combination weights | LOO predictive densities | predictive model combination | — |

## Variants

### Marginal likelihood and Bayes factors

- $`p(y\mid M) = \int p(y\mid\theta, M)\,p(\theta\mid M)\,d\theta`$: the prior-weighted average fit. Automatically penalises complexity (Occam factor).
- Bayes factor $`B_{12} = p(y\mid M_1)/p(y\mid M_2)`$; posterior odds = $`B_{12}\times`$ prior odds. Rough scale (Kass–Raftery): $`2\ln B`$ of 2–6 positive, 6–10 strong, $`\gt 10`$ very strong.
- *Prior sensitivity:* widening the prior on a parameter by 10× reduces the evidence by up to \~10×, even though the posterior barely changes.

### Nested sampling

1. Draw $`N`$ live points from the prior.
2. Repeatedly remove the point with the lowest likelihood $`L^*`$; the enclosed prior mass shrinks by \~$`e^{-1/N}`$ per step.
3. Replace it by a new prior sample with $`L \gt L^*`$ (slice sampling, ellipsoidal or region sampling).
4. Evidence $`Z \approx \sum_iL_i\,\Delta X_i`$; stop when the remaining mass can't change $`Z`$ noticeably. Dead points with weights $`L_i\Delta X_i/Z`$ are posterior samples.

- Cost grows with the KL divergence prior → posterior; uncertainty of $`\log Z`$ is $`\approx\sqrt{\text{KL}/N}`$.

### Thermodynamic integration

- $`\log p(y) = \int_0^1\mathbb E_{\beta}[\log p(y\mid\theta)]\,d\beta`$ with power posteriors $`p_\beta \propto p(y\mid\theta)^\beta p(\theta)`$.
- Run MCMC at a ladder of $`\beta`$ (concentrated near 0, e.g. $`\beta_k = (k/K)^5`$), integrate numerically.

### PSIS-LOO

1. From $`S`$ posterior draws, compute pointwise log-likelihoods $`\log p(y_i\mid\theta^s)`$.
2. LOO importance ratios $`r_i^s \propto 1/p(y_i\mid\theta^s)`$; stabilise by fitting a generalised Pareto distribution to the largest ratios (PSIS).
3. $`\widehat{\text{elpd}}_{\text{loo}} = \sum_i\log\frac{\sum_sw_i^sp(y_i\mid\theta^s)}{\sum_sw_i^s}`$, with standard errors from pointwise values.
4. Check Pareto-$`\hat k`$ per point: $`\hat k \gt 0.7`$ means that point's estimate is unreliable (influential observation); refit without it or use K-fold.

- Compare models via differences in elpd and their standard error.

### Model averaging and stacking

- *BMA:* weight predictions by posterior model probabilities; assumes one of the models is true; dominated by the highest-evidence model.
- *Stacking:* choose weights to maximise LOO predictive performance of the mixture; better calibrated when all models are approximations.

## Other lenses

- *Information criteria.* BIC approximates $`-2\log`$ evidence; AIC and WAIC approximate predictive performance; the AIC/BIC disagreement is the evidence/prediction distinction (see [model selection](/ref/freq/model-selection)).
- *Information theory.* Evidence relates to minimum description length: the code length of the data under the model with parameters integrated out. KL from prior to posterior measures information gained (and nested sampling's cost).
- *Physics.* Evidence = partition function; thermodynamic integration and nested sampling come from statistical mechanics (free energy computation).

## Pitfalls & diagnostics

- *Vague priors + Bayes factors* = arbitrary results; use evidence only with meaningful priors.
- *Harmonic mean estimator* has infinite variance in typical cases; don't use it.
- *LOO with time series or grouped data:* use leave-future-out or leave-group-out CV matching the prediction task.
- *Small elpd differences* (within \~2 standard errors) are not meaningful.
- *Evidence uncertainty:* run nested sampling/SMC several times and report the spread.

## See also

- [Bayesian foundations & priors](/ref/bayes/foundations-priors)
- [Sequential Monte Carlo](/ref/bayes/sequential-monte-carlo)
- [Model selection (frequentist)](/ref/freq/model-selection)
- [Resampling & cross-validation](/ref/freq/resampling)

*References:*

- Kass & Raftery, "Bayes factors", JASA (1995).
- Skilling, "Nested sampling for general Bayesian computation", Bayesian Analysis (2006).
- Vehtari, Gelman & Gabry, "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC", Stat. Comput. (2017).
