---
title: 'Likelihood-free inference'
description: 'Approximate Bayesian computation (rejection, SMC-ABC), synthetic likelihood, and neural simulation-based inference (NPE, NLE, NRE) for models you can simulate but not evaluate.'
aliases:
  [
    likelihood-free inference,
    simulation-based inference,
    SBI,
    approximate Bayesian computation,
    ABC,
    rejection ABC,
    ABC-SMC,
    summary statistics,
    synthetic likelihood,
    neural posterior estimation,
    NPE,
    neural likelihood estimation,
    NLE,
    neural ratio estimation,
    NRE,
    amortized inference,
    simulator,
    stochastic models,
    Bayesian optimization for likelihood-free inference
  ]
tags: [Bayesian, simulation-based inference, approximate inference]
order: 6
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- For stochastic simulators (SSA, agent-based models, complex pipelines) where $`p(y\mid\theta)`$ cannot be evaluated but $`y \sim p(\cdot\mid\theta)`$ can be sampled.
- *ABC:* accept parameters whose simulated data are close to the observed data (distance between summary statistics ≤ $`\epsilon`$). Simple, embarrassingly parallel, needs many simulations; results depend on summaries and $`\epsilon`$.
- *ABC-SMC:* sequentially shrink $`\epsilon`$ with populations of particles; far more efficient than rejection.
- *Synthetic likelihood:* assume summaries are Gaussian given $`\theta`$, estimate mean/covariance by simulation, plug into MCMC.
- *Neural SBI* (NPE/NLE/NRE): train neural density estimators or classifiers on simulations; typically 10–100× fewer simulations than ABC, and *amortised* (NPE gives posteriors for new data instantly).
- Validate with *simulation-based calibration* / coverage tests on synthetic data.

## Decision table

| Method | Simulations needed | Summaries | Output | Use when |
| --- | --- | --- | --- | --- |
| Rejection ABC | $`10^5`$–$`10^8`$ | hand-crafted | approximate posterior samples | cheap simulators, few parameters, simple |
| ABC-SMC / ABC-PMC | $`10^4`$–$`10^6`$ | hand-crafted | approximate posterior | default classical ABC |
| ABC-MCMC | many, sequential | hand-crafted | approximate posterior | when proposals can be local |
| Regression-adjusted ABC | as rejection | hand-crafted | corrected samples | improve rejection ABC cheaply |
| Synthetic likelihood | $`10^2`$–$`10^3`$ per MCMC step | approx. Gaussian summaries | MCMC posterior | summaries nearly Gaussian |
| NPE (neural posterior) | $`10^3`$–$`10^5`$ | learned (embedding net) | amortised posterior | many observations, fast repeated inference |
| NLE (neural likelihood) | $`10^3`$–$`10^5`$ | learned / given | likelihood surrogate + MCMC | multiple iid observations, model comparison |
| NRE (neural ratio) | $`10^3`$–$`10^5`$ | learned | likelihood-to-evidence ratio + MCMC | flexible, classifier-based |
| Sequential NPE/NLE | fewer | learned | posterior for one observation | expensive simulators, single dataset |
| Particle MCMC | particles × MCMC steps | none (exact likelihood estimate) | exact posterior | state-space structure available |

## Variants

### Rejection ABC

1. Sample $`\theta_i`$ from the prior.
2. Simulate $`y_i \sim p(\cdot\mid\theta_i)`$; compute summaries $`s_i = S(y_i)`$.
3. Accept $`\theta_i`$ if $`d(s_i, s_{\text{obs}}) \le \epsilon`$ (or keep the closest fraction).

- As $`\epsilon \to 0`$ with sufficient summaries, the accepted samples follow the true posterior; with insufficient summaries, information is lost permanently.
- Scale summaries (e.g. by their prior-predictive MAD) before computing distances.

### ABC-SMC

1. Populations $`t = 1..T`$ with decreasing tolerances $`\epsilon_t`$ (e.g. quantile of previous distances).
2. Sample from the previous weighted population, perturb with a kernel, simulate, accept if within $`\epsilon_t`$.
3. Importance weights correct for the perturbation: $`w \propto p(\theta)/\sum_jw_jK(\theta\mid\theta_j)`$.

### Synthetic likelihood

- At each $`\theta`$, simulate $`n`$ datasets, estimate $`\hat\mu(\theta)`$, $`\hat\Sigma(\theta)`$ of the summaries, use $`\mathcal N(s_{\text{obs}}; \hat\mu, \hat\Sigma)`$ as the likelihood inside MCMC.
- Robust to moderate non-Gaussianity; costly per likelihood evaluation.

### Neural SBI

- *NPE:* train a conditional density estimator $`q_\phi(\theta\mid y)`$ (normalising flow, mixture density network) on pairs $`(\theta_i, y_i)`$ from prior × simulator by maximum likelihood. Amortised: evaluate for any new observation.
- *NLE:* learn $`q_\phi(y\mid\theta)`$, then run MCMC with it as likelihood (handles multiple iid observations naturally).
- *NRE:* train a classifier to distinguish joint pairs $`(\theta, y)`$ from shuffled pairs; its logit estimates $`\log p(y\mid\theta)/p(y)`$.
- *Embedding networks* learn summary statistics from raw data (time series, images) jointly with the estimator.
- *Sequential* variants concentrate simulations near the observation (more efficient, not amortised).

## Other lenses

- *Importance sampling and SMC.* ABC-SMC is SMC with indicator likelihoods; synthetic likelihood and NLE replace the indicator by a smooth surrogate (see [sequential Monte Carlo](/ref/bayes/sequential-monte-carlo)).
- *Surrogates.* GP surrogates of the discrepancy (BOLFI) choose simulations with Bayesian optimisation when each simulation is very expensive (see [surrogates](/ref/approx/surrogates)).
- *Variational inference.* NPE minimises the forward KL averaged over the prior predictive: amortised, mass-covering VI (see [variational inference](/ref/bayes/variational-inference)).

## Pitfalls & diagnostics

- *Summary statistics* decide everything in classical ABC; poor summaries give confident but wrong posteriors. Check posterior predictive fits on raw data.
- *Tolerance too large:* posterior ≈ prior inflated; too small: no acceptances. Plot results against $`\epsilon`$.
- *Model misspecification:* neural SBI can extrapolate badly when the observation is unlike any simulation; check that the observed data lie within the prior predictive distribution.
- *Calibration:* run SBC or coverage tests (expected coverage of credible intervals on synthetic data); neural estimators can be overconfident.
- *Simulation budget:* start with a pilot study to estimate cost and choose the method.

## See also

- [Stochastic simulation](/ref/integrate/stochastic-simulation)
- [Sequential Monte Carlo](/ref/bayes/sequential-monte-carlo)
- [Variational inference](/ref/bayes/variational-inference)
- [Generative models](/ref/learn/generative-models) (normalising flows)

*References:*

- Sisson, Fan & Beaumont (eds.), _Handbook of Approximate Bayesian Computation_.
- Cranmer, Brehmer & Louppe, "The frontier of simulation-based inference", PNAS (2020).
- Toni et al., "Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems", J. R. Soc. Interface (2009).
