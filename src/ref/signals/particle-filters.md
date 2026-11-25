---
title: 'Particle filters'
description: 'Bootstrap and auxiliary particle filters, resampling schemes, weight degeneracy and path degeneracy, proposals, smoothing, and likelihood estimation for parameter inference.'
aliases:
  [
    particle filter,
    sequential importance resampling,
    SIR filter,
    bootstrap filter,
    auxiliary particle filter,
    APF,
    resampling,
    systematic resampling,
    effective sample size,
    weight degeneracy,
    path degeneracy,
    optimal proposal,
    particle smoother,
    forward filtering backward smoothing,
    likelihood estimate,
    Rao-Blackwellized particle filter
  ]
tags: [state estimation, sequential Monte Carlo, filtering]
order: 2
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- Represent the filtering distribution $`p(x_t\mid y_{1:t})`$ by weighted particles; propagate, reweight by the likelihood, resample. Works for *any* nonlinear, non-Gaussian state-space model you can simulate.
- *Bootstrap filter* (propose from the dynamics) is the simple default; better proposals (using the current observation) or the auxiliary PF help when observations are informative.
- *Systematic resampling* when ESS drops below \~$`N/2`$.
- Particle filters scale badly with state dimension (weight degeneracy grows exponentially); for high-dimensional states use EnKF or Rao–Blackwellisation.
- The product of average weights is an *unbiased likelihood estimate*: the engine of particle MCMC for parameter inference.

## Decision table

| Variant | Proposal | Use when |
| --- | --- | --- |
| Bootstrap (SIR) filter | transition density $`p(x_t\mid x_{t-1})`$ | *default*, simulators without tractable densities |
| Optimal / locally optimal proposal | $`p(x_t\mid x_{t-1}, y_t)`$ (or approximation, e.g. EKF/UKF step) | informative observations, low noise |
| Auxiliary particle filter | pre-select particles by predicted likelihood | outliers, informative observations |
| Rao–Blackwellised PF | sample nonlinear part, Kalman filter for linear part | conditionally linear-Gaussian models |
| Particle smoother (FFBSi) | backward simulation | smoothed trajectories, $`O(NT)`$–$`O(N^2T)`$ |
| Fixed-lag smoother | resampled ancestry over a lag | cheap approximate smoothing |
| Guided / twisted PFs | lookahead-informed proposals | very informative data, rare transitions |

## Variants

### Bootstrap particle filter

1. Initialise $`x_0^{(i)} \sim p(x_0)`$, weights $`1/N`$.
2. For each $`t`$:
   1. *Propagate:* $`x_t^{(i)} \sim p(x_t\mid x_{t-1}^{(i)})`$ (simulate the model, e.g. SSA or SDE step).
   2. *Weight:* $`w_t^{(i)} \propto w_{t-1}^{(i)}\,p(y_t\mid x_t^{(i)})`$; normalise.
   3. *Likelihood increment:* $`\hat p(y_t\mid y_{1:t-1}) = \sum_iw_{t-1}^{(i)}p(y_t\mid x_t^{(i)})`$ (with normalised previous weights).
   4. *Resample* if $`\text{ESS} = 1/\sum_i(w_t^{(i)})^2 \lt N/2`$.
3. Filtering estimate: $`\mathbb E[g(x_t)] \approx \sum_iw_t^{(i)}g(x_t^{(i)})`$.

### Resampling schemes

- *Systematic:* one uniform $`u \sim\mathcal U(0, 1/N)`$, pointers $`u + k/N`$ through the cumulative weights. Lowest variance in practice, $`O(N)`$.
- Stratified, residual and multinomial alternatives; avoid multinomial (highest variance).
- Resampling introduces correlation and path degeneracy (all particles share early ancestors).

### Better proposals

- Weight update for proposal $`q`$: $`w \propto w_{\text{prev}}\,\frac{p(y_t\mid x_t)\,p(x_t\mid x_{t-1})}{q(x_t\mid x_{t-1}, y_t)}`$.
- Use an EKF/UKF linearisation per particle as $`q`$ when observations are precise.

### Smoothing

- *Path degeneracy* makes naive trajectory smoothing from the filter poor for early times.
- *Forward filtering backward simulation* (FFBSi) samples trajectories backwards using transition densities; requires $`p(x_{t+1}\mid x_t)`$ to be evaluable.

### Likelihood and parameters

- $`\hat p(y_{1:T}\mid\theta) = \prod_t\hat p(y_t\mid y_{1:t-1})`$ is unbiased; its log has variance growing with $`T/N`$.
- Plug into PMMH (pseudo-marginal MCMC) or use iterated filtering (IF2) for maximum likelihood (see [sequential Monte Carlo](/ref/bayes/sequential-monte-carlo)).

## Other lenses

- *Importance sampling.* Each step is importance sampling with the previous particles as proposal; resampling controls variance (see [variance reduction](/ref/random/variance-reduction)).
- *Kalman filtering.* For linear-Gaussian models the PF converges to the Kalman filter; the EnKF is a particle method with a Gaussian update instead of reweighting (see [Kalman filters](/ref/signals/kalman-filters)).
- *Evolutionary algorithms.* Propagate–select–resample mirrors mutation–selection; particle methods in optimisation (e.g. CEM) share the structure.

## Pitfalls & diagnostics

- *Weight degeneracy:* ESS collapses to \~1 at every step; observations too informative relative to the proposal, or model misspecification (outliers). Use better proposals, heavier-tailed observation models, or more particles.
- *High state dimension:* required $`N`$ grows exponentially; PFs are practical for ≲ 10–20 dimensions without structure.
- *Sample impoverishment* with deterministic dynamics (zero process noise): particles collapse to few distinct values; add jitter or use static-parameter methods.
- *Variance of likelihood estimates* for PMCMC: tune $`N`$ so the log-likelihood standard deviation is \~1–2.
- *Report ESS over time* and check filter estimates against simulated ground truth.

## See also

- [Kalman filters](/ref/signals/kalman-filters)
- [Sequential Monte Carlo](/ref/bayes/sequential-monte-carlo)
- [Stochastic simulation](/ref/integrate/stochastic-simulation)
- [Likelihood-free inference](/ref/bayes/likelihood-free)

*References:*

- Doucet & Johansen, "A tutorial on particle filtering and smoothing: fifteen years later" (2009).
- Chopin & Papaspiliopoulos, _An Introduction to Sequential Monte Carlo_.
- Särkkä, _Bayesian Filtering and Smoothing_ (2nd ed.), ch. 11.
