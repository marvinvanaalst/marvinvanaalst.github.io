---
title: 'Sequential Monte Carlo'
description: 'SMC samplers with tempering, resampling and MCMC moves, evidence estimation for free, and particle MCMC for models with intractable likelihoods.'
aliases:
  [
    SMC,
    sequential Monte Carlo,
    SMC sampler,
    tempering,
    annealed importance sampling,
    AIS,
    likelihood tempering,
    data tempering,
    resampling,
    effective sample size,
    particle MCMC,
    PMCMC,
    particle marginal Metropolis-Hastings,
    PMMH,
    particle Gibbs,
    pseudo-marginal,
    evidence estimation,
    adaptive tempering
  ]
tags: [Bayesian, sampling, SMC, particles]
order: 4
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- *SMC samplers* move a population of weighted particles through a sequence of distributions from the prior to the posterior (tempering $`p(\theta)\,p(y\mid\theta)^{\beta}`$, $`\beta: 0 \to 1`$), with reweighting, resampling and MCMC moves at each stage.
- Strengths: handles *multimodal* posteriors, embarrassingly *parallel* over particles, gives the *evidence* $`p(y)`$ for free, adapts its own schedule.
- *Particle filters* (SMC over time for state-space models) are covered in [particle filters](/ref/signals/particle-filters); their unbiased likelihood estimates power *particle MCMC* for parameters of stochastic dynamic models.
- Cost: number of particles × number of stages × MCMC steps per stage; often comparable to MCMC in total evaluations, but parallel.

## Decision table

| Method | Target | Output | Use when |
| --- | --- | --- | --- |
| SMC sampler, likelihood tempering | static posterior | weighted samples + evidence | multimodal posteriors, model comparison, parallel hardware |
| SMC, data tempering (batch Bayes) | posterior as data arrive | sequence of posteriors | sequential/online updating |
| Annealed importance sampling | evidence | evidence estimate | evidence only, no resampling |
| Particle filter | latent states of a state-space model | filtering distributions, likelihood estimate | time series, hidden Markov / nonlinear SSMs |
| PMMH (particle marginal MH) | parameters of SSMs | posterior over parameters | stochastic dynamic models with intractable likelihood |
| Particle Gibbs (conditional SMC) | parameters and latent paths | joint posterior | high-dimensional latent trajectories |
| SMC² | parameters + states, online | sequential joint posterior | online inference for SSM parameters |

## Variants

### SMC sampler with adaptive tempering

1. Sample $`N`$ particles from the prior, weights $`w_i = 1/N`$, $`\beta = 0`$.
2. Choose the next $`\beta'`$ by bisection so that the ESS of the incremental weights $`p(y\mid\theta_i)^{\beta' - \beta}`$ equals a target (e.g. $`N/2`$).
3. Reweight: $`w_i \leftarrow w_i\,p(y\mid\theta_i)^{\beta' - \beta}`$; accumulate the evidence increment $`\log\sum_iw_i`$ (normalised weights).
4. Resample (systematic) to equal weights.
5. Move each particle with a few MCMC steps targeting the current tempered posterior; tune proposals from the particle cloud (covariance, or HMC).
6. Repeat until $`\beta = 1`$.

- The evidence estimate $`\log p(y) = \sum_{\text{stages}}\log\big(\sum_iw_i^{\text{norm}}\,p(y\mid\theta_i)^{\Delta\beta}\big)`$ is unbiased (on the natural scale).

### Resampling schemes

- *Multinomial* (simple, highest variance), *stratified*, *systematic* (one uniform, lowest variance in practice, default), *residual*.
- Resample only when ESS drops below a threshold to avoid unnecessary loss of diversity.

### Particle MCMC

- A particle filter gives an *unbiased estimate* $`\hat p(y\mid\theta)`$ of the likelihood of a state-space model.
- *PMMH:* run MH on $`\theta`$ using $`\hat p(y\mid\theta)`$ in place of the exact likelihood (pseudo-marginal): still targets the exact posterior. Keep the estimate's log-variance around 1–2 by choosing the number of particles.
- *Particle Gibbs:* alternate a conditional SMC update of the latent path with updates of $`\theta`$ given the path; ancestor sampling improves mixing.

## Other lenses

- *Importance sampling.* Each stage is importance sampling from the previous tempered distribution, kept healthy by small steps; resampling prevents weight degeneracy (see [variance reduction](/ref/random/variance-reduction)).
- *Optimisation.* Tempering is homotopy/continuation in the likelihood strength; a population version of simulated annealing that also returns a proper posterior.
- *Model comparison.* The evidence comes as a by-product, making SMC an alternative to nested sampling and thermodynamic integration (see [model comparison](/ref/bayes/model-comparison)).

## Pitfalls & diagnostics

- *Too few particles* or too few MCMC moves per stage: particle degeneracy, underestimated posterior spread, high-variance evidence. Check by rerunning with different seeds and more particles.
- *Bad mutation kernels* leave duplicated particles after resampling; monitor acceptance rates and the number of unique particles.
- *Evidence variance* grows with the number of stages and the information gained; report its spread across repeated runs.
- *PMMH with noisy likelihood estimates* gets stuck (one lucky high estimate is rarely beaten); increase particles or correlate the random numbers between proposals.

## See also

- [Particle filters](/ref/signals/particle-filters)
- [MCMC](/ref/bayes/mcmc)
- [Model comparison](/ref/bayes/model-comparison)
- [Likelihood-free inference](/ref/bayes/likelihood-free)

*References:*

- Del Moral, Doucet & Jasra, "Sequential Monte Carlo samplers", JRSS B (2006).
- Andrieu, Doucet & Holenstein, "Particle Markov chain Monte Carlo methods", JRSS B (2010).
- Chopin & Papaspiliopoulos, _An Introduction to Sequential Monte Carlo_.
