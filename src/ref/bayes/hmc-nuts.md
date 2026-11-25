---
title: 'Hamiltonian Monte Carlo & NUTS'
description: 'Leapfrog integration, mass matrix adaptation, step size tuning, the No-U-Turn sampler, divergences and what they tell you about your model.'
aliases:
  [
    HMC,
    Hamiltonian Monte Carlo,
    NUTS,
    No-U-Turn sampler,
    leapfrog,
    mass matrix,
    metric,
    step size adaptation,
    dual averaging,
    divergent transitions,
    energy BFMI,
    Riemannian HMC,
    Stan,
    warmup,
    trajectory length,
    MALA,
    Langevin
  ]
tags: [Bayesian, sampling, MCMC, gradients]
order: 3
updated: '2026-11-17'
published: true
template: ref
---

## TL;DR

- HMC treats $`-\log p(\theta\mid y)`$ as a potential energy, adds random momenta and simulates Hamiltonian dynamics with the *leapfrog* integrator; a Metropolis step corrects discretisation error.
- Distant, low-correlation proposals with high acceptance: efficient in hundreds to thousands of dimensions. Needs *gradients* of the log density (AD, adjoint sensitivities for ODE models).
- *NUTS* picks the trajectory length automatically; warmup adapts the *step size* (dual averaging, target acceptance \~0.8) and a diagonal or dense *mass matrix* (≈ inverse posterior covariance).
- *Divergences* signal regions of high curvature the sampler can't explore (funnels, poorly identified parameters). Fix the model (reparametrise) before raising `adapt_delta`.
- Only for continuous parameters; marginalise discrete ones or combine with Gibbs.

## Decision table

| Setting / diagnostic | Typical value / action | Meaning |
| --- | --- | --- |
| Target acceptance (`adapt_delta`) | 0.8 default; 0.9–0.99 for difficult posteriors | higher = smaller steps, fewer divergences, slower |
| Mass matrix | diagonal (default); dense for strongly correlated, moderate $`d`$ | preconditioning ≈ posterior covariance |
| Max tree depth | 10 (1023 leapfrog steps) | hitting it often = long correlation lengths, poor scaling |
| Divergent transitions | 0 is the goal | biased exploration near high curvature |
| E-BFMI | ≳ 0.3 | low = heavy tails, poor momentum resampling |
| R-hat, ESS | ≲ 1.01, ESS ≳ 400 | as for any MCMC |
| Warmup iterations | \~1000 (default) | adaptation of step size and metric |
| Gradient cost | \~2–5× density evaluation (reverse AD) | dominates runtime with leapfrog steps per iteration |

## Variants

### HMC

1. Draw momentum $`r \sim \mathcal N(0, M)`$.
2. Simulate $`H(\theta, r) = U(\theta) + \tfrac12r^\top M^{-1}r`$ with $`U = -\log p`$ for $`L`$ leapfrog steps of size $`\epsilon`$:
   - $`r \leftarrow r - \tfrac\epsilon2\nabla U(\theta)`$
   - $`\theta \leftarrow \theta + \epsilon M^{-1}r`$
   - $`r \leftarrow r - \tfrac\epsilon2\nabla U(\theta)`$
3. Accept the endpoint with probability $`\min(1, e^{-\Delta H})`$.

- Leapfrog is volume-preserving and reversible, so the acceptance probability only involves the energy error (see [geometric integrators](/ref/integrate/geometric-integrators)).
- Energy error grows with $`\epsilon`$; acceptance collapses if $`\epsilon`$ exceeds the stability limit set by the stiffest posterior direction.

### NUTS

- Build a trajectory by doubling forwards or backwards in time (binary tree) until it starts to turn back on itself (U-turn criterion: momentum points back towards the start), or the energy error is too large.
- Sample a point from the trajectory with multinomial weights $`\propto e^{-H}`$.
- Removes the need to tune $`L`$; cost per iteration adapts to the local geometry.

### Adaptation (warmup)

- *Step size:* dual averaging towards a target acceptance statistic.
- *Mass matrix:* estimated from warmup draws in windows of increasing length; diagonal = per-parameter scales, dense = full covariance ($`O(d^2)`$ memory, $`O(d^3)`$ factorisation).
- Good parametrisation matters more than any tuning: a constant mass matrix cannot fix geometry that varies across the posterior (funnels).

### Divergences and how to fix them

- A divergence = the simulated energy blew up during a trajectory, typically in a region of high curvature.
- *Fixes (in order):*
  1. Reparametrise: non-centred hierarchical models, standardised predictors, log scales (see [priors](/ref/bayes/foundations-priors)).
  2. Stronger priors on poorly identified parameters (e.g. scale parameters near zero).
  3. Increase `adapt_delta` (smaller steps) — helps only for mild cases.
- Plot divergences in pair plots: they cluster where the problem is.

### Related samplers

- *MALA:* a single leapfrog step = Langevin proposal with MH correction.
- *Riemannian HMC:* position-dependent metric (Fisher information) handles varying curvature; expensive (implicit integrators, Hessians).
- *Microcanonical / MCHMC, ChEES, MEADS:* newer tuning-free or GPU-friendly variants for many parallel chains.

## Other lenses

- *Physics.* A frictionless particle on the potential $`U`$; resampling the momentum = coupling to a heat bath. The sampler explores energy level sets, with the momentum refresh moving between them (E-BFMI measures how well).
- *ODEs.* Leapfrog stability limits the step size by the largest eigenvalue of $`M^{-1}\nabla^2U`$: stiffness of the posterior geometry. The mass matrix is a preconditioner (see [integrate ODEs](/ref/integrate/ode-initial-value)).
- *Differentiable programming.* HMC for ODE models needs gradients of the log-likelihood through the solver: forward sensitivities for few parameters, adjoints for many (see [ODE sensitivities](/ref/diff/ode-sensitivities)).

## Pitfalls & diagnostics

- *Ignoring divergences:* results may look fine but be biased.
- *Discontinuous densities* (hard constraints without transforms, piecewise models) break the dynamics; transform constrained parameters to unconstrained space.
- *Multimodality:* HMC doesn't jump between well-separated modes either; use multiple chains to detect it, tempering or SMC to handle it.
- *Gradient errors* (wrong AD, loose ODE tolerances) cause low acceptance and divergences; check gradients and tighten tolerances.
- *Tree depth saturation* suggests extreme correlations or scale differences; reparametrise or use a dense metric.

## See also

- [MCMC](/ref/bayes/mcmc)
- [Geometric integrators](/ref/integrate/geometric-integrators)
- [Bayesian foundations & priors](/ref/bayes/foundations-priors)
- [Automatic differentiation](/ref/diff/automatic-differentiation)
- [Probabilistic programming](/ref/programs/probabilistic-programming)

*References:*

- Neal, "MCMC using Hamiltonian dynamics", in _Handbook of MCMC_ (2011).
- Hoffman & Gelman, "The No-U-Turn sampler", JMLR (2014).
- Betancourt, "A conceptual introduction to Hamiltonian Monte Carlo" (2017).
