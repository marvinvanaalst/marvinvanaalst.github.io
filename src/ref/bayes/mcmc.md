---
title: 'Markov chain Monte Carlo'
description: 'Metropolis–Hastings, Gibbs, adaptive MH, slice sampling and ensemble samplers, plus convergence diagnostics (R-hat, ESS, trace plots) and when to use which.'
aliases:
  [
    MCMC,
    Metropolis-Hastings,
    random walk Metropolis,
    Gibbs sampling,
    adaptive Metropolis,
    slice sampling,
    ensemble sampler,
    affine-invariant ensemble,
    emcee,
    differential evolution MCMC,
    MALA,
    parallel tempering,
    replica exchange,
    R-hat,
    effective sample size,
    ESS,
    autocorrelation,
    burn-in,
    warmup,
    thinning,
    trace plot,
    detailed balance
  ]
tags: [Bayesian, sampling, MCMC]
order: 2
updated: '2026-11-17'
published: true
layout: ref
---

## TL;DR

- MCMC builds a Markov chain whose stationary distribution is the posterior; averages over the chain estimate posterior expectations.
- **Gradients available and continuous parameters:** use **HMC/NUTS** (see [HMC & NUTS](/ref/bayes/hmc-nuts)); it scales to hundreds or thousands of dimensions.
- **No gradients, low dimension (≲ 10–20):** adaptive random-walk Metropolis or an affine-invariant ensemble sampler (emcee). Simple, but mixing degrades fast with dimension.
- **Conditionally conjugate models / discrete variables:** Gibbs sampling.
- **Multimodal posteriors:** parallel tempering or SMC (see [sequential Monte Carlo](/ref/bayes/sequential-monte-carlo)).
- **Always:** run ≥ 4 chains from dispersed starts, check **R-hat ≲ 1.01** and **bulk/tail ESS ≳ 400**, look at trace plots.

## Decision table

| Sampler                           | Needs                        | Scaling with dimension $d$                   | Tuning                       | Use when                                       |
| --------------------------------- | ---------------------------- | -------------------------------------------- | ---------------------------- | ---------------------------------------------- |
| Random-walk Metropolis (RWM)      | log density                  | steps $\propto d$ (optimal acceptance 0.234) | proposal covariance          | low $d$, no gradients                          |
| Adaptive Metropolis (Haario)      | log density                  | as RWM, learns covariance                    | automatic                    | correlated low-$d$ posteriors                  |
| Gibbs                             | full conditionals            | depends on correlations                      | none                         | conjugate structure, discrete latents          |
| Slice sampling                    | log density                  | moderate                                     | step width (robust)          | 1D updates inside Gibbs, few parameters        |
| Affine-invariant ensemble (emcee) | log density                  | poor beyond ~20–50                           | number of walkers            | low $d$, badly scaled posteriors, no gradients |
| DE-MC / DREAM                     | log density                  | moderate                                     | population size              | multimodal low-$d$ problems                    |
| MALA                              | gradient                     | steps $\propto d^{1/3}$                      | step size                    | moderate $d$                                   |
| HMC / NUTS                        | gradient                     | steps $\propto d^{1/4}$                      | adaptive (step, mass matrix) | **default** with gradients                     |
| Parallel tempering                | log density (+ gradients)    | several chains                               | temperature ladder           | multimodality                                  |
| Pseudo-marginal / PMCMC           | unbiased likelihood estimate | as underlying sampler                        | number of particles          | intractable likelihoods (stochastic models)    |

## Variants

### Metropolis–Hastings

1. Propose $\theta' \sim q(\theta'\mid\theta)$.
2. Accept with probability $\alpha = \min\Big(1, \dfrac{p(\theta')\,q(\theta\mid\theta')}{p(\theta)\,q(\theta'\mid\theta)}\Big)$ (only the unnormalised posterior is needed).
3. Otherwise stay at $\theta$ (repeat the current value in the chain).

- **Random walk:** $q = \mathcal N(\theta, \lambda^2\Sigma)$ with $\Sigma$ ≈ posterior covariance and $\lambda \approx 2.38/\sqrt d$; target acceptance ~0.23 (0.44 in 1D).
- Work in unconstrained, well-scaled coordinates (log for positive parameters, with Jacobian adjustment of the density).

### Gibbs sampling

- Cycle through blocks, sampling each from its full conditional $p(\theta_j\mid\theta_{-j}, y)$.
- No tuning; very efficient with conjugacy; slow when parameters are strongly correlated (zig-zag moves). Block correlated parameters together.
- Data augmentation (latent variables) can make conditionals conjugate (probit, mixtures, Pólya-gamma for logistic).

### Slice sampling

- Sample uniformly from the region under the density: draw a height $u \sim \mathcal U(0, p(\theta))$, then sample uniformly from the slice $\{\theta : p(\theta) \gt u\}$ via stepping out and shrinkage.
- Self-tuning in 1D; used coordinate-wise or along random directions.

### Ensemble samplers

- Many walkers; the stretch move proposes $\theta' = \theta_j + Z(\theta_k - \theta_j)$ using another walker $k$.
- Affine invariant: unaffected by linear correlations/scaling. Fails in higher dimensions; initialise walkers in a small ball and discard early iterations.

### Parallel tempering

- Run chains at tempered posteriors $p^{1/T_i}$; swap states between neighbouring temperatures with an MH criterion. Hot chains cross between modes, cold chain samples the posterior.

### Diagnostics

- **R-hat (rank-normalised, split):** compares within- and between-chain variance; ≲ 1.01 indicates mixing.
- **Effective sample size:** $N/(1 + 2\sum_k\rho_k)$ from the autocorrelation; report bulk and tail ESS.
- **Monte Carlo standard error** = posterior sd / $\sqrt{\text{ESS}}$: tells how many digits of a posterior mean are meaningful.
- **Trace and rank plots:** chains should overlap and look like "hairy caterpillars".
- **Warmup:** discard the adaptation phase; thinning only saves memory, it never increases ESS per unit of computation.

## Other lenses

- **Markov chains.** Detailed balance $\pi(\theta)K(\theta, \theta') = \pi(\theta')K(\theta', \theta)$ guarantees the target is stationary; ergodicity guarantees convergence; the spectral gap sets mixing speed (see [stochastic processes](/ref/dynamics/stochastic-processes)).
- **Optimisation.** RWM with decreasing temperature becomes simulated annealing; Langevin dynamics is noisy gradient descent; NUTS uses Hamiltonian dynamics (see [geometric integrators](/ref/integrate/geometric-integrators)).
- **Variance reduction.** MCMC estimates are correlated averages; control variates and Rao–Blackwellisation still apply (see [variance reduction](/ref/random/variance-reduction)).

## Pitfalls & diagnostics

- **Single chains hide non-convergence;** multiple dispersed chains are the minimum.
- **Multimodality** can go undetected if all chains start in the same mode.
- **Expensive likelihoods** (ODE solves): budget MCMC as $10^4$–$10^6$ evaluations; use gradients (HMC), surrogates, or SMC to keep it feasible.
- **ODE solver failures** inside the likelihood (stiffness at extreme parameters) should return $-\infty$ log density rather than crash; use stiff solvers.
- **Bad scaling** (parameters differing by orders of magnitude) cripples RWM; reparametrise or adapt the covariance.

## See also

- [HMC & NUTS](/ref/bayes/hmc-nuts)
- [Bayesian foundations & priors](/ref/bayes/foundations-priors)
- [Sequential Monte Carlo](/ref/bayes/sequential-monte-carlo)
- [Stochastic processes](/ref/dynamics/stochastic-processes)

**References:**

- Brooks, Gelman, Jones & Meng (eds.), _Handbook of Markov Chain Monte Carlo_.
- Vehtari et al., "Rank-normalization, folding, and localization: an improved R-hat", Bayesian Analysis (2021).
- Foreman-Mackey et al., "emcee: the MCMC hammer", PASP (2013).
